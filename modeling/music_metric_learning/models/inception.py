from typing import Dict, List, Tuple
from warnings import filterwarnings

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from numpy.core.numeric import full
from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.losses import NTXentLoss, ProxyAnchorLoss
from pytorch_metric_learning.miners.batch_easy_hard_miner import BatchEasyHardMiner
from pytorch_metric_learning.testers import GlobalEmbeddingSpaceTester
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from music_metric_learningdata.dataset import SgramDataModule  # type: ignore
from music_metric_learningmodules.inception import Encoder as ConditionalInceptionLikeEncoder # type: ignore
from music_metric_learningutils.dev_utils import delete  # type: ignore
from music_metric_learningutils.model_utils import (  # type: ignore
    histogram_from_weights,
    visualizer_hook,
)
from umap import UMAP
from pytorch_lightning.utilities import AMPType

filterwarnings("ignore", category=UserWarning)


DEFAULT_HPARAMS = {
    "latent_dim": 256,
    "batch_size": 128,
    "lr": 0.004,
    "b1": 0.88,
    "b2": 0.995,
    "npairs_temperature": 0.8,
    "track_reg": 20.,
    "normalize_embeddings": True,
    "margin": 0.1,
    "embeddings_l2_lambda": 5e-3,
    "mask_l1_lambda": 5e-4,
    "checkpoint_path": "model_checkpoints/train--{epoch}--{train_loss}",
}


def data_and_label_getter(batch):
    return batch[0], batch[1]


class MusicInception(pl.LightningModule):
    def __init__(
        self, datamodule: SgramDataModule = None, hparams: dict = DEFAULT_HPARAMS
    ) -> None:
        super().__init__()
        self.hparams = hparams  # type: ignore
        if datamodule is None:
            raise ValueError("Datamodule cannot be none")
        self.dm = datamodule  # Don't call this self.datamodule b/c pytorch lightning looks for this first and we want to hide it from the trainer
        self.automatic_optimization = False
        self.weight_decay_steps = 0
        self.dm.batch_size = self.hparams.batch_size  # type: ignore
        self.spectrogram_shape = (1, 128, 130)  # (channels, height, width)
        self.model = ConditionalInceptionLikeEncoder(
            latent_dim=self.hparams.latent_dim,  # type: ignore
            normalize_embeddings=self.hparams.normalize_embeddings,  # type: ignore
        )
        self.make_category_losses()
        self.track_miner = BatchEasyHardMiner(
            distance=DotProductSimilarity(), pos_strategy="all", neg_strategy="hard"
        )
        self.track_loss_func = NTXentLoss(distance=DotProductSimilarity(), temperature=self.hparams.npairs_temperature)  # type: ignore

        self.tester = GlobalEmbeddingSpaceTester(
            data_and_label_getter=data_and_label_getter,
            dataloader_num_workers=2,
            visualizer=UMAP(),
            visualizer_hook=visualizer_hook,
            accuracy_calculator=AccuracyCalculator(),
        )
        self.save_hyperparameters()

    def make_category_losses(self):
        for n, loader in enumerate(self.train_dataloader()):
            setattr(
                self,
                f"loss_func_{n}",
                ProxyAnchorLoss(
                    num_classes=loader.dataset.n_classes,
                    embedding_size=self.hparams.latent_dim,
                    margin=self.hparams.margin,
                    alpha=32,
                ),
            )

    def forward(self, x):
        return self.model(x, torch.tensor(4, dtype=torch.long))

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Training batch is a tuple of 3-tuples of (sgrams, labels, matched_sgrams), one for
        each of four similarity conditions

        matched_sgrams should be `batch_size, 2, 1, H, W` for Siamese regularization
        """

        model_opt, loss_opt = self.optimizers()


        model_opt.zero_grad()
        loss_opt.zero_grad()

        full_loss = 0
        
        assert optimizer_idx != 1  # optimizer_idx shouldn't do anything

        for condition in range(len(batch)):
            """ Use a closure to free tensors between conditions """
            #  @debug_cuda
            def handle_condition():

                nonlocal full_loss
                conditional_sgrams, conditional_labels, track_sgrams = batch[condition]
                assert (
                    conditional_sgrams.shape[1:] == self.spectrogram_shape
                )  # (N, 1, H, W)
                batch_size = conditional_sgrams.shape[0]
                

                # Conditional proxy loss
                conditional_mask_in = torch.zeros(
                    (batch_size, len(batch)),
                    dtype=conditional_sgrams.dtype,
                    device=conditional_sgrams.device,
                )
                conditional_mask_in[:, condition] = 1.0
                conditional_embeddings, embeddings_norm = self.model(
                    conditional_sgrams #, conditional_mask_in
                )
                conditional_loss = getattr(self, f"loss_func_{condition}")(
                    conditional_embeddings, conditional_labels
                )

                if hasattr(self.logger.experiment, "log"):
                    self.logger.experiment.log(
                        {
                            f"cond_{condition}_loss": conditional_loss,
                            f"cond_{condition}_embed_l2": embeddings_norm,
                       #     f"cond_{condition}_mask_l1": mask_weight_norm,
                        },
                    )

                # Track loss
                track_labels = torch.tensor(
                    list(zip(range(batch_size), range(batch_size))),
                    dtype=conditional_labels.dtype
                ).flatten()

                track_mask_in = torch.full(
                    (batch_size * 2, len(batch)),
                    1 / len(batch),
                    device=track_sgrams.device,
                    dtype=track_sgrams.dtype,
                )

                track_embeddings, embeddings_norm = self.model(
                    track_sgrams.flatten(0, 1)#, track_mask_in
                )

                hard_triplets = self.track_miner(track_embeddings, track_labels)
                track_loss = self.track_loss_func(
                    track_embeddings, track_labels, hard_triplets
                ) * (self.hparams.track_reg)
                loss = (
                    conditional_loss
              #      + self.hparams.mask_l1_lambda * mask_weight_norm
                    + track_loss
                    + embeddings_norm.mean() * self.hparams.embeddings_l2_lambda
                ) / 100  # for 16-bit training
                with autocast(enabled=False):
                    # self.manual_backward(loss, loss_opt, retain_graph=True)
                    self.manual_backward(loss)
                    
                if hasattr(self.logger.experiment, "log"):
                     self.logger.experiment.log(
                        {
                            f"track_{condition}_loss": track_loss,
                            f"track_{condition}_l2": embeddings_norm.mean(),
                        }
                    )
                full_loss += track_loss.item()
                
            handle_condition()
            with autocast(enabled=False):
                model_opt.step()
                loss_opt.step()
            # TODO: in manual optimization mode, do the callbacks even run?
        self.log("train_loss", full_loss)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        with torch.no_grad():
            full_loss = 0
            conditional_sgrams, conditional_labels, track_sgrams = batch
            assert (
                conditional_sgrams.shape[1:] == self.spectrogram_shape
            )  # (N, 1, H, W)
            batch_size = conditional_sgrams.shape[0]

            # Conditional proxy loss
            conditional_mask_in = torch.zeros(
                (batch_size, 4),
                dtype=conditional_sgrams.dtype,
                device=conditional_sgrams.device,
            )
            conditional_mask_in[:, dataloader_idx] = 1.0
            conditional_embeddings, embeddings_norm, mask_weight_norm = self.model(
                conditional_sgrams, conditional_mask_in
            )
            conditional_loss = getattr(self, f"loss_func_{dataloader_idx}")(
                conditional_embeddings, conditional_labels
            )
            loss = (
                conditional_loss
                + self.hparams.embeddings_l2_lambda * embeddings_norm
                + self.hparams.mask_l1_lambda * mask_weight_norm
            )


            if hasattr(self.logger.experiment, "log"):            
                self.logger.experiment.log(
                    {
                        f"cond_{dataloader_idx}_val_loss": conditional_loss,
                        f"cond_{dataloader_idx}_val_embed_l2": embeddings_norm,
                        f"cond_{dataloader_idx}_val_mask_l1": mask_weight_norm,
                    },
                )
            full_loss += loss.item()
            delete(
                loss,
                conditional_loss,
                conditional_embeddings,
                embeddings_norm,
                mask_weight_norm,
            )

            # Track loss
            track_labels = torch.tensor(
                list(zip(range(batch_size), range(batch_size))),
                dtype=conditional_labels.dtype,
                requires_grad=False,
            ).flatten()

            track_mask_in = torch.full(
                (batch_size * 2, 4),
                1 / 4,
                device=track_sgrams.device,
                dtype=track_sgrams.dtype,
            )
            delete(conditional_mask_in)

            track_embeddings, embeddings_norm, mask_weight_norm = self.model(
                track_sgrams.flatten(0, 1), track_mask_in
            )

            hard_triplets = self.track_miner(track_embeddings, track_labels)
            track_loss = self.track_loss_func(
                track_embeddings, track_labels, hard_triplets
            ) * (self.hparams.track_reg)
            loss = track_loss + embeddings_norm * self.hparams.embeddings_l2_lambda
            self.log(
                f"track_{dataloader_idx}_val_loss",
                track_loss,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                f"track_{dataloader_idx}_val_l2",
                embeddings_norm,
                on_step=True,
                on_epoch=True,
            )
            full_loss += track_loss.item()

            self.log("val_loss", full_loss)

    def on_validation_end(self, *args, **kwargs) -> None:  # type: ignore
        with torch.no_grad():
            self.model.eval()
            embeddings, labels, label_maps = self.get_all_embeddings()
            visualizations = {}
            accuracies = []
            for category, (embedding, label, label_map) in enumerate(
                zip(embeddings, labels, label_maps)
            ):
                visualizations[f"val_proj_{category}"] = visualizer_hook(
                    self.tester.visualizer,
                    embedding,
                    label,
                    label_map,
                    split_name="val",
                    show_plot=False,
                )
                accuracy = self.tester.accuracy_calculator.get_accuracy(
                    embedding,
                    embedding,
                    label,
                    label,
                    embeddings_come_from_same_source=True,
                )
                accuracies.append({k + f"_{category}": v for k, v in accuracy.items()})
            accuracies_log = {
                k: v for dct in accuracies for k, v in dct.items()
            }  # type ignore
            if hasattr(self.logger.experiment, "log"):
                self.logger.experiment.log(visualizations)
                self.logger.experiment.log(accuracies_log)
                self.logger.experiment.log(
                    histogram_from_weights(self.model.masks.weight)
                )
            self.model.train()

    def get_all_embeddings(
        self, stage: str = "val"
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[int, str]]]:
        if stage == "val":
            loaders = self.val_dataloader()
        elif stage == "test":
            loaders = self.test_dataloader()
        else:
            loaders = self.train_dataloader()
        embeddings = []
        labels = []
        label_maps = []
        batch_size = loaders[0].batch_size
        mask = torch.full(
            (batch_size, len(loaders)),
            1 / len(loaders),
            device=self.model.masks.weight.device,
        )
        for loader in loaders:
            embed, lab = self.tester.compute_all_embeddings(
                loader, lambda x: x, lambda x: self.model(x, mask)[0]
            )
            embeddings.append(embed)
            labels.append(lab.squeeze().squeeze())
            label_maps.append(loader.dataset.label_map)
        return embeddings, labels, label_maps

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        model_opt = torch.optim.AdamW(
            self.model.parameters(), lr=lr, betas=(b1, b2), amsgrad=True
        )
        sched = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                model_opt, factor=0.2, patience=5, threshold=1e-3, verbose=True
            ),
            "monitor": "train_loss",
        }
        loss_params = []
        for n in range(len(self.train_dataloader())):
            loss_params += list(getattr(self, f"loss_func_{n}").parameters())
        loss_opt = torch.optim.SGD(
            loss_params,
            lr=self.hparams.lr
            )
        return [
            {
                "optimizer": model_opt,
                "lr_scheduler": sched,
            },
            {"optimizer": loss_opt},
        ]

    def configure_callbacks(self):
        early_stop = pl.callbacks.EarlyStopping("train_loss", patience=6, verbose=True)
        checkpoint = pl.callbacks.ModelCheckpoint(
            self.hparams.checkpoint_path,
            monitor="train_loss",
            save_last=True,
            save_top_k=5,
        )
        lr_monitor = pl.callbacks.LearningRateMonitor()
        return [early_stop, checkpoint, lr_monitor]

    def train_dataloader(self, *args, **kwargs):
        loaders = [self.dm.train_dataloader(i, *args, **kwargs) for i in range(4)]
        return loaders

    def val_dataloader(self, *args, **kwargs):
        loaders = [self.dm.val_dataloader(i, *args, **kwargs) for i in range(4)]
        return loaders

    def test_dataloader(self, *args, **kwargs):
        loaders = [self.dm.test_dataloader(i, *args, **kwargs) for i in range(4)]
        return loaders
