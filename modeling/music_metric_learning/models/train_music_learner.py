from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import gc

# Because Colab is still on Python 3.7
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from music_metric_learning.data.dataset import MusicBatchPart, MusicMetricDatamodule
from music_metric_learning.losses.contrastive import SelectivelyContrastiveLoss
from music_metric_learning.modules.embedding import EmbeddingMLP
from music_metric_learning.optimizers.adabound import AdaBound
from music_metric_learning.utils.model_utils import (
    batch_shuffle_single_gpu,
    batch_unshuffle_single_gpu,
    copy_parameters,
    make_encoder,
    visualizer_hook,
)
from omegaconf import OmegaConf
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.optim.lr_scheduler import MultiplicativeLR  # type: ignore
from torch.optim.optimizer import Optimizer
from umap import UMAP
from sklearn.manifold import TSNE

DEFAULT_HPARAMS = {
    "optimizer": "sgd",  # "sgd", "asgd", "adam"
    "learning_rate": 0.001,
    "final_learning_rate": 0.01,
    "momentum": 0.9,
    "batch_size": 512,
    "epoch_length": 100_000,
    "dropout": 0.5,
    "m_per_class": 32,
    "embedding_dim": 256,
    "category_embedding_dim": 32,
    "encoder": "mobilenet",
    "encoder_params": {
        "pretrained": True,
        "freeze_weights": False,
        "max_pool": True,
    },
    "loss_func": "xent_all",  # "selective", "xent_all", "softmax_only"
    "loss_params": {
        "track_loss_alpha": 0.5,
        "hn_lambda": 1.0,
        "temperature": 0.1,
        "hard_margin": 0.8,
    },
    "key_encoder_momentum": 0.99,
    "queue_size": (512 * 70),
    "checkpoint_path": "model_checkpoints",
}


class MusicMetricLearner(pl.LightningModule):
    def __init__(
        self, datamodule: MusicMetricDatamodule, hparams: dict = DEFAULT_HPARAMS
    ):
        super().__init__()
        conf = OmegaConf.create(hparams)

        ### Lightning Config ###
        self.save_hyperparameters(conf)
        self.automatic_optimization = False

        ### Setup Dataset ###
        self.dm: MusicMetricDatamodule = datamodule
        self.dm.epoch_length = self.hparams.epoch_length  # type: ignore
        if not self.dm.is_setup:
            self.dm.setup()
        self.dm.batch_size = cast(int, self.hparams.get("batch_size"))
        self.dm.m_per_class = cast(int, self.hparams.get("m_per_class"))
        assert conf.queue_size % conf.batch_size == 0  # For simplicity

        ### Instantiate Model ###
        self.encoder = make_encoder(
            kind=conf.encoder,
            pretrained=conf.encoder_params.pretrained,
            freeze_weights=conf.encoder_params.freeze_weights,
            max_pool=conf.encoder_params.max_pool,
        )
        self.embedder = EmbeddingMLP(
            category_embedding_dim=conf.category_embedding_dim,
            out_dim=conf.embedding_dim,
            normalize_embeddings=False,  # We'll normalize in the loss function
            dropout=conf.dropout
        )

        ### Instantiate Model Copy for MoCo Track Queue ###
        self.key_encoder = make_encoder(
            kind=conf.encoder,
            pretrained=conf.encoder_params.pretrained,
            freeze_weights=cast(bool, self.hparams.get("freeze_weights")),
            max_pool=conf.encoder_params.max_pool,
        )
        self.key_embedder = EmbeddingMLP(
            category_embedding_dim=conf.category_embedding_dim,
            out_dim=conf.embedding_dim,
            normalize_embeddings=False,
        )
        for model, key_model in (
            (self.encoder, self.key_encoder),
            (self.embedder, self.key_embedder),
        ):
            copy_parameters(model, key_model)

        self.key_encoder_momentum = cast(float, self.hparams.key_encoder_momentum)  # type: ignore

        ### Create the Queue ###
        self.register_buffer("queue", torch.randn(conf.embedding_dim, conf.queue_size))
        self.queue: torch.Tensor
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer(
            "queue_labels", torch.zeros((conf.queue_size,), dtype=torch.long)
        )
        self.queue_labels: torch.Tensor
        self.queue_is_full: bool = False
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_ptr: torch.Tensor

        ### Create Category Label Buffers for the Queue, inited to -1, meaning missing ###
        for i in range(4):
            self.register_buffer(f"queue_{i}_labels", torch.full_like(self.queue_labels, -1))

        ### Instantiate Loss Function ###
        loss_func = self.hparams.loss_func  # type: ignore
        if loss_func in ["selective", "xent", "xent_all", "bce"]:
            xent_only = True if loss_func in ["xent", "xent_all", "bce"] else False
            bce_all = True if loss_func == "bce" else False
            softmax_all = True if loss_func == "xent_all" else False
            self.criterion = SelectivelyContrastiveLoss(
                hn_lambda=cast(float, self.hparams.loss_params.hn_lambda),  # type: ignore
                temperature=cast(float, self.hparams.loss_params.temperature),  # type: ignore
                hard_cutoff=cast(float, self.hparams.loss_params.hard_margin),  # type: ignore
                xent_only=xent_only,
                softmax_all=softmax_all,
                bce_all=bce_all,
            )
        else:
            raise ValueError("Loss function f{loss_func} not implemented")

        ### Add utilities for logging ###
        self.visualizer = UMAP(n_neighbors=10, min_dist=0.2, metric='cosine')
        self.accuracy = AccuracyCalculator()

    def forward_step(
        self, images: torch.Tensor, category_n: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.encoder(images)
        embeddings, _ = self.embedder(encoded, category_n)
        return embeddings

    def track_forward(
        self,
        images: torch.Tensor,
        track_category_n: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images_query, images_key = images[:, 0], images[:, 1]
        query_embeddings = self.forward_step(images_query, track_category_n)
        key_embeddings = self.key_forward_step(images_key, track_category_n)
        return query_embeddings, key_embeddings

    @torch.no_grad()
    def key_forward_step(
        self, key_images: torch.Tensor, category_n: torch.Tensor
    ) -> torch.Tensor:
        """(1) update the parameters of the reference model, (2) shuffle
        the input images along the batch dimension to simulate 
        batch-norm across GPUs, and (3) unshuffle the returned embeddings.
        
        This prevents the model from learning query-key relationships by leaking
        information across the batch dimension through the batch norm parameters.

        Note that this only makes sense for models where we use a replacement for 
        batchnorm that emulates multi-gpu behavior, with separate parameters across
        a split or splits. Not all model architectures have this implemented.
        """
        for query_model, key_model in (
            (self.encoder, self.key_encoder),
            (self.embedder, self.key_embedder),
        ):
            copy_parameters(query_model, key_model, momentum=self.key_encoder_momentum)

        key_images_shuffled, idx_unshuffle = batch_shuffle_single_gpu(key_images)
        key_encoded_shuffled = self.key_encoder(key_images_shuffled)
        key_embeddings_shuffled, _ = self.key_embedder(key_encoded_shuffled, category_n)
        key_embeddings = batch_unshuffle_single_gpu(
            key_embeddings_shuffled, idx_unshuffle
        )
        return key_embeddings
    
    def retrieve_embeddings_labels_from_queue(self, category_idx: Optional[int] = None):
        """Return the correctly-formatted embeddings and labels from the queue
        for downstream evaluation. Pass a category label index to retrieve class labels instead of
        track IDs.
        """
        if category_idx is None:
            label_queue = self.queue_labels
        else:
            label_queue = getattr(self, f"queue_{category_idx}_labels")
        if not self.queue_is_full:
            ptr = cast(int, self.queue_ptr.item())
            key_embeddings_from_queue = self.queue.T[:ptr].clone().detach()
            key_labels_from_queue = label_queue[:ptr].clone().detach()
        else:
            key_embeddings_from_queue = self.queue.T.clone().detach()
            key_labels_from_queue = label_queue.T.clone().detach()
        if category_idx is None:
            return key_embeddings_from_queue, key_labels_from_queue
        else:
            return self.remove_missing_labels_from_queue_segment(key_embeddings_from_queue, key_labels_from_queue)
    
    def remove_missing_labels_from_queue_segment(self, key_embeddings: torch.Tensor, key_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        missing_mask = (key_labels != -1)
        return key_embeddings[missing_mask], key_labels[missing_mask]

    def moco_track_loss(
        self,
        query_embeddings: torch.Tensor,
        track_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Track loss using the queue to supply reference embeddings
        """
        key_embeddings_from_queue, key_labels_from_queue = self.retrieve_embeddings_labels_from_queue()

        # log_callback = self.make_log_callback("train_track") if self.training else None
        log_callback = None

        track_loss, _ = self.criterion.forward(
            query_embeddings,
            track_labels,
            key_embeddings=key_embeddings_from_queue,
            key_labels=key_labels_from_queue,
            log_callback=log_callback,
        )
        return track_loss
    
    def moco_category_loss(
        self,
        query_category_embeddings: torch.Tensor,
        class_labels: torch.Tensor,
        category_idx: int,
    ):
        key_embeddings_for_category, key_labels_for_category = self.retrieve_embeddings_labels_from_queue(category_idx=category_idx)

        # log_callback = self.make_log_callback(f"train_cat{category_idx}") if self.training else None
        log_callback = None

        category_loss, _ = self.criterion.forward(
            query_category_embeddings,
            class_labels,
            key_embeddings=key_embeddings_for_category,
            key_labels=key_labels_for_category,
            log_callback=log_callback
        )

        return category_loss

    @torch.no_grad()
    def dequeue_enqueue(self, new_keys: torch.Tensor, new_labels: torch.Tensor, category_idx: int, new_category_labels: torch.Tensor) -> None:
        """Add new embeddings and labels to the queue, evicting oldest keys
        if the queue is full
        """
        batch_size = new_keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr : ptr + batch_size] = new_keys.T  # store batch-last
        self.queue_labels[ptr : ptr + batch_size] = new_labels

        category_label_queue = getattr(self, f"queue_{category_idx}_labels") 
        category_label_queue[ptr : ptr + batch_size] = new_category_labels
        
        if not self.queue_is_full and ptr + batch_size >= self.hparams.queue_size:  # type: ignore
            self.queue_is_full = True
        ptr = (ptr + batch_size) % cast(int, self.hparams.queue_size)  # type: ignore
        self.queue_ptr[0] = ptr  # move pointer
        return None

    def validation_track_loss(
        self, images, track_labels, track_category_n
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Don't use this during training, because it doesn't add the new keys to the queue
        before calculating the loss.
        """
        assert not self.training
        assert images.shape[2] == 1 and len(images.shape) == 5
        query_embeddings, _ = self.track_forward(
            images=images, track_category_n=track_category_n
        )
        track_loss = self.moco_track_loss(
            query_embeddings=query_embeddings,
            track_labels=track_labels,
        )
        return track_loss, query_embeddings

    def category_loss(
        self, images: torch.Tensor, class_labels: torch.Tensor, category_n: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the loss and embeddings for a given category.
        """
        assert images.shape[1] == 1 and len(images.shape) == 4
        category_embeddings = self.forward_step(images=images, category_n=category_n)
        category_idx = cast(int, category_n[0].item())
        category_loss = self.moco_category_loss(
            query_category_embeddings=category_embeddings,
            class_labels=class_labels,
            category_idx=category_idx
        )
        return category_loss, category_embeddings

    def get_label_maps(
        self, stage: Literal["train", "val", "test"]
    ) -> List[Dict[int, str]]:
        """Helper function to retrieve names of classes to make logged values
        more easily interpretable.
        """
        if stage == "val":
            loaders = self.val_dataloader()
        elif stage == "test":
            loaders = self.test_dataloader()
        elif stage == "train":
            loaders = self.train_dataloader()
        label_maps: List[Dict[int, str]] = []
        for loader in loaders:
            label_maps.append(loader.dataset.class_label_map)
        return label_maps

    def make_log_callback(
        self,
        prefix: str,
        also_log: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """Returns a function that, when called, sends key value pairs to the logger,
        with the key prefixed with the supplied prefix. Key-value pairs in `also_log`
        will be logged simultaneously with the other values (for instance, you can supply
        step, category, or epoch information here.)
        """
        def log_callback(
            logging_dict: Dict[str, Union[float, int, torch.Tensor, np.ndarray]],
            do_print=False,
        ):
            """Pass to loss functions, datasets, etc., to log arbitrary values using available logger
            """
            to_log = {f"{prefix}_{key}": value for key, value in logging_dict.items()}
            if also_log is not None:
                to_log.update(also_log)
            if hasattr(self.logger.experiment, "log"):
                self.logger.experiment.log(to_log)
            else:
                for key, value in to_log.items():
                    self.log(key, value)
            if do_print:
                for key, value in to_log.items():
                    self.print(f"{key}: {value}")

        return log_callback

    @torch.no_grad()
    def evaluate_and_log(
        self,
        stage: Literal["train", "val", "test"],
        batch_part: Optional[MusicBatchPart] = None,
        visualize: bool = False,
        compute_accuracies: bool = True,
        batch_idx: Optional[int] = None,
        track_loss: Optional[torch.Tensor] = None,
        track_embeddings: Optional[torch.Tensor] = None,
        category_loss: Optional[torch.Tensor] = None,
        category_embeddings: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None,
    ):
        """Prepare and send values to the logger. With the exception of "stage", every argument
        is optional, but some arguments must be passed together to have any effect.

        Arguments:
            stage {"train", "val", "test"} -- Used as prefix to logging keys

        Keyword Arguments:
            batch_part {Optional[MusicBatchPart]} -- Training input; optionally, concat
            multiple batches to log a larger set. (default: {None})
            visualize {bool} -- Whether to log a visulization of the embeddings (default: {False})
            compute_accuracies {bool -- Whether compute and log accuracy metrics (default: {True})
            batch_idx {Optional[int]} -- Current batch index (default: {None})
            track_loss {Optional[torch.Tensor]} -- Loss just for track embeddings (default: {None})
            track_embeddings {Optional[torch.Tensor]} -- The actual track embeddings (default: {None})
            category_loss {Optional[torch.Tensor]} -- Loss just for category embeddings (default: {None})
            category_embeddings {Optional[torch.Tensor]} -- The actual category embeddings (default: {None})
            loss {Optional[torch.Tensor]} -- The final value used for optimization (default: {None})
        """
        if track_loss is not None:
            self.log(f"{stage}_track_loss", track_loss)
        if category_loss is not None:
            self.log(f"{stage}_category_loss", category_loss)
        if loss is not None:
            self.log(f"{stage}_loss", loss, prog_bar=True)
        if batch_idx is not None:
            self.log(f"{stage}_batch_idx", batch_idx)
        if all((compute_accuracies, track_embeddings is not None, batch_part)):
            assert batch_part is not None
            assert track_embeddings is not None
            track_labels = batch_part["track_labels"]
            normalized = F.normalize(track_embeddings, p=2, dim=1).clone().contiguous()
            if stage == "train":
                key_embeddings, key_labels = self.retrieve_embeddings_labels_from_queue()
                accuracy =self.accuracy.get_accuracy(
                    normalized,
                    F.normalize(key_embeddings, p=2, dim=1).clone().contiguous(),
                    track_labels,
                    key_labels,
                    embeddings_come_from_same_source=False,
                )
            else:
                accuracy = self.accuracy.get_accuracy(
                    normalized,
                    normalized,
                    track_labels,
                    track_labels,
                    embeddings_come_from_same_source=True,
                )
            accuracy_log = {f"{stage}_track_{k}": v for k, v in accuracy.items()}
            if hasattr(self.logger.experiment, "log"):
                self.logger.experiment.log(accuracy_log)
        if all((compute_accuracies, category_embeddings is not None, batch_part)):
            assert batch_part is not None
            assert category_embeddings is not None
            category = cast(int, batch_part["category_n"][0].item())
            class_labels = batch_part["class_labels"]
            normalized = F.normalize(category_embeddings, p=2, dim=1).clone().contiguous()
            accuracy = self.accuracy.get_accuracy(
                normalized,
                normalized,
                class_labels,
                class_labels,
                embeddings_come_from_same_source=True,
            )
            accuracy_log = {
                f"{stage}_cat{category}_{k}": v for k, v in accuracy.items()
            }
            if hasattr(self.logger.experiment, "log"):
                self.logger.experiment.log(accuracy_log)
        if all((visualize, category_embeddings is not None, batch_part)):
            assert batch_part is not None
            assert category_embeddings is not None
            labels = batch_part["class_labels"].clone().detach()
            category = cast(int, batch_part["category_n"][0].clone().detach().item())
            label_map = self.get_label_maps(stage)[category]
            visualization = visualizer_hook(
                visualizer=self.visualizer,
                embeddings=F.normalize(category_embeddings),
                labels=labels,
                label_map=label_map,
                split_name=stage,
                show_plot=False,
            )
            if hasattr(self.logger.experiment, "log"):
                self.logger.experiment.log({f"{stage}_{category}": visualization})
        return None

    def training_step(self, batch, batch_idx):
        self.train()
        total_loss = 0
        opt = cast(Optimizer, self.optimizers())
        batch_part: MusicBatchPart
        for batch_part in batch:

            def closure():
                opt.zero_grad()
                images: torch.Tensor = batch_part["images"]
                category_n: torch.Tensor = batch_part["category_n"]
                category_idx: int = category_n[0].item()
                assert torch.all(
                    category_n == category_idx
                )  # Mixed-category minibatches won't work
                class_labels: torch.Tensor = batch_part["class_labels"]
                track_category_n: torch.Tensor = batch_part["track_category_n"]
                track_labels: torch.Tensor = batch_part["track_labels"]

                query_embeddings, key_embeddings = self.track_forward(
                    images=images, track_category_n=track_category_n
                )
                self.dequeue_enqueue(
                    new_keys=key_embeddings, new_labels=track_labels, category_idx=category_idx, new_category_labels=class_labels,
                )
                track_loss = self.moco_track_loss(
                    query_embeddings=query_embeddings,
                    track_labels=track_labels,
                )

                first_images = images[:, 0]
                category_loss, category_embeddings = self.category_loss(
                    images=first_images,
                    class_labels=class_labels,
                    category_n=category_n,
                )
                loss: torch.Tensor = category_loss + (self.hparams.loss_params.track_loss_alpha * track_loss)  # type: ignore

                self.manual_backward(loss, opt)

                self.evaluate_and_log(
                    stage="train",
                    compute_accuracies=False,  # This is very slow, but provides useful info while tuning hyperparameters
                    visualize=False,
                    batch_part=batch_part,
                    batch_idx=batch_idx,
                    track_loss=track_loss,
                    track_embeddings=query_embeddings,
                    category_loss=category_loss,
                    category_embeddings=category_embeddings,
                    loss=None,  # Lightning doesn't like logging something called "loss" more than once per (its concept of) step
                )

                return loss

            loss = closure()
            opt.step()
            total_loss += loss.clone().detach().item()
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)

    def reset_saved_batches(self) -> None:
        self.saved_batches: Dict[int, List[MusicBatchPart]] = {k: [] for k in range(4)}
        self.saved_category_embeddings: Dict[int, List[torch.Tensor]] = {
            k: [] for k in range(len(self.saved_batches))
        }
        self.saved_track_embeddings: Dict[int, List[torch.Tensor]] = {
            k: [] for k in range(len(self.saved_batches))
        }
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def on_validation_epoch_start(self) -> None:
        self.reset_saved_batches()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        batch_part = cast(MusicBatchPart, batch)
        track_loss, track_embeddings = self.validation_track_loss(
            images=batch_part["images"],
            track_labels=batch_part["track_labels"],
            track_category_n=batch_part["track_category_n"],
        )
        category_loss, category_embeddings = self.category_loss(
            images=batch_part["images"][:, 0],
            class_labels=batch_part["class_labels"],
            category_n=batch_part["category_n"],
        )
        loss = category_loss + (
            self.hparams.loss_params.track_loss_alpha * track_loss
        )

        self.log("val_loss", loss, on_epoch=True)

        self.saved_batches[dataloader_idx].append(batch_part)
        self.saved_category_embeddings[dataloader_idx].append(category_embeddings)
        self.saved_track_embeddings[dataloader_idx].append(track_embeddings)

    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        for category in range(len(self.saved_batches)):
            accumulated_batches: MusicBatchPart = {
                k: torch.cat(
                    [batch_part[k] for batch_part in self.saved_batches[category]]
                )
                for k in self.saved_batches[category][0]
            }
            # Pytorch doesn't free tensors with 0 references in containers; you have to clear them manually
            for batch_part in self.saved_batches[category]:
                for _, v in batch_part.items():
                    del v
            accumulated_category_embeddings: torch.Tensor
            accumulated_category_embeddings = torch.cat(
                self.saved_category_embeddings[category]
            )
            for category_embedding in self.saved_category_embeddings[category]:
                del category_embedding
            accumulated_track_embeddings: torch.Tensor
            accumulated_track_embeddings = torch.cat(
                self.saved_track_embeddings[category]
            )
            for track_embedding in self.saved_track_embeddings[category]:
                del track_embedding
            self.evaluate_and_log(
                stage="val",
                batch_part=accumulated_batches,
                compute_accuracies=True,
                visualize=True,
                category_embeddings=accumulated_category_embeddings,
                track_embeddings=accumulated_track_embeddings,
            )
            for _, v in accumulated_batches.items():
                del v
            del accumulated_batches
            del accumulated_category_embeddings
            del accumulated_track_embeddings
            gc.collect()

        self.reset_saved_batches()

    def configure_callbacks(self):
        lr_monitor = pl.callbacks.LearningRateMonitor()
        checkpoint = pl.callbacks.ModelCheckpoint(
            self.hparams.checkpoint_path,
            monitor="train_loss_step",
            save_last=True,
            save_top_k=8,
        )
        return [checkpoint, lr_monitor]

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        final_lr = self.hparams.final_learning_rate
        momentum = self.hparams.momentum
        if self.hparams.optimizer == "adabound":
            opt = AdaBound(self.parameters(), lr=lr, final_lr=final_lr)
        elif self.hparams.optimizer == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=lr)
        elif self.hparams.optimizer == "sgd":
            opt = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, nesterov=True)
        elif self.hparams.optimizer == "asgd":
            opt = torch.optim.ASGD(self.parameters(), lr=lr)
        else:
            raise ValueError("optimizer not implemented")

        def lr_lambda(epoch: int):
            return 0.1 if epoch in (50, 85) else 1

        sched = MultiplicativeLR(opt, lr_lambda)
        return [opt], [sched]

    def train_dataloader(self, *args, **kwargs):
        loaders = [self.dm.train_dataloader(i, *args, **kwargs) for i in range(4)]
        return loaders

    def val_dataloader(self, *args, **kwargs):
        loaders = [self.dm.val_dataloader(i, *args, **kwargs) for i in range(4)]
        return loaders

    def test_dataloader(self, *args, **kwargs):
        loaders = [self.dm.test_dataloader(i, *args, **kwargs) for i in range(4)]
        return loaders
