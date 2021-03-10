import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import mean_squared_error, ssim
from src.data.dataset import SgramDataModule  # type: ignore
from src.modules.inception import ConditionalInceptionLikeEncoder  # type: ignore
from src.utils.model_utils import pairwise_distances, anchor_positive_triplet_mask, anchor_negative_triplet_mask  # type: ignore
from torch.utils.data import dataloader

DEFAULT_HPARAMS = {
    "latent_dim": 256,
    "lr": 0.01,
    "track_reg": 0.5,
    "margin": 0.1,
    "soft": True,
    "checkpoint_path": "model_checkpoints/{epoch}--{val_loss:.2f}"
}


class MusicInception(pl.LightningModule):
    def __init__(
        self, datamodule: SgramDataModule, hparams: dict = DEFAULT_HPARAMS
    ) -> None:
        super().__init__()
        self.hparams = hparams
        self.datamodule = datamodule
        self.weight_decay_steps = 0

        self.spectrogram_shape = (1, 128, 129)  # (channels, height, width)
        self.model = ConditionalInceptionLikeEncoder(latent_dim=self.hparams.latent_dim)
        self.train_dataloader()

    def forward(self, x):
        latent_params = self.encoder(x).view(-1, 2, self.hparams.latent_size)
        mu = latent_params[:, 0, :]
        std = latent_params[:, 1, :]
        latent_sample = self.sample(mu, std)
        generated = self.generator(latent_sample)
        return latent_sample, generated

    def triplet_hard_loss(self, embeddings: torch.Tensor, labels: torch.Tensor, scale):
        """Triplet loss using hard negative and hard positive mining.
        Uses the *unsquared* distances
        See Harmans, et al., https://arxiv.org/pdf/1703.07737.pdf

        Arguments:
            preds {torch.Tensor} -- (batch_size, latent_dim)
            labels {torch.Tensor} -- (batch_size,)
        """
        distances = (
            pairwise_distances(embeddings, squared=False) * scale
        )  # (batch_size, batch_size)

        # For positive distances, set negative to zero
        positive_anchor_mask = anchor_positive_triplet_mask(labels)
        positive_distances = distances.mult(positive_anchor_mask)

        # For each anchor, get the hardest positive
        hardest_positive_dist, _ = positive_distances.max(
            dim=1, keepdim=True
        )  # (batch_size, 1)

        # For negative distances, add the maximum distance per item to each of non-negatives
        negative_anchor_mask = anchor_negative_triplet_mask(labels)
        max_distances, _ = distances.max(dim=1, keepdim=True)  # (batch_size, 1)
        negative_distances = distances + max_distances * (~negative_anchor_mask)

        # For each anchor, get the hardest negative
        hardest_negative_dist, _ = negative_distances.min(
            dim=1, keepdim=True
        )  # (batch_size, 1)

        if self.hparams.soft:
            triplet_loss = torch.log1p(
                torch.exp(hardest_positive_dist - hardest_negative_dist)
            )
        else:
            triplet_loss = (
                hardest_positive_dist - hardest_negative_dist + self.hparams.margin
            ).clamp(0)

        return triplet_loss.mean()

    def training_step(self, batch, batch_idx):
        """Training batch is a 4-tuple of 2-tuples of (sgrams, labels), one for each similarity
        condition.
        """
        assert len(batch) == 4
        loss = 0.0
        for condition in range(len(batch)):
            sgrams, labels = batch[condition]
            assert sgrams.shape[1:] == self.spectrogram_shape  # (N, 1, H, W)
            batch_size = sgrams.shape[0]

            embeddings, masked_embeddings, mask_norm, embeddings_norm = self.model(
                sgrams, torch.tensor(condition, dtype=torch.long)
            )

            scale_masked_norm = 1.0
            conditional_loss = self.triplet_hard_loss(
                masked_embeddings, labels, scale_masked_norm
            )

            scale_track_norm = mask_norm / self.hparams.latent_dim
            track_labels = torch.arange(
                batch_size, dtype=torch.long, device=labels.device
            )
            track_loss = self.triplet_hard_loss(
                embeddings, track_labels, scale_track_norm
            )

            loss += conditional_loss + self.hparams.track_reg * track_loss

        self.log('train_loss', loss)
        self.log('embeddings_norm', embeddings_norm)
        return loss
    
    def on_train_epoch_end(self, outputs) -> None:
        return super().on_train_epoch_end(outputs)
    
    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt = torch.optim.Adam(self.parameters(), lr=lr, betas=(b1, b2))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.2, patience=5, threshold=1e-3, verbose=True
        )
        return {"optimizer": opt, "lr_scheduler": sched}
    
    def configure_callbacks(self):
        early_stop = pl.callbacksEarlyStopping(patience=6, vebose=True)
        checkpoint = pl.callbacks.ModelCheckpoint(self.hparams.checkpoint_path, monitor="val_loss", save_last=True, save_top_k=5)
        lr_monitor = pl.callbacks.LearningRateMonitor()
        return [early_stop, checkpoint, lr_monitor]

    def early_stop_on(self):
        pass

    def train_dataloader(self, *args, **kwargs) -> dataloader:
        return self.datamodule.train_dataloader(*args, **kwargs)

    def val_dataloader(self, *args, **kwargs) -> dataloader:
        return self.datamodule.val_dataloader(*args, **kwargs)

    def test_dataloader(self, *args, **kwargs) -> dataloader:
        return self.datamodule.test_dataloader(*args, **kwargs)
