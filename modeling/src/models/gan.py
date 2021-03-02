import torch
from torch import optim
import torch.nn.functional as F

from warnings import warn
import pytorch_lightning as pl
from src.models.encoder import MobileNetLikeEncoder  # type: ignore
from src.models.generator import SuperResolutionGenerator  # type: ignore
from src.models.discriminator import MulticlassDiscriminator  # type: ignore


DEFAULT_HPARAMS = {"latent_size": 60, "dropout": 0.5, "lr": 3e-4}


class MusicGALI(pl.LightningModule):
    def __init__(self, hparams: dict = DEFAULT_HPARAMS) -> None:
        super().__init__()
        self.hparams = hparams

        self.latent_size = hparams["latent_size"]
        self.dropout = hparams["dropout"]
        self.lr = hparams["lr"]

        self.spectrogram_shape = (1, 128, 624)
        self.encoder = MobileNetLikeEncoder(latent_size=self.latent_size)
        self.generator = SuperResolutionGenerator(latent_size=self.latent_size)
        self.discriminator = MulticlassDiscriminator(
            n_classes=4, latent_size=self.latent_size, dropout_prob=self.dropout
        )

    def forward(self, x):
        warn("Forward called on MusicGALI. Model not intended for inference.")
        return x

    def product_of_terms_loss(self, log_probs: torch.Tensor, true_idx: int):
        """
        Computes partial product of terms loss for a single position.
        Does not perform a reduction (i.e., returns tensor of (batch_size,))
        Product is negated for minimization objective.
        """
        neg_log_probs = log_probs.mul(-1)
        return neg_log_probs[:, :true_idx].sum(dim=1) + neg_log_probs[
            :, (true_idx + 1) :
        ].sum(dim=1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        sgrams, _ = batch
        batch_size = sgrams.shape[0]
        assert sgrams.shape[1:] == self.spectrogram_shape

        """
        Four classes with product of terms objective.
        See Dandi et al., "Generalized Adversarially Learned Inference"
        https://arxiv.org/pdf/2006.08089.pdf
        0: x, E(x)
        1: z, G(z)
        2: x, E(G(E(x)))
        3: G(E(G(z))), z
        """
        logits = [None] * 4
        # Class 0:
        true_latent = self.encoder(sgrams)
        logits[0] = self.discriminator((sgrams, true_latent))
        if optimizer_idx == 0:
            log_probs = F.log_softmax(logits[0], dim=1)
            pt_loss = self.product_of_terms_loss(log_probs, 0)

        # Class 1:
        fake_latent = torch.randn(batch_size, self.latent_size, device=self.device)
        fake_sgrams = self.generator(fake_latent)
        logits[1] = self.discriminator((fake_sgrams, fake_latent))
        if optimizer_idx == 0:
            log_probs = F.log_softmax(logits[1], dim=1)
            pt_loss += self.product_of_terms_loss(log_probs, 1)

        # Class 2:
        true_reencoded_latent = self.encoder(self.generator(true_latent))
        logits[2] = self.discriminator((sgrams, true_reencoded_latent))
        if optimizer_idx == 0:
            log_probs = F.log_softmax(logits[2], dim=1)
            pt_loss += self.product_of_terms_loss(log_probs, 2)

        # Class 3:
        fake_regenerated_sgrams = self.generator(self.encoder(fake_sgrams))
        logits[3] = self.discriminator((fake_regenerated_sgrams, fake_latent))
        if optimizer_idx == 0:
            log_probs = F.log_softmax(logits[3], dim=1)
            pt_loss += self.product_of_terms_loss(log_probs, 3)

            return pt_loss.mean()  # Reduce across batch dimension

        if optimizer_idx == 1:
            d_loss = torch.zeros((batch_size,), dtype=logits[0].dtype, device=self.device)
            for true_idx in range(4):
                target = torch.full(
                    (batch_size,), true_idx, dtype=torch.long, device=self.device
                )
                d_loss += F.cross_entropy(logits[true_idx], target, reduction="none")
            return d_loss.mean()  # Reduce across batch dimension
