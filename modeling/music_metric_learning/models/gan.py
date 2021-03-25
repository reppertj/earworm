import torch
from torch import optim
import torch.nn.functional as F

from warnings import warn
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import mean_squared_error, ssim
from torch.utils.data import dataloader
from music_metric_learningmodels.encoder import MobileNetLikeEncoder  # type: ignore
from music_metric_learningmodels.generator import DCGANLikeGenerator  # type: ignore
from music_metric_learningmodels.discriminator import MulticlassDiscriminator  # type: ignore
from music_metric_learningdata.dataset import SgramDataModule  # type: ignore


DEFAULT_HPARAMS = {
    "latent_size": 64,
    "dropout": 0.5,
    "lr": 3e-4,
    "b1": 0.9,
    "b2": 0.999,
    "noise_variance": 0.2,
    "noise_decay_C": -5.0,
    "label_smoothing_epsilon": 0.1,
}


class MusicGALI(pl.LightningModule):
    def __init__(
        self, datamodule: SgramDataModule, hparams: dict = DEFAULT_HPARAMS
    ) -> None:
        super().__init__()
        self.hparams = hparams
        self.datamodule = datamodule
        self.weight_decay_steps = 0

        self.spectrogram_shape = (1, 128, 624)  # (channels, height, width)
        self.encoder = MobileNetLikeEncoder(latent_size=self.hparams.latent_size * 2)
        self.generator = DCGANLikeGenerator(latent_size=self.hparams.latent_size)
        self.discriminator = MulticlassDiscriminator(
            n_classes=4,
            latent_size=self.hparams.latent_size,
            dropout_prob=self.hparams.dropout,
        )
        self.train_dataloader()

    def forward(self, x):
        latent_params = self.encoder(x).view(-1, 2, self.hparams.latent_size)
        mu = latent_params[:, 0, :]
        std = latent_params[:, 1, :]
        latent_sample = self.sample(mu, std)
        generated = self.generator(latent_sample)        
        return latent_sample, generated

    def product_of_terms_loss(self, log_probs: torch.Tensor, true_idx: int):
        """
        Computes partial product of terms loss for a single position.
        Does not perform a reduction (i.e., returns tensor of (batch_size,))
        """
        return log_probs[:, :true_idx].sum(dim=1) + log_probs[
            :, (true_idx + 1) :
        ].sum(dim=1)

    def sample(self, mu: torch.Tensor, log_var: torch.Tensor):
        """Use the reparameterization trick to separate the source of randomness
        from the latent variables, to allow efficient backprop through random
        samples from the latent embeddings.
        See Domoulin, et al., "Adversarially Learned Inference"
        https://arxiv.org/pdf/1606.00704.pdf

        Arguments:
            mu {torch.Tensor} -- means from the encoder's latent space (batch_size, latent_dim)
            log_var {torch.Tensor} -- log variances from the encoder's latent space (batch_size, latent_dim)

        Returns:
            torch.tensor -- random samples in the latent space
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    def add_gaussian_noise(self, sgram: torch.Tensor):
        noise = torch.rand_like(sgram) * (self.hparams.noise_variance ** (0.5))
        noise = noise.mul(
            torch.exp(
                torch.tensor(self.weight_decay_steps / self.hparams.noise_decay_C)
            )
        )
        if self.training:
            self.weight_decay_steps += 1
        return sgram.add(noise)

    def smoothing_cross_entropy_loss(self, log_probs, y):
        """Cross entropy loss with true label sampled
        between 0.7-1.2 as a hack to improve stability

        Arguments:
            log_probs {torch.Tensor} -- (batch_size, n_classes)
            y {torch.Tensor} -- (batch_size,)

        Returns:
            loss -- loss with true labels changed 0.7-1.2
        """
        num_classes = log_probs.shape[-1]
        eps = torch.rand(1, device=self.device) * (0.5) - 0.2  # Sample true prob (0.7-1.2)
        loss = -log_probs.sum(dim=1) / num_classes
        nll = F.nll_loss(log_probs, y, reduction="none")
        return (eps * loss) + (1 - eps) * nll

    def training_step(self, sgrams, batch_idx, optimizer_idx):
        batch_size = sgrams.shape[0]
        assert sgrams.shape[1:] == self.spectrogram_shape  # (N, 1, H, W)

        """
        Four classes with product of terms objective.
        See Dandi et al., "Generalized Adversarially Learned Inference"
        https://arxiv.org/pdf/2006.08089.pdf
        0: x, E(x)
        1: z, G(z)
        2: x, E(G(E(x)))
        3: G(E(G(z))), z
        """
        log_probs = [None] * 4
        # Class 0:
        true_latent_params = self.encoder(sgrams).view(-1, 2, self.hparams.latent_size)
        true_mu = true_latent_params[:, 0, :]
        true_std = true_latent_params[:, 1, :]
        true_latent_sample = self.sample(true_mu, true_std)
        logits = self.discriminator((sgrams, true_latent_sample))
        log_probs[0] = F.log_softmax(logits, dim=1)
        if optimizer_idx == 0:
            pt_loss = self.product_of_terms_loss(log_probs[0], 0)

        # Class 1:
        fake_latent = torch.randn(
            batch_size, self.hparams.latent_size, device=self.device
        )
        fake_sgrams = self.generator(fake_latent)
        logits = self.discriminator((fake_sgrams, fake_latent))
        log_probs[1] = F.log_softmax(logits, dim=1)
        if optimizer_idx == 0:
            pt_loss += self.product_of_terms_loss(log_probs[1], 1)

        # Class 2:
        true_reencoded_params = self.encoder(self.generator(true_latent_sample)).view(
            -1, 2, self.hparams.latent_size
        )
        true_reencoded_mu = true_reencoded_params[:, 0, :]
        true_reencoded_std = true_reencoded_params[:, 1, :]
        true_reencoded_latent_sample = self.sample(
            true_reencoded_mu, true_reencoded_std
        )
        logits = self.discriminator((sgrams, true_reencoded_latent_sample))
        log_probs[2] = F.log_softmax(logits, dim=1)
        if optimizer_idx == 0:
            pt_loss += self.product_of_terms_loss(log_probs[2], 2)

        # Class 3:
        fake_reencoded_params = self.encoder(fake_sgrams).view(
            -1, 2, self.hparams.latent_size
        )
        fake_reencoded_mu = fake_reencoded_params[:, 0, :]
        fake_reencoded_std = fake_reencoded_params[:, 1, :]
        fake_reencoded_latent_sample = self.sample(
            fake_reencoded_mu, fake_reencoded_std
        )
        fake_regenerated_sgrams = self.generator(fake_reencoded_latent_sample)
        self.log("reconstruction-mse", mean_squared_error(fake_regenerated_sgrams, sgrams))
        self.log("reconstruction-ssim", ssim(fake_regenerated_sgrams, sgrams))
        logits = self.discriminator((fake_regenerated_sgrams, fake_latent))
        log_probs[3] = F.log_softmax(logits, dim=1)
        # Encoder-Generator loss

        if optimizer_idx == 0:
            pt_loss += self.product_of_terms_loss(log_probs[3], 3)


        last_update = "D" if optimizer_idx == 0 else "EG"
        step_type = "t" if self.training else "v"
        self.log(f"x,E(x)|{step_type}|{last_update}", torch.exp(log_probs[0][:, 0]).mean())
        self.log(f"z,G(z)|{step_type}|{last_update}", torch.exp(log_probs[1][:, 1]).mean())
        self.log(f"x,E(G(E(x)))|{step_type}|{last_update}", torch.exp(log_probs[2][:, 2]).mean())
        self.log(f"G(E(G(z))),z|{step_type}|{last_update}", torch.exp(log_probs[3][:, 3]).mean())

        
        if optimizer_idx == 0:
            # Weight the objective containing m=4 log terms by 2/m=0.5, corresponding to a weight
            # of 1 for ALI's generator and discriminator objective, keeping the objectives
            # in a similar range for both optimizers
            pt_loss = pt_loss.mean() / 2
            self.log("eg_loss", pt_loss)
            return pt_loss.mul(torch.tensor(-1, device=self.device))

        # Discriminator loss
        
        if optimizer_idx == 1:
            d_loss = torch.zeros(
                (batch_size,), dtype=log_probs[0].dtype, device=self.device
            )
            for true_idx in range(4):
                target = torch.full(
                    (batch_size,), true_idx, dtype=torch.long, device=self.device
                )
                # Maybe randomly swap labels instead
                d_loss += self.smoothing_cross_entropy_loss(log_probs[true_idx], target)
                # d_loss += F.cross_entropy(logits[true_idx], target, reduction="none")
            
            d_loss = d_loss.mean()
            self.log("d_loss", d_loss)
            return d_loss.mean()  # Reduce across batch dimension

    # def validation_step(self, *args, **kwargs):
    #     return self.training_step(*args, **kwargs)

    # def test_step(self, *args, **kwargs):
    #     return self.training_step(*args, **kwargs)

    def on_epoch_end(self) -> None:
        # TODO: Validation MSE or similar
        return super().on_epoch_end()

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        encoder_generator_params = list(self.encoder.parameters()) + list(
            self.generator.parameters()
        )
        opt_pt = torch.optim.Adam(encoder_generator_params, lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        eg_sched = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(opt_pt, 0.99),
            "interval": "step",
        }
        d_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=10)

        return [opt_pt, opt_d], [eg_sched, d_sched]

    def train_dataloader(self, *args, **kwargs) -> dataloader:
        return self.datamodule.train_dataloader(*args, **kwargs)

    def val_dataloader(self, *args, **kwargs) -> dataloader:
        return self.datamodule.val_dataloader(*args, **kwargs)

    def test_dataloader(self, *args, **kwargs) -> dataloader:
        return self.datamodule.test_dataloader(*args, **kwargs)