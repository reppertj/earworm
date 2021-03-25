from torch import nn
import torch
from music_metric_learningmodules.residual import SqueezeExcite, Identity, MobileBlock, ResidualBlock  # type: ignore
from music_metric_learningmodules.convolution import make_conv, HardSwish  # type: ignore


class MulticlassDiscriminator(nn.Module):
    def __init__(
        self, n_classes: int = 4, latent_size: int = 60, dropout_prob=0.5
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.latent_size = latent_size
        self.sgram_side = [
            make_conv(
                transpose=False,
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        ]
        middle_layer_params = [
            # Cf. MobilenetV3 Table 2
            # kernel, hidden_ch, out_ch, squeeze_excite, activation, stride
            (3, 16, 16, SqueezeExcite, nn.ReLU, 2),
            (3, 72, 24, Identity, nn.ReLU, 2),
            (3, 88, 24, Identity, nn.ReLU, 1),
            (5, 96, 40, SqueezeExcite, HardSwish, 2),
            (5, 240, 40, SqueezeExcite, HardSwish, 1),
            (5, 240, 40, SqueezeExcite, HardSwish, 1),
            (5, 120, 48, SqueezeExcite, HardSwish, 1),
            (5, 144, 48, SqueezeExcite, HardSwish, 1),
            (5, 288, 96, SqueezeExcite, HardSwish, 2),
            (5, 576, 96, SqueezeExcite, HardSwish, 1),
            (5, 576, 96, SqueezeExcite, HardSwish, 1),
        ]
        # add middle layers
        i_ch = 16
        for k, h_ch, out_ch, se, nl, s in middle_layer_params:
            if s == 1 and i_ch == out_ch:
                self.sgram_side.append(ResidualBlock(i_ch, h_ch, out_ch, k, s, se, nl))
            else:
                self.sgram_side.append(MobileBlock(i_ch, h_ch, out_ch, k, s, se, nl))
            i_ch = out_ch
        # add last few layers
        self.sgram_side.append(
            make_conv(
                transpose=False,
                in_channels=i_ch,
                out_channels=576,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.sgram_side.append(nn.AdaptiveAvgPool2d(1))
        self.sgram_side.append(nn.Conv2d(576, 576, 1, 1, 0))
        self.sgram_side = nn.Sequential(*self.sgram_side)

        self.latent_side = nn.Sequential(
            nn.Linear(self.latent_size, 512),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
        )

        self.combined_classifier = nn.Sequential(
            nn.Linear(704, 512),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, self.n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        sgram, latent = x
        x_sgram = self.sgram_side(sgram).squeeze(3).squeeze(2)
        x_latent = self.latent_side(latent)
        x = torch.cat((x_sgram, x_latent), dim=1)
        x = self.combined_classifier(x)
        return x