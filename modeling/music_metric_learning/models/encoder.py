import torch
from torch import nn
from music_metric_learningmodules.residual import SqueezeExcite, Identity, MobileBlock, ResidualBlock  # type: ignore
from music_metric_learningmodules.convolution import make_conv, HardSwish  # type: ignore


class MobileNetLikeEncoder(nn.Module):
    def __init__(self, latent_size: int = 32):
        super().__init__()
        self.latent_size = latent_size
        self.sample_input = torch.randn((1, 1, 128, 624), dtype=torch.float32)
        # first layer
        self.layers = [
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
                self.layers.append(ResidualBlock(i_ch, h_ch, out_ch, k, s, se, nl))
            else:
                self.layers.append(MobileBlock(i_ch, h_ch, out_ch, k, s, se, nl))
            i_ch = out_ch
        # add last few layers
        self.layers.append(
            make_conv(
                transpose=False,
                in_channels=i_ch,
                out_channels=576,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.layers.append(nn.AdaptiveAvgPool2d(1))
        self.layers.append(nn.Conv2d(576, self.latent_size, 1, 1, 0))

        self.model = nn.Sequential(*self.layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        return self.model(x).squeeze(2).squeeze(2)
