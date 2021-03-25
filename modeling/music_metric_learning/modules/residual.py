import torch
from music_metric_learningmodules.convolution import HardSigmoid # type: ignore
from torch import nn


class SqueezeExcite(nn.Module):
    def __init__(self, thru_channels: int, squeeze_factor=4) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(thru_channels, thru_channels // squeeze_factor, bias=False),
            nn.ReLU(),
            nn.Linear(thru_channels // squeeze_factor, thru_channels, bias=False),
            HardSigmoid(),
        )

    def forward(self, x: torch.Tensor):
        batch_size, channels, _, _ = x.shape
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x


class MobileBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        squeeze_excite: nn.Module,
        activation: nn.Module,
        transpose=False
    ) -> None:
        super().__init__()
        assert stride in (1, 2)
        assert kernel_size in (3, 5)
        padding = (kernel_size - 1) // 2
        if not transpose:
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                activation(),
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                squeeze_excite(hidden_channels),
                activation(),
                nn.Conv2d(
                    hidden_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                activation(),
                nn.ConvTranspose2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                squeeze_excite(hidden_channels),
                activation(),
                nn.ConvTranspose2d(
                    hidden_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            ) 

    def forward(self, x: torch.Tensor):
        return self.block(x)


class ResidualBlock(MobileBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        return x + self.block(x)
