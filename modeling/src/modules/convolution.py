from typing import Optional, Tuple
from torch import nn
import torch.nn.functional as F


def make_conv(
    transpose: bool,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    norm_layer: nn.Module = nn.BatchNorm2d,
    activation_layer: nn.Module = nn.ReLU,
    kaiming_init: bool = True,
):
    if transpose:
        conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
    else:
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
    if kaiming_init:
        nn.init.kaiming_normal_(conv.weight)
    return nn.Sequential(conv, norm_layer(out_channels), activation_layer())


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0
