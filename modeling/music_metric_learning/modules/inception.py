from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

# Implementation from https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/MoCoCIFAR10.ipynb
# - Simulates split behavior of performing batch norm across GPUs; to prevent leakage of information
# across the batch dimension during MoCo-style training


class SplitBatchNorm(torch.nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kwargs):
        super().__init__(num_features, **kwargs)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = F.batch_norm(
                input.view(-1, C * self.num_splits, H, W),
                running_mean_split,
                running_var_split,
                self.weight.repeat(self.num_splits),
                self.bias.repeat(self.num_splits),
                True,
                self.momentum,
                self.eps,
            ).view(N, C, H, W)
            self.running_mean.data.copy_(
                running_mean_split.view(self.num_splits, C).mean(dim=0)
            )
            self.running_var.data.copy_(
                running_var_split.view(self.num_splits, C).mean(dim=0)
            )
            return outcome
        else:
            return torch.nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )


def make_inception_conv(
    in_channels, out_channels, kernel_size, stride, padding, num_splits
):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True),
        nn.ReLU(inplace=True),
    )


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_splits):
        """Maintains the spatial dimension through different sized filters,
        with a fixed dimension output. Modeled after Inception-A block
        of the Inception-v4 paper
        """
        super().__init__()
        assert out_channels % 8 == 0
        each_out = out_channels // 4
        hidden = each_out * 3 // 2
        self.conv1 = make_inception_conv(in_channels, each_out, 1, 1, 0, num_splits)
        self.conv3 = nn.Sequential(
            make_inception_conv(in_channels, hidden, 1, 1, 0, num_splits),
            make_inception_conv(hidden, each_out, 3, 1, 1, num_splits),
        )
        self.conv5 = nn.Sequential(
            make_inception_conv(in_channels, hidden, 1, 1, 0, num_splits),
            make_inception_conv(hidden, each_out, 3, 1, 1, num_splits),
            make_inception_conv(each_out, each_out, 3, 1, 1, num_splits),
        )
        self.pooling = nn.Sequential(
            nn.AvgPool2d(3, 1, 1, count_include_pad=False),
            make_inception_conv(in_channels, each_out, 1, 1, 0, num_splits),
        )

    def forward(self, x):
        return torch.cat(
            (self.conv1(x), self.conv3(x), self.conv5(x), self.pooling(x)), dim=1
        )


class ReductionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_splits):
        super().__init__()
        assert out_channels % 16 == 0
        inner = out_channels // 4
        outer = inner * 3 // 2
        self.conv3 = make_inception_conv(in_channels, outer, 3, 2, 0, num_splits)
        self.conv5 = nn.Sequential(
            make_inception_conv(in_channels, inner * 3 // 4, 1, 1, 0, num_splits),
            make_inception_conv(inner * 3 // 4, inner * 7 // 8, 3, 1, 1, num_splits),
            make_inception_conv(inner * 7 // 8, inner, 3, 2, 0, num_splits),
        )
        self.pooling = nn.Sequential(
            nn.MaxPool2d(3, stride=2),
            make_inception_conv(in_channels, outer, 1, 1, 0, num_splits),
        )

    def forward(self, x):
        return torch.cat((self.conv3(x), self.conv5(x), self.pooling(x)), dim=1)


class Encoder(nn.Module):
    def __init__(
        self,
        num_splits=4,
    ):
        super().__init__()
        self.sample_input = (torch.randn((1, 1, 128, 130)), torch.randn((1, 4)))

        layers = [
            nn.Conv2d(1, 64, 5, stride=1, padding=2),
            nn.MaxPool2d(2, 2, 0),  # 64, 64, 64
            InceptionBlock(64, 256, num_splits),
            ReductionBlock(256, 256, num_splits),
        ]
        for _ in range(4):
            layers.append(InceptionBlock(256, 256, num_splits))
            layers.append(ReductionBlock(256, 256, num_splits))
        self.inception = nn.Sequential(*layers)

    def logits(self, x):
        batch_size, kernel_size = x.shape[0], x.shape[2]
        x = F.avg_pool2d(x, kernel_size=kernel_size).view(batch_size, -1)
        return self.fc(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, 1, H, W)

        Out:
        embeddings: the masked embedding (batch_size, latent_dim)
        embeddings_norm: the squared l2 norm of the embeddings (batch_size,)
        """
        batch_size = x.shape[0]
        x = self.inception(x).view(batch_size, -1)
        return x, torch.linalg.norm(x, ord=2, dim=-1).pow(2)
