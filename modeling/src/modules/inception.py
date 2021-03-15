from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


def make_inception_conv(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True),
        nn.ReLU(inplace=True),
    )


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Maintains the spatial dimension through different sized filters,
        with a fixed dimension output. Modeled after Inception-A block
        of the Inception-v4 paper
        """
        super().__init__()
        assert out_channels % 8 == 0
        each_out = out_channels // 4
        hidden = each_out * 3 // 2
        self.conv1 = make_inception_conv(in_channels, each_out, 1, 1, 0)
        self.conv3 = nn.Sequential(
            make_inception_conv(in_channels, hidden, 1, 1, 0),
            make_inception_conv(hidden, each_out, 3, 1, 1),
        )
        self.conv5 = nn.Sequential(
            make_inception_conv(in_channels, hidden, 1, 1, 0),
            make_inception_conv(hidden, each_out, 3, 1, 1),
            make_inception_conv(each_out, each_out, 3, 1, 1),
        )
        self.pooling = nn.Sequential(
            nn.AvgPool2d(3, 1, 1, count_include_pad=False),
            make_inception_conv(in_channels, each_out, 1, 1, 0),
        )

    def forward(self, x):
        return torch.cat(
            (self.conv1(x), self.conv3(x), self.conv5(x), self.pooling(x)), dim=1
        )


class ReductionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 16 == 0
        inner = out_channels // 4
        outer = inner * 3 // 2
        self.conv3 = make_inception_conv(in_channels, outer, 3, 2, 0)
        self.conv5 = nn.Sequential(
            make_inception_conv(in_channels, inner * 3 // 4, 1, 1, 0),
            make_inception_conv(inner * 3 // 4, inner * 7 // 8, 3, 1, 1),
            make_inception_conv(inner * 7 // 8, inner, 3, 2, 0),
        )
        self.pooling = nn.Sequential(
            nn.MaxPool2d(3, stride=2), make_inception_conv(in_channels, outer, 1, 1, 0)
        )

    def forward(self, x):
        return torch.cat((self.conv3(x), self.conv5(x), self.pooling(x)), dim=1)


class ConditionalInceptionLikeEncoder(nn.Module):
    def __init__(
        self,
        latent_dim=256,
        n_masks=4,
        normalize_embeddings=False,
        mask_nonlinearity=nn.ReLU,
    ):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.sample_input = (torch.randn((1, 1, 128, 130)), torch.randn((1, 4)))

        self.masks = nn.Linear(n_masks, latent_dim, bias=False)
        self.mask_nonlinearity = mask_nonlinearity()

        layers = [
            nn.Conv2d(1, 64, 5, stride=1, padding=2),
            nn.MaxPool2d(2, 2, 0),  # 64, 64, 64
            InceptionBlock(64, 256),
            ReductionBlock(256, 256),
        ]
        for _ in range(4):
            layers.append(InceptionBlock(256, 256))
            layers.append(ReductionBlock(256, 256))
        self.inception = nn.Sequential(*layers)
        self.fc = nn.Linear(256, latent_dim, bias=True)

    def logits(self, x):
        batch_size, kernel_size = x.shape[0], x.shape[2]
        x = F.avg_pool2d(x, kernel_size=kernel_size).view(batch_size, -1)
        return self.fc(x)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, 1, H, W)
        condition: (batch_size, n_masks)

        Out:
        embeddings: the masked embedding (batch_size, latent_dim)
        embeddings_norm: the (unreduced, non-squared) l2 norm of the embeddings (batch_size,)
        mask_weight_norm: the scalar l1 norm of the masking function's weights
        """
        batch_size = x.shape[0]
        x = self.fc(self.inception(x).view(batch_size, -1))
        mask = self.mask_nonlinearity(self.masks(condition))
        embeddings = x.mul(mask)
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, 2, -1)
        embeddings_norm = torch.linalg.norm(x, ord=2, dim=-1)
        mask_weight_norm = torch.linalg.norm(self.masks.weight, ord=1)
        return embeddings, embeddings_norm, mask_weight_norm
