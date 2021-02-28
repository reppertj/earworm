from typing import Tuple

import torch
from torch import nn


class SuperResolutionGenerator(nn.Module):
    def __init__(self, latent_size: int = 60):
        super().__init__()
        feature_depth = 64  # Results in (512, 752) output
        self.channels_out = 1
        self.latent_size = latent_size
        self.layers = [
            nn.ConvTranspose2d(self.latent_size, feature_depth * 16, (4, 5), 1, 0, bias=False),
            nn.BatchNorm2d(feature_depth * 16),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_depth * 16, feature_depth * 8, (4, 5), 2, 1, bias=False),
            nn.BatchNorm2d(feature_depth * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_depth * 8, feature_depth * 4, (4, 5), 2, 1, bias=False),
            nn.BatchNorm2d(feature_depth * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_depth * 4, feature_depth * 4, (4, 5), 2, 1, bias=False),
            nn.BatchNorm2d(feature_depth * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_depth * 4, feature_depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_depth * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_depth * 2, feature_depth, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_depth),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_depth, self.channels_out * 4, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.PixelShuffle(2),
        ]
        self.model = nn.Sequential(*self.layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        # (batch_size, latent_size)
        x = x.unsqueeze(-1).unsqueeze(-1)
        return self.model(x)
