import torch
from torch import nn


class MobileNetEncoder(nn.Module):
    def __init__(self, pretrained=True, freeze_weights=False, max_pool=True):
        super().__init__()
        self.sample_input = torch.randn((1, 1, 128, 130))
        mobilenet = torch.hub.load(
            "pytorch/vision:v0.8.2", "mobilenet_v2", pretrained=pretrained
        )
        new_first = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        first_bn_relu = list(
            [list(layer.children()) for layer in list(mobilenet.children())][0][0]
        )[1:]
        rest = list(
            [list(layer.children()) for layer in list(mobilenet.children())][0][1:]
        )
        if freeze_weights:
            for layer in rest:
                if hasattr(layer, "requires_grad"):
                    layer.requires_grad(False)
        pool = (
            nn.AdaptiveMaxPool2d((1, 1)) if max_pool else nn.AdaptiveAvgPool2d((1, 1))
        )
        self.model = nn.Sequential(new_first, *(first_bn_relu + rest), pool)

    def forward(self, x):
        return self.model(x).squeeze().squeeze()
