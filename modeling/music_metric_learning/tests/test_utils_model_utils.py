import pytest
import torch
from music_metric_learning.utils.model_utils import make_encoder


def test_make_encoder():
    sample_batch = torch.randn(5, 1, 128, 130)

    mobilenet = make_encoder(
        kind="mobilenet", pretrained=True, freeze_weights=False, max_pool=True
    )

    inception = make_encoder(kind="inception")

    mobilenet_out = mobilenet(sample_batch)
    inception_out = inception(sample_batch)
    assert mobilenet_out.shape == (5, 1280)
    assert inception_out.shape == (5, 1280)
