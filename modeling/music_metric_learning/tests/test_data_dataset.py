import torch
from music_metric_learning.data.dataset import InverseMELNormalize, MELNormalize


def test_normalize_inverse_roundtrip():
    normalizer = MELNormalize()
    inverse_normalizer = InverseMELNormalize()
    torch.manual_seed(42)
    sgram = torch.randn(1, 128, 130)
    normalized = normalizer(sgram)
    inverse_normalized = inverse_normalizer(normalized)
    assert torch.allclose(sgram, inverse_normalized, atol=1e-6, rtol=1e-3)
