import pytest
import torch
from music_metric_learning.modules.embedding import EmbeddingMLP


def test_embedding_mlp():
    sample_input = torch.randn(5, 1280)
    words = torch.ones((5,), dtype=torch.long)
    embedder = EmbeddingMLP(num_embeddings=5, category_embedding_dim=64, in_dim=1280, hidden_dim=512, out_dim=256, normalize_embeddings=True)

    embeddings, norm = embedder(sample_input, words)
    assert embeddings.shape == (5, 256)
    assert torch.allclose(norm, torch.ones_like(norm))
