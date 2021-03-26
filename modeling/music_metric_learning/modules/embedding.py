from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class EmbeddingMLP(nn.Module):
    def __init__(
        self,
        num_embeddings=5,
        category_embedding_dim=64,
        in_dim=1280,
        hidden_dim=512,
        out_dim=256,
        normalize_embeddings=False,
    ):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=category_embedding_dim
        )
        self.mlp = nn.Sequential(
            nn.Linear(category_embedding_dim + in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self, x: torch.Tensor, words: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of MLP with single embedding tensor

        Arguments:
            x {torch.Tensor}-- (batch_size, in_dim)
            words {torch.Tensor (long)} -- (batch_size, 1)
        """
        embedded = self.embedding(words)
        x = self.mlp(torch.cat((embedded, x), dim=1))
        if self.normalize_embeddings:
            x = F.normalize(x, p=2, dim=1)
        return x, torch.linalg.norm(x, ord=2, dim=-1).pow(2)
