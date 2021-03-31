from typing import Dict, Optional, Callable, Tuple, Union

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from music_metric_learning.losses.contrastive import cosine_similarity_matrix, keyed_cosine_similarity_matrix, same_label_mask, keyed_same_label_mask, mine_easy_positives


class MoCoCrossEntropyLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        # TODO: Look at thresholding in xent versus triplet loss


    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        key_embeddings: Optional[torch.Tensor] = None,
        key_labels: Optional[torch.Tensor] = None,
        log_callback: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MoCo-style cross-entropy loss, but for arbitrarily labeled keys.
        You don't need to guarantee only one positive pair. The positive class is found
        by mining the easiest positive per anchor from the key embeddings

        Arguments:
            embeddings {torch.Tensor} -- (batch_size, embedding_dim)
            labels {torch.Tensor} -- (batch_size,)

        Keyword Arguments:
            key_embeddings {Optional[torch.Tensor]} -- (n_keys, embedding_dim) (default: {None})
            key_labels {Optional[torch.Tensor]} -- (n_keys,) (default: {None})
            log_callback {Optional[Callable]} -- function that takes a dict containing
            keys and labels to send to the logger (default: {None})

        Returns:
            Tuple[torch.Tensor, torch.Tensor] -- scalar loss and tensor of easy positive scores
        """
        batch_size = embeddings.shape[0]

        if key_embeddings is None:
            assert key_labels is None
            distances = cosine_similarity_matrix(embeddings)
            same_mask = same_label_mask(labels)  # (batch_size, batch_size)
        else:
            assert key_labels is not None
            distances = keyed_cosine_similarity_matrix(
                embeddings, key_embeddings=key_embeddings
            )
            same_mask = keyed_same_label_mask(labels, key_labels=key_labels)

        pos_max_idxs, pos_maxes_valid_mask = mine_easy_positives(distances, same_mask)
        easy_positive_scores = distances[torch.arange(batch_size), pos_max_idxs]

        distances[same_mask] = torch.finfo(
            distances.dtype
        ).min  # set anchor-positives in anchor-negative space to negative infinity
        # logits have shape (batch_size - n_invalid_positives, 1 + n_key_embeddings)
        logits = torch.cat(
            [
                easy_positive_scores[pos_maxes_valid_mask].unsqueeze(1),
                distances[pos_maxes_valid_mask, :],
            ],
            dim=1,
        ) / self.temperature
        # Positive pairs have label 0 for cross-entropy loss
        ground_truth = torch.zeros(
            logits.shape[0], dtype=torch.long, device=logits.device
        )
        loss = F.cross_entropy(logits, ground_truth)

        if log_callback is not None:
            logging_dict: Dict[str, Union[torch.Tensor, int, float]] = {}
            positive_log = easy_positive_scores.clone().detach().cpu().numpy()
            logging_dict["mean_easy_positive"] = max(-1, min(1, np.mean(positive_log)))
            log_callback(logging_dict)
        
        return loss, easy_positive_scores
