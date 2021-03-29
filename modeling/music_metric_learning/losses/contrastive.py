from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from pytorch_metric_learning.utils.common_functions import neg_inf
import torch
import torch.nn.functional as F
from torch import nn
from pytorch_metric_learning.losses.contrastive_loss import (
    ContrastiveLoss as PMLContrastiveLoss,
)
from pytorch_metric_learning.distances.dot_product_similarity import (
    DotProductSimilarity as PMLDotProductSimilarity,
)
from pytorch_metric_learning.utils.loss_and_miner_utils import get_all_pairs_indices


def cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Return cosine similarity matrix for a set of embeddings, with -1 on the diagonal (for
    purposes of the loss, we don't want to consider embeddings to be self-similar)

    Arguments:
        embeddings {torch.Tensor} -- Does not need to be normalized (batch_size, embedding_dim)

    Returns:
        torch.Tensor -- (batch_size, batch_size)
    """
    embeddings_normed = F.normalize(embeddings, p=2, dim=1)
    distances = embeddings_normed.mm(embeddings_normed.t())  # (batch_size, batch_size)
    distances.fill_diagonal_(-1)  # prevent maxing on anchor-anchor pairs
    return distances


def keyed_cosine_similarity_matrix(
    query_embeddings: torch.Tensor, key_embeddings: torch.Tensor
) -> torch.Tensor:
    """Like `cosine_similarity_matrix`, but for separate query (anchor) and key
    (positive/negative) embeddings

    Arguments:
        query_embeddings {query_embeddings} -- Does not need to be normalized (batch_size,
        embeddings_dim)
        key_embeddings {key_embeddings} -- Does not need to be normalized (key_size, embeddings_dim)


    Returns:
        torch.Tensor -- Cosine similarity matrix of (batch_size, key_size)
    """
    key_embeddings_normed = F.normalize(key_embeddings, p=2, dim=1)
    query_embeddings_normed = F.normalize(query_embeddings, p=2, dim=1)
    distances = query_embeddings_normed.mm(key_embeddings_normed.t())
    return distances


def same_label_mask(labels: torch.Tensor) -> torch.Tensor:
    """Returns a symmetric boolean mask indicating where pairwise labels are the same. An instance
    is considered to not share a label with itself, so the diagonal is false

    Arguments:
        labels {torch.Tensor} -- long tensor containing labels (batch_size,)

    Returns:
        torch.Tensor -- bool tensor (batch_size, batch_size)
    """
    batch_size = labels.shape[0]
    repeated = labels.repeat(batch_size, 1)
    mask = repeated == repeated.t()
    return mask.fill_diagonal_(0)


def keyed_same_label_mask(
    query_labels: torch.Tensor, key_labels: torch.Tensor
) -> torch.Tensor:
    """Return a boolean mask indicating where key and query labels are the same. For this to make
    sense, the key labels and query labels should not correspond to identical items.

    Arguments:
        query_labels {torch.Tensor} -- long tensor containing query (i.e., anchor) labels (batch_size,)
        key_labels {torch.Tensor} -- long tensor containing key labels for positive/negative pairings

    Returns:
        torch.Tensor -- bool tensor (batch_size, key_size)
    """
    batch_size = query_labels.shape[0]
    key_size = key_labels.shape[0]
    query_repeated = query_labels.repeat(key_size, 1)
    key_repeated = key_labels.repeat(batch_size, 1)
    mask = key_repeated == query_repeated.t()
    return mask


def mine_easy_positives(
    distances: torch.Tensor, same_label_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mine the highest scoring positive instance for each anchor, along with a mask
    indicating positions where that score was either 1 or -1. If there are no positive instances,
    -1 will be returned for that anchor, so it is important to apply the mask downstream.

    Arguments:
        distances {torch.Tensor} -- Cosine similarity matrix (batch_size, batch_size)
        same_label_mask {torch.Tensor} -- boolean tensor indicating where labels are the same,
        with zeros on the diagonal (batch_size,)

    Returns:
        Tuple[torch.Tensor, torch.Tensor] -- [description]
    """
    distances = (
        distances.clone().detach()
    )  # We'll modify this, and it's not needed for backprop from here forward
    distances[~same_label_mask] = -1  # Only consider positive pairings
    distances[distances > 0.9999] = 1  # For numerical reasons?
    pos_maxes, pos_max_idxs = distances.max(dim=1)  # (batch_size,)
    pos_maxes_valid_mask = (pos_maxes > -1) & (
        pos_maxes < 1
    )  # We'll mask out degenerate cases
    return pos_max_idxs, pos_maxes_valid_mask


def mine_hard_negatives(
    distances: torch.Tensor, same_label_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mine the highest scoring negative instance for each anchor, along with a mask indicating
    positions where that score was either 1 or -1, or where there were no negative instances.

    Arguments:
        distances {torch.Tensor} -- Cosine similarity matrix (batch_size, batch_size)
        same_label_mask {torch.Tensor} -- boolean tensor indicating where labels are different
        (batch_size,)

    Returns:
        Tuple[torch.Tensor, torch.Tensor] -- [description]
    """

    distances = (
        distances.clone().detach()
    )  # We'll modify this, and it's not needed for backprop from here forward
    distances[same_label_mask] = -1  # Only consider negative pairings
    neg_maxes, neg_max_idxs = distances.max(1)  # (batch_size,)
    neg_maxes_valid_mask = (neg_maxes > -1) & (neg_maxes < 1)
    return neg_max_idxs, neg_maxes_valid_mask


class SelectivelyContrastiveLoss(nn.Module):
    def __init__(
        self,
        hn_lambda: float = 1.0,
        temperature: float = 0.1,
        hard_cutoff: float = 0.8,
        xent_only: bool = False,
    ):
        """Selectively contrastive loss, from https://arxiv.org/pdf/2007.12749.pdf
        See reference implementation at https://github.com/littleredxh/HardNegative/blob/master/_code/Loss.py

        Arguments:
            hn_lambda: weighting factor for the hard negative loss
            temperature: temperature hyperparamter for NCXEnt loss
            hard_cutoff: above this threshold for cosine similarity, negative examples will be
            considered hard even if the easiest example is closer
        """
        super().__init__()
        self.hn_lambda = hn_lambda
        self.temperature = temperature
        self.hard_cutoff = hard_cutoff
        self.xent_only = xent_only

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        key_embeddings: Optional[torch.Tensor] = None,
        key_labels: Optional[torch.Tensor] = None,
        log_callback: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the selectively contrastive loss for a batch of embeddings and labels, using the
        cosine similarity as the distance metric.

        Arguments:
            embeddings {torch.Tensor} -- (batch_size, embedding_dim)
            labels {torch.Tensor} -- (batch_size,)

        Returns:
            loss {torch.Tensor} -- (1,)
            triplets {torch.Tensor} -- (batch_size, 2) easy_score, hard_score indexed by anchor;
            not equivalent to input to the loss function, since the loss function masks invalid
            triplets
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

        neg_max_idxs, neg_maxes_valid_mask = mine_hard_negatives(distances, same_mask)
        hard_negative_scores = distances[torch.arange(batch_size), neg_max_idxs]

        combined_valid_mask = pos_maxes_valid_mask & neg_maxes_valid_mask

        hard_triplet_mask = (
            (hard_negative_scores > easy_positive_scores)
            | (hard_negative_scores > self.hard_cutoff)
        ) & combined_valid_mask

        easy_triplet_mask = (
            (hard_negative_scores < easy_positive_scores)
            & (hard_negative_scores < self.hard_cutoff)
        ) & combined_valid_mask

        triplets = torch.stack((easy_positive_scores, hard_negative_scores), dim=1)

        if not self.xent_only:
            hard_triplet_loss = hard_negative_scores[
                hard_triplet_mask
            ].sum()  # This is the contrastive loss from the paper
            easy_triplet_loss = -F.log_softmax(
                triplets[easy_triplet_mask, :] / self.temperature, dim=1
            )[:, 0].sum()
            n_triplets = (
                hard_triplet_mask.float().sum() + easy_triplet_mask.float().sum()
            )
            if n_triplets == 0:
                n_triplets = 1
            loss = (self.hn_lambda * hard_triplet_loss + easy_triplet_loss) / n_triplets
        else:
            loss = -F.log_softmax(
                triplets[combined_valid_mask, :] / self.temperature, dim=1
            )[:, 0].mean()

        if log_callback is not None:
            logging_dict: Dict[str, Union[torch.Tensor, np.ndarray, int, float]] = {}
            positive_log = easy_positive_scores.clone().detach().cpu().numpy()
            logging_dict["easy_positive_scores"] = positive_log
            logging_dict["mean_easy_positive"] = np.mean(positive_log)
            negative_log = hard_negative_scores.clone().detach().cpu().numpy()
            logging_dict["hard_negative_scores"] = negative_log
            logging_dict["mean_hard_negative"] = np.mean(negative_log)
            logging_dict["hn_ratio"] = (
                ((hard_negative_scores > easy_positive_scores)[combined_valid_mask])
                .clone()
                .detach()
                .float()
                .mean()
                .cpu()
            )
            logging_dict["n_valid_hard"] = hard_triplet_mask.detach().float().sum()
            logging_dict["n_valid_easy"] = easy_triplet_mask.detach().float().sum()

            log_callback(logging_dict)

        return loss, triplets


class ContrastiveLoss(nn.Module):
    def __init__(self, pos_margin: float = 1.0, neg_margin: float = 0.0):
        super().__init__()
        self.criterion = PMLContrastiveLoss(
            pos_margin=pos_margin,
            neg_margin=neg_margin,
            distance=PMLDotProductSimilarity(normalize_embeddings=True),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        key_embeddings: Optional[torch.Tensor] = None,
        key_labels: Optional[torch.Tensor] = None,
    ):
        all_pairs = get_all_pairs_indices(labels, key_labels)

        return self.criterion.compute_loss(embeddings, labels)
