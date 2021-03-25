from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F


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


def same_label_mask(labels: torch.Tensor) -> torch.Tensor:
    """Returns a symmetric boolean mask indicating where pairwise labels are the same. An instance is
    considered to not share a label with itself, so the diagonal is false

    Arguments:
        labels {torch.Tensor} -- long tensor containing labels (batch_size,)

    Returns:
        torch.Tensor -- bool tensor (batch_size, batch_size)
    """
    batch_size = labels.shape[0]
    repeated = labels.repeat(batch_size, 1)
    mask = repeated == repeated.t()
    return mask.fill_diagonal_(0)


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
    assert same_label_mask[0, 0] == torch.tensor(0, dtype=torch.bool)
    distances[
        ~same_label_mask
    ] = -1  # Only consider positive pairings; also, ignore diagonal
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
    def __init__(self, hn_lambda: float = 1.0):
        """Selectively contrastive loss, from https://arxiv.org/pdf/2007.12749.pdf
        See reference implementation at https://github.com/littleredxh/HardNegative/blob/master/_code/Loss.py

        Arguments:
            hn_lambda: weighting factor for the hard negative loss
        """
        super().__init__()
        self.hn_lambda = hn_lambda

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        distances = cosine_similarity_matrix(embeddings)
        same_mask = same_label_mask(labels)  # (batch_size, batch_size)

        pos_max_idxs, pos_maxes_valid_mask = mine_easy_positives(
            distances, same_mask
        )
        easy_positive_scores = distances[torch.arange(batch_size), pos_max_idxs]

        neg_max_idxs, neg_maxes_valid_mask = mine_hard_negatives(
            distances, same_mask
        )
        hard_negative_scores = distances[torch.arange(batch_size), neg_max_idxs]

        combined_valid_mask = pos_maxes_valid_mask & neg_maxes_valid_mask

        hard_triplet_mask = (
            (hard_negative_scores > easy_positive_scores) | (hard_negative_scores > 0.8)
        ) & combined_valid_mask

        easy_triplet_mask = (
            (hard_negative_scores < easy_positive_scores) & (hard_negative_scores < 0.8)
        ) & combined_valid_mask

        hard_triplet_loss = hard_negative_scores[hard_triplet_mask].sum()  # This is the contrastive loss from the paper
        triplets = torch.stack((easy_positive_scores, hard_negative_scores), dim=1)
        easy_triplet_loss = -F.log_softmax(triplets[easy_triplet_mask, :] * 10, dim=1)[:, 0].sum()
        
        n_triplets = hard_triplet_mask.float().sum() + easy_triplet_mask.float().sum()
        if n_triplets == 0:
            n_triplets = 1
        
        loss = (self.hn_lambda * hard_triplet_loss + easy_triplet_loss) / n_triplets

        return loss, triplets
        