import numpy as np
import pytest
import torch
from music_metric_learning.losses.contrastive import (
    SelectivelyContrastiveLoss,
    cosine_similarity_matrix,
    mine_easy_positives,
    mine_hard_negatives,
    same_label_mask,
)


def angle_to_coord(angle: float):
    return np.cos(np.radians(angle)), np.sin(np.radians(angle))


def test_same_label_mask():
    labels = torch.tensor([0, 0, 1, 1, 2])
    mask = same_label_mask(labels)
    same_labels = [(0, 1), (1, 0), (2, 3), (3, 2)]
    assert all([mask[pair] == 1 for pair in same_labels])
    for pair in same_labels:
        mask[pair] = 0
    assert torch.all(mask == torch.zeros_like(mask))


def test_mine_easy_positives():
    embeddings = torch.tensor(
        [angle_to_coord(float(a)) for a in [10, 15, 25, 35, 40, 45]]
    )
    labels = torch.tensor([0, 0, 1, 1, 1, 2])
    easy_positives = {
        0: 1,
        1: 0,
        2: 3,
        3: 4,
        4: 3,
    }
    mask = same_label_mask(labels)
    maxes, validities = mine_easy_positives(cosine_similarity_matrix(embeddings), mask)
    for query, key in easy_positives.items():
        assert maxes[query] == key
        assert validities[query]
    assert not validities[-1]


def test_mind_hard_negatives():
    embeddings = torch.tensor(
        [angle_to_coord(float(a)) for a in [10, 15, 25, 35, 40, 45]]
    )
    labels = torch.tensor([0, 0, 1, 1, 1, 2])
    hard_negatives = {
        0: 2,
        1: 2,
        2: 1,
        3: 5,
        4: 5,
        5: 4,
    }
    mask = same_label_mask(labels)
    maxes, validities = mine_hard_negatives(cosine_similarity_matrix(embeddings), mask)
    for query, key in hard_negatives.items():
        assert maxes[query] == key
    assert torch.all(validities)


def test_selectively_contrastive_loss():
    criterion = SelectivelyContrastiveLoss()
    embeddings = torch.tensor(
        [angle_to_coord(float(a)) for a in [10, 15, 25, 35, 40, 80, 100]]
    )
    labels = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    easy_positives = torch.tensor(
        [(5, 6), (6, 5)], dtype=torch.long
    ).t()  # the others don't count because distance is below 0.8 threshold
    hard_negatives = torch.tensor(
        [
            (0, 2),
            (1, 2),
            (2, 1),
            (3, 1),
            (4, 1),
        ],
        dtype=torch.long,
    ).t()  # last two left out, again, because of 0.8 threshold
    distances = cosine_similarity_matrix(embeddings)
    loss, _ = criterion(embeddings, labels)
    hard_triplet_loss = distances[hard_negatives[0], hard_negatives[1]].sum()
    ap = (
        distances[easy_positives[0], easy_positives[1]] * 10
    )  # Scalar multiple of 10 from reference implementation
    an = distances[[5, 6], [4, 4]] * 10
    easy_triplet_loss = -torch.log(
        torch.exp(ap).div(torch.exp(ap) + torch.exp(an))
    ).sum()
    manual_loss = (easy_triplet_loss + criterion.hn_lambda * hard_triplet_loss) / 7.0
    assert torch.allclose(loss, manual_loss)