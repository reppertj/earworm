import os
import random
from typing import Dict, Literal, Optional, Union, cast

from music_metric_learning.modules.mobilenet import MobileNetEncoder
from music_metric_learning.modules.inception import InceptionEncoder

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import umap


def seed_everything(seed: int = 42) -> int:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def pairwise_distances(embeddings: torch.Tensor, squared=False) -> torch.Tensor:
    dot_product = embeddings.mm(embeddings.t())
    square_norm = dot_product.diagonal()
    # |a - b|**2 = |a|**2 - 2ab + |b|^2
    distances = (
        square_norm.unsqueeze(0) - (2.0 * dot_product) + square_norm.unsqueeze(1)
    )
    distances = distances.clamp(0)  # To handle floating point errors

    if not squared:
        # avoid infinite gradient of sqrt when distance == 0 by adding an epsilon
        mask = distances == 0
        distances = distances + mask * 1e-6

        distances = distances.pow(0.5)

        # correct the epsilon
        distances = distances * (~mask)

    return distances


def anchor_positive_triplet_mask(labels: torch.LongTensor) -> torch.Tensor:
    """Returns a square boolean matrix where (x, y) is True iff x and y are distinct and have the
    same label.
    """
    batch_size = labels.shape[0]
    distinct = torch.logical_not(
        torch.eye(batch_size, dtype=torch.bool, device=labels.device)
    )
    same_label = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    return distinct.logical_and(same_label)


def anchor_negative_triplet_mask(labels: torch.LongTensor) -> torch.Tensor:
    """Returns a square boolean matrix where (x, y) is True iff x and y have different labels."""
    return torch.logical_not(labels.unsqueeze(0).eq(labels.unsqueeze(1)))


def visualizer_hook(
    visualizer: umap.UMAP,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    label_map: Optional[Dict[int, str]] = None,
    split_name: Optional[str] = None,
    show_plot: bool = False,
) -> plt.figure:
    """Returns a pyplot figure of a 2d projection of the supplied embeddings

    Arguments:
        visualizer {umap.UMAP} -- must have a `fit_transform` method that returns a 2d embedding
        embeddings {torch.Tensor} -- batch_size, latent_dim
        labels {torch.Tensor} -- batch_size
        label_map {dict} -- dict mapping integers to label descriptions
        split_name {str} -- name of split to use in image caption
    """
    embeddings_2d = visualizer.fit_transform(embeddings.detach().cpu().numpy())
    x, y = embeddings_2d[:, 0], embeddings_2d[:, 1]
    split_name = "" if split_name is None else split_name
    label_types = np.unique(labels.detach().cpu().numpy())
    colors = [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, len(label_types))]
    friendly_labels = ["" if label_map is None else label_map[i] for i in label_types]
    labels = labels.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    ax.set_prop_cycle(color=colors)
    for i, fl in enumerate(friendly_labels):
        idxs = labels == label_types[i]
        ax.plot(x[idxs], y[idxs], ".", markersize=2, label=fl)
    plt.legend(
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        fontsize="xx-small",
        frameon=False,
        labelspacing=0.2,
    )
    plt.tight_layout()
    if show_plot:
        plt.show()
    return fig


def histogram_from_weights(
    weights: torch.Tensor, tol: float = 1e-4
) -> Dict[str, Union[plt.figure, np.array, float]]:
    """Plot of sparsity structure of weights

    Arguments:
        tensor {torch.Tensor} -- weights tensor for visualization (latent_dim, n_masks)
        tol {float} -- how close to zero should be considered zero

    Returns:
        Dict containing histogram figure, array of values sorted by magnitude, and sparsity
    """
    weights_np = cast(np.array, weights.detach().cpu().numpy().T)
    sort_idx = np.argsort(np.abs(weights_np).sum(axis=0))[::-1]
    weights_np = weights_np[:, sort_idx]

    fig, axs = plt.subplots(4)

    for idx, condition in enumerate(weights_np):
        axs[idx].hist(np.abs(condition))

    sparsity = float((np.abs(weights_np) < tol).mean())

    return {"plot": fig, "weights": weights_np, "sparsity": sparsity}


def copy_params(encQ, encK, m=None):
    """Copy parameters with momentum from query to key models for MoCo implementation"""
    if m is None:
        for param_q, param_k in zip(encQ.parameters(), encK.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # don't backprop on keys
    else:
        for param_q, param_k in zip(encQ.parameters(), encK.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)


def make_encoder(
    kind: Literal["mobilenet", "inception"],
    pretrained: bool = True,
    freeze_weights: bool = False,
    max_pool: bool = True,
) -> nn.Module:
    """Return encoder without a final embedding layer

    Arguments:
        kind {"mobilenet", "inception"} -- Architecture to use

    Keyword Arguments:
        pretrained {bool} -- use weights from ImageNet pretrained model (only for MobileNet) (default: {True})
        freeze_weights {bool} -- freeze weights from pretrained model (only for MobileNet) (default: {False})
        max_pool {bool} -- use max pooling instead of average pooling after final convolutional layer (only for MobileNet) {default: {True}}

    Returns:
        nn.Module -- encoder model
    """
    if kind == "mobilenet":
        model = MobileNetEncoder(
            pretrained=pretrained, freeze_weights=freeze_weights, max_pool=max_pool
        )
    elif kind == "inception":
        model = InceptionEncoder()
    else:
        raise ValueError("Kind must be `mobilenet` or `inception`")
    return model


def copy_parameters(
    query_encoder: nn.Module, key_enconder: nn.Module, momentum: Optional[float] = None
) -> None:
    """Copy or update parameters between key and query encoders to use for MoCo queue. Modifies parameters in place.
    Implementation adapted from https://github.com/facebookresearch/moco

    Arguments:
        query_encoder {nn.Module} -- Model to copy parameters from
        key_enconder {nn.Module} -- Model to copy paramters to

    Keyword Arguments:
        momentum {Optional[float]} -- Momentum to applyto updates. If None, copy parameters wholesale and do not backprop on key parameters (use to setup initial key encoder) (default: {None})
    """
    if momentum is None:
        for param_q, param_k in zip(
            query_encoder.parameters(), key_enconder.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    else:
        for param_q, param_k in zip(
            query_encoder.parameters(), key_enconder.parameters()
        ):
            param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)
