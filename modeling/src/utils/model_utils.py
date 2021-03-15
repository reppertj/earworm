import io
from typing import Dict, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap


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
    plt.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize="xx-small", frameon=False, labelspacing=0.2)
    plt.tight_layout()
    if show_plot:
        plt.show()
    return fig

def histogram_from_weights(weights: torch.Tensor, tol: float=1e-4) -> Dict[str, Union[plt.figure, np.array, float]]:
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
        
    

