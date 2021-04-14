from typing import Union, List, Optional
import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean

sphere = Hypersphere(dim=127)


def spherical_mean(
    embeddings: Union[List[np.ndarray], np.ndarray],
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Expects list of (128,) embeddings or (bs, 128) array of embeddings
    """
    if isinstance(embeddings, list):
        embeddings = np.vstack(embeddings)
    mean = FrechetMean(metric=sphere.metric)
    return mean.fit(embeddings, weights=weights).estimate_
