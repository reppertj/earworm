from typing import Union, List, Optional
import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean

_sphere = Hypersphere(dim=127)
_mean = FrechetMean(metric=_sphere.metric)

def spherical_mean(
    embeddings: Union[List[List[float]], List[np.ndarray], np.ndarray],
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Expects (bs, 128) structure"""
    if isinstance(embeddings, list):
        if isinstance(embeddings[0], list):
            embeddings = np.array(embeddings, dtype=np.float32)
        else:
            embeddings = np.vstack(embeddings).astype(np.float32)
    return _mean.fit(embeddings, weights=weights).estimate_.astype(np.float32)
