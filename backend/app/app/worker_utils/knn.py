import numpy as np
from typing import List, Optional, Tuple, Union
from app import crud
from app.core.config import settings
from app.api.deps import SessionLocal
from app.worker_utils.metrics import spherical_mean
import faiss

import logging


class KNN:
    def __init__(self, db):
        embeddings = crud.embedding.get_embeddings_by_embedding_model_name(
            db, embed_model_name=settings.ACTIVE_MODEL_NAME
        )

        EMBEDDING_DIM = 128
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))
        self.index.add_with_ids(
            x=np.array(
                [embedding.values for embedding in embeddings], dtype=np.float32
            ), ids=np.array([embedding.track_id for embedding in embeddings], dtype=np.int64)
        )
        self.num_embeddings = self.index.ntotal
        logging.info(f"The KNN Index contains {self.index.ntotal} vectors")

    def __call__(
        self,
        query: Union[List[List[float]], List[float], np.ndarray],
        k,
        weights: Optional[List[float]] = None,
    ) -> Tuple[List[int], List[float]]:
        """Returns KNN by inner product for query vector and percent
        similarities, with assumption that queries are normalized to the unit hypersphere.

        Input numpy array should already be in shape [bs, 128]. This includes where
        bs = 1, so input shape should be [1, 128].

        If multiple queries are provided in a list of lists or a numpy array,
        queries are treated as multiple embeddings to be reduced to a single query
        via their (possibly weighted) Frechet mean with a 127-sphere metric.
        """
        if isinstance(query[0], list):
            as_array = spherical_mean(query, weights=weights)[np.newaxis, :]
        elif isinstance(query) and query.shape[0] > 1:
            as_array = spherical_mean(query, weights=weights)[np.newaxis, :]
        else:
            as_array = np.array(query, dtype=np.float32)[np.newaxis, :]
        k = min(k, self.num_embeddings)
        simals, track_ids = self.index.search(as_array, k)
        pcts = [
            pct.item() for pct in ((simals[0] + 1) * 50).round()
        ]  # Cosine similarity -> %
        return [tid.item() for tid in track_ids[0]], pcts


knn = None


def get_knn(db):
    global knn
    if knn is None:
        knn = KNN(db)
    return knn


def reset_knn(db):
    global knn
    knn = KNN(db)
