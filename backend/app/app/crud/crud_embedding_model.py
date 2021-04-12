from typing import List

from app.crud.base import CRUDBase
from app.models.embedding_model import Embedding_Model
from app.schemas.embedding_model import (
    EmbeddingModel,
    EmbeddingModelCreate,
    EmbeddingModelUpdate,
)

embedding_model = CRUDBase[EmbeddingModel, EmbeddingModelCreate, EmbeddingModelUpdate](
    Embedding_Model
)
