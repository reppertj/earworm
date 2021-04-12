from typing import List

from app.crud.base import CRUDBase
from app.models.embedding import Embedding
from app.schemas.embedding import (
    Embedding,
    EmbeddingCreate,
    EmbeddingUpdate,
)

embedding = CRUDBase[Embedding, EmbeddingCreate, EmbeddingUpdate](
    Embedding
)
