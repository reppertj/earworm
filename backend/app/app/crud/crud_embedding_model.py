from typing import List

from sqlalchemy.orm.session import Session

from app.crud.base import CRUDBase
from app.models.embedding_model import Embedding_Model
from app.schemas.embedding_model import (
    EmbeddingModelCreate,
    EmbeddingModelUpdate,
)

class EmbeddingModelBase(CRUDBase[Embedding_Model, EmbeddingModelCreate, EmbeddingModelUpdate]):
    def get_by_name(self, db: Session, *, name: str):
        return db.query(self.model).filter(Embedding_Model.name == name).first()


embedding_model = EmbeddingModelBase(Embedding_Model)