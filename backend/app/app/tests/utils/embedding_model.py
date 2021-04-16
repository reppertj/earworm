from typing import Optional

from sqlalchemy.orm import Session

from app import crud, models
from app.schemas.embedding_model import EmbeddingModelCreate
from app.tests.utils.utils import random_lower_string, random_url


def create_embedding_model(db: Session, *, name: str=None) -> models.Embedding_Model:
    if not name:
        name = random_lower_string()
    embedding_model_in = EmbeddingModelCreate(name=name)
    return crud.embedding_model.create(db=db, obj_in=embedding_model_in)
