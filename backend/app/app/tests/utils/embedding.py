from typing import Optional

from sqlalchemy.orm import Session

from app import crud, models
from app.schemas.embedding import EmbeddingCreate
from .track import create_random_track
from .utils import random_unit_vector
from .embedding_model import create_embedding_model


def create_random_embedding(
    db: Session,
    *,
    track_id: Optional[int] = None,
    model_name: Optional[str] = None
) -> models.Embedding:
    if track_id is None:
        track = create_random_track(db)
        track_id = track.id
    if model_name is None:
        embedding_model = create_embedding_model(db)
        model_name = embedding_model.name
    embedding_in = EmbeddingCreate(
        track_id=track_id,
        model_name=model_name,
        values=random_unit_vector(),
    )
    return crud.embedding.create(db=db, obj_in=embedding_in)
