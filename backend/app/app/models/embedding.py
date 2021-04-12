from typing import TYPE_CHECKING

from sqlalchemy import Column, Integer, String, ARRAY, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql.schema import ForeignKey

from app.db.base_class import Base

if TYPE_CHECKING:
    from .embedding_model import Embedding_Model  # noqa: F401


class Embedding(Base):
    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, ForeignKey("track.id"))
    track = relationship("Track", back_populates="embeddings")
    embedding_model_id = Column(Integer, ForeignKey("embedding_model.id"))
    embedding_model = relationship("Embedding_Model", back_populates="embeddings")
    values = Column(ARRAY(Float(precision=32)), nullable=False)
