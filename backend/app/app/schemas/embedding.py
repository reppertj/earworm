from __future__ import annotations

from typing import Optional, List

from pydantic import BaseModel

from .upload import UploadStatusBase
from .embedding_model import EmbeddingModel



# Shared properties
class EmbeddingBase(BaseModel):
    values: List[float]
    class Config:
        orm_mode = True


# Properties to receive via API on creation
class EmbeddingCreate(EmbeddingBase):
    track_id: int
    model_name: str


# Properties to receive via API on update
class EmbeddingUpdate(EmbeddingBase):
    track_id: int


class EmbeddingInDBBase(EmbeddingBase):
    id: int
    model: EmbeddingModel


# Additional properties to return via API
class Embedding(EmbeddingBase):
    id: int
    model: EmbeddingModel


# Special async return types
class EmbeddingUploadStatus(UploadStatusBase):
    embeddings: Optional[List[Embedding]]
    details: Optional[str]


# Additional properties stored in DB
class EmbeddingInDB(EmbeddingInDBBase):
    pass
