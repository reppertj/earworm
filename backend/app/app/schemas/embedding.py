from typing import Optional, List

from pydantic import BaseModel, HttpUrl

from .upload import UploadStatusBase
from .track import Track
from .embedding_model import EmbeddingModel


# Shared properties
class EmbeddingBase(BaseModel):
    track: Track
    model: EmbeddingModel
    values: List[float]
    class Config:
        orm_mode = True


# Properties to receive via API on creation
class EmbeddingCreate(EmbeddingBase):
    pass


# Properties to receive via API on update
class EmbeddingUpdate(EmbeddingBase):
    pass


class EmbeddingInDBBase(EmbeddingBase):
    pass


# Additional properties to return via API
class Embedding(EmbeddingBase):
    id: int


# Special async return types
class EmbeddingUploadStatus(UploadStatusBase):
    embeddings: Optional[List[Embedding]]
    details: Optional[str]


# Additional properties stored in DB
class EmbeddingInDB(EmbeddingInDBBase):
    pass
