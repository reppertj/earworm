from typing import Optional, List

from pydantic import BaseModel, HttpUrl

from .upload import UploadStatusBase


# Shared properties
class EmbeddingModelBase(BaseModel):
    name: str

    class Config:
        orm_mode = True


# Properties to receive via API on creation
class EmbeddingModelCreate(EmbeddingModelBase):
    pass


# Properties to receive via API on update
class EmbeddingModelUpdate(EmbeddingModelBase):
    pass


class EmbeddingModelInDBBase(EmbeddingModelBase):
    pass


# Additional properties to return via API
class EmbeddingModel(EmbeddingModelBase):
    id: int


# Special async return types
class EmbeddingModelUploadStatus(UploadStatusBase):
    embeddings: Optional[List[EmbeddingModel]]
    details: Optional[str]


# Additional properties stored in DB
class EmbeddingModelInDB(EmbeddingModelInDBBase):
    pass
