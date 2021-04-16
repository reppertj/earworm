from typing import List, Optional

from pydantic import BaseModel, HttpUrl

from .license import License
from .provider import Provider
from .embedding import Embedding
from .upload import UploadStatusBase


# Shared properties
class TrackBase(BaseModel):
    title: Optional[str]
    artist: Optional[str]
    url: Optional[HttpUrl]
    license: Optional[License]
    provider: Optional[Provider]

    class Config:
        orm_mode = True


# Properties to receive via API on creation
class TrackCreate(TrackBase):
    title: str
    artist: str
    url: HttpUrl
    media_url: Optional[HttpUrl]
    license_name: str
    provider_name: str
    s3_preview_key: Optional[str]



# Properties to receive via API on update
class TrackUpdate(TrackBase):
    id: int


class TrackInDBBase(TrackBase):
    id: int
    title: str
    url: HttpUrl
    media_url: Optional[HttpUrl]
    s3_preview_key: Optional[str]


# Additional properties to return via API
class Track(TrackBase):
    id: int
    temp_preview_url: Optional[HttpUrl]


class TrackKNNResult(Track):
    percent_match: Optional[float]


# Special async return types
class TrackUploadStatus(UploadStatusBase):
    tracks: Optional[List[Track]]
    details: Optional[str]


# Additional properties stored in DB
class TrackInDB(TrackInDBBase):
    pass
