from typing import List, Optional

from pydantic import BaseModel, HttpUrl

from .license import License
from .provider import Provider
from .upload import UploadStatusBase


# Shared properties
class TrackBase(BaseModel):
    title: str
    artist: Optional[str]
    url: HttpUrl
    license: Optional[License]
    provider: Optional[Provider]

    class Config:
        orm_mode = True


# Properties to receive via API on creation
class TrackCreate(TrackBase):
    media_url: Optional[HttpUrl]
    license_name: str
    provider_name: str
    s3_preview_key: Optional[str]



# Properties to receive via API on update
class TrackUpdate(TrackCreate):
    id: int


class TrackInDBBase(TrackBase):
    id: int
    media_url: Optional[HttpUrl]
    s3_preview_key: Optional[str]


# Additional properties to return via API
class Track(TrackBase):
    id: int
    temp_preview_url: Optional[HttpUrl]


# Special async return types
class TrackUploadStatus(UploadStatusBase):
    tracks: Optional[List[Track]]
    details: Optional[str]


# Additional properties stored in DB
class TrackInDB(TrackInDBBase):
    pass
