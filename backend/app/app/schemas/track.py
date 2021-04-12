from typing import Optional, List

from pydantic import BaseModel, HttpUrl

from .upload import UploadStatusBase
from .license import License
from .provider import Provider


# Shared properties
class TrackBase(BaseModel):
    title: str
    artist: Optional[str]
    internal_media_url: Optional[str]
    external_link: HttpUrl
    license: License
    provider: Provider
    class Config:
        orm_mode = True


# Properties to receive via API on creation
class TrackCreate(TrackBase):
    pass


# Properties to receive via API on update
class TrackUpdate(TrackBase):
    pass


class TrackInDBBase(TrackBase):
    pass


# Additional properties to return via API
class Track(TrackBase):
    id: int


# Special async return types
class TrackUploadStatus(UploadStatusBase):
    tracks: Optional[List[Track]]
    details: Optional[str]


# Additional properties stored in DB
class TrackInDB(TrackInDBBase):
    pass
