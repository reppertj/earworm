from typing import Optional, List

from pydantic import BaseModel, HttpUrl

from .upload import UploadStatusBase


# Shared properties
class ProviderBase(BaseModel):
    name: str
    url: HttpUrl

    class Config:
        orm_mode = True


# Properties to receive via API on creation
class ProviderCreate(ProviderBase):
    pass


# Properties to receive via API on update
class ProviderUpdate(ProviderBase):
    pass


class ProviderInDBBase(ProviderBase):
    pass


# Additional properties to return via API
class Provider(ProviderBase):
    id: int


# Special async return types
class ProviderUploadStatus(UploadStatusBase):
    providers: Optional[List[Provider]]
    details: Optional[str]


# Additional properties stored in DB
class ProviderInDB(ProviderInDBBase):
    pass
