from typing import Optional, List

from pydantic import BaseModel, HttpUrl

from .upload import UploadStatusBase


# Shared properties
class LicenseBase(BaseModel):
    name: str
    url: HttpUrl

    class Config:
        orm_mode = True


# Properties to receive via API on creation
class LicenseCreate(LicenseBase):
    pass


# Properties to receive via API on update
class LicenseUpdate(LicenseBase):
    pass


class LicenseInDBBase(LicenseBase):
    pass


# Additional properties to return via API
class License(LicenseBase):
    id: int


# Special async return types
class LicenseUploadStatus(UploadStatusBase):
    licenses: Optional[List[License]]
    details: Optional[str]


# Additional properties stored in DB
class LicenseInDB(LicenseInDBBase):
    pass
