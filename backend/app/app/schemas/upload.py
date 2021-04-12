
from typing import Literal

from pydantic import BaseModel


class UploadStatusBase(BaseModel):
    task_id: str
    status: Literal["processing", "failed", "success"]


class UploadStatus(UploadStatusBase):
    pass