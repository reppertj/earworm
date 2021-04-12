from typing import List

from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.provider import Provider as ProviderModel
from app.schemas.provider import Provider, ProviderCreate, ProviderUpdate

provider = CRUDBase[Provider, ProviderCreate, ProviderUpdate](ProviderModel)
