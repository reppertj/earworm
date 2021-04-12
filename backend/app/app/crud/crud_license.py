from typing import List

from app.crud.base import CRUDBase
from app.models.license import License as LicenseModel
from app.schemas.license import License, LicenseCreate, LicenseUpdate

license = CRUDBase[License, LicenseCreate, LicenseUpdate](LicenseModel)
