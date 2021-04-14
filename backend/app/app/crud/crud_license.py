from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.license import License
from app.schemas.license import LicenseCreate, LicenseUpdate


class CRUDLicense(CRUDBase[License, LicenseCreate, LicenseUpdate]):
    def get_by_name(self, db: Session, name: str) -> License:
        return db.query(self.model).filter(self.model.name == name).first()


license = CRUDLicense(License)
