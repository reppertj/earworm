from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.provider import Provider
from app.schemas.provider import ProviderCreate, ProviderUpdate


class CRUDProvider(CRUDBase[Provider, ProviderCreate, ProviderUpdate]):
    def get_by_name(self, db: Session, name: str) -> Provider:
        return db.query(self.model).filter(self.model.name == name).first()


provider = CRUDProvider(Provider)
