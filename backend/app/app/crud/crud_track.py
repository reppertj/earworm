from typing import List

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.track import Track as TrackModel
from app.schemas.track import Track, TrackCreate, TrackUpdate


class CRUDTrack(CRUDBase[Track, TrackCreate, TrackUpdate]):
    def create_with_license_provider(
        self, db: Session, *, obj_in: TrackCreate, license_id: int, provider_id: int
    ) -> Track:
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data, license_id=license_id, provider_id=provider_id)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_multi_by_provider(
        self, db: Session, *, provider_id: int, skip: int = 0, limit: int = 100
    ) -> List[Track]:
        return (
            db.query(self.model)
            .filter(Track.provider_id == provider_id)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_multi_by_license(
        self, db: Session, *, license_id: int, skip: int = 0, limit: int = 100
    ) -> List[Track]:
        return (
            db.query(self.model)
            .filter(Track.license_id == license_id)
            .offset(skip)
            .limit(limit)
            .all()
        )


track = CRUDTrack(TrackModel)
