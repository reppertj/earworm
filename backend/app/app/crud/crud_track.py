from typing import List

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.track import Track
from app.schemas.track import TrackCreate, TrackUpdate


class CRUDTrack(CRUDBase[Track, TrackCreate, TrackUpdate]):
    def create(self, db: Session, *, obj_in: TrackCreate) -> Track:
        obj_in_data = jsonable_encoder(obj_in)
        # TODO: Figure out generic way for nested pydantic models & sqlalchemy to play nice
        obj_in_data["provider_id"] = obj_in_data["provider"]["id"]
        del obj_in_data["provider"]
        obj_in_data["license_id"] = obj_in_data["license"]["id"]
        del obj_in_data["license"]
        db_obj = self.model(**obj_in_data)  # type: ignore
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def create_with_license_provider(
        self, db: Session, *, obj_in: TrackCreate, license_id: int, provider_id: int
    ) -> Track:
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_multi_by_provider(
        self, db: Session, *, provider_id: int, skip: int = 0, limit: int = 100
    ) -> List[Track]:
        return (
            db.query(self.model)
            .filter(Track.provider.id == provider_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_by_preview_url(self, db: Session, *, s3_preview_key: str):
        return (
            db.query(self.model)
            .filter(Track.s3_preview_key == s3_preview_key)
            .first()
        )

    def get_multi_by_license(
        self, db: Session, *, license_id: int, skip: int = 0, limit: int = 100
    ) -> List[Track]:
        return (
            db.query(self.model)
            .filter(Track.license.id == license_id)
            .offset(skip)
            .limit(limit)
            .all()
        )


track = CRUDTrack(Track)
