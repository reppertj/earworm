from typing import Any, Dict, List, Union

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.crud.crud_license import license
from app.crud.crud_provider import provider
from app.models.track import Track
from app.schemas.track import TrackCreate, TrackUpdate


class CRUDTrack(CRUDBase[Track, TrackCreate, TrackUpdate]):
    def create(self, db: Session, *, obj_in: TrackCreate) -> Track:
        obj_in_data = jsonable_encoder(obj_in)
        db_license = obj_in_data.get("license") or license.get_by_name(
            db, name=obj_in.license_name
        )
        db_provider = obj_in_data.get("provider") or provider.get_by_name(
            db, name=obj_in.provider_name
        )
        db_obj = self.model(
            title=obj_in.title,
            artist=obj_in.artist,
            s3_preview_key=obj_in.s3_preview_key,
            url=obj_in.url,
            media_url=obj_in.media_url,
            license_id=db_license.id,
            provider_id=db_provider.id,
        )  # type: ignore
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self, db: Session, *, db_obj: Track, obj_in: Union[TrackUpdate, Dict[str, Any]]
    ) -> Track:
        obj_data = jsonable_encoder(db_obj)
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        if db_license := update_data.get("license"):
            update_data["license_id"] = db_license["id"]
            del update_data["license"]
        if db_provider := update_data.get("provider"):
            update_data["provider_id"] = db_provider["id"]
            del update_data["provider"]
        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_by_title_artist_provider_name(
        self, db: Session, *, title: str, artist: str, provider_name: str
    ):
        db_provider = provider.get_by_name(db, name=provider_name)
        if db_provider:
            provider_id = db_provider.id

            return (
                db.query(self.model)
                .filter_by(title=title, artist=artist, provider_id=provider_id)
                .first()
            )

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
            db.query(self.model).filter(Track.s3_preview_key == s3_preview_key).first()
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
