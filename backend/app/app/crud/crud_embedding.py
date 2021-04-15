from typing import List
from fastapi.encoders import jsonable_encoder

from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.embedding import Embedding
from app.models.embedding_model import Embedding_Model
from app.schemas.embedding import (
    EmbeddingCreate,
    EmbeddingUpdate,
)


class EmbeddingBase(CRUDBase[Embedding, EmbeddingCreate, EmbeddingUpdate]):
    def create(self, db: Session, *, obj_in: EmbeddingCreate):
        obj_in_data = jsonable_encoder(obj_in)
        db_embedding_model = (
            db.query(Embedding_Model).filter_by(name=obj_in_data["model_name"]).first()
        )
        db_obj = self.model(
            track_id=obj_in_data["track_id"],
            embedding_model_id=db_embedding_model.id,
            values=obj_in_data["values"],
        )  # type: ignore
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(self, db: Session, *, db_obj, obj_in):
        obj_data = jsonable_encoder(db_obj)
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        if "model_name" in update_data:
            db_embedding_model = (
                db.query(Embedding_Model)
                .filter_by(name=update_data["model_name"])
                .first()
            )
            update_data["embedding_model_id"] = db_embedding_model.id
            del update_data["model_name"]
        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_embeddings_by_embedding_model_name(
        self, db: Session, *, embed_model_name: str
    ) -> List[Embedding]:
        subq = db.query(Embedding_Model).filter_by(name=embed_model_name).subquery()
        return (
            db.query(Embedding)
            .join(subq, Embedding.embedding_model_id == subq.c.id)
            .all()
        )

    def get_by_embedding_model_name_track_id(
        self,
        db: Session,
        *,
        embed_model_name: str,
        track_id: int,
    ) -> Embedding:
        subq = db.query(Embedding_Model).filter_by(name=embed_model_name).subquery()
        return (
            db.query(Embedding)
            .join(subq, Embedding.embedding_model_id == subq.c.id)
            .filter_by(track_id=track_id)
            .first()
        )


# subq = session.query(Address).\
#     filter(Address.email_address == 'ed@foo.com').\
#     subquery()

# q = session.query(User).join(
#     subq, User.id == subq.c.user_id
# )

embedding = EmbeddingBase(Embedding)