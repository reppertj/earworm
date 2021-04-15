from sqlalchemy.orm import Session

from app import crud, schemas
from app.core.config import settings
from app.db import base  # noqa: F401

# make sure all SQL Alchemy models are imported (app.db.base) before initializing DB
# otherwise, SQL Alchemy might fail to initialize relationships properly
# for more details: https://github.com/tiangolo/full-stack-fastapi-postgresql/issues/28


def init_db(db: Session) -> None:

    user = crud.user.get_by_email(db, email=settings.FIRST_SUPERUSER)
    if not user:
        user_in = schemas.UserCreate(
            email=settings.FIRST_SUPERUSER,
            password=settings.FIRST_SUPERUSER_PASSWORD,
            is_superuser=True,
        )
        crud.user.create(db, obj_in=user_in)
    embedding_model = crud.embedding_model.get_by_name(db, name=settings.ACTIVE_MODEL_NAME)
    if not embedding_model:
        model_in = schemas.EmbeddingModelCreate(
            name=settings.ACTIVE_MODEL_NAME
        )
        crud.embedding_model.create(db, obj_in=model_in)
