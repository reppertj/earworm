from typing import Optional

from sqlalchemy.orm import Session

from app import crud, models
from app.schemas.provider import ProviderCreate
from app.tests.utils.utils import random_lower_string, random_url


def create_random_provider(db: Session) -> models.License:
    name = random_lower_string()
    url = random_url()
    provider_in = ProviderCreate(name=name, url=url)
    return crud.provider.create(db=db, obj_in=provider_in)
