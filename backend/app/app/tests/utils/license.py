from typing import Optional

from sqlalchemy.orm import Session

from app import crud, models
from app.schemas.license import LicenseCreate
from app.tests.utils.utils import random_lower_string, random_url


def create_random_license(db: Session) -> models.License:
    name = random_lower_string()
    external_link = random_url()
    license_in = LicenseCreate(name=name, external_link=external_link)
    return crud.license.create(db=db, obj_in=license_in)
