from typing import Optional

from sqlalchemy.orm import Session

from app import crud, models
from app.schemas.license import LicenseCreate
from app.tests.utils.utils import random_lower_string, random_url


def create_random_license(db: Session) -> models.License:
    name = random_lower_string()
    url = random_url()
    license_in = LicenseCreate(name=name, url=url)
    return crud.license.create(db=db, obj_in=license_in)
