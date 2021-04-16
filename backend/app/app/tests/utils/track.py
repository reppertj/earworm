from typing import Optional

from sqlalchemy.orm import Session

from app import crud, models
from app.schemas.track import TrackCreate
from app.tests.utils.utils import random_lower_string, random_url
from app.utils import canonical_preview_uri

from .provider import create_random_provider
from .license import create_random_license


def create_random_track(db: Session, *, provider_name: Optional[str] = None, license_name: Optional[str] = None) -> models.Track:
    if provider_name is None:
        provider= create_random_provider(db)
        provider_name = provider.name
    if license_name is None:
        license = create_random_license(db)
        license_name = license.name
    track_in = TrackCreate(
        title = random_lower_string(),
        artist = random_lower_string(),
        url = random_url(),
        provider_name = provider_name,
        license_name = license_name,
        media_url = random_url()
    )
    track_in.s3_preview_key = canonical_preview_uri(track_in)
    return crud.track.create(db, obj_in=track_in)

    # item_in = ItemCreate(title=title, description=description, id=id)
    # return crud.item.create_with_owner(db=db, obj_in=item_in, owner_id=owner_id)
