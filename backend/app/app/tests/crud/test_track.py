from sqlalchemy.orm import Session

from app import crud
from app.schemas.track import TrackCreate, TrackUpdate
from app.tests.utils.provider import create_random_provider
from app.tests.utils.license import create_random_license
from app.tests.utils.utils import random_lower_string, random_url


def test_create_track(db: Session) -> None:
    title = random_lower_string()
    artist = random_lower_string()
    provider = create_random_provider()
    license = create_random_license()
    url = random_url()
    internal_media_uri = random_url()
    item_in = TrackCreate(title=title, artist=artist, )
    item = crud.item.create_with_owner(db=db, obj_in=item_in, owner_id=user.id)
    assert item.title == title
    assert item.description == description
    assert item.owner_id == user.id


def test_get_item(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    item_in = ItemCreate(title=title, description=description)
    user = create_random_user(db)
    item = crud.item.create_with_owner(db=db, obj_in=item_in, owner_id=user.id)
    stored_item = crud.item.get(db=db, id=item.id)
    assert stored_item
    assert item.id == stored_item.id
    assert item.title == stored_item.title
    assert item.description == stored_item.description
    assert item.owner_id == stored_item.owner_id


def test_update_item(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    item_in = ItemCreate(title=title, description=description)
    user = create_random_user(db)
    item = crud.item.create_with_owner(db=db, obj_in=item_in, owner_id=user.id)
    description2 = random_lower_string()
    item_update = ItemUpdate(description=description2)
    item2 = crud.item.update(db=db, db_obj=item, obj_in=item_update)
    assert item.id == item2.id
    assert item.title == item2.title
    assert item2.description == description2
    assert item.owner_id == item2.owner_id


def test_delete_item(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    item_in = ItemCreate(title=title, description=description)
    user = create_random_user(db)
    item = crud.item.create_with_owner(db=db, obj_in=item_in, owner_id=user.id)
    item2 = crud.item.remove(db=db, id=item.id)
    item3 = crud.item.get(db=db, id=item.id)
    assert item3 is None
    assert item2.id == item.id
    assert item2.title == title
    assert item2.description == description
    assert item2.owner_id == user.id
