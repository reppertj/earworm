from sqlalchemy.orm import Session

from app import crud
from app.models import track
from app.schemas.track import TrackCreate, TrackUpdate
from app.tests.utils.provider import create_random_provider
from app.tests.utils.license import create_random_license
from app.tests.utils.track import create_random_track
from app.tests.utils.utils import random_lower_string, random_url
from app.utils import canonical_preview_uri


def test_create_track(db: Session) -> None:
    title = random_lower_string()
    artist = random_lower_string()
    provider = create_random_provider(db)
    license = create_random_license(db)
    url = random_url()
    media_url = random_url()
    track_in = TrackCreate(
        title=title,
        artist=artist,
        provider_name=provider.name,
        license_name=license.name,
        url=url,
        media_url=media_url,
    )
    track_in.s3_preview_key = canonical_preview_uri(track_in)
    db_track = crud.track.create(db, obj_in=track_in)
    assert db_track.id
    assert db_track.title == title
    assert db_track.artist == artist
    assert db_track.provider_id == provider.id
    assert db_track.license_id == license.id
    assert db_track.url == url
    assert db_track.media_url == media_url
    assert db_track.license == license
    assert db_track.provider == provider
    assert db_track.embeddings == []
    assert db_track.s3_preview_key == track_in.s3_preview_key


def test_get_track(db: Session) -> None:
    title = random_lower_string()
    artist = random_lower_string()
    provider = create_random_provider(db)
    license = create_random_license(db)
    url = random_url()
    media_url = random_url()
    track_in = TrackCreate(
        title=title,
        artist=artist,
        provider_name=provider.name,
        license_name=license.name,
        url=url,
        media_url=media_url,
    )
    track_in.s3_preview_key = canonical_preview_uri(track_in)
    db_track = crud.track.create(db, obj_in=track_in)
    stored_track = crud.track.get(db=db, id=db_track.id)
    assert stored_track
    assert db_track.id == stored_track.id
    assert db_track.title == stored_track.title
    assert db_track.artist == stored_track.artist
    assert db_track.provider_id == stored_track.provider_id
    assert db_track.license_id == stored_track.license_id
    assert db_track.url == stored_track.url
    assert db_track.media_url == stored_track.media_url
    assert db_track.s3_preview_key == stored_track.s3_preview_key



def test_update_track(db: Session) -> None:
    title = random_lower_string()
    artist = random_lower_string()
    provider = create_random_provider(db)
    license = create_random_license(db)
    url = random_url()
    media_url = random_url()
    track_in = TrackCreate(
        title=title,
        artist=artist,
        provider_name=provider.name,
        license_name=license.name,
        url=url,
        media_url=media_url,
    )
    db_track = crud.track.create(db, obj_in=track_in) 
    url2 = random_url()
    provider2 = create_random_provider(db)
    track_update = TrackUpdate(id=db_track.id, url=url2, provider=provider2)
    track2 = crud.track.update(db=db, db_obj=db_track, obj_in=track_update)
    assert db_track.id == track2.id
    assert db_track.title == track2.title
    assert track2.url == url2
    assert track2.provider == provider2
    assert db_track.license == track2.license


def test_delete_track(db: Session) -> None:
    track_db = create_random_track(db)
    track_del_db = crud.track.remove(db=db, id=track_db.id)
    track3 = crud.track.get(db=db, id=track_db.id)
    assert track3 is None
    assert track_del_db.id == track_db.id
    assert track_del_db.title == track_db.title
    assert track_del_db.url == track_db.url
    assert track_del_db.license_id == track_db.license_id
