import os
import tempfile

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app import crud
from app.core.config import settings
from app.tests.utils.track import create_random_track


def test_tracks_get(client: TestClient, db: Session) -> None:
    db.connection().execute("TRUNCATE TABLE track CASCADE")
    db.commit()
    tracks = crud.track.get_multi(db, skip=0, limit=None)
    for track in tracks:
        crud.track.remove(db, id=track.id)
    track_dbs = [create_random_track(db) for _ in range(8)]
    response = client.get(f"{settings.API_V1_STR}/tracks/")
    assert response.status_code == 200
    content = response.json()
    assert len(content) == 8
    assert content[4]["title"] == track_dbs[4].title
    assert content[2]["url"] == track_dbs[2].url
    assert content[0]["id"] == track_dbs[0].id

    response = client.get(f"{settings.API_V1_STR}/tracks/{track_dbs[3].id}")
    content = response.json()
    assert isinstance(content, dict)
    assert content["id"] == track_dbs[3].id
    assert content["title"] == track_dbs[3].title

