import io
from tempfile import TemporaryDirectory

import responses
from sqlalchemy.orm import Session
import numpy as np

from app.schemas import TrackCreate
from app.tests.utils.license import create_random_license
from app.tests.utils.provider import create_random_provider
from app.tests.utils.utils import random_lower_string, random_url
from app.utils import canonical_preview_uri
from app.core import object_store
from app.worker_utils.io import import_track


@responses.activate
def test_import_track(db: Session):
    with open("app/tests/support/example.mp3", "rb") as f:
        responses.add(
            responses.GET,
            "http://example.com/sample.mp3",
            content_type="audio/mpeg",
            body=f,  # type: ignore
        )

        title = random_lower_string()
        artist = random_lower_string()
        provider = create_random_provider(db)
        license = create_random_license(db)
        url = random_url()
        media_url = "http://example.com/sample.mp3"
        track_in = TrackCreate(
            title=title,
            artist=artist,
            provider_name=provider.name,
            license_name=license.name,
            url=url,
            media_url=media_url,
        )
        db_track = import_track(db, track_in)
        on_s3 = object_store.object_on_s3(db_track.s3_preview_key)
        assert on_s3 in [None, True]
        if on_s3:
            object_store.remove_from_s3(db_track.s3_preview_key)
        assert db_track.title == title
        assert db_track.url == url
        assert db_track.s3_preview_key == canonical_preview_uri(track_in)
        assert db_track.embeddings
        assert db_track.embeddings[0].track_id == db_track.id
        as_ary = np.array(db_track.embeddings[0].values)
        assert np.isclose(np.linalg.norm(as_ary), 1.)


