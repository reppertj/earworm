import logging
import os
import warnings
import shutil
import tempfile
from typing import List, Literal, Optional, Union, cast
from pydantic import HttpUrl
import librosa
import ffmpeg

from app.core.config import settings
from app.worker_utils.metrics import spherical_mean
from app.worker_utils.inference import get_inferer

import requests
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app import crud, schemas
from app.core import object_store
from app.utils import canonical_preview_uri


def provider_license_in_db(db: Session, track: schemas.TrackCreate):
    provider = crud.provider.get_by_name(db, name=track.provider_name)
    license = crud.license.get_by_name(db=db, name=track.license_name)
    return True if provider and license else False


def convert_trim_30_seconds_overwrite(audio_path) -> float:
    """ Converts to mp3, trims to middle 30 seconds, and returns the final length """
    length = librosa.get_duration(filename=audio_path)
    if length > 30:
        start, end = (length / 2 - 15), (length / 2 + 15)
    else:
        start, end = 0, length
    with tempfile.TemporaryDirectory() as temp_dir:
        _, extension = os.path.splitext(audio_path)
        temp_in_path = os.path.join(temp_dir, "in" + extension)
        shutil.copyfile(audio_path, temp_in_path)
        audio_in = ffmpeg.input(temp_in_path, ss=start, to=end)["a"]  # Just get audio
        audio_out = ffmpeg.output(audio_in, audio_path)
        audio_out = ffmpeg.overwrite_output(audio_out)
        ffmpeg.run(audio_out)
    return end - start


def get_up_to_n_samples(audio_path, n):
    """Return up to n ndarrays of shape [1, 66150] (fewer depending on length)"""
    TARGET_SAMPLE_RATE = 22050
    SPECTROGRAM_SECONDS = 3
    length = librosa.get_duration(filename=audio_path)
    # We don't get good results for < 10 seconds
    if length < 10:
        raise ValueError("Audio too short for model")
    n = min(
        int(length // SPECTROGRAM_SECONDS - 2), n
    )  # Conservative value here to anticipate FP/rounding errors
    starts = [(c + 1) * length // (n + 1) for c in range(0, n)]
    return [
        librosa.load(
            path=audio_path,
            sr=TARGET_SAMPLE_RATE,
            mono=True,
            offset=start,
            duration=SPECTROGRAM_SECONDS,
            dtype="float32",
        )[0].reshape(1, -1)
        for start in starts
    ]


def download_media(file_path: str, media_url: str) -> Literal[True]:
    """Downloads media to `file_path`;"""
    sess = requests.Session()
    try:
        resp = sess.get(media_url, timeout=0.01)
        with open(file_path, "wb") as f:
            f.write(resp.content)
        return True
    except:
        raise
    finally:
        sess.close()


def download_from_s3_or_original(
    dest_path: str,
    already_on_s3: Optional[bool],
    s3_key: Optional[str] = None,
    orig_url: Optional[HttpUrl] = None,
) -> Literal["original", "s3"]:
    """ Prioritizes S3 over original """
    if already_on_s3 and s3_key:
        downloaded_from_s3 = object_store.download_from_s3(
            key=s3_key, file_path=dest_path
        )
        return "s3"
    if not orig_url:
        raise ValueError("No media available")
    else:  # Only download if not already on s3
        download_media(file_path=dest_path, media_url=orig_url)
        return "original"


def import_track(db: Session, track: schemas.TrackCreate) -> Union[schemas.Track, str]:
    """Adds track to DB, puts preview on S3, and adds embedding to DB
    Does not perform inference/add embeddings to DB.
    Raises error and does not add to DB if:
    - the provider or license are not in the DB
    - S3 environment variables are not present and FALLBACK_TO_ORIGINAL_PREVIEWS flag not set
    - a preview is already on S3 and a corresponding track exists in the DB
    - a preview is not already on S3 and the audio file is <10 seconds long
    - a preview is not already on S3 and there is an error downloading the original
    """
    if not provider_license_in_db(db=db, track=track):
        raise ValueError("Provider or license not in DB")
    with tempfile.TemporaryDirectory() as tempdir:
        if not track.s3_preview_key:
            track.s3_preview_key = canonical_preview_uri(track)
        temppath = cast(str, os.path.join(tempdir, track.s3_preview_key))
        already_on_s3 = object_store.object_on_s3(track.s3_preview_key)
        if already_on_s3 is None and not settings.FALLBACK_TO_ORIGINAL_PREVIEWS:
            raise ValueError("S3 not configured")
        source = download_from_s3_or_original(
            dest_path=temppath,
            already_on_s3=already_on_s3,
            s3_key=track.s3_preview_key,
            orig_url=track.media_url,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveforms = get_up_to_n_samples(temppath, n=10)
        embeddings = [get_inferer()(w) for w in waveforms]
        mean_embedding = list(spherical_mean(embeddings))
        try:
            db_track = crud.track.create(db=db, obj_in=track)
        except IntegrityError:
            db.rollback()
            db_track = crud.track.get_by_title_artist_provider_name(
                db=db,
                title=track.title,
                artist=track.artist,
                provider_name=track.provider_name,
            )
            db_track = crud.track.update(db=db, db_obj=db_track, obj_in=track)
        if source == "original":
            convert_trim_30_seconds_overwrite(temppath)
            already_on_s3 = object_store.upload_to_s3(
                key=track.s3_preview_key, file_path=temppath, check_first=False
            )
            if already_on_s3 is None:
                track.s3_preview_key = None
        embedding_in = schemas.EmbeddingCreate(
            track_id=db_track.id,
            model_name=settings.ACTIVE_MODEL_NAME,
            values=mean_embedding,
        )
        try:
            db_embedding = crud.embedding.create(db, obj_in=embedding_in)
        except IntegrityError:
            db.rollback()
            db_embedding = crud.embedding.get_by_embedding_model_name_track_id(
                db=db, embed_model_name=settings.ACTIVE_MODEL_NAME, track_id=db_track.id
            )
            db_embedding = crud.embedding.update(db=db, db_obj=db_embedding, obj_in=embedding_in)
        return db_track
