import logging
import os
import shutil
import tempfile
from typing import List, Optional, Union, cast
import librosa
import ffmpeg
import numpy as np

import requests
from sqlalchemy.orm import Session

from app import crud, schemas
from app.core import object_store
from app.utils import canonical_preview_uri


def provider_license_in_db(db: Session, track: schemas.TrackCreate):
    provider = crud.provider.get(db=db, id=track.provider.id)
    license = crud.license.get(db=db, id=track.license.id)
    return True if provider and license else False


def convert_trim_30_seconds_overwrite(audio_path) -> float:
    """ Converts to mp3, trims to middle 30 seconds, and returns the final length """
    length = librosa.get_duration(filename=audio_path)
    if length > 30:
        start, end = (length / 2 - 15), (length / 2 + 15)
    else:
        start, end = 0, length
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_in_path = os.path.join(temp_dir, "in.mp3")
        shutil.copyfile(audio_path, temp_in_path)
        audio_in = ffmpeg.input(temp_in_path, ss=start, to=end)["a"]  # Just get audio
        audio_out = ffmpeg.output(audio_in, audio_path)
        audio_out = ffmpeg.overwrite_output(audio_out)
        ffmpeg.run(audio_out)
    return end - start


def get_up_to_n_samples(audio_path, n):
    """Return up to n np arrays of shape [1, 66150] (less depending on length)"""
    TARGET_SAMPLE_RATE = 22050
    length = librosa.get_duration(filename=audio_path)
    # We don't get good results for < 10 seconds
    if length < 10:
        raise ValueError("Audio too short for model")
    n = min(int(length // 3 - 2), n)  # Conservative value here to anticipate FP/rounding errors
    starts = [(c + 1) * length // (n + 1) for c in range(0, n)]
    return [
        librosa.load(
            path=audio_path,
            sr=TARGET_SAMPLE_RATE,
            mono=True,
            offset=start,
            duration=3,
            dtype="float32",
        )[0].reshape(1, -1)
        for start in starts
    ]


def download_preview(file_path: str, mp3: str) -> bool:
    """Downloads mp3 to `file_path`;
    logs a warning and returns False upon requests exception
    """
    sess = requests.Session()
    try:
        resp = sess.get(mp3)
        with open(file_path, "wb") as f:
            f.write(resp.content)
        return True
    except requests.exceptions.RequestException as e:
        logging.warning(
            f"Request exception getting {mp3}",
            exc_info=e,
        )
        return False
    finally:
        sess.close()


def import_track(
    db: Session, track: schemas.TrackCreate, mp3: Optional[str] = None
) -> Union[schemas.Track, str]:
    """Adds track to DB and puts preview on S3.
    Does not perform inference/add embeddings to DB.
    Raises error and does not add to DB if:
    - the provider or license are not in the DB
    - S3 environment variables are not present
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
        if already_on_s3 is None:  # S3 not configured
            raise ValueError("S3 not configured")
        elif already_on_s3:
            if crud.track.get_by_preview_url(
                db=db, s3_preview_key=track.s3_preview_key
            ):  # s3 key already in DB
                raise ValueError("Track with key already in DB")
            object_store.download_from_s3(key=track.s3_preview_key, filename=temppath)
        else:
            raise Exception(
                f"{object_store.download_from_s3(key=track.s3_preview_key, filename=temppath)}"
            )
            if not mp3 or not download_preview(
                file_path=temppath, mp3=mp3
            ):  # http error/no mp3
                raise ValueError("Error downloading audio file")
            elif convert_trim_30_seconds_overwrite(temppath) < 10:  # too short
                raise ValueError("audio file shorter than 10 seconds")
            object_store.upload_to_s3(temppath)
        track_db = crud.track.create(db=db, obj_in=track)
        return track_db
