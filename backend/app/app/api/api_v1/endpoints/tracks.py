from fastapi import APIRouter, Depends, Response, Query, UploadFile, File
from fastapi.exceptions import HTTPException
import typing as t

from app.api import deps
from app import crud, schemas, worker, utils
from app.core import object_store
from celery import group
from celery.result import AsyncResult

router = r = APIRouter()

@r.get(
    "/",
    response_model=t.List[schemas.Track],
    response_model_exclude_unset=True,
)
def tracks_list(
    response: Response,
    skip: int = 0,
    limit: int = Query(100, le=100),
    db=Depends(deps.get_db),
):
    """
    Get tracks
    """
    tracks = crud.track.get_multi(db, skip=skip, limit=limit)
    # This is necessary for react-admin to work
    response.headers["Content-Range"] = f"0-9/{len(tracks)}"
    return tracks


@r.get(
    "/{track_id}",
    response_model=schemas.Track,
    response_model_exclude_unset=True,
)
def track_details(
    track_id: int,
    db=Depends(deps.get_db),
):
    """
    Get any track details
    """
    track = crud.track.get(db, track_id)
    return track

@r.get(
    "/signed_preview_url/{track_id}",
)
def signed_preview_url(
    track_id: int,
    db=Depends(deps.get_db),
) -> t.Optional[str]:
    """
    Get a url to download a 30-second track preview, valid for 60 minutes
    """
    track = crud.track.get(db, track_id)
    if track:
        preview_url = object_store.create_presigned_url(track.s3_preview_key)
        return preview_url
    else:
        return None


@r.post(
    "/upload",
    response_model=schemas.TrackUploadStatus,
    dependencies=[Depends(deps.get_current_active_superuser)],
)
async def upload_tracks(csv_file: UploadFile = File(...)):
    """Upload a utf-8 encoded CSV file containing columns (with headings):
        - `provider`
        - `license`
        - `title`
        - `artist`
        - `external_url`
        - `mp3` (as url of audio file in format ffmpeg can read)
        - `internal_preview_uri` (optional)
    """
    if not csv_file.content_type.startswith("text"):
        raise HTTPException(415, "Should be CSV")
    content_bytes = await csv_file.read()
    content_str = t.cast(bytes, content_bytes).decode("utf-8")
    tracks_in = utils.parse_csv_string(content_str)  # Do this synchronously
    task = group((worker.import_track_task.s(track) for track in tracks_in))()
    return {"task_id": str(task), "status": "processing"}
