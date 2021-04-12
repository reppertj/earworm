from fastapi import APIRouter, Depends, Response, Query
import typing as t

from app.api import deps
from app import crud, schemas

router = r = APIRouter()

@r.get(
    "/",
    response_model=t.List[schemas.Track],
    response_model_exclude_unset=True,
)
async def tracks_list(
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
async def track_details(
    track_id: int,
    db=Depends(deps.get_db),
):
    """
    Get any track details
    """
    track = crud.track.get(db, track_id)
    return track

