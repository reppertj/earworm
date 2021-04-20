from fastapi import APIRouter, Depends, Response, Body, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
import typing as t

from starlette.responses import JSONResponse

from app.api import deps
from app import crud, schemas, worker, utils
from app.schemas.track import TrackKNNResult
from app.worker_utils.knn import get_knn, reset_knn
from app.core import object_store
from celery import group
from celery.result import AsyncResult

router = r = APIRouter()


@r.post(
    "/search",
    response_model=t.List[schemas.TrackKNNResult],
    response_model_exclude_none=True,
)
def knn_tracks(
    embeddings: t.List[t.List[float]] = Body(...),
    k: int = Body(30, le=300),
    db=Depends(deps.get_db),
) -> t.List[schemas.TrackKNNResult]:
    """Search for k nearest neighbors of spherical mean of embeddings"""
    # TODO: How compute intensive is this? Will I need to offload to a worker?
    if len(embeddings[0]) != 128:
        raise HTTPException(status_code=422, detail="Embedding size should be 128")
    elif len(embeddings) > 10:
        raise HTTPException(
            status_code=422, detail="At most 10 embeddings supported in query"
        )
    track_ids, pcts = get_knn(db)(embeddings, k)
    id_pct_map = dict(zip(track_ids, pcts))
    db_tracks = crud.track.get_multi_by_ids(db=db, track_ids=track_ids)
    tracks_out = [schemas.TrackKNNResult.from_orm(track) for track in db_tracks]
    for idx, track in enumerate(tracks_out):
        tracks_out[idx].percent_match = id_pct_map[track.id]
        tracks_out[idx].temp_preview_url = object_store.create_presigned_url(
            key=db_tracks[idx].s3_preview_key
        )
    return tracks_out

@r.get(
    "/reset",
    response_model=str,
    dependencies=[Depends(deps.get_current_active_superuser)]
)
def reset_knn_index(db=Depends(deps.get_db)):
    """Reset the in-memory knn index on the server with the latest DB data"""
    reset_knn(db)
    knn = get_knn(db)
    return f"Reset successful; the index currently has {knn.num_embeddings} tracks."