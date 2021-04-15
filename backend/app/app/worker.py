from typing import Dict, List, Union, Any, cast


import sentry_sdk
from celery.signals import worker_process_init, worker_process_shutdown, task_failure
from fastapi.encoders import jsonable_encoder
from sentry_sdk import Hub
from sqlalchemy.orm import Session

from app import crud, schemas, utils
from app.core.celery_app import celery_app
from app.core.config import settings
from celery.utils.log import get_task_logger as _get_task_logger
from app.db.session import SessionLocal
from app.utils import canonical_preview_uri
from app.worker_utils.io import import_track
from app.worker_utils.knn import get_knn

if settings.SENTRY_DSN:
    sentry_sdk.init(settings.SENTRY_DSN)

_logger = _get_task_logger(__name__)

db_conn = None


@worker_process_init.connect
def init_worker(**kwargs):
    pass


@worker_process_shutdown.connect
def _shutdown_worker(**kwargs):
    # Try to flush sentry_sdk logs before shutting down
    client = Hub.current.client
    if client:
        client.close(timeout=5.0)


@celery_app.task(acks_late=True)
def test_celery(word: str) -> str:
    return f"test task return {word}"


# @celery_app.task(ignore_result=True)
# def create_provider_task(name: str, url: str):
#     create_provider(
#         create_provider_task.db,
#         provider=schemas.ProviderCreate(
#             name=name,
#             url=url,
#         ),
#     )

# @celery_app.task()
# def check_object_in_bucket(obj_name: str) -> bool:


@celery_app.task(acks_late=True, priority=0)
def knn_task(query_embeddings: List[List[float]], k):
    track_ids, pcts = get_knn()(query_embeddings, k)


@celery_app.task(acks_late=True, priority=9)
def import_track_task(track: Dict[str, str]):
    db = SessionLocal()
    try:
        track_in = schemas.TrackCreate(
            title=track.get("title") or track.get("name"),
            artist=track.get("artist"),
            url=track.get("url"),
            provider_name=track.get("provider_name") or track.get("provider"),
            license_name=track.get("license_name") or track.get("license"),
            media_url=track.get("media_url") or track.get("mp3"),
            s3_preview_key=track.get("s3_preview_key")
            or track.get("internal_preview_uri"),
        )
        if track_in.s3_preview_key is None:
            track_in.s3_preview_key = canonical_preview_uri(track_in)
        track_db = import_track(db=db, track=track_in)
        return jsonable_encoder(track_db)
    except Exception as e:
        _logger.error(
            f"Skipped track {track['title']} because of error",
            exc_info=e,
        )
    finally:
        db.close()


@celery_app.task()
def parse_csv_task(csv_content: str) -> List[Dict[str, Any]]:
    return utils.parse_csv_string(csv_content)


@celery_app.task()
def create_licenses_task(licenses: List[Dict[str, str]]) -> List[Dict[str, str]]:
    db = SessionLocal()
    try:
        licenses_db = crud.license.create_multi(
            db=db,
            objs_in=[
                schemas.LicenseCreate(name=license["name"], url=license["url"])
                for license in licenses
            ],
        )
        return jsonable_encoder(licenses_db)
    finally:
        db.close()


@celery_app.task()
def create_providers_task(providers: List[Dict[str, str]]):
    db = SessionLocal()
    try:
        providers_db = crud.provider.create_multi(
            db=db,
            objs_in=[
                schemas.ProviderCreate(name=provider["name"], url=provider["url"])
                for provider in providers
            ],
        )
        return jsonable_encoder(providers_db)
    finally:
        db.close()
