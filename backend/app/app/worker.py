from typing import Dict, List, Union

import faiss
import sentry_sdk
from celery.signals import worker_process_init, worker_process_shutdown, task_failure
from fastapi.encoders import jsonable_encoder
from sentry_sdk import Hub

from app import crud, schemas
from app.core.celery_app import celery_app
from app.core.config import settings
from app.db.session import SessionLocal

sentry_sdk.init(settings.SENTRY_DSN)


db_conn = None

@worker_process_init.connect
def init_worker(**kwargs):
    global db_conn
    db_conn = SessionLocal()


@worker_process_shutdown.connect
def shutdown_worker(**kwargs):
    global db_conn
    if db_conn:
        db_conn.close()
    # Try to flush sentry_sdk logs before shutting down
    client = Hub.current.client
    if client:
        client.close(timeout=2.0)

@task_failure.connect
def on_task_failure(**kwargs):
    global db_conn
    db_conn.rollback()

@celery_app.task(acks_late=True)
def test_celery(word: str) -> str:
    return f"test task return {word}"



class FAISSTask(celery_app.Task):
    def __init__(self) -> None:
        super().__init__()
        self.index = faiss.IndexFlatIP(128)
        # self.index.add(xb)


class InferenceTask(celery_app.Task):
    def __init__(self) -> None:
        import torch


# @celery_app.task(ignore_result=True)
# def create_provider_task(name: str, url: str):
#     create_provider(
#         create_provider_task.db,
#         provider=schemas.ProviderCreate(
#             name=name,
#             url=url,
#         ),
#     )


@celery_app.task()
def parse_csv_task(
    csv_contents: str,
) -> List[Dict[str, Union[str, int, float]]]:
    """Returns a list of dicts corresponding to rows in that CSV"""
    from io import StringIO

    import pandas as pd

    return pd.read_csv(StringIO(csv_contents)).to_dict("records")


@celery_app.task()
def create_licenses_task(licenses: List[Dict[str, str]]) -> List[Dict[str, str]]:
    global db_conn
    licenses_db = crud.license.create_multi(
        db=db_conn,
        objs_in=[
            schemas.LicenseCreate(
                name=license["name"], external_link=license["external_link"]
            )
            for license in licenses
        ],
    )
    return jsonable_encoder(licenses_db)


@celery_app.task()
def create_providers_task(providers: List[Dict[str, str]]):
    global db_conn
    providers_db = crud.provider.create_multi(
        db=db_conn,
        objs_in=[
            schemas.ProviderCreate(name=provider["name"], url=provider["url"])
            for provider in providers
        ],
    )
    return jsonable_encoder(providers_db)
