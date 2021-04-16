import typing as t

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, Response, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from app import crud, schemas, worker
from app.api import deps

router = r = APIRouter()


@r.get(
    "/", response_model=t.List[schemas.Provider], response_model_exclude_unset=True,
)
def providers_list(
    response: Response, skip: int = 0, limit: int = 100, db=Depends(deps.get_db),
):
    """
    Get all providers (paginated)
    """
    providers = crud.provider.get_multi(db, skip=skip, limit=limit)
    # This is necessary for react-admin to work
    response.headers["Content-Range"] = f"0-9/{len(providers)}"
    return providers


@r.get(
    "/{provider_id}", response_model=schemas.Provider, response_model_exclude_none=True,
)
def provider_details(
    provider_id: int, db=Depends(deps.get_db),
):
    """
    Get name and link to any provider
    """
    provider = crud.provider.get(db, provider_id)
    return provider


@r.post(
    "/upload",
    response_model=schemas.ProviderUploadStatus,
    dependencies=[Depends(deps.get_current_active_superuser)],
)
async def upload_providers(csv_file: UploadFile = File(...)):
    """Upload a utf-8 encoded CSV file containing `name` and `url` columns"""
    if not csv_file.content_type.startswith("text"):
        raise HTTPException(415, "Should be CSV")
    content_bytes = await csv_file.read()
    content_str = t.cast(bytes, content_bytes).decode("utf-8")
    task = (worker.parse_csv_task.s(content_str) | worker.create_providers_task.s())()
    return JSONResponse(
        status_code=202, content={"task_id": str(task), "status": "processing"}
    )

@r.get(
    "/upload/status/{task_id}",
    response_model=schemas.ProviderUploadStatus,
    dependencies=[Depends(deps.get_current_active_superuser)],
    response_model_exclude_none=True,
)
def upload_providers_status(task_id: str):
    task = AsyncResult(task_id)
    if task.failed():
        return JSONResponse(
            status_code=409,
            content={
                "task_id": task_id,
                "status": "failed",
                "details": task.result.args,
            },
        )
    elif not task.ready():
        return JSONResponse(
            status_code=202, content={"task_id": task_id, "status": "processing"}
        )
    result = task.get()
    print(result)
    return {
        "task_id": task_id,
        "status": "success",
        "providers": jsonable_encoder(result, exclude_defaults=True, exclude_none=True),
    }
