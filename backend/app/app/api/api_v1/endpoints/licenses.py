import typing as t

from app.api import deps
from app import crud, schemas, worker
from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, Response, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

router = r = APIRouter()


@r.get(
    "/",
    response_model=t.List[schemas.License],
    response_model_exclude_unset=True,
)
async def licenses_list(
    response: Response,
    skip: int = 0,
    limit: int = 100,
    db=Depends(deps.get_db),
):
    """
    Get all licenses (paginated)
    """
    licenses = crud.license.get_multi(db, skip=skip, limit=limit)
    # This is necessary for react-admin to work
    response.headers["Content-Range"] = f"0-9/{len(licenses)}"
    return licenses


@r.get(
    "/{license_id}",
    response_model=schemas.License,
    response_model_exclude_none=True,
)
async def license_details(
    license_id: int,
    db=Depends(deps.get_db),
):
    """
    Get name and link to any license
    """
    license = crud.license.get(db, license_id)
    return license


@r.post(
    "/upload",
    response_model=schemas.LicenseUploadStatus,
    dependencies=[Depends(deps.get_current_active_superuser)],
)
async def upload_licenses(csv_file: UploadFile = File(...)):
    """Upload a utf-8 encoded CSV file containing `name` and `external_link` columns"""
    if not csv_file.content_type.startswith("text"):
        raise HTTPException(415, "Should be CSV")
    content_bytes = await csv_file.read()
    content_str = t.cast(bytes, content_bytes).decode("utf-8")
    task = (
        worker.parse_csv_task.s(content_str) | worker.create_licenses_task.s()
    )()
    return {"task_id": str(task), "status": "processing"}


@r.get(
    "/upload/status/{task_id}",
    response_model=schemas.LicenseUploadStatus,
    dependencies=[Depends(deps.get_current_active_superuser)],
    response_model_exclude_none=True,
)
async def upload_licenses_status(task_id: str):
    task = AsyncResult(task_id)
    if task.failed():
        return JSONResponse(
            status_code=500,
            content={
                "task_id": task_id,
                "status": "failed",
                "details": task.result.args,
            },
        )
    elif not task.ready():
        return JSONResponse(status_code=202, content={"task_id": task_id, "status": "processing"})
    result = task.get()
    print(result)
    return {
        "task_id": task_id,
        "status": "success",
        "licenses": jsonable_encoder(
            result, exclude_defaults=True, exclude_none=True
        ),
    }
