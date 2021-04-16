import os
import tempfile

from time import sleep

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app import crud
from app.core.config import settings
from app.tests.utils.license import create_random_license


def test_licenses_get(client: TestClient, db: Session) -> None:
    licenses = crud.license.get_multi(db, skip=0, limit=None)
    for license in licenses:
        crud.license.remove(db, id=license.id)
    license_dbs = [create_random_license(db) for _ in range(8)]
    response = client.get(f"{settings.API_V1_STR}/licenses/")
    assert response.status_code == 200
    content = response.json()
    assert len(content) == 8
    assert content[4]["name"] == license_dbs[4].name
    assert content[2]["url"] == license_dbs[2].url
    assert content[0]["id"] == license_dbs[0].id

    response = client.get(f"{settings.API_V1_STR}/licenses/{license_dbs[3].id}")
    content = response.json()
    assert isinstance(content, dict)
    assert content["id"] == license_dbs[3].id
    assert content["name"] == license_dbs[3].name


def test_upload_licenses(
    client: TestClient, superuser_token_headers: dict, db: Session, celery_worker
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        csv = os.path.join(tmpdir, "lic.csv")
        with open(csv, "w") as f_in:
            f_in.writelines([",name,url\n", ",sample,www.example.com\n"])
        with open(csv, "rb") as f_out:
            unauth = client.post(
                f"{settings.API_V1_STR}/licenses/upload",
                files={"csv_file": f_out},
            )
            assert unauth.status_code == 401
            f_out.seek(0)
            badtype = client.post(
                f"{settings.API_V1_STR}/licenses/upload",
                headers=superuser_token_headers,
                files={"csv_file": ("in.csv", f_out, "audio/mpeg")},
            )
            assert badtype.status_code == 415
            response = client.post(
                f"{settings.API_V1_STR}/licenses/upload",
                headers=superuser_token_headers,
                files={"csv_file": ("in.csv", f_out, "text/csv")},
            )
    assert response.status_code == 202
    content = response.json()
    assert content["status"] == "processing"

    response2 = client.get(
        f"{settings.API_V1_STR}/licenses/upload/status/{content['task_id']}",
        headers=superuser_token_headers,
    )
    assert response2.status_code in [200, 202]


