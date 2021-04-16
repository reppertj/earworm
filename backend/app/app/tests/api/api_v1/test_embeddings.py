from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app import crud
from app.core.config import settings
from app.tests.utils.embedding import create_random_embedding
from app.tests.utils.embedding_model import create_embedding_model
from app.worker_utils.knn import get_knn, reset_knn


def test_random_search(client: TestClient, db: Session) -> None:
    embedding_model = crud.embedding_model.get_by_name(
        db, name=settings.ACTIVE_MODEL_NAME
    )
    if not embedding_model:
        create_embedding_model(db, name=settings.ACTIVE_MODEL_NAME)
    first = create_random_embedding(db, model_name=settings.ACTIVE_MODEL_NAME)
    model_name = first.embedding_model.name

    for _ in range(300):
        create_random_embedding(db, model_name=model_name)
      
    reset_knn(db)

    data = {"k": 40, "embeddings": [first.values]}
    response = client.post(f"{settings.API_V1_STR}/embeddings/search", json=data)
    assert response.status_code == 200
    content = response.json()
    assert len(content) == 40
    assert content[0]["id"] == first.track_id
    assert content[0]["percent_match"] == 100

    multi_embedding = [create_random_embedding(db, model_name=settings.ACTIVE_MODEL_NAME).values for _ in range(8)]

    data2 = {"k": 25, "embeddings": multi_embedding}
    response = client.post(f"{settings.API_V1_STR}/embeddings/search", json=data2)
    assert response.status_code == 200
    content = response.json()
    assert len(content) == 25
    assert content[0]["license"]["name"]
    assert content[0]["provider"]["url"]
    