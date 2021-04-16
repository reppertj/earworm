import random
from numpy.random import default_rng
import numpy as np
import string
from typing import Dict, List

from fastapi.testclient import TestClient

from app.core.config import settings

rng = default_rng(42)

def random_lower_string() -> str:
    return "".join(random.choices(string.ascii_lowercase, k=32))


def random_email() -> str:
    return f"{random_lower_string()}@{random_lower_string()}.com"

def random_url() -> str:
    return f"https://{random.choice(['www.', ''])}{random_lower_string()}.com"

def random_unit_vector(dim=128) -> List[float]:
    def normalize(ary):
        norm = np.linalg.norm(ary)
        return ary if norm == 0 else ary / norm
    return list(normalize(rng.standard_normal(dim)))

def get_superuser_token_headers(client: TestClient) -> Dict[str, str]:
    login_data = {
        "username": settings.FIRST_SUPERUSER,
        "password": settings.FIRST_SUPERUSER_PASSWORD,
    }
    r = client.post(f"{settings.API_V1_STR}/login/access-token", data=login_data)
    tokens = r.json()
    a_token = tokens["access_token"]
    headers = {"Authorization": f"Bearer {a_token}"}
    return headers
