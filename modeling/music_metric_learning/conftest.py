import pytest
from dotenv import load_dotenv
import os

assert load_dotenv(dotenv_path="local.modeling.env") == True

@pytest.fixture(scope="session")
def tensor_dir():
    return os.getenv("YT_TENSOR_DATA_DIR")

@pytest.fixture(scope="session")
def audio_dir():
    return os.getenv("YT_AUDIO_DATA_DIR")

@pytest.fixture(scope="session")
def audioset_json():
    return os.getenv("AUDIOSET_JSON_PATH")

@pytest.fixture(scope="session")
def audioset_csv():
    return os.getenv(
        "AUDIOSET_TRAIN_CSV_PATH"
    )
