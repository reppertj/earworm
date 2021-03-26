import pytest
from dotenv import load_dotenv
import os

assert load_dotenv(dotenv_path="local.modeling.env") == True

@pytest.fixture(scope="session")
def tensor_dir():
    assert os.getenv("YT_TENSOR_DATA_DIR") is not None
    return os.getenv("YT_TENSOR_DATA_DIR")

@pytest.fixture(scope="session")
def audio_dir():
    assert os.getenv("YT_AUDIO_DATA_DIR") is not None
    return os.getenv("YT_AUDIO_DATA_DIR")

@pytest.fixture(scope="session")
def audioset_json():
    assert os.getenv("AUDIOSET_JSON_PATH") is not None
    return os.getenv("AUDIOSET_JSON_PATH")

@pytest.fixture(scope="session")
def audioset_csv():
    assert os.getenv("AUDIOSET_TRAIN_CSV_PATH") is not None
    return os.getenv(
        "AUDIOSET_TRAIN_CSV_PATH"
    )
