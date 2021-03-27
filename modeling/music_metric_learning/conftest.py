import pytest
from dotenv import load_dotenv
from music_metric_learning.data.dataset import MusicMetricDatamodule
from music_metric_learning.data.metadata import (
    make_dataset_df,
    make_ontology_df,
    make_combined_tensor_df,
    save_tensor_df,
)
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
    return os.getenv("AUDIOSET_TRAIN_CSV_PATH")


@pytest.fixture(scope="session")
def train_csv(tmpdir_factory, tensor_dir, audioset_json, audioset_csv):
    dataset_df = make_dataset_df(audioset_csv)
    ontology_df = make_ontology_df(audioset_json)
    df = make_combined_tensor_df(tensor_dir, dataset_df, ontology_df)
    out_path = tmpdir_factory.mktemp("data").join("train.csv")
    save_tensor_df(out_path, df)
    return out_path


@pytest.fixture(scope="session")
def datamodule(train_csv, tensor_dir):
    dm = MusicMetricDatamodule(
        dataset_csv=train_csv, tensor_dir=tensor_dir, m_per_class=2, random_state=42
    )
    dm.setup()
    return dm
