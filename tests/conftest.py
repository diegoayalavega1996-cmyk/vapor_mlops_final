import pytest

from vapor_mlops.config import MODEL_PATH
from vapor_mlops.data import load_data
from vapor_mlops.predict import load_model_info
from vapor_mlops.train import main as train_main


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    if not MODEL_PATH.exists():
        train_main()


@pytest.fixture(scope="session")
def train_data():
    return load_data("train")


@pytest.fixture(scope="session")
def test_data():
    return load_data("test")


@pytest.fixture(scope="session")
def model_info():
    return load_model_info()
