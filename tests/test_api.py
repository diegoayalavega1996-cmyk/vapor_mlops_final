from fastapi.testclient import TestClient

from vapor_mlops import api
from vapor_mlops.config import FEATURES

client = TestClient(api.app)


def test_health_and_readiness():
    assert client.get("/").json()["status"] == "ok"
    assert client.get("/ready").status_code == 200


def test_valid_prediction_and_units(train_data):
    response = client.post("/predict", json=train_data[FEATURES].iloc[0].to_dict())
    assert response.status_code == 200
    result = response.json()
    assert abs(result["steam_consumption_t_h"] * 1000 - result["steam_consumption_kg_h"]) < 1e-7


def test_api_rejects_missing_input():
    assert client.post("/predict", json={}).status_code == 422


def test_api_rejects_infinity_and_unknown_fields(train_data):
    sample = train_data[FEATURES].iloc[0].to_dict()
    sample[FEATURES[0]] = "NaN"
    assert client.post("/predict", json=sample).status_code == 422
    sample = train_data[FEATURES].iloc[0].to_dict()
    sample["unknown"] = 5
    assert client.post("/predict", json=sample).status_code == 422


def test_model_info_exposes_actual_evaluation():
    result = client.get("/model-info")
    assert result.status_code == 200
    assert result.json()["evaluation_type"] == "future_holdout"


def test_missing_model_returns_503(monkeypatch, train_data):
    def unavailable(**kwargs):
        raise FileNotFoundError("Modelo no encontrado")
    monkeypatch.setattr(api, "predict", unavailable)
    assert client.post("/predict", json=train_data[FEATURES].iloc[0].to_dict()).status_code == 503
