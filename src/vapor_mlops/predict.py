"""Inferencia con un solo modelo en memoria por proceso."""

from functools import lru_cache
import json
from pathlib import Path

import joblib
import pandas as pd

from vapor_mlops.config import FEATURES, METRICS_PATH, MODEL_PATH
from vapor_mlops.data import validate_features


def load_model_info() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _read_model(path: str, modified_ns: int):
    model = joblib.load(path)
    model.n_jobs = 1
    if list(model.feature_names_in_) != FEATURES:
        raise ValueError("El modelo guardado no coincide con las 14 variables configuradas.")
    return model


def load_model():
    path = Path(MODEL_PATH)
    if not path.is_file():
        raise FileNotFoundError("Modelo no encontrado. Ejecute: uv run entrenar")
    return _read_model(str(path), path.stat().st_mtime_ns)


def predict_frame(frame: pd.DataFrame) -> pd.DataFrame:
    features = validate_features(frame)
    predictions = load_model().predict(features)
    result = pd.DataFrame(index=frame.index)
    result["Predicted_Steam_Consumption_kg_h"] = predictions
    result["Predicted_Steam_Consumption_t_h"] = predictions / 1000
    return result


def predict(**sample) -> dict:
    result = predict_frame(pd.DataFrame([sample])).iloc[0]
    return {"steam_consumption_kg_h": float(result.iloc[0]), "steam_consumption_t_h": float(result.iloc[1])}


def main() -> None:
    info = load_model_info()
    sample = {feature: info["feature_ranges"][feature]["median"] for feature in FEATURES}
    print(json.dumps(predict(**sample), indent=2))


if __name__ == "__main__":
    main()
