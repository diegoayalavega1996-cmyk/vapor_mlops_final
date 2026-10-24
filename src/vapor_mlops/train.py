"""Validación temporal de parámetros fijos y entrenamiento del Random Forest."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from vapor_mlops import __version__
from vapor_mlops.config import (
    CV_SPLITS, DATA_DIR, DATE_COLUMN, FEATURES, METRICS_PATH, MODEL_PARAMS,
    MODEL_PATH, PAPER_METRICS, PREDICTIONS_PATH, TARGET, TEST_FILENAME, TRAIN_FILENAME,
)
from vapor_mlops.data import file_sha256, load_data, temporal_relationship


def train_model(X_train, y_train) -> RandomForestRegressor:
    return RandomForestRegressor(**MODEL_PARAMS).fit(X_train, y_train)


def calculate_metrics(y_true, y_pred) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae_kg_h": mae, "mae_t_h": mae / 1000,
        "rmse_kg_h": rmse, "rmse_t_h": rmse / 1000,
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    return calculate_metrics(y_test, model.predict(X_test))


def cross_validate(train: pd.DataFrame) -> dict:
    folds = []
    for number, (fit_idx, val_idx) in enumerate(TimeSeriesSplit(n_splits=CV_SPLITS).split(train), 1):
        fit, val = train.iloc[fit_idx], train.iloc[val_idx]
        model = train_model(fit[FEATURES], fit[TARGET])
        metrics = evaluate_model(model, val[FEATURES], val[TARGET])
        folds.append({
            "fold": number, "train_rows": len(fit), "validation_rows": len(val),
            "train_end": fit[DATE_COLUMN].max().isoformat(),
            "validation_start": val[DATE_COLUMN].min().isoformat(),
            "validation_end": val[DATE_COLUMN].max().isoformat(), **metrics,
        })
        print(f"  Fold {number}/{CV_SPLITS}: MAE = {metrics['mae_t_h']:.4f} t/h", flush=True)
    return {"n_splits": CV_SPLITS, "mean_mae_t_h": float(np.mean([f["mae_t_h"] for f in folds])), "folds": folds}


def dataset_info(frame: pd.DataFrame, filename: str) -> dict:
    return {
        "file": filename, "rows": len(frame),
        "start": frame[DATE_COLUMN].min().isoformat(),
        "end": frame[DATE_COLUMN].max().isoformat(),
        "sha256": file_sha256(DATA_DIR / filename),
        "negative_balance_rows": int((frame[TARGET] < 0).sum()),
    }


def save_model(model, metadata: dict, predictions: pd.DataFrame) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=("xz", 1))
    predictions.to_csv(PREDICTIONS_PATH, index=False, float_format="%.8f")
    metadata["model_sha256"] = file_sha256(MODEL_PATH)
    metadata["predictions_sha256"] = file_sha256(PREDICTIONS_PATH)
    METRICS_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main() -> None:
    print("1) Cargar y validar TRAIN y TEST...", flush=True)
    train, test = load_data("train"), load_data("test")
    relationship = temporal_relationship(train, test)
    print("2) Validación temporal sobre TRAIN con los parámetros del paper...", flush=True)
    cv = cross_validate(train)
    print("3) Entrenar el modelo final utilizando únicamente TRAIN...", flush=True)
    model = train_model(train[FEATURES], train[TARGET])
    print("4) Evaluar en TEST...", flush=True)
    predictions = model.predict(test[FEATURES])
    metrics = calculate_metrics(test[TARGET], predictions)
    metadata = {
        "model_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_commit": os.getenv("GITHUB_SHA"),
        "model_type": type(model).__name__, "params": MODEL_PARAMS,
        "features": FEATURES, "target": TARGET, "target_unit": "kg/h",
        "training_dataset": dataset_info(train, TRAIN_FILENAME),
        "test_dataset": dataset_info(test, TEST_FILENAME),
        "evaluation_type": relationship,
        "cross_validation": cv, "test_metrics": metrics,
        "paper_metrics_reference": PAPER_METRICS,
        "feature_ranges": {
            feature: {"min": float(train[feature].min()), "max": float(train[feature].max()), "median": float(train[feature].median())}
            for feature in FEATURES
        },
        "feature_importances_mdi": dict(zip(FEATURES, map(float, model.feature_importances_))),
        "versions": {p: version(p) for p in ["scikit-learn", "numpy", "pandas", "joblib"]},
    }
    output = pd.DataFrame({
        DATE_COLUMN: test[DATE_COLUMN], "Actual_kg_h": test[TARGET], "Predicted_kg_h": predictions,
    })
    save_model(model, metadata, output)
    print(f"MAE TEST: {metrics['mae_t_h']:.4f} t/h | RMSE: {metrics['rmse_t_h']:.4f} t/h | R²: {metrics['r2']:.4f}")
    print(f"Modelo guardado: {MODEL_PATH}")


if __name__ == "__main__":
    main()
