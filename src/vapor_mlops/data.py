"""Lectura y validación de los CSV; nunca se sobrescriben los originales."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from vapor_mlops.config import (
    DATA_DIR, DATE_COLUMN, FEATURES, TARGET, TEST_FILENAME, TRAIN_FILENAME,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("El archivo no contiene registros.")
    if df.columns.duplicated().any():
        raise ValueError("Hay nombres de columnas duplicados.")
    missing = sorted(set(FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan variables predictoras: {', '.join(missing)}")
    try:
        numeric = df.loc[:, FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Las 14 variables predictoras deben ser numéricas.") from exc
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Las variables predictoras contienen nulos o valores infinitos.")
    return numeric


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = {DATE_COLUMN, TARGET} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas de datos: {sorted(missing)}")
    result = df.copy()
    result[FEATURES] = validate_features(result)
    try:
        result[TARGET] = pd.to_numeric(result[TARGET], errors="raise")
        result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("El consumo o la fecha tienen un formato inválido.") from exc
    if not np.isfinite(result[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("El consumo contiene nulos o valores infinitos.")
    if result[DATE_COLUMN].isna().any() or result[DATE_COLUMN].duplicated().any():
        raise ValueError("Hay fechas nulas o repetidas.")
    if "Operating_State" in result and not result["Operating_State"].eq("IN_SERVICE").all():
        raise ValueError("Los datos deben corresponder al estado IN_SERVICE.")
    if "Turbine_Steam_Consumption_t_h" in result:
        target_t_h = pd.to_numeric(result["Turbine_Steam_Consumption_t_h"], errors="raise")
        if not np.allclose(target_t_h * 1000, result[TARGET], rtol=1e-6, atol=1e-3):
            raise ValueError("Las columnas de consumo en kg/h y t/h no son consistentes.")
    return result.sort_values(DATE_COLUMN).reset_index(drop=True)


def load_data(role: str = "train") -> pd.DataFrame:
    filenames = {"train": TRAIN_FILENAME, "test": TEST_FILENAME}
    if role not in filenames:
        raise ValueError("El conjunto debe ser 'train' o 'test'.")
    path = DATA_DIR / filenames[role]
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    return validate_data(pd.read_csv(path, encoding="utf-8-sig"))


def temporal_relationship(train: pd.DataFrame, test: pd.DataFrame) -> str:
    if pd.Index(train[DATE_COLUMN]).intersection(test[DATE_COLUMN]).size:
        raise ValueError("TRAIN y TEST comparten fechas. Revise la separación de datos.")
    if train[DATE_COLUMN].max() < test[DATE_COLUMN].min():
        return "future_holdout"
    if test[DATE_COLUMN].max() < train[DATE_COLUMN].min():
        return "retrospective"
    raise ValueError("Los periodos de TRAIN y TEST se intercalan.")


def split_data():
    """Conserva los archivos independientes, sin una división aleatoria."""
    train, test = load_data("train"), load_data("test")
    temporal_relationship(train, test)
    return train[FEATURES], test[FEATURES], train[TARGET], test[TARGET]


def main() -> None:
    train, test = load_data("train"), load_data("test")
    relationship = temporal_relationship(train, test)
    for role, filename, frame in [("TRAIN", TRAIN_FILENAME, train), ("TEST", TEST_FILENAME, test)]:
        print(f"{role}: {filename} | {len(frame):,} filas | {len(FEATURES)} predictores")
        print(f"  {frame[DATE_COLUMN].min()} a {frame[DATE_COLUMN].max()}")
    print(f"Evaluación: {relationship}")


if __name__ == "__main__":
    main()
