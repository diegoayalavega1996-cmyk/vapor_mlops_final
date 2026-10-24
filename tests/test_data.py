import numpy as np
import pandas as pd
import pytest

from vapor_mlops.config import DATA_DIR, DATE_COLUMN, FEATURES, TARGET
from vapor_mlops.data import file_sha256, temporal_relationship, validate_data, validate_features


def test_dataset_roles_and_original_bytes(train_data, test_data, model_info):
    assert model_info["training_dataset"]["file"] == "df_operacion.csv"
    assert model_info["test_dataset"]["file"] == "df_operacion2.csv"
    assert len(train_data) == 82146
    assert len(test_data) == 13261
    for dataset in [model_info["training_dataset"], model_info["test_dataset"]]:
        assert dataset["sha256"] == file_sha256(DATA_DIR / dataset["file"])


def test_temporal_separation_is_explicit(train_data, test_data):
    assert temporal_relationship(train_data, test_data) == "future_holdout"
    assert train_data[DATE_COLUMN].max() < test_data[DATE_COLUMN].min()
    assert train_data[DATE_COLUMN].is_monotonic_increasing
    assert test_data[DATE_COLUMN].is_monotonic_increasing


def test_overlapping_dates_are_rejected(train_data):
    with pytest.raises(ValueError, match="comparten fechas"):
        temporal_relationship(train_data.iloc[:3], train_data.iloc[2:5])


def test_missing_predictor_is_rejected(train_data):
    with pytest.raises(ValueError, match="Faltan variables"):
        validate_features(train_data.head(2).drop(columns=FEATURES[0]))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, "texto"])
def test_non_numeric_or_non_finite_inputs_are_rejected(train_data, invalid):
    frame = train_data.head(2).copy().astype({FEATURES[0]: object})
    frame.loc[0, FEATURES[0]] = invalid
    with pytest.raises(ValueError):
        validate_features(frame)


def test_duplicate_dates_are_rejected(train_data):
    frame = train_data.head(2).copy()
    frame.loc[1, DATE_COLUMN] = frame.loc[0, DATE_COLUMN]
    with pytest.raises(ValueError, match="fechas"):
        validate_data(frame)


def test_inconsistent_target_units_are_rejected(train_data):
    frame = train_data.head(2).copy()
    frame.loc[0, "Turbine_Steam_Consumption_t_h"] += 5
    with pytest.raises(ValueError, match="consistentes"):
        validate_data(frame)


def test_inference_accepts_csv_without_target(train_data):
    features = validate_features(train_data[FEATURES].head(2))
    assert features.shape == (2, 14)
    assert TARGET not in features


def test_original_negative_balance_is_reported(train_data, test_data, model_info):
    assert (train_data[TARGET] < 0).sum() == 1
    assert model_info["training_dataset"]["negative_balance_rows"] == 1
    assert (test_data[TARGET] < 0).sum() == 0
