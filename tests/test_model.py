import numpy as np
import pandas as pd
import pytest

from vapor_mlops.config import FEATURES, MODEL_PATH, PREDICTIONS_PATH, TARGET
from vapor_mlops.data import file_sha256
from vapor_mlops.predict import load_model, predict, predict_frame


def test_saved_model_uses_paper_configuration():
    model = load_model()
    expected = {"n_estimators": 200, "max_depth": 20, "min_samples_split": 2, "min_samples_leaf": 2, "max_features": 0.7}
    for key, value in expected.items():
        assert model.get_params()[key] == value
    assert list(model.feature_names_in_) == FEATURES
    assert len(FEATURES) == 14
    assert TARGET not in FEATURES
    assert "Turbine_Steam_Consumption_t_h" not in FEATURES


def test_predict_uses_correct_units_and_cached_model(train_data):
    sample = train_data[FEATURES].iloc[0].to_dict()
    result = predict(**sample)
    assert np.isfinite(result["steam_consumption_kg_h"])
    assert result["steam_consumption_t_h"] * 1000 == pytest.approx(result["steam_consumption_kg_h"])
    assert load_model() is load_model()


def test_batch_results_preserve_index_and_feature_order(train_data):
    frame = train_data.iloc[[2, 9, 15]][list(reversed(FEATURES))]
    batch = predict_frame(frame)
    assert batch.index.equals(frame.index)
    np.testing.assert_allclose(batch.iloc[:, 0], load_model().predict(frame[FEATURES]))


def test_recorded_artifacts_match_model_and_predictions(model_info, test_data):
    assert model_info["model_sha256"] == file_sha256(MODEL_PATH)
    assert model_info["predictions_sha256"] == file_sha256(PREDICTIONS_PATH)
    saved = pd.read_csv(PREDICTIONS_PATH)
    expected = load_model().predict(test_data[FEATURES].iloc[:20])
    np.testing.assert_allclose(saved["Predicted_kg_h"].iloc[:20], expected, rtol=0, atol=1e-7)
    assert len(saved) == len(test_data)


def test_validation_folds_never_include_future_training_data(model_info):
    folds = model_info["cross_validation"]["folds"]
    assert len(folds) == 4
    for fold in folds:
        assert pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["validation_start"])
        assert np.isfinite(fold["mae_t_h"])
