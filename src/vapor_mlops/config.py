"""Variables, rutas e hiperparámetros del modelo."""

from pathlib import Path
import os

ROOT_DIR = Path(os.environ.get("VAPOR_MLOPS_ROOT", Path(__file__).resolve().parents[2])).resolve()
DATA_DIR = ROOT_DIR / "data" / "raw"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "vapor_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
PREDICTIONS_PATH = MODELS_DIR / "test_predictions.csv"

# La asignación se conserva en todos los comandos y flujos de trabajo.
TRAIN_FILENAME = "df_operacion.csv"
TEST_FILENAME = "df_operacion2.csv"
DATE_COLUMN = "DateTime"
TARGET = "Turbine_Steam_Consumption_kg_h"
RANDOM_STATE = 42
CV_SPLITS = 4

FEATURES = [
    "Condenser_Absolute_Pressure_kg_cm2A",
    "Blower_Air_Flow_kg_h",
    "Blower_Discharge_Temperature_C",
    "Turbine_Speed_rpm",
    "Turbine_Inlet_Steam_Pressure_kg_cm2",
    "Turbine_Inlet_Steam_Temperature_C",
    "Seawater_Flow_m3_h",
    "Seawater_Inlet_Temperature_C",
    "Seawater_DeltaT_C",
    "Seawater_Inlet_Pressure_kg_cm2",
    "Regenerator_Pressure_kg_cm2",
    "Antisurge_Valve_Opening_pct",
    "HS_Ejector_Steam_Pressure_kg_cm2",
    "HS_Ejector_Steam_Temperature_C",
]

FEATURE_LABELS = {
    "Condenser_Absolute_Pressure_kg_cm2A": "Presión absoluta del condensador (kgf/cm² A)",
    "Blower_Air_Flow_kg_h": "Caudal de aire del soplador (kg/h)",
    "Blower_Discharge_Temperature_C": "Temperatura de descarga del soplador (°C)",
    "Turbine_Speed_rpm": "Velocidad de la turbina (rpm)",
    "Turbine_Inlet_Steam_Pressure_kg_cm2": "Presión de vapor a la turbina (kgf/cm²)",
    "Turbine_Inlet_Steam_Temperature_C": "Temperatura de vapor a la turbina (°C)",
    "Seawater_Flow_m3_h": "Caudal de agua de mar (m³/h)",
    "Seawater_Inlet_Temperature_C": "Temperatura de entrada del agua de mar (°C)",
    "Seawater_DeltaT_C": "Diferencia de temperatura del agua de mar (°C)",
    "Seawater_Inlet_Pressure_kg_cm2": "Presión de entrada del agua de mar (kgf/cm²)",
    "Regenerator_Pressure_kg_cm2": "Presión del regenerador (kgf/cm²)",
    "Antisurge_Valve_Opening_pct": "Apertura de la válvula antisurge (%)",
    "HS_Ejector_Steam_Pressure_kg_cm2": "Presión de vapor a los eyectores (kgf/cm²)",
    "HS_Ejector_Steam_Temperature_C": "Temperatura de vapor a los eyectores (°C)",
}

# Paper, sección IV.A. random_state procede del notebook de análisis.
# n_jobs=1 conserva los parámetros estadísticos y limita el paralelismo.
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": 0.7,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}

PAPER_METRICS = {
    "cv_mae_t_h": 0.548,
    "mae_t_h": 0.786,
    "rmse_t_h": 1.077,
    "r2": 0.8710,
}
