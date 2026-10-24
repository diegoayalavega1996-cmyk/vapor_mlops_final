"""API opcional. Streamlit utiliza la inferencia directamente."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model

from vapor_mlops import __version__
from vapor_mlops.config import FEATURES, FEATURE_LABELS
from vapor_mlops.predict import load_model, load_model_info, predict

app = FastAPI(title="Predicción de consumo de vapor", version=__version__)

SteamInput = create_model(
    "SteamInput",
    __config__=ConfigDict(extra="forbid"),
    **{name: (float, Field(..., allow_inf_nan=False, description=FEATURE_LABELS[name])) for name in FEATURES},
)


class PredictionOutput(BaseModel):
    steam_consumption_kg_h: float
    steam_consumption_t_h: float


@app.get("/")
def health():
    return {"status": "ok", "service": "vapor-mlops", "version": __version__}


@app.get("/ready")
def readiness():
    try:
        load_model()
        load_model_info()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready"}


@app.get("/model-info")
def model_info():
    try:
        return load_model_info()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Modelo no encontrado o no entrenado.") from exc


@app.post("/predict", response_model=PredictionOutput)
def predict_consumption(sample: SteamInput):
    try:
        return predict(**sample.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
