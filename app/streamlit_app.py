"""Interfaz para inferencia, evaluación histórica y predicción por lotes."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

import pandas as pd
import streamlit as st

from vapor_mlops.config import DATE_COLUMN, FEATURES, FEATURE_LABELS, MODEL_PATH, PREDICTIONS_PATH
from vapor_mlops.predict import load_model, load_model_info, predict, predict_frame
from vapor_mlops.train import calculate_metrics

st.set_page_config(page_title="Consumo de vapor FCC", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner="Cargando modelo…", max_entries=1)
def cached_model(modified_ns: int):
    return load_model()


@st.cache_data(show_spinner=False, max_entries=1)
def cached_history(modified_ns: int):
    return pd.read_csv(PREDICTIONS_PATH, parse_dates=[DATE_COLUMN])


st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1380px;}
.vapor-header {padding: 1.6rem 1.8rem; border-radius: 14px; background: #123047; color: white; margin-bottom: 1.5rem;}
.vapor-header h1 {font-size: 2rem; color: white; margin: 0; padding: 0;}
.vapor-header p {color: #d4e3ee; margin: .7rem 0 0;}
</style>
<div class="vapor-header"><h1>Consumo de vapor</h1>
<p>Predicción de la turbina del soplador principal FCC</p></div>
""", unsafe_allow_html=True)

try:
    info = load_model_info()
    cached_model(MODEL_PATH.stat().st_mtime_ns)
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.info("Ejecute `uv run preparar-datos` y `uv run entrenar` antes de iniciar la aplicación.")
    st.stop()

with st.sidebar:
    st.subheader("Modelo")
    st.write("Random Forest · 200 árboles")
    st.caption("14 variables de proceso · Salida en t/h y kg/h")
    for label, key in [("Entrenamiento", "training_dataset"), ("Prueba", "test_dataset")]:
        dataset = info[key]
        st.markdown(f"**{label}**")
        st.caption(f"{dataset['file']} · {dataset['rows']:,} registros")
        st.caption(f"{dataset['start'][:10]} a {dataset['end'][:10]}")
    st.caption(f"Versión {info['model_version']} · Datos históricos de operación")

if info["evaluation_type"] == "retrospective":
    st.info("Evaluación retrospectiva: el periodo de prueba es anterior al de entrenamiento.")

manual_tab, history_tab, batch_tab, model_tab = st.tabs([
    "Predicción", "Histórico", "Predicción por CSV", "Resultados del modelo",
])

with manual_tab:
    st.subheader("Condiciones de operación")
    st.caption("Las entradas iniciales corresponden a las medianas del conjunto de entrenamiento.")
    with st.form("prediction_form"):
        columns = st.columns(2)
        values = {}
        for index, feature in enumerate(FEATURES):
            bounds = info["feature_ranges"][feature]
            with columns[index % 2]:
                values[feature] = st.number_input(
                    FEATURE_LABELS[feature], value=float(bounds["median"]),
                    format="%.4f", key=feature,
                    help=f"Rango observado en TRAIN: {bounds['min']:.4f} a {bounds['max']:.4f}",
                )
        submitted = st.form_submit_button("Predecir consumo", type="primary", key="predict_button")
    if submitted:
        try:
            outside = [FEATURE_LABELS[f] for f in FEATURES if not info["feature_ranges"][f]["min"] <= values[f] <= info["feature_ranges"][f]["max"]]
            if outside:
                st.warning("Entradas fuera del rango observado en TRAIN: " + "; ".join(outside))
            prediction = predict(**values)
            left, right = st.columns(2)
            left.metric("Consumo predicho (t/h)", f"{prediction['steam_consumption_t_h']:.3f}")
            right.metric("Consumo predicho (kg/h)", f"{prediction['steam_consumption_kg_h']:,.1f}")
        except ValueError as exc:
            st.error(str(exc))

with history_tab:
    st.subheader("Consumo real y predicho en TEST")
    if not PREDICTIONS_PATH.is_file():
        st.warning("No se encontró el histórico de predicciones. Ejecute el entrenamiento.")
    else:
        history = cached_history(PREDICTIONS_PATH.stat().st_mtime_ns)
        first, last = history[DATE_COLUMN].min().date(), history[DATE_COLUMN].max().date()
        default_start = max(first, (pd.Timestamp(last) - pd.Timedelta(days=29)).date())
        date_range = st.date_input("Periodo", value=(default_start, last), min_value=first, max_value=last)
        if len(date_range) == 2:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
            selected = history.loc[history[DATE_COLUMN].ge(start) & history[DATE_COLUMN].lt(end)].copy()
            if selected.empty:
                st.info("No hay registros en el periodo seleccionado.")
            else:
                metrics = calculate_metrics(selected["Actual_kg_h"], selected["Predicted_kg_h"]) if len(selected) > 1 else None
                if metrics:
                    cols = st.columns(3)
                    cols[0].metric("MAE del periodo (t/h)", f"{metrics['mae_t_h']:.3f}")
                    cols[1].metric("RMSE del periodo (t/h)", f"{metrics['rmse_t_h']:.3f}")
                    cols[2].metric("R² del periodo", f"{metrics['r2']:.3f}")
                chart = selected.set_index(DATE_COLUMN)[["Actual_kg_h", "Predicted_kg_h"]].div(1000)
                chart.columns = ["Real (t/h)", "Predicho (t/h)"]
                # Promedios para visualizar; las métricas anteriores usan cada registro.
                chart = chart.resample("15min").mean()
                st.line_chart(chart, color=["#087c9d", "#e88b28"], height=380)
                st.caption(f"{len(selected):,} registros. La gráfica muestra promedios de 15 minutos.")
                st.download_button("Descargar periodo", selected.to_csv(index=False).encode("utf-8"), "predicciones_periodo.csv", "text/csv")

with batch_tab:
    st.subheader("Predecir a partir de un CSV")
    st.caption("Incluya las 14 columnas predictoras con sus nombres originales. La columna de consumo no es necesaria.")
    template = pd.DataFrame([{f: info["feature_ranges"][f]["median"] for f in FEATURES}])
    st.download_button("Descargar plantilla", template.to_csv(index=False).encode("utf-8"), "plantilla_prediccion.csv", "text/csv")
    uploaded = st.file_uploader("Archivo CSV", type=["csv"])
    if uploaded is not None and st.button("Procesar archivo", key="batch_predict"):
        try:
            frame = pd.read_csv(uploaded, encoding="utf-8-sig")
            if len(frame) > 100_000:
                raise ValueError("Procese hasta 100 000 filas por archivo.")
            result = predict_frame(frame)
            if DATE_COLUMN in frame:
                result.insert(0, DATE_COLUMN, frame[DATE_COLUMN])
            st.session_state["batch_result"] = result
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.session_state.pop("batch_result", None)
            st.error(str(exc))
    if "batch_result" in st.session_state:
        result = st.session_state["batch_result"]
        st.success(f"{len(result):,} predicciones generadas.")
        st.dataframe(result.head(100), hide_index=True, width="stretch")
        st.download_button("Descargar predicciones", result.to_csv(index=False).encode("utf-8"), "predicciones_vapor.csv", "text/csv")

with model_tab:
    st.subheader("Evaluación del modelo entrenado")
    result = info["test_metrics"]
    cols = st.columns(4)
    cols[0].metric("MAE TEST (t/h)", f"{result['mae_t_h']:.3f}")
    cols[1].metric("RMSE TEST (t/h)", f"{result['rmse_t_h']:.3f}")
    cols[2].metric("R² TEST", f"{result['r2']:.3f}")
    cols[3].metric("MAE CV (t/h)", f"{info['cross_validation']['mean_mae_t_h']:.3f}")
    st.caption("La validación temporal utiliza cuatro divisiones dentro de TRAIN. TEST se reserva para la evaluación final.")
    st.dataframe(pd.DataFrame(info["cross_validation"]["folds"])[["fold", "train_rows", "validation_rows", "mae_t_h", "rmse_t_h", "r2"]], hide_index=True, width="stretch")
    st.markdown("**Hiperparámetros**")
    st.json(info["params"], expanded=True)
    importance = pd.DataFrame({"Variable": [FEATURE_LABELS[f] for f in FEATURES], "Importancia MDI": [info["feature_importances_mdi"][f] for f in FEATURES]}).sort_values("Importancia MDI")
    st.markdown("**Importancia de variables del bosque (MDI)**")
    st.bar_chart(importance.set_index("Variable"), horizontal=True, color="#087c9d", height=480)
    st.caption("MDI mide la reducción de impureza en los árboles. Es una medida distinta de SHAP.")
