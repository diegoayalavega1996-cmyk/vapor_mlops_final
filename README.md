# Predicción del consumo de vapor

Proyecto de regresión para estimar el consumo de vapor de la turbina del soplador principal FCC. Utiliza Random Forest con 14 variables y los hiperparámetros de la sección IV.A del paper incluido en `docs/`.

## 1. Ejecutar localmente

Instalar [uv](https://docs.astral.sh/uv/getting-started/installation/) y abrir esta carpeta en Visual Studio Code. Ejecutar desde la terminal:

```bash
uv sync --locked
uv run preparar-datos
uv run entrenar
uv run pytest -v
uv run streamlit run app/streamlit_app.py
```

Abrir `http://localhost:8501` después de ejecutar el entrenamiento. Esta descarga incluye el código, los dos CSV y las métricas de referencia. El comando `uv run entrenar` genera `models/vapor_model.joblib`, necesario para iniciar la aplicación.

## 2. Datos y modelo

| Conjunto | Archivo en `data/raw/` | Registros |
|---|---|---:|
| TRAIN | `df_operacion.csv` | 82 146 |
| TEST | `df_operacion2.csv` | 13 261 |

TRAIN abarca del 1 de junio al 18 de agosto de 2026. TEST corresponde al periodo posterior, del 28 de agosto al 6 de septiembre de 2026. Esta separación coincide con la utilizada en el paper. Las métricas calculadas se guardan en `models/metrics.json`.

El modelo tiene 200 árboles, profundidad máxima 20, `min_samples_split=2`, `min_samples_leaf=2` y `max_features=0.7`. Se utiliza `random_state=42` y un solo hilo. El objetivo se procesa en kg/h y la interfaz también muestra t/h.

## 3. GitHub y Streamlit Community Cloud

1. Publicar esta carpeta desde Visual Studio Code o Git en la rama `main` de un repositorio de GitHub.
2. Permitir **Read and write permissions** en **Settings → Actions → General → Workflow permissions**, si la política del repositorio lo requiere.
3. Esperar a que terminen **CI** y **CD** en la pestaña **Actions**. CI valida los datos, entrena, prueba y guarda el modelo. CD copia el mismo modelo validado y su código a la rama `deploy`.
4. En [Streamlit Community Cloud](https://share.streamlit.io), seleccionar el repositorio, la rama **deploy** y el archivo **app/streamlit_app.py**. En las opciones avanzadas, elegir **Python 3.12**.
5. Publicar la aplicación. Los siguientes cambios se suben a `main`; la rama `deploy` se actualiza cuando CI aprueba la versión.

Streamlit carga el modelo en memoria una sola vez por proceso. No vuelve a entrenarlo ni necesita un servicio de Render. Su estado de publicación y los errores de arranque se consultan en Streamlit; CD confirma la actualización de la rama, no la disponibilidad del servicio.

## 4. API y Docker (opcionales)

```bash
uv run uvicorn vapor_mlops.api:app --host 127.0.0.1 --port 8000
```

Documentación de la API: `http://127.0.0.1:8000/docs`.

Con Docker instalado y los archivos de `models/` generados:

```bash
docker compose up --build
```

Para iniciar también la API: `docker compose --profile api up --build`.

La estructura conserva `app/`, `data/raw/`, `src/`, `tests/`, `Dockerfile`, `compose.yaml`, `pyproject.toml`, `uv.lock` y los flujos `.github/workflows/ci.yml` y `cd.yml`.
