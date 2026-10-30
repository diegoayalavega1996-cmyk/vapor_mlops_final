# Predicción del consumo de vapor

El consumo de vapor de la turbina del soplador principal de la unidad FCC varía según las condiciones de operación, como el flujo de aire, el vacío del condensador y la presión y temperatura del vapor de entrada. La interacción entre estas variables dificulta establecer cuánto vapor debería consumir el equipo en una condición determinada. Contar con una estimación permite comparar el consumo observado con una referencia y estudiar posibles desviaciones en su comportamiento.

Este proyecto utiliza datos históricos de operación para entrenar un modelo Random Forest que predice el consumo de vapor a partir de 14 variables del proceso. Mediante una aplicación en Streamlit, el usuario ingresa los valores de operación y obtiene el consumo estimado en toneladas por hora (t/h) y kilogramos por hora (kg/h).

El objetivo es facilitar el análisis del consumo de vapor y aportar información para evaluar el desempeño energético de la turbina del soplador.

## Ejecución local

- Tener Git y uv instalados. El proyecto utiliza Python 3.12.
- Descomprimir el proyecto ligero y abrir la carpeta `vapor_mlops` en Visual Studio Code: **Archivo → Abrir carpeta**.
- Verificar los archivos en `data/raw/`: `df_operacion.csv` para entrenamiento (82 146 filas) y `df_operacion2.csv` para prueba (13 261 filas).
- Abrir **Terminal → Nuevo terminal**, desde la carpeta que contiene `pyproject.toml`, y ejecutar en orden:

```powershell
uv --version
uv sync --locked
uv run preparar-datos
uv run entrenar
uv run pytest -v
uv run streamlit run app/streamlit_app.py
```

- El entrenamiento genera `models/vapor_model.joblib` y guarda las métricas del modelo.
- Abrir http://localhost:8501 y pulsar **Predecir consumo**. Para detener la aplicación local, presionar `Ctrl + C` en su terminal.

## Subir el proyecto a GitHub

- Crear el repositorio privado `vapor_mlops_final`, sin agregar README, licencia ni `.gitignore` desde GitHub.
- Abrir otra terminal en Visual Studio Code y ejecutar, desde la carpeta del proyecto:

```powershell
git init -b main
git add .
git commit -m "Proyecto MLOps de prediccion de consumo de vapor"
git remote add origin https://github.com/diegoayalavega1996-cmyk/vapor_mlops_final.git
git push -u origin main
```

- En GitHub, abrir **Actions** y esperar que **CI** y **CD** terminen correctamente. CI valida los datos, entrena y ejecuta las pruebas; CD publica el código y el modelo en la rama `deploy`.

## Publicar en Streamlit

- Entrar a https://share.streamlit.io/ y seleccionar **Create app → Deploy now**.
- En **Repository**, ingresar `diegoayalavega1996-cmyk/vapor_mlops_final`.
- En **Branch**, seleccionar `deploy`.
- En **Main file path**, ingresar `app/streamlit_app.py`. Dejar **App URL** vacío para generar el enlace automáticamente.
- Abrir **Advanced settings**, elegir **Python 3.12**, pulsar **Save** y después **Deploy**.
- Para permitir el acceso mediante el enlace, abrir **⋮ → Settings → Sharing** y seleccionar **This app is public and searchable**. Guardar y comprobar el enlace en una ventana de incógnito.

## Actualizar el proyecto

- Modificar los archivos en Visual Studio Code y subir los cambios a `main`:

```powershell
git add .
git commit -m "Actualizar proyecto"
git push origin main
```

- Esperar a que CI y CD finalicen. Streamlit actualizará la aplicación desde la rama `deploy`.

