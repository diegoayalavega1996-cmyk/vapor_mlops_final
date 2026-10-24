FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VAPOR_MLOPS_ROOT=/app \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir --no-deps . \
    && test -f models/vapor_model.joblib \
    && test -f models/metrics.json \
    && test -f models/test_predictions.csv \
    && useradd --create-home appuser

USER appuser
EXPOSE 8501
CMD ["sh", "-c", "exec streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
