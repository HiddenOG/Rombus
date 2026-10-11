# Backend: FastAPI + scikit-learn. Works on Railway (which sets $PORT) and with docker compose.

FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8000

COPY --from=builder /install /usr/local

# Run as an ordinary user, not root: if anything ever went wrong, it can't touch the system
RUN useradd --create-home appuser && mkdir -p /app/data && chown appuser:appuser /app/data
COPY --chown=appuser:appuser app/ ./app/
USER appuser

EXPOSE 8000

# ONE worker on purpose: rate limits, the per-notebook queue and the in-memory notebook cache
# live inside the process. Scale up with a bigger machine, not more workers.
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75"]
