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

# libgomp: LightGBM's multithreading. libseccomp: the code sandbox uses it to take the network and starting programs away from code cells
RUN apt-get update && apt-get install -y --no-install-recommends libseccomp2 libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# 'appuser' runs the app. Code cells each run as a throwaway user id of their own (see
# app/sandbox_runner.py), so they can't read the app's data (owner-only) or its process
RUN useradd --create-home appuser \
    && mkdir -p /app/data && chown appuser:appuser /app/data && chmod 700 /app/data
COPY --chown=appuser:appuser app/ ./app/
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 755 /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000

# Starts as root only to launch the code sandbox server, then runs the app as 'appuser'.
# ONE app worker on purpose: rate limits, the per-notebook queue and the in-memory notebook cache
# live inside the process. Scale up with a bigger machine, not more workers.
CMD ["/usr/local/bin/docker-entrypoint.sh"]
