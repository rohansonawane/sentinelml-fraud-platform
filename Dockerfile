FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY artifacts /app/artifacts
COPY scripts /app/scripts

RUN pip install --no-cache-dir .

EXPOSE 8102
HEALTHCHECK --interval=20s --timeout=5s --retries=5 CMD curl -f http://localhost:8102/health || exit 1
CMD ["uvicorn", "sentinelml.api.app:app", "--host", "0.0.0.0", "--port", "8102"]
