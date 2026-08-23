# PlantAI backend — FastAPI service that also serves the console.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    ULTRALYTICS_RUNS_DIR=/tmp/runs \
    DATA_DIR=/data/plantai

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl ca-certificates git \
      gdal-bin libgdal-dev proj-bin proj-data \
      libgl1 libglib2.0-0 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU wheels by default; override with a CUDA index for a GPU host.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --upgrade pip && \
    pip install --index-url ${TORCH_INDEX} torch torchvision

COPY requirements.txt /app/requirements.txt
# torch/torchvision are already installed from the index above.
RUN grep -vE '^(torch|torchvision)' /app/requirements.txt > /tmp/reqs.txt && \
    pip install -r /tmp/reqs.txt

COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

RUN mkdir -p /data/plantai
WORKDIR /app/backend

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=5)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "65"]
