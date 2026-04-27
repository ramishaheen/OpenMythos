# --- Stage 1: build deps in a fat image ----------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# System libs needed by opencv-python-headless and Pillow.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg-dev \
        zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
COPY web/requirements.txt web/requirements.txt
COPY fraud_detection ./fraud_detection

# Install into a relocatable prefix to keep the runtime image small.
RUN pip install --prefix=/install --upgrade pip \
 && pip install --prefix=/install \
        anthropic openai Pillow numpy \
        opencv-python-headless scikit-image pypdfium2 \
        fastapi 'uvicorn[standard]' python-multipart pydantic


# --- Stage 2: minimal runtime --------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    FRAUD_LOG_LEVEL=INFO

# Non-root user.
RUN useradd --create-home --uid 10001 --shell /bin/bash mythos

# Runtime libs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        libgl1 \
        libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local
COPY fraud_detection /app/fraud_detection
COPY web /app/web

USER mythos

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=4s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
