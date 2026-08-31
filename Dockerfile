# Single-stage build: sentence-transformers/torch pull in enough native deps that a
# slim multi-stage split buys little here, and this keeps the image simple to reason about.
FROM python:3.11-slim

# System deps: libgomp1 is required by torch/onnxruntime at runtime (embedder + reranker);
# build-essential is needed to compile a couple of wheels (e.g. rank-bm25's C extensions
# on some platforms) and is removed after pip install to keep the final image smaller.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached across code-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential

# App code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Runtime dirs the app writes to at request time (upload temp files, BM25 pickle) —
# created up front so the container doesn't need root to create them on first write
RUN mkdir -p /app/temp

EXPOSE 8000

# Basic container-level health check hitting the real /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Reload is a dev feature (settings.debug) — the container always runs the production
# server invocation; set DEBUG=False in the environment for a prod deployment.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
