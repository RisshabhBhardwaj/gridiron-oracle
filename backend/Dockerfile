FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# WORKDIR /app means /app is on Python's sys.path.
# Code layout inside the container mirrors the project root:
#   /app/backend/app/   ← from backend.app.XXX import YYY  ✓
#   /app/pipeline/       ← from pipeline.XXX import YYY     ✓
#   /app/ml/             ← from ml.XXX import YYY           ✓
#   /app/scraper/        ← from scraper.XXX import YYY      ✓
WORKDIR /app

# Install deps first (better layer caching)
COPY backend/requirements.txt /app/requirements.txt
# Add build-essential and cmake for compiling the C++ engine
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*
# Install the CPU-only torch build first. The default PyPI wheel for linux/amd64
# drags in ~2.5 GB of nvidia-* CUDA runtime packages that a Railway container can
# never use; pinning the cpu index keeps the image inside Railway's size budget.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project packages — layout matches import paths in source code
COPY backend/app  /app/backend/app
COPY pipeline     /app/pipeline
COPY ml           /app/ml
COPY scraper      /app/scraper
COPY engine       /app/engine
COPY alembic      /app/alembic
COPY alembic.ini  /app/alembic.ini

# Release evidence read at runtime:
#   releases/artifacts/MANIFEST.json  ← runtime_status /integrity
#   releases/current_baseline.json    ← BASELINE_MANIFEST_PATH
#   releases/candidates/…/CONSTRAINED_STACK_SELECTION.json ← ml.served_learner
COPY releases     /app/releases
COPY user_config.json /app/user_config.json

# Build C++ engine
RUN cd /app/engine && rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

EXPOSE 8000
# Railway injects $PORT; fall back to 8000 for local docker-compose.
CMD ["sh", "-c", "python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
