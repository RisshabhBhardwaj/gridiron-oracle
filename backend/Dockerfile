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
# Add build-essential and cmake for compiling the C++ engine.
# libgomp1 is LightGBM's OpenMP runtime. It must be installed explicitly:
# otherwise it arrives only as a build-essential dependency and the
# `autoremove` below deletes it, so `import lightgbm` fails at run time.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Install the CPU-only torch build first. The default PyPI wheel for linux/amd64
# drags in several GB of nvidia-* CUDA runtime packages a CPU box never uses.
# The version MUST be the one pinned in requirements.txt: an unpinned install
# takes the newest CPU wheel, the next step sees the `torch==` pin unsatisfied
# and replaces it with the CUDA wheel from PyPI (same bug CI had). With the
# pinned version here, pip treats the pin as satisfied (X+cpu matches ==X).
RUN set -eu; \
    pin=$(grep -E '^torch==' /app/requirements.txt | sed -E 's/[[:space:]]*#.*$//'); \
    test -n "$pin"; \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "$pin"
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project packages — layout matches import paths in source code
# The commit this image was built from (compose passes the .env variable CODE_GIT_SHA, not GIT_COMMIT_SHA, which stays pinned to the freeze commit). The forward producer records
# it in every run and refuses to write when it is empty or "unknown" (decision 0012, D8).
ARG CODE_GIT_SHA=
ENV CODE_GIT_SHA=${CODE_GIT_SHA}
# Declared after the dependency layers so a new commit does not rebuild them.
COPY backend/app  /app/backend/app
COPY pipeline     /app/pipeline
COPY ml           /app/ml
COPY scraper      /app/scraper
COPY scripts      /app/scripts
COPY engine       /app/engine
COPY bench        /app/bench
COPY alembic      /app/alembic
COPY alembic.ini  /app/alembic.ini

# Release evidence read at runtime:
#   releases/artifacts/MANIFEST.json  ← runtime_status /integrity
#   releases/current_baseline.json    ← BASELINE_MANIFEST_PATH
#   releases/candidates/…/CONSTRAINED_STACK_SELECTION.json ← ml.served_learner
COPY releases     /app/releases
COPY user_config.json /app/user_config.json
COPY backend/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Build C++ engine
RUN cd /app/engine && rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

# Drop build tools from the runtime layer. curl stays: the compose
# healthcheck probes /health with it. git is safe to remove — the
# materializer reads the platform commit variable, never `.git`.
RUN apt-get purge -y build-essential cmake git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
# Fail the build, not a training run, if the purge took a native library.
RUN python3 -c "import lightgbm, catboost, xgboost, sklearn, shap, torch"

# Run as non-root (audit A-04). 8000 is unprivileged, so uvicorn needs no
# capabilities. /app stays owned by appuser so entrypoint.sh can seed
# SETTINGS_FILE on a fresh volume.
RUN useradd --system --uid 10001 --create-home appuser \
    && chown -R appuser:appuser /app
# Mount point for the vps1 `training` service's persistent `train_out` volume.
# Docker seeds an empty named volume with this directory's ownership.
RUN mkdir -p /train_out && chown appuser:appuser /train_out
USER appuser

EXPOSE 8000
# Seeds SETTINGS_FILE onto its volume when empty, then execs uvicorn.
# Railway injects $PORT; falls back to 8000 for local docker-compose.
CMD ["/app/entrypoint.sh"]
