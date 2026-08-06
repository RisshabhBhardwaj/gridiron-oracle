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
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project packages — layout matches import paths in source code
COPY backend/app  /app/backend/app
COPY pipeline     /app/pipeline
COPY ml           /app/ml
COPY scraper      /app/scraper
COPY engine       /app/engine

# Build C++ engine
RUN cd /app/engine && rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
