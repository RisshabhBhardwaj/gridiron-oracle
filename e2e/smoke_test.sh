#!/bin/bash
# e2e/smoke_test.sh
#
# End-to-end smoke test: starts all services, hits every API endpoint,
# checks the React build, and reports pass/fail.
#
# PREREQUISITES:
#   - Docker running
#   - C++ engine built: cd engine/build && cmake .. && make -j4
#   - Python deps installed: pip install -r backend/requirements.txt
#   - Node deps installed: cd frontend && npm install
#
# Run from project root:  bash e2e/smoke_test.sh

set -uo pipefail   # -e intentionally omitted so checks can fail without exit

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export DATABASE_URL=${DATABASE_URL:-postgresql://oracle:oracle@localhost:5432/oracle}
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"

PYTHON=/opt/homebrew/bin/python3.11
FASTAPI_PORT=8000
MLFLOW_PORT=5000
ENGINE_CONFIG="engine/config.json"

PASS=0
FAIL=0
FASTAPI_PID=""
MLFLOW_PID=""
ENGINE_PID=""

# ── helpers ────────────────────────────────────────────────────────────────────

check() {
  local label="$1"
  shift
  if eval "$@" 2>/dev/null; then
    echo "  PASS  $label"
    PASS=$((PASS + 1))
  else
    echo "  FAIL  $label"
    FAIL=$((FAIL + 1))
  fi
}

cleanup() {
  echo ""
  echo "── Cleaning up background processes ──"
  [ -n "$FASTAPI_PID" ] && kill "$FASTAPI_PID" 2>/dev/null && echo "  stopped FastAPI (pid=$FASTAPI_PID)"
  [ -n "$MLFLOW_PID"  ] && kill "$MLFLOW_PID"  2>/dev/null && echo "  stopped MLflow  (pid=$MLFLOW_PID)"
  [ -n "$ENGINE_PID"  ] && kill "$ENGINE_PID"  2>/dev/null && echo "  stopped engine  (pid=$ENGINE_PID)"
  # Remove engine IPC socket if leftover
  rm -f /tmp/gridiron.sock
}
trap cleanup EXIT

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Gridiron Oracle — End-to-End Smoke Test"
echo "════════════════════════════════════════════════════════════════"

# ── Step 1: Docker services ────────────────────────────────────────────────────
echo ""
echo "── [1/6] Starting Docker services ──"
docker compose -f infra/docker-compose.yml up -d 2>/dev/null || true
echo "  Waiting 10s for DB to be ready..."
sleep 10
check "PostgreSQL reachable" \
  "docker compose -f infra/docker-compose.yml exec -T db psql -U oracle -d oracle -c 'SELECT 1' -q"

# ── Step 2: FastAPI backend ────────────────────────────────────────────────────
echo ""
echo "── [2/6] Starting FastAPI backend ──"
# Remove any stale IPC socket from a previous run
rm -f /tmp/gridiron.sock
$PYTHON -m uvicorn backend.app.main:app \
  --host 127.0.0.1 --port $FASTAPI_PORT \
  --log-level warning &
FASTAPI_PID=$!
echo "  FastAPI started (pid=$FASTAPI_PID) — waiting 8s..."
sleep 8
check "FastAPI process alive" "kill -0 $FASTAPI_PID"

# ── Step 3: MLflow server ──────────────────────────────────────────────────────
echo ""
echo "── [3/6] Starting MLflow server ──"
mkdir -p ml/mlartifacts
$PYTHON -m mlflow server \
  --backend-store-uri sqlite:///ml/mlruns.db \
  --default-artifact-root ml/mlartifacts \
  --host 127.0.0.1 --port $MLFLOW_PORT \
  --serve-artifacts &
MLFLOW_PID=$!
echo "  MLflow started (pid=$MLFLOW_PID) — waiting 5s..."
sleep 5
check "MLflow process alive" "kill -0 $MLFLOW_PID"

# ── Step 4: C++ engine ────────────────────────────────────────────────────────
echo ""
echo "── [4/6] Starting C++ engine ──"
check "Engine binary exists" "test -x engine/build/engine"
if [ -x engine/build/engine ]; then
  # Engine reads config from argv[1] (not --config flag)
  ./engine/build/engine "$ENGINE_CONFIG" &
  ENGINE_PID=$!
  sleep 3
  check "Engine process alive" "kill -0 $ENGINE_PID"
else
  echo "  SKIP  Engine not built — run: cd engine/build && cmake .. && make -j4"
  FAIL=$((FAIL + 1))
fi

# ── Step 5: API endpoint checks ───────────────────────────────────────────────
echo ""
echo "── [5/6] API endpoint checks ──"
BASE="http://127.0.0.1:$FASTAPI_PORT"

check "/health" \
  "curl -sf $BASE/health"

check "/season/current returns {season,week}" \
  "curl -sf $BASE/season/current | grep -q 'season'"

check "/projections/week/1 returns array" \
  "curl -sf '$BASE/projections/week/1?season=2025' | grep -qE '\[|\]'"

check "/predict returns projection" \
  "curl -sf '$BASE/predict?player_id=00-0031344&week=1&season=2025&stat=receiving_yards&position=WR' | grep -q 'projection'"

check "/backtest returns results" \
  "curl -sf $BASE/backtest | grep -q 'mae'"

check "/settings returns weights" \
  "curl -sf $BASE/settings | grep -q 'xgb_weight'"

check "/alerts returns array" \
  "curl -sf $BASE/alerts | grep -qE '\[|\]'"

check "MLflow /health" \
  "curl -sf http://127.0.0.1:$MLFLOW_PORT/health"

# ── Step 6: Frontend build ─────────────────────────────────────────────────────
echo ""
echo "── [6/6] Frontend build check ──"
if [ -d frontend ] && [ -f frontend/package.json ]; then
  check "React build (npm run build)" \
    "cd frontend && npm run build --silent"
else
  echo "  SKIP  frontend/ not found"
  FAIL=$((FAIL + 1))
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  SMOKE TEST RESULTS"
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -eq 0 ]; then
  echo "  ALL CHECKS PASSED — system is production-ready"
else
  echo "  $FAIL checks failed — review output above"
fi
echo "════════════════════════════════════════════════════════════════"

# Exit code mirrors failure count
[ "$FAIL" -eq 0 ]
