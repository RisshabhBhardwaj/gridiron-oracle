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
export DATABASE_URL=${DATABASE_URL:-postgresql://oracle:oracle@localhost:15439/oracle}
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

# Response-shape assertions live in a sourced library so they can be tested
# directly — see backend/tests/test_http_assert.py (audit C-32).
# shellcheck source=lib/http_assert.sh
source "$(cd "$(dirname "$0")" && pwd)/lib/http_assert.sh"

cleanup() {
  echo ""
  echo "── Cleaning up background processes ──"
  [ -n "$FASTAPI_PID" ] && kill "$FASTAPI_PID" 2>/dev/null && echo "  stopped FastAPI (pid=$FASTAPI_PID)"
  [ -n "$MLFLOW_PID"  ] && kill "$MLFLOW_PID"  2>/dev/null && echo "  stopped MLflow  (pid=$MLFLOW_PID)"
  [ -n "$ENGINE_PID"  ] && kill "$ENGINE_PID"  2>/dev/null && echo "  stopped engine  (pid=$ENGINE_PID)"
  # Remove engine IPC socket if leftover
  rm -f /tmp/gridiron.sock
  http_assert_cleanup
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

# /predict takes `player` (a name; partial match is resolved server-side),
# `week`, `season`, and `stat`. It does NOT take `player_id` or `position` —
# position is derived from the resolved player. The old request sent the
# obsolete parameters and therefore could not exercise the current contract
# (audit C-32).
#
# The cell must be one the materializer actually covered: only 10 of the 15
# declared (stat, position) cells were materialized, and `receiving_yards` —
# what this check used to request — is not among them (audit C-10).
PREDICT_PLAYER="${SMOKE_PLAYER:-Amon-Ra St. Brown}"
PREDICT_STAT="${SMOKE_STAT:-fantasy_ppr}"
PREDICT_SEASON="${SMOKE_SEASON:-2024}"
PREDICT_WEEK="${SMOKE_WEEK:-2}"

check "/health" \
  "curl -sf $BASE/health"

api_check "/season/current returns {season, week}" \
  "$BASE/season/current" "
assert isinstance(d, dict), type(d)
assert isinstance(d['season'], int), d
assert isinstance(d['week'], int), d
"

api_check "/projections/week/$PREDICT_WEEK returns populated rows" \
  "$BASE/projections/week/$PREDICT_WEEK?season=$PREDICT_SEASON&stat=$PREDICT_STAT" "
rows = d['projections']
assert isinstance(rows, list), type(rows)
assert rows, 'empty projections list — an empty array used to pass this check'
assert d['count'] == len(rows), (d['count'], len(rows))
assert d['stat'] == '$PREDICT_STAT' and d['season'] == $PREDICT_SEASON, d['stat']
r = rows[0]
for field in ('player_id', 'player_name', 'position', 'stat', 'projection', 'floor', 'ceiling'):
    assert field in r, f'missing {field} in {sorted(r)}'
assert isinstance(r['projection'], (int, float)), r['projection']
assert r['floor'] <= r['projection'] <= r['ceiling'], r
"

api_check "/predict returns a full projection for the current contract" \
  "$BASE/predict?player=$(printf %s "$PREDICT_PLAYER" | sed 's/ /%20/g')&week=$PREDICT_WEEK&season=$PREDICT_SEASON&stat=$PREDICT_STAT" "
for field in ('player', 'player_id', 'week', 'season', 'position', 'stat',
              'projection', 'percentiles', 'model_version', 'data_freshness'):
    assert field in d, f'missing {field} in {sorted(d)}'
assert d['stat'] == '$PREDICT_STAT', d['stat']
assert d['season'] == $PREDICT_SEASON and d['week'] == $PREDICT_WEEK, d
# The requested stat must appear as a key of the stat projection.
assert '$PREDICT_STAT' in d['projection'], sorted(d['projection'])
value = d['projection']['$PREDICT_STAT']
assert isinstance(value, (int, float)) and value == value, value   # not None, not NaN
p = d['percentiles']
assert p['p10'] is not None and p['p50'] is not None and p['p90'] is not None, p
assert p['p10'] <= p['p50'] <= p['p90'], p
assert isinstance(d['model_version'], str) and d['model_version'], d['model_version']
"

# Regression lock: the pre-C-32 request shape must not quietly succeed. A 422
# is the correct answer — `player` is required, and `player_id`/`position` are
# not parameters at all.
status_check "/predict rejects the obsolete player_id/position contract" \
  "$BASE/predict?player_id=00-0031344&week=1&season=2025&stat=receiving_yards&position=WR" \
  422

api_check "/backtest returns results" \
  "$BASE/backtest" "
rows = d['results'] if isinstance(d, dict) and 'results' in d else d
assert rows, 'empty backtest payload'
blob = json.dumps(d)
assert 'mae' in blob, sorted(d) if isinstance(d, dict) else type(d)
"

api_check "/settings returns weights" \
  "$BASE/settings" "
assert isinstance(d, dict), type(d)
assert 'xgb_weight' in d, sorted(d)
"

api_check "/alerts returns a list" \
  "$BASE/alerts" "
rows = d['alerts'] if isinstance(d, dict) and 'alerts' in d else d
assert isinstance(rows, list), type(rows)
"

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
