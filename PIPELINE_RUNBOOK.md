# Gridiron Oracle — Pipeline Runbook

**Purpose:** End-to-end guide from raw data → trained models → predictions. Designed for multi-day runs with **progress saved at every step** so a crash does not lose work.

---

## 1. Pipeline Order

```
ETL (run_full_etl.sh)
    ↓
Training (train_all_models.sh)   ← Use --resume after any crash
    ↓
Predictions (served via FastAPI /predict)
```

---

## 2. Step 1 — ETL

**Command:** `bash pipeline/run_full_etl.sh`

**What it does:**
1. Starts Docker (PostgreSQL)
2. Orchestrator: ingest → normalize → feature_engineer (per season)
3. Enrich: Elo, player embeddings, PBP, injury
4. Optional: weather + odds (need API keys)
5. Row count verification

For a dry-run sanity check of the CLI wiring:
```bash
.venv_311/bin/python -m pipeline.ingest --dry-run --seasons 2025
```

**Progress saved:**
- **Per source:** staging rows marked `processed=true` after each normalize step
- **Per season:** feature_matrix committed in batches of 2000 rows
- **On crash:** Re-run `run_full_etl.sh` — it continues from unprocessed staging rows; normalize/feature steps are idempotent (upsert)

**Runtime:** ~60–120 min (7 seasons)

**Prerequisites:** Docker, `DATABASE_URL` set

---

## 3. Step 2 — Training

**Command:** `bash ml/train_all_models.sh`  
**Resume after crash:** `bash ml/train_all_models.sh --resume`

**Sub-steps (each saves before moving on):**

| Step | What runs | Saves | Runtime |
|------|-----------|-------|---------|
| 1 | XGB (38 stat×pos) | OOF → `ml/oof/xgb_*.csv`, checkpoint → `ml/checkpoints/done/xgb_{stat}_{pos}.done` | ~3–4 hrs |
| 2 | LightGBM (38) | OOF → `ml/oof/lgbm_*.csv`, checkpoint | ~2–3 hrs |
| 3 | CatBoost (38) | OOF → `ml/oof/catboost_*.csv`, checkpoint | ~3–5 hrs |
| 4 | TFT (15 stats) | **Per-stat checkpoint** → `ml/checkpoints/done/tft_{stat}.done` + OOF; best model each fold → `ml/checkpoints/tft/{stat}_fold{N}/`; resume skips any stat with checkpoint OR OOF file | ~18–22 hrs |
| 5 | Stacking | Ridge coefs → `ml/oof/ridge_*_coefs.json`, checkpoint | ~10 min |
| 6 | Projection pipeline | **Per week** → `projections` table in DB | ~4–8 hrs |

**Progress saved:**
- **Base learners:** OOF CSV written at end of each (stat, position) run; checkpoint file written on success
- **With `--resume`:** Skips any task that has a checkpoint file; skips (season, week) already in `projections`
- **TFT:** `ModelCheckpoint` saves best model every epoch to `ml/checkpoints/tft/{stat}_fold{N}/`
- **Projections:** DB commit after **each week** — if it crashes at week 5, weeks 1–4 are persisted

**Total runtime:** ~30–40 hrs (full run)

**Prerequisites:**
- PostgreSQL + feature_matrix populated (Step 1)
- MLflow server: Docker publishes the service at `http://localhost:15091`
  (the container itself listens on `5001`). Set host processes to
  `MLFLOW_TRACKING_URI=http://localhost:15091`; do not use the container port
  from the host.
- `DATABASE_URL`, `MLFLOW_TRACKING_URI` set

---

## 4. Step 3 — Predictions

**How:** FastAPI backend reads from `projections` table and feature_matrix.

**Command to start backend:** `make up` or `uvicorn backend.app.main:app --reload`

**Endpoints:**
- `GET /predict?season=2025&week=1` — projections for a week
- `GET /health` — liveness

**Data freshness:** `/predict` includes `data_freshness`; frontend should warn if >24 hrs old.

---

## 5. Recovery After a Crash

### ETL crash
Re-run `run_full_etl.sh`. It will:
- Process only unprocessed staging rows
- Upsert into production tables (idempotent)

### Training crash (e.g. after 20 hrs)
```bash
bash ml/train_all_models.sh --resume
```
This will:
- Skip XGB/LGB/CatBoost/TFT/Stacking tasks that have checkpoint files
- **TFT:** Skips any stat that has `tft_{stat}.done` OR an OOF file (`tft_{stat}_*.csv`). Checkpoint is written by the Python process immediately when that stat completes.
- Skip (season, week) pairs already in `projections`
- Continue from the first incomplete step

**TFT specifically:** If TFT crashes during stat 8 of 15, stats 1–7 are already checkpointed. On `--resume`, it will skip stats 1–7 and start at stat 8. No 18+ hour re-run.

### To force full re-run (ignore checkpoints)
```bash
rm -rf ml/checkpoints/done/
# Optionally clear projections for a specific season/week in DB
bash ml/train_all_models.sh
```

---

## 6. Verification Checklist

Fast offline checks before committing:

```bash
ruff check .
OTEL_TRACES_EXPORTER=none .venv_311/bin/python -m pytest backend/tests/ -q -m "not integration and not slow and not network"
npm run typecheck
npm test -- --run
make diagnose-backtest
```

Before a long run:

- [ ] `DATABASE_URL` set and PostgreSQL reachable
- [ ] `MLFLOW_TRACKING_URI` set (e.g. `http://127.0.0.1:15091`)
- [ ] MLflow server running
- [ ] `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` (Apple Silicon)
- [ ] `.venv_311` exists with `pip install -r requirements.txt`
- [ ] `ml/oof` and `ml/checkpoints` exist (or will be created)

After ETL:

- [ ] `game_logs` row count ~426k
- [ ] `feature_matrix` row count ~129k

After training:

- [ ] `projections` row count ~161k
- [ ] `ml/oof/ridge_*_coefs.json` files exist
- [ ] MLflow runs visible at `$MLFLOW_TRACKING_URI`

Before freezing a release:

- [ ] `make verify-readiness` returns `overall_status: ok`
- [ ] `make freeze-baseline` writes `releases/current_baseline.json`
- [ ] If readiness is `blocked`, do not freeze unless you intentionally set `ALLOW_BLOCKED_BASELINE=1` for an emergency override.
- [ ] `make diagnose-backtest` has been reviewed, especially passing_yards and rushing_yards regressions.

---

## 7. Tracing

The API now propagates W3C trace context (`traceparent` / `tracestate`) from
the HTTP edge through live projection fallback:

`HTTP request -> ProjectionService -> PipelineRunner -> Kalman -> Stacking -> Bayesian -> Monte Carlo`

Runtime env vars:

- `OTEL_SERVICE_NAME=gridiron-oracle-api`
- `OTEL_TRACES_EXPORTER=none|console|otlp`
- `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318/v1/traces`
- `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf|grpc`
- `OTEL_EXPORTER_OTLP_HEADERS=authorization=Bearer ...`
- `OTEL_EXPORTER_OTLP_INSECURE=true|false`

Notes:

- Default is `OTEL_TRACES_EXPORTER=none`, which keeps propagation and local
  trace IDs active without exporting spans anywhere.
- `/integrity` now reports tracing mode so you can tell whether spans are only
  local or actively exportable.
- In the Docker stack, the backend exports OTLP HTTP to `otel-collector`,
  which forwards traces to Jaeger UI at `http://localhost:18686`.

---

## 8. Incident Response

This section covers what to do when the system reports a problem. Check `/integrity` first — it
returns `overall_status: ok | degraded | blocked` and per-domain details.

---

### 8.1 `/integrity` returns `overall_status: blocked`

**Immediate steps:**
1. Open `/integrity` in a browser or `curl localhost:18017/integrity | jq .` to identify which
   domain is blocked (database, mlflow, backtest_assets, baseline_manifest, tracing).
2. For each blocked domain, follow the relevant section below.
3. Once resolved, re-check `/integrity` until status returns `ok`.

**If frontend is already showing stale/broken data**, stop new traffic via the load balancer
before investigating to prevent serving bad projections.

---

### 8.2 Database unreachable

**Symptoms:** `/integrity` shows `database: blocked`. `/predict` returns 503.

**Recovery:**
```bash
make up           # restart Docker stack (restarts PostgreSQL)
docker ps         # verify gridiron_db container is running
psql $DATABASE_URL -c "SELECT 1"   # confirm connectivity
```

If the database volume is corrupted:
```bash
docker-compose down -v            # WARNING: destroys data
make up
bash pipeline/run_full_etl.sh     # re-ingest from nflreadpy (~60-120 min)
bash ml/train_all_models.sh       # full retrain (~30-40 hrs)
```

---

### 8.3 Feature matrix stale (>24h)

**Symptoms:** Frontend shows data freshness warning. `/integrity` shows `feature_matrix: degraded`.
Alert feed shows "Data Freshness Warning".

**Recovery:**
```bash
make ingest    # runs ETL orchestrator, typically 60-120 min
```

After completion verify:
```bash
psql $DATABASE_URL -c "SELECT MAX(computed_at) FROM feature_matrix"
```
Expected: timestamp within the last hour.

---

### 8.4 MLflow unreachable

**Symptoms:** `/integrity` shows `mlflow: degraded`. Live fallback predictions may still work
if the DB has existing projections.

**Recovery:**
```bash
mlflow server \
  --backend-store-uri sqlite:///ml/mlruns.db \
  --default-artifact-root ml/mlartifacts \
  --host 127.0.0.1 --port 15091
```

If `mlruns.db` is missing (data loss), restart from the last model checkpoint:
```bash
# Re-register artifacts from the OOF and checkpoint files that survived
bash ml/train_all_models.sh --resume
```

---

### 8.5 Backtest assets missing

**Symptoms:** `/integrity` shows `backtest_assets: blocked`. `/backtest` returns 503.

**Recovery:**
```bash
make backtest    # re-runs walk-forward backtest suite (~1-2 hrs)
```

---

### 8.6 C++ execution engine disconnected

**Symptoms:** IPC bridge error in logs (`ipc_client` connection refused). Season projections
return empty or 503. A `SYSTEM` alert is published to the alert feed on disconnect.

**Recovery:**
```bash
make engine           # rebuild C++ engine binary
./engine/build/engine engine/config.json   # start engine process
```

Verify reconnection in the API logs:
```
INFO  ipc_client — connected to engine socket
```

If the engine crashes repeatedly, check for:
- Missing `engine/config.json`
- Port conflict on the Unix domain socket path
- Build failure: `make engine` output for compiler errors

---

### 8.7 Training crash mid-run

See **§5 (Recovery After a Crash)** above. Always use `--resume`:
```bash
bash ml/train_all_models.sh --resume
```

**Do not** delete checkpoint files unless you intend a full retrain.

---

### 8.8 API_KEY not set (authentication disabled)

**Symptoms:** Startup log shows `WARNING: API_KEY is not set — authentication is DISABLED`.

**Recovery:** Set the environment variable before starting the API:
```bash
export API_KEY="$(openssl rand -hex 32)"
# or add to .env and restart: make up
```

---

## 9. Incident Response Drill

**Purpose:** Verify that the §8 recovery procedures still work before they're
needed in anger. Run this checklist quarterly against a local Docker stack.

**On-call context:** This is a solo project — there is no on-call rotation.
The developer is both the responder and the stakeholder. The drill below is
a tabletop exercise (no production traffic at risk).

### Drill checklist

Run `make up` before starting. Work through each scenario in order; tick it
off only after confirming the described recovery path resolves the symptom.

- [ ] **8.1 Blocked integrity** — Stop MLflow (`docker stop mlflow`). Hit `/integrity`.
  Confirm `mlflow` domain shows `degraded`. Restart MLflow, confirm recovery.
- [ ] **8.2 DB unreachable** — Stop postgres (`docker stop postgres`). Hit `/health`.
  Confirm `status: blocked`. Restart, confirm `/health` returns `ok`.
- [ ] **8.3 Stale feature matrix** — Run `UPDATE feature_matrix SET computed_at = NOW() - INTERVAL '25 hours'` on one row.
  Trigger `_check_data_freshness()` manually (or wait for hourly job). Confirm
  "Data Freshness Warning" appears in the alerts feed. Run `make ingest` to clear.
- [ ] **8.4 MLflow unreachable during training** — Stop MLflow, run a single
  `python ml/train.py --stat passing_yards --position QB --seasons 2024`.
  Confirm training degrades gracefully (logs WARNING, does not crash).
- [ ] **8.5 Missing backtest assets** — Rename `ml/oof/` temporarily.
  Hit `/integrity`. Confirm `backtest_assets` shows `blocked`. Restore dir.
- [ ] **8.6 Engine disconnect** — If engine is running, kill it. Confirm
  "C++ Engine Disconnected" system alert appears in `/api/alerts`.
- [ ] **8.7 Training crash** — Kill a `train.py` run mid-flight with Ctrl-C.
  Confirm `--resume` picks up from the last completed stat/position checkpoint.
- [ ] **8.8 API_KEY unset** — Start API without `API_KEY` set. Confirm startup
  log shows the `authentication is DISABLED` warning. Set key and restart.

**Drill cadence:** Once per quarter, or after any change to §8 procedures.
**Last drilled:** 2026-05-03 _(documentation review — run `make up` then execute each scenario above to complete a live drill)_

---

## 10. Quick Reference

| Command | Purpose |
|---------|---------|
| `make up` | Start all services (DB + backend) |
| `make ingest` | Run ETL orchestrator only |
| `bash ml/train_all_models.sh` | Full training (30–40 hrs) |
| `bash ml/train_all_models.sh --resume` | Resume after crash |
| `make test` | Run non-integration Python tests + the full C++ CTest suite |
| `make test-integration` | Run the DB-backed FastAPI integration tests explicitly |
| `make mutmut` | Run phase-1 mutation testing for backend runtime paths |
| `make mutmut-results` | Print the current mutmut status summary |

---

*Last updated: 2026-05-02*
