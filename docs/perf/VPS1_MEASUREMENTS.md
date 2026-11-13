# VPS-1 → VPS-4 measurements (no Docker)

Date: 2026-11-12. Machine: Apple M2 Pro, 10 cores, 16 GB RAM. Region-free:
per the BHS decision the numbers project to the VPS-4 spec (8 vCores, 24 GB
RAM, shared box with ATS) regardless of region (VPS-4 in Canada-East/
Beauharnois).
Worktree: `agent/ovh-vps1-measure` (base `b069384`; local `main` has since
moved to `c6f07de` — not rebased, manager handles merge).

Venv note: this worktree has no `.venv_311`; all runs used the main checkout's
`.venv_311/bin/python` (same base commit) with cwd in this worktree.
`DATABASE_URL` pointed at scratch DB `aw_15c225fec8_ovh_vps1_measure` in the
local brew Postgres 17 (the launcher's TEST_DATABASE_NAMESPACE). No prod data
touched. Env everywhere: `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
OTEL_TRACES_EXPORTER=none`, `PRODUCT_MODE=artifact_backed` for serving runs.

Blocker (honest): the local snapshot `gor_prod_20261010` (pg18,
127.0.0.1:15439) is down and its dump file is gone from disk; the data
persists only in docker volume `gor-pg18-data`, which this worker may not
mount (no-Docker rule). Serving/ETL numbers below are therefore lower bounds
(empty DB → 404/503 paths). Real-data serving + ETL numbers should come from
the manager's `agent-docker` rehearsal.

## 1. Serving import trace

Method: static grep of top-level imports + runtime `sys.modules` probes:
`/tmp/vps1_probe_imports.py` (top-level `import backend.app.main`),
`/tmp/vps1_probe_trace.py` (TestClient lifespan + real `GET /health`,
`GET /predict`, `GET /explain`), `/tmp/vps1_probe_lazy.py` (SHAPService.explain
with MLflow unreachable; `run_onnx_inference` on committed `ml/onnx/*.onnx`).

| Library | At startup | On first request | Notes (file:line) |
|---|---|---|---|
| torch | never | never¹ | lazy only in `ml/tft_model.py:106` (TFT path) |
| catboost | never | never¹ | native lib never imported; ONNX files used instead |
| lightgbm | never | never¹ | same as catboost |
| xgboost | never | never¹ | not in any served ridge `learner_order` |
| mlflow | never | never² | lazy: `ml/inference_client.py:86-88,416,455,458`; `ml/shap_service.py:288` |
| shap | never | never² | lazy: `ml/shap_service.py:348` (`_real_shap` only) |
| optuna | never | never | training-only (`ml/lgbm_model.py`, `ml/xgb_model.py`) |
| pymc | never | never | never imported on serving path (grep) |
| onnxruntime | never | fallback-only | lazy `ml/utils.py:609`; loads only on pipeline-fallback path when an `ml/onnx/*.onnx` file matches — verified returning real predictions (`catboost_carries_QB.onnx` → `[2.688, 2.688]`) |
| pytorch_forecasting | never | never | training-only (grep; no serving import) |

¹ Normal path (precomputed `projections` row found): zero heavy imports —
`/predict` → `ProjectionService` (top-level imports: numpy, fastapi,
`ml.served_learner`) → `_build_shap_factors` (`backend/app/api/predict.py:506`)
imports `ml.shap_service` module (light) and `SHAPService.explain` fail-closes
with `ArtifactRequiredError` → factors `"artifact_required"`, no `mlflow`/`shap`
import. Verified at runtime.
² `mlflow`/`shap` packages load only if MLflow is reachable AND a registered
model exists — impossible in `artifact_backed` mode (fail-closed before any
import). `torch` loads only for TFT-learner fallback; no served ridge cell
uses TFT (all 15 pinned cells are `ridge_stack`/`lgbm_identity` over
catboost/lgbm learners — `releases/candidates/causal_20260821_fixed/`).

Implication for the serving image: the lean image can drop torch, catboost,
lightgbm, xgboost, mlflow, shap, optuna, pymc, pytorch_forecasting. Keep
`onnxruntime` ONLY if the fallback path must work in serving; otherwise the
fallback returns 404/`NoForecastAvailable` without it. `ml/train.py`
(pipeline fallback) is itself light at import (numpy/pandas only).

## 2. API RSS (single process, port 4318, empty-DB lower bound)

Method: `/usr/bin/time -l` around the server process, plus direct `ps -o rss`
sampling of the server pid. Early supervisor-mode runs were discarded (the
wrapper, not the worker, was measured). Final runs use single-process
`python -m uvicorn backend.app.main:app --port 4318` (no `--workers`, no
`--reload`), which is the same single serving process. Scripts:
`/tmp/vps1_rss_both.sh` (idle + 50x load), `/tmp/vps1_rss_sample.sh`,
`/tmp/vps1_rss_boot.sh` (0.2 s sampling from launch).

Warmup call statuses (empty scratch DB, all expected): `GET /health` → 503
`blocked` (no artifact data), `GET /predict` → 404, `GET /explain/{id}` →
503 `"SHAP model artifact unavailable while PRODUCT_MODE=artifact_backed"`
(fail-closed over real HTTP — matches the probe).

- Idle (warmup + 1xhealth + 1xpredict + 1xexplain + 3 s settle):
  `148,979,712 B max RSS (~142 MiB)`, 5.28 s wall for the run window.
- Load (warmup + 50 sequential `/predict`, all served — 200 request log
  lines): `88,080,384 B max RSS (~84 MiB)`, 2.37 s server lifetime.
- Direct sampling (0.2 s cadence from launch through boot + 51 requests):
  peak `86,016 KB (~84 MiB)`, stable across boot transient and requests.

The 142 vs 84 MiB spread is unexplained (same binary, same machine; likely
allocator/page-cache variance on first boot). Budget the API at **0.25 GB**
(conservative, ~1.7x the highest observed). With real data, responses do real
work (row fetch, SHAP fallback) — re-measure against the snapshot in the
Docker rehearsal; treat 0.25 GB as a lower bound.

## 3. Weekly ETL dry run

Command:

    python scripts/weekly_refresh.py --dry-run \
      --database-url postgresql://risshabh@localhost:5432/aw_15c225fec8_ovh_vps1_measure

Result: `1.08s real, max RSS 126,255,104 B (~120 MB)`. The script got as far
as `resolve_season_and_week` and failed on the missing `games` table in the
empty scratch DB (dead_letter write also failed — no such table). This
measures interpreter import + failed-DB path only, NOT a real ETL. Real ETL
needs the snapshot; measure in the `agent-docker` rehearsal (volume
`gor-pg18-data`).

## 4. One-fold training profile (FIRST priority) — FITS, margin ~28x

Command (mirrors `scripts/retrain_causal_oof.sh` one cell, lgbm, n-trials 20;
`/tmp/vps1_probe_fold.py`):

    /usr/bin/time -l .venv_311/bin/python /tmp/vps1_probe_fold.py
    # train(df from ml/cache WR matrix 12256x133, seasons 2019-2025,
    #       target receiving_yards, position WR, n_optuna_trials=20,
    #       mlflow off, out under task tmp)

Substitution: matrix read from the existing `ml/cache` parquet instead of
re-querying the snapshot DB (same `train()` code path; DB fetch is IO, not the
RSS driver). Largest position matrix chosen (WR, 12,256 rows x 133 cols) so
the verdict is conservative.

Raw `/usr/bin/time -l`:

    OOF rows: 10641, folds: 6 (val 2020..2025)
    fold MAE 25.3-27.6, RMSE 32.0-35.6
    192.76 real, 124.62 user, 322.90 sys
    366,133,248 maximum resident set size (~349 MB)
    382,682,120 peak memory footprint

**Verdict: FITS the 10 GB training-slice cap (0.35 GB, ~3.5% of cap). No stop.
**

Projection to 5 capped VPS-4 vCores (assumption: a shared OVH vCore sustains
~0.4x an M2 Pro core → ~2.5x wall time; LightGBM averaged ~2.3 cores here,
well under the 5-CPU quota):
- Tree-OOF full retrain (`retrain_causal_oof.sh`: 15 cells x lgbm+catboost @
  20 trials + stack): 15 x ~193s (lgbm) + catboost ~2-4x lgbm (assumption,
  unmeasured) → ~3-4 h Mac → **~8-10 h on 5 capped vCores. Fits Fri-night–
  weekend window (~58 h) with large margin.**
- One small refit (single cell): ~193s Mac → **~8 min on VPS. Fits the
  01:00–07:00 window trivially.**
- NOT measured: TFT/GNN/SHAP full retrain (needs its own fold profile before
  scheduling; owner's prior figure 30-40 h Mac — unconfirmed here).
- Memory: cells run sequentially → full retrain peaks at ~one-cell RSS even
  allowing 5x for catboost (~2 GB) — still far under 10 GB.

## 5. Shared VPS-4 budget (24 GB)

| Envelope | Size | Basis |
|---|---|---|
| Gridiron site (API + PG18 + cloudflared) | ~2.5 GB | API 0.25 GB budget (84-142 MiB observed, empty-DB lower bound), PG `shared_buffers` 256 MB + conns (~0.5 GB), cloudflared ~0.1 GB, rest headroom |
| ATS (other project) | ~9.5 GB | reserved, not measured here |
| Gridiron training slice | ~10 GB cap | one fold peaks 0.35 GB; catboost allowance included |
| Host/Docker/cloudflared spare | ~2 GB | 24 - (2.5+9.5+10) |

`pg_dump -Fc` (~28 MB dump per prior handoff): unmeasured, tens of MB —
schedule with the nightly backup, measure on first live run. Weekly ETL real
RSS: unmeasured (blocked, see above) — cap the one-shot at ~2 GB until the
Docker rehearsal measures it.

What does NOT fit / stays off: TFT/GNN full retrains until profiled (schedule
only after a torch-CPU fold profile); observability profile (MLflow, Jaeger,
OTel, Prometheus, Alertmanager) only inside the 2.5 GB site envelope — turn on
selectively. Nothing may grow one envelope into another; if measurement busts
an envelope, report it.
