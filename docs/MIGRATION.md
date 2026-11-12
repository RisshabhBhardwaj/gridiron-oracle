# Migration to the VPS

Dated 2026-10-17. Status: **planned, not executed.** Nothing here has run
against a live host. This file describes the move; it does not perform it.

Target: an owner-acquired OVH VPS (not yet ordered) replacing
Railway (api, weekly-refresh) and Neon (Postgres). The frontend moved to
Cloudflare Pages on 2026-11-12 and does not move to the VPS
(`docs/runbooks/frontend-cloudflare-pages.md`). After cutover, point
`BACKEND_URL` in `frontend/wrangler.toml` at the VPS API and redeploy.
No worker provisions, deploys, or touches any live host or DNS entry.

## 1. VPS bootstrap assumptions

Confirm each on the day; none are done.

- Fresh Linux host with git and the Docker Compose plugin installed.
- This repo cloned at the migrated commit, on `main`.
- Ports from `infra/docker-compose.yml` stay on loopback (they are bound to
  `127.0.0.1`); public traffic arrives through a reverse proxy the owner
  installs. There is no proxy config in this repo yet — that is planned work,
  not built work.
- Postgres image is `postgres:18` (see `infra/docker-compose.yml`).

Bring-up:

```bash
git clone <repo-url> gridiron_oracle && cd gridiron_oracle
cp .env.example .env   # then fill in section 2 values
make up
PYTHONPATH=. .venv_311/bin/python scripts/migrate.py upgrade
```

`make up` runs `docker compose -f infra/docker-compose.yml up -d`.
`make migrate` applies the Alembic schema (`scripts/migrate.py upgrade`).

## 2. Env and secrets inventory (names only, never values)

`.env` is never committed (`.githooks/pre-push` and CI `push-gate` block
secrets). Every name below is read from the environment; see `.env.example`
for defaults and `backend/app/core/config.py` for the authoritative list.

Secrets to generate fresh on the VPS (do not reuse dev values):

- `DATABASE_URL` — change the `oracle:oracle` dev password.
- `API_KEY` — generate: `python3 -c "import secrets; print(secrets.token_hex(32))"`
- `ADMIN_API_KEY` — separate value from `API_KEY` in production.
- `ODDS_API_KEY` — the-odds-api.com key (500 req/month free tier).
- `OPENWEATHER_API_KEY` — openweathermap.org key (1,000 req/day free tier).

Non-secret config (copy from `.env.example`, adjust hosts as needed):

- `MLFLOW_TRACKING_URI`, `MODEL_VERSION`, `PRODUCT_MODE`,
  `BASELINE_MANIFEST_PATH`, `ARTIFACT_INVALIDATION_PATH`,
  `ENGINE_CONFIG_PATH`, `CORS_ORIGINS`, `SETTINGS_FILE`, `LOG_LEVEL`,
  `KMP_DUPLICATE_LIB_OK`, `OMP_NUM_THREADS`,
  `OTEL_SERVICE_NAME`, `OTEL_TRACES_EXPORTER`,
  `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_PROTOCOL`,
  `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_EXPORTER_OTLP_INSECURE`.
- Scripts-only: `ALLOW_BLOCKED_BASELINE`, `APPROVED_PIPELINE_RUN_IDS`,
  `GIT_COMMIT_SHA` (used when no platform commit variable exists).
- Frontend build-time: `VITE_SETTINGS_EDITABLE` — the public build must NOT
  set it (Settings page stays view-only).

## 3. Database move (Neon Postgres 18 -> VPS Postgres 18)

Both ends are Postgres 18, so plain `pg_dump` / `pg_restore` applies.
Run from a machine that can reach both databases (owner's laptop or the VPS).

```bash
# 1. Snapshot the source (custom format, single file):
pg_dump -Fc "$NEON_DATABASE_URL" -f /tmp/gridiron_prod.dump
# 2. Create the target database on the VPS stack, then restore:
docker compose -f infra/docker-compose.yml up -d db
pg_restore -d "$VPS_DATABASE_URL" --no-owner --role=oracle /tmp/gridiron_prod.dump
# 3. Bring the schema to the repo head (no-op if the dump is current):
make migrate
# 4. Sanity check row counts, e.g.:
#    games 2019-2026, pbp_plays, game_logs, feature_matrix, depth_charts,
#    projections, team_game_predictions (see docs/status/CURRENT.md for the
#    last known-good counts).
```

Keep `/tmp/gridiron_prod.dump` until section 6 (rollback) is retired.

## 4. Artifact and data volumes

Restore or re-transfer these onto the VPS (all are gitignored; none travel
with the clone):

- Docker volumes `db_data`, `mlflow_artifacts`, `prometheus_data` — recreated
  by `make up`; contents come from the section 3 restore (db) and MLflow
  re-tracking (artifacts). Back them up before cutover:
  `docker run --rm -v db_data:/data -v /tmp:/backup alpine tar czf /backup/db_data.tgz /data`
  (repeat per volume).
- Training inputs `ml/cache/*.parquet` (2019-2025 feature matrices) — copy
  with `rsync`/`scp` from the machine that holds them; verify with
  `.venv_311/bin/python scripts/verify_artifact_prerequisites.py`.
- Pinned serving artifacts under `releases/` travel with git, but any
  absolute `/Users/...` paths baked into `releases/*.json` and
  `ml/onnx/*.onnx` are fixed only by re-freezing on the VPS
  (`make freeze-baseline`), never by hand-editing.
- Shipped evidence under ignored paths (`reports/`, `ml/oof/`,
  `ml/experiments/`) must stay listed in `releases/artifacts/MANIFEST.json`;
  check with `make verify-evidence`.

## 5. DNS cutover checklist — OWNER-ONLY

No worker performs these. Owner runs them after sections 1-4 verify green:

1. Point the domain at the VPS (registrar/DNS change).
2. Confirm the reverse proxy serves the frontend and `/health` answers.
3. Run the smoke test: `bash e2e/smoke_test.sh` (needs `make up` first).
4. Confirm `PRODUCT_MODE=artifact_backed` with `/integrity` green:
   `make verify-readiness`.
5. Only then stop or delete the Railway services and Vercel project. Keep
   the Neon project read-only until section 6 expires.

## 6. Rollback

If the VPS misbehaves after cutover:

1. Point DNS back at the old hosting (section 5, step 1 in reverse).
2. Restart Railway/Vercel services if they were stopped (they deploy from
   `main`; no rebuild needed if `main` did not move).
3. The VPS database is disposable until the old hosting is deleted: the
   section 3 dump file plus the Neon source remain the record.
4. Retire rollback only when the owner deletes the old hosting.

## 7. Future VPS training run (do not train yet)

This section is a runbook, not a report: no training ran for this change.
On the VPS, after sections 1-4:

```bash
# 1. Confirm inputs: ml/cache/*.parquet for 2019-2025 plus the games table.
.venv_311/bin/python scripts/verify_artifact_prerequisites.py
# 2. Train (long job — run in background with a log, or hand the owner the command):
docker compose -f infra/docker-compose.yml run --rm backend python -m ml.train
# 3. Re-freeze the baseline (rewrites releases/*.json with repo-relative paths):
make freeze-baseline
# 4. Regenerate the causal eval summary:
.venv_311/bin/python scripts/summarize_causal_evals.py
# 5. Re-verify serving integrity:
make verify-evidence && make verify-readiness
```

Note (honest): `scripts/summarize_causal_evals.py` is currently hard-coded to
old input files (see docs/status/CURRENT.md); it may need updating before
step 4 reproduces the baseline-beating claim. Until a clean rebuild
reproduces them, evidence entries in `releases/artifacts/MANIFEST.json` stay
"provisional".
