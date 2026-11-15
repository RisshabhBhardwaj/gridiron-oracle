# Gridiron Oracle: development guide

Type: Runbook
Status: Accepted
Last verified against code: 2026-10-07 (commands in AGENTS.md were run; `make bootstrap-local` and `make up` were not). The integration section was added 2026-11-14.

## Run it

```bash
make bootstrap-local            # Python env and git hooks
make up                         # docker stack: Postgres, MLflow, API, frontend
cd frontend && npm run dev      # or use .claude/launch.json
```

The Settings page is view-only by default, because the public site's proxy refuses `PUT /api/settings` (`frontend/api/proxy-policy.js`). To edit settings against your own API, put `VITE_SETTINGS_EDITABLE=true` in `frontend/.env.local` and restart `npm run dev`.

Tests and commands are listed in [AGENTS.md](../../AGENTS.md). On Apple Silicon set `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` before ML code.

## Integration tests (Postgres)

Tests marked `integration` need a real Postgres. CI runs five of the six files in the `integration` job ("Postgres integration tests") against a `postgres:18` service after `alembic upgrade head`, with `FAIL_ON_SKIP=1` so a skipped test fails the job instead of passing silently (`backend/tests/fail_on_skip.py`). `backend/tests/test_integration.py` is not in that job: it needs trained 2026 projections, materialized team predictions and the built C++ engine, so on an empty database it only skips. Run it locally with `make test-integration` (compose database).

To run them without Docker, with Homebrew Postgres (TCP only; a socket path in a deep temp dir can pass the 103-byte limit):

```bash
initdb -D /tmp/go-pg -U oracle --auth=trust -E UTF8
pg_ctl -D /tmp/go-pg -o "-p 5440 -c unix_socket_directories='' -c listen_addresses=localhost" -l /tmp/go-pg.log -w start
createdb -h localhost -p 5440 -U oracle oracle
export DATABASE_URL=postgresql://oracle@localhost:5440/oracle KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OTEL_TRACES_EXPORTER=none
alembic upgrade head
FAIL_ON_SKIP=1 python -m pytest backend/tests/test_weekly_refresh_integration.py -m integration -rs   # about 6.5 minutes
pg_ctl -D /tmp/go-pg -m fast stop
```

Leave `FAIL_ON_SKIP` unset to let a missing database skip instead of fail.

Releases: [releases/README.md](../../releases/README.md). Evidence policy: `releases/artifacts/README.md`.

## Layout

`scraper/` sources, `pipeline/` ETL and features, `ml/` models, `backend/` API and tests, `frontend/` website, `engine/` C++, `releases/` pinned artifacts, `reports/` evaluation outputs, `infra/` docker and monitoring, `scripts/` jobs.

