# Gridiron Oracle: development guide

Type: Runbook
Status: Accepted
Last verified against code: 2026-10-07 (moved from README; verify commands before relying on them)

## Run it

```bash
make bootstrap-local            # Python env and git hooks
make up                         # docker stack: Postgres, MLflow, API, frontend
cd frontend && npm run dev      # or use .claude/launch.json
```

Tests and commands are listed in [AGENTS.md](AGENTS.md). On Apple Silicon set `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` before ML code.

Releases: `releases/README.md`. Evidence policy: `releases/artifacts/README.md`.

## Layout

`scraper/` sources, `pipeline/` ETL and features, `ml/` models, `backend/` API and tests, `frontend/` website, `engine/` C++, `releases/` pinned artifacts, `reports/` evaluation outputs, `infra/` docker and monitoring, `scripts/` jobs.

