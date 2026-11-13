# Migration to the shared OVH VPS-4

Dated 2026-11-12. Status: **LIVE (2026-11-12).** Gridiron serves from the
box in Canada-East (Beauharnois/BHS), monthly, no commitment: backend
through `https://gridiron-api.risshabh.com` (Cloudflare Tunnel), frontend
on Pages from `main`. Seed = restored `gor_prod_20261010` snapshot
(2232 games); owner secrets set by the owner. Open items: R2 backup
credentials + healthchecks URLs still placeholders (23:00 UTC backup
timer fails loud until then), `TRAIN_*_CMD` unset, baseline re-freeze
awaits the manager/owner (health reports degraded,
`artifact_mode_ready:false`, predictions serving). First connection
verified the ED25519 host key against the fingerprint in the internal brief
(`_internal-docs/gridiron/briefs/ovh-vps1.md`, host record — never in this
repo; the repo is public).

This file describes the move; the executable layout is
`infra/hosts/vps1/` (README + compose + systemd + bootstrap). The exact
first-connection sequence is `docs/runbooks/vps4-shared-host.md`.

Target: **one shared VPS-4 (8 vCores, 24 GB RAM, 200 GB disk, Ubuntu 24.04)**
for Gridiron AND the separate ATS project. The frontend stays on Cloudflare
Pages at `gridiron.risshabh.com` (decision 0009 on `main`); it does not move
to the VPS. After cutover, `BACKEND_URL` in `frontend/wrangler.toml` points
at the tunnel hostname and the frontend redeploys from `main` (with the
calm-state port — see §6).

## 1. Host envelopes (owner decision 2026-11-12)

Measure, then adjust WITHIN envelopes — never grow one envelope into another.

| Envelope | Size | What |
|---|---|---|
| Gridiron site | ~2.5 GB | `api` + `db` + `cloudflared`. First claim on CPU. |
| ATS (reserved) | ~9.5 GB | RESERVED for ATS. Left free; no ATS services designed here. |
| Gridiron training | ~10 GB, 5 CPUs | Low-priority systemd slice. |
| Host slack | ~2 GB | OS, Docker, one-shot jobs. |

2.5 + 9.5 + 10 = 22 of 24 GB.

## 2. Schedule (all UTC; EST = UTC−5 in November; box runs on UTC)

| Job | Window | Notes |
|---|---|---|
| Small retrains / SHAP / backfills | daily 01:00–07:00 UTC | `RuntimeMaxSec=6h` kills at 07:00 |
| Full retrain | Fri 22:00 UTC → Mon 06:00 UTC | `RuntimeMaxSec=56h`; TFT/GNN torch CPU |
| Weekly ETL | Tue 13:00 UTC (08:00 EST) | Keeps today's slot; provably outside training windows |
| Nightly R2 backup | daily 23:00 UTC | Overlaps the full-retrain weekend by design (pg_dump is minutes-long) |

ETL exclusion is enforced three ways: disjoint schedules, `Conflicts=` on
the ETL unit (ETL wins, training stops), and `RuntimeMaxSec` window caps.
`Persistent=false` everywhere: missed runs surface via missing dead-man
pings, never catch-up pileups.

## 3. Measurements (2026-11-12, Apple M2 Pro; full table in `docs/perf/VPS1_MEASUREMENTS.md`)

- One lgbm training fold peaks **~0.35 GB → FITS the 10 GB cap (~3.5%)**. No stop.
- Full tree-OOF retrain (15 cells, lgbm + catboost) projects to **~8–10 h on
  5 capped vCores** — fits the Fri-night–weekend window with large margin.
- Serving imports **zero heavy libraries** in artifact-backed mode
  (`torch`/`catboost`/`lightgbm`/`xgboost`/`mlflow`/`shap` never load) — the
  lean serving image drops them and keeps native boosters + `onnxruntime`.
- API budget **0.25 GB** (84–142 MB observed on an empty DB — a lower bound;
  re-measure against the snapshot in the on-box rehearsal).
- **Unmeasured:** real ETL peak RSS (local snapshot was down; `etl` stays
  capped at 1.5 GB until the rehearsal measures it) and TFT/GNN full-retrain
  time (owner's 30–40 h Mac figure is unconfirmed — profile a fold before
  scheduling; `TRAIN_FULL_CMD` stays unset until then).
- `pg_dump -Fc` (~28 MB per prior handoff): tens of MB — measure on first live run.

## 4. Bring-up (runbook: `docs/runbooks/vps4-shared-host.md`)

1. First SSH as `ubuntu` (key `~/.ssh/ovh_boxes`), verify host fingerprint.
2. `scripts/bootstrap.sh` as root with the owner's **public** key: `deploy`
   user, SSH hardening (`AllowUsers deploy`, password/root off), ufw
   22/tcp only, unattended-upgrades (reboot off), 4 GB swap (swappiness 10),
   Docker + log rotation. Confirm a NEW `deploy` session BEFORE closing the old one.
3. Check out this repo at the pinned commit under `/srv/gridiron`,
   `cp .env.example .env` (mode 0600), fill secrets (§5).
4. `docker compose build && up -d`, smoke test, ETL dry run against the
   restored snapshot (measure real peak RSS), pin image digests.
5. Install `systemd/gridiron-training.slice` + timers; leave
   `TRAIN_SMALL_CMD`/`TRAIN_FULL_CMD` unset (fail-loud) until recipes are pinned.
6. Tunnel + frontend cutover (§6), then `/srv/HOST.md` + notify ATS (§8).

## 5. Env and secrets (names only, never values)

`.env` is never committed (pre-push hook + CI `push-gate`). Secrets the
**owner sets himself** on the box / in Pages — exact commands in the
runbook/handoff, values never pass through an agent:

- `API_KEY` (generate: `python3 -c "import secrets; print(secrets.token_hex(32))"`)
- `ADMIN_API_KEY` (separate value from `API_KEY`)
- `ODDS_API_KEY`, `OPENWEATHER_API_KEY`
- Tunnel: `TUNNEL_TOKEN` (from `cloudflared tunnel create`, owner-side DNS route)
- R2: `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_BACKUPS`
- Dead-man pings: `HEALTHCHECKS_*_URL` (ETL, backup, train-small, train-full)
- Training recipes: `TRAIN_SMALL_CMD`, `TRAIN_FULL_CMD` (unset until pinned)

Non-secrets: `POSTGRES_*`, `DATABASE_URL`, `MODEL_VERSION`,
`PRODUCT_MODE=artifact_backed`, `CORS_ORIGINS`, `MODEL_VERSION`,
`BASELINE_MANIFEST_PATH`, plus build-time `VITE_SETTINGS_EDITABLE` (public
build must NOT set it). Full list: `infra/hosts/vps1/.env.example`.

## 6. Tunnel + frontend cutover — OWNER-GATED STEPS MARKED

1. Tunnel hostname via the `cloudflare-personal` Keychain token
   (manager-side; Wrangler OAuth has no DNS scope).
2. Point `BACKEND_URL` in `frontend/wrangler.toml` at the tunnel hostname
   (no trailing slash).
3. **Do not deploy a `main` frontend without the calm-state port**
   (cloudflare-live `8c3d1e0`: `ApiError.code`, `ErrorBanner` calm line,
   `useSeasonProjections`). Cutover condition, tracked by the coordinator.
4. OWNER sets Pages secrets (`BACKEND_API_KEY` = new VPS `API_KEY`,
   optional `ODDS_API_KEY`), then deploys the frontend from `main`.
5. Re-run the verification list: Pages render, calm state only when backend
   is down, settings PUT → 403, `/api/docs` → 404, `/api` reads with real data.

## 7. Data seeding (the ONLY production-data write allowed)

Restore our own snapshot (`gor_prod_20261010`) into the box Postgres, then
verify row counts (games 2019–2026, `pbp_plays`, `game_logs`,
`feature_matrix`, `depth_charts`, `projections`, `team_game_predictions` —
last known-good counts in `docs/status/CURRENT.md`), run `make migrate`
(`alembic upgrade head`), and copy `ml/cache/*.parquet` (2019–2025) for
training inputs. The weekly ETL resumes from there. Never touch any other
production store.

## 8. Shared-host record

When the host is ready, write `/srv/HOST.md` **on the box**: envelopes, who
owns what (Gridiron manager owns the host layer; Gridiron site
`/srv/gridiron`; ATS `/srv/ats` 9.5 GB with its own tunnel — ATS deploys
itself, envelopes only), the training slice, the UTC schedule windows, and
the ATS Canadian-IP note (job sites see a BHS datacenter IP — ATS tests
reachability itself). Then notify the ATS worker. Do not deploy ATS services.

## 9. Rollback

Keep the pre-cutover dump until the owner deletes the old hosting. If the
VPS misbehaves: point `BACKEND_URL` back (Pages redeploy), restart the old
path if it still exists. The box Postgres is disposable until cutover is
accepted. Retire rollback only on the owner's word.

## 10. Portability (BHS today, US-East later)

No BHS-specific config exists in the stack (no regional mirrors, endpoints,
locales, or firewall rules); the region appears only in docs and bootstrap
notes. A later US-East move is: re-bootstrap (~1 h, script is idempotent),
restore the latest R2 dump, re-freeze the baseline, point the tunnel
hostname. Order-time options on this region are final, so backups are R2
only — no OVH backup dependency anywhere.

## 11. What stays off / open issues

- `TRAIN_FULL_CMD`/`TRAIN_SMALL_CMD` unset until recipes are pinned on the box.
- TFT/GNN full retrains wait for their own fold profile.
- `mlflow`/`jaeger` stay OFF (don't fit the 2.5 GB site line); trio only.
- `mlflow_release` database needs one `CREATE DATABASE` before first trio enable.
- R2 lifecycle retention rules still `[VERIFY]` (14/8/12, never expire newest).
- Second-copy backups (B2) and the off-provider whole-service drill (gate G3).
- `releases/*.json` absolute `/Users/...` paths: fix only by re-freezing on
  the box (`make freeze-baseline`), never by hand.
- Evidence entries in `releases/artifacts/MANIFEST.json` stay "provisional"
  until a clean rebuild reproduces them.
