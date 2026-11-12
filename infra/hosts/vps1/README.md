# Gridiron on VPS-1 — one-runbook host layout

Dated 2026-11-12. Status: **built, not executed.** The server is not ordered
yet; nothing here has run against a live host. This directory is the complete
VPS-1 layout (fresh Ubuntu 24.04 OVH VPS-1, 2 vCores, 4 GB RAM, tunnel-only
ingress). It supersedes the big-box plan in `docs/MIGRATION.md` in *layout*
only — that file is untouched and still describes the data-move runbook.

## What runs by default vs profiles

| Command | What starts |
|---|---|
| `docker compose up -d` (no profile) | `api` + `db` + `cloudflared`. This is the steady state. |
| `--profile jobs run --rm etl` | Weekly refresh one-shot (host cron, Tue 13:00 UTC). |
| `--profile jobs run --rm backup` | Nightly R2 backup one-shot (host cron, daily 03:30 UTC). |
| `--profile training run --rm training ...` | Full-image light-refit shell. Interactive only. |
| `--profile observability up -d` | MLflow + OTel collector + Jaeger + Prometheus + Alertmanager. Diagnostic windows ONLY (does not fit in 4 GB alongside serving) — bring it down afterwards. |

`etl` / `backup` carry `profiles: [jobs]` + `restart: "no"`, so plain `up -d`
never starts them. The `training` service idles on `sleep infinity` under
`restart: "no"`; it exists so refits run in the exact full image.

## Memory budget (provisional)

Until the parallel serving-measurement lands, these are the planning caps
enforced via `mem_limit` (always-on target: 2–2.5 GB of 4 GB):

| Service | `mem_limit` | Notes |
|---|---|---|
| `api` (lean serving) | 1 GB | Worst-case assumption: heavy ML present. The lean image (no torch/CUDA) should land far below; keep the cap until measured. |
| `db` (postgres:18) | 700 MB | `shared_buffers` 256 MB + 20 conns + vacuum headroom. |
| `cloudflared` | 128 MB | Single tunnel, one hostname. |
| **Always-on total** | **~1.8 GB** | Leaves ~2 GB headroom on a 4 GB box. |
| `etl` one-shot | 1.5 GB | Weekly refresh window (Tue 13:00 UTC, no overlap with backup). |
| `backup` one-shot | 512 MB | Nightly 03:30 UTC (no overlap with ETL). |
| `training` (profile) | 2.5 GB | Light refits only; stop serving first if the box strains — full TFT/GNN retrains are UPGRADE-ONLY (see "What waits"). |
| `observability` (profile) | ~1.9 GB combined | Diagnostic windows only. Never steady-state on VPS-1. |

Pointer: the serving-import measurement doc (parallel worker) decides whether
the lean image can shrink further or must grow back to full.

## Bring-up order (owner, on the box)

```bash
# 1. Bootstrap (as root, with your key):
SSH_PUBKEY="$(cat ~/.ssh/id_ed25519.pub)" bash scripts/bootstrap.sh
#    Then open a NEW ssh deploy@<vps-ip> session and confirm it works
#    BEFORE closing this one.

# 2. Check out the repo at the pinned commit and enter this directory:
git clone <repo-url> gridiron && cd gridiron
git checkout <manager-pinned-commit>   # manager names the commit at handoff
cd infra/hosts/vps1
cp .env.example .env && chmod 600 .env   # fill in — names in .env.example

# 3. Tunnel config: replace api.<your-domain> in cloudflared.config.yml
#    (owner: `cloudflared tunnel create gridiron-vps1` + DNS route first).

# 4. Pin image digests: `docker pull <each image>` and fill the
#    TODO(deploy) digest comments in compose.yml / Dockerfile.backup.

# 5. Build + start serving:
docker compose build api
docker compose up -d
curl -f http://localhost:8000/health 2>/dev/null || \
  docker compose exec api curl -f http://localhost:8000/health

# 6. Schema (one-shot from the FULL image — the lean image skips alembic
#    binaries deliberately):
docker compose build etl   # backend/Dockerfile (see "Image choice" below)
docker compose --profile jobs run --rm etl \
  python3 /app/scripts/migrate.py upgrade

# 7. Readiness: PRODUCT_MODE=artifact_backed, /integrity green
#    (repo root: make verify-readiness), then restore data per
#    docs/MIGRATION.md §§3–4 (dump/restore, volumes, re-freeze baseline).

# 8. Scheduler (host cron, not a sidecar — see "Scheduler choice"):
sudo cp scheduler/gridiron.cron /etc/cron.d/gridiron
sudo chmod 644 /etc/cron.d/gridiron
#    Logs: /var/log/gridiron-etl.log, /var/log/gridiron-backup.log.
#    Create the two healthchecks.io checks and put their URLs in .env;
#    a missing ping after a scheduled run is the alert.

# 9. Observability first-enable (diagnostic window only):
docker compose exec db psql -U oracle -c 'CREATE DATABASE mlflow_release;'
docker compose --profile observability up -d
#    UIs are loopback-bound (15091/19090/19093/18686) — reach them over an
#    SSH tunnel, then `docker compose --profile observability down`.
```

Verify from an outside machine: only 22/tcp is reachable (no 80/443 —
tunnel-only by design; `nmap` should show nothing else).

## Backup / restore

- **Nightly (Tier 2):** `pg_dump -Fc --no-owner` → gzip → age (when
  `AGE_RECIPIENT` is set) → R2 at
  `<bucket>/gridiron/YYYY/MM/DD-HHMMSS.dump.gz[.age]` → size travels with the
  upload (rclone errors fail the job) → healthchecks ping **on success only**.
  Retention: R2 lifecycle rules on the bucket (14 daily / 8 weekly / 12 monthly
  — owner-side, still [VERIFY] per PLATFORM_SETUP §7.4; never expire the
  newest verified restore).
- **Restore drill (monthly):** restore into a scratch container, run the SQL
  smoke file (row counts per table, alembic head), compare with the counts
  recorded at backup time, then destroy the scratch container. A drill that
  has not passed in 35 days is an open incident.
- **Rollback:** the pre-cutover dump plus the old hosting stay until the
  owner deletes them (docs/MIGRATION.md §6).

## Decisions (deviations from PLATFORM_SETUP, applied deliberately)

- **Ingress is tunnel-only.** §§4–5 assume Caddy + 80/443. On this box ufw
  allows inbound 22/tcp and nothing else; no Caddy, no port bindings except
  loopback observability UIs. Public traffic arrives over the outbound
  `cloudflared` tunnel. If a public web port is ever needed, that is a new
  decision (Caddy + 80/443 + nmap gate), not a tweak.
- **Secrets: plain `.env`, mode 0600, accepted for v1.** §8 prescribes
  sops/age per box. Rationale: one app, one box, one deploy user — the
  encryption buys little until a second secret-holder exists. The layout is
  sops-ready: secrets enter only via `.env` (never images, compose files, or
  logs), so adopting `sops -d secrets/gridiron.sops.env > .env` later needs
  no compose changes. Revisit when a second box or operator appears.
- **No digest pins yet.** Every image carries a `TODO(deploy)` digest
  comment. Pins are filled at deploy time from `docker pull` output — pinning
  to digests observed on this Mac would be cargo-culting.
- **Scheduler: host cron, not a sidecar.** A cron sidecar would occupy RAM
  24/7 to sleep until Tuesday. Host cron (`/etc/cron.d/gridiron`) invokes
  `docker compose --profile jobs run --rm ...`; silence is caught by the
  dead-man pings either way.
- **ETL image: reuse `backend/Dockerfile` (no separate Dockerfile.etl).**
  The full image is the documented training/refit environment; forking a
  second full-image Dockerfile would drift. Compose pins it by path with a
  comment; only `Dockerfile.serving` and `Dockerfile.backup` are new.
- **Observability reuses the shared `infra/` configs** (`prometheus.yml`,
  `alerts.yml`, `alertmanager.yml`, `otel-collector-config.yaml`) unchanged.
  The `backend:8000` scrape target keeps resolving via the `api` network
  alias. Jaeger rides along in the profile because the shared collector
  config exports to it. First enable needs `CREATE DATABASE mlflow_release`
  (the shared dev stack has the same latent gap — noted, not fixed here).
- **`user:` omitted for `cloudflared`, `prometheus`, `db`.** The upstream
  images' supported run-as uids are unverified from here; a wrong uid breaks
  tunnel credentials, TSDB chown, or PGDATA ownership on first start. Verify
  uids on the scratch box, then harden. `no-new-privileges` is set
  everywhere regardless.
- **`api` is `read_only: true`.** The entrypoint only writes when
  `SETTINGS_FILE` points at a missing path (unset here → no writes), and
  `PYTHONDONTWRITEBYTECODE=1` stops `.pyc` writes. `etl`/`training` stay
  writable (staging scratch) — they are ephemeral one-shots.

## What waits for a VPS-3/4

- Full TFT/GNN retrains (memory + 8 GB-box tuning in PLATFORM_SETUP §7.2).
- WAL archiving / PITR (Tier-1 physical base + WAL; this box is Tier-2
  nightly dumps only).
- Caddy + public 80/443, if direct ingress is ever wanted.
- Loki / searchable logs (today: `docker logs` with rotation).
- sops/age per-box keys (see "Secrets" above).
- Second-copy backups (B2) and the off-provider whole-service drill (gate G3).
