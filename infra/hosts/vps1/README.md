# Gridiron on the shared VPS-4 — one-runbook host layout

Dated 2026-11-12. Status: **built, not executed.** No server access yet;
nothing here has run against a live host. This directory is the complete
Gridiron layout for ONE shared OVH VPS-4 (8 vCores, 24 GB RAM, 200 GB disk,
4 GB swap, Ubuntu 24.04, tunnel-only ingress), host to Gridiron AND the
separate ATS project. The `ovh-vps1` slug names the task, not the box. This
file is the executable companion to the rewritten `docs/MIGRATION.md`
(shared VPS-4 plan): MIGRATION describes the move end to end, this directory
is the exact layout it deploys.

## Host envelopes (owner decision 2026-11-12)

Measure, then adjust WITHIN envelopes — never grow one envelope into another.
If measurement busts an envelope, report it.

| Envelope | Size | What |
|---|---|---|
| Gridiron site | ~2.5 GB | `api` + `db` + `cloudflared`. First claim on CPU (uncapped; training is the capped one). |
| ATS (reserved) | ~9.5 GB | RESERVED for the separate ATS project. This layout leaves it free and designs no ATS services — coordination on shared resources only. |
| Gridiron training | ~10 GB, 5 CPUs | Low-priority slice (see "Training slice"). |
| Host slack | ~2 GB | OS, Docker, one-shot jobs. |

2.5 + 9.5 + 10 = 22 of 24 GB.

### Shared-host notes (ATS coordination only)

- Gridiron's Postgres is its own (`db` service) — never shared with ATS.
- One-shot jobs (ETL 1.5 GB, backup 512 MB) run in the training-idle window
  and draw from host slack: never alongside training, never alongside each
  other.
- **Canadian-IP note:** this box sits in Beauharnois/BHS, so job sites will
  see a Canadian datacenter IP. The ATS team must test reachability from it
  before relying on it. No ATS testing is done from this layout.

## Schedule (all timers UTC; EST = UTC−5 in November)

The box runs on UTC (bootstrap.sh forces it); units set no `Timezone=`, so
there is no DST ambiguity in the schedule. EST equivalents shift by an hour
under EDT — the UTC column is authoritative.

| Job | Timer (UTC) | UTC | EST |
|---|---|---|---|
| Small retrains / SHAP / backfills | daily 01:00, `RuntimeMaxSec=6h` kills at 07:00 | 01:00–07:00 | prev day 20:00 – 02:00 |
| Full retrain (TFT/GNN torch CPU, SHAP) | Fri 23:30, `RuntimeMaxSec=54h 30min` kills Mon 06:00 | Fri 23:30 → Mon 06:00 | Fri 18:30 → Mon 01:00 |
| Friday availability refresh (not enabled by default) | Fri 22:00, `TimeoutStartSec=20min` hard limit | Fri 22:00 → at most 22:20 | Fri 17:00 |
| Weekly ETL | Tue 13:00 | 13:00 | Tue 08:00 |
| Nightly R2 backup | daily 23:00 | 23:00 | 18:00 |

- ETL hour is deliberate: 13:00 UTC keeps today's standing slot
  (post-MNF data availability), preserves the day, and sits provably outside
  the 01:00–07:00 training window with wide clearance on both sides.
- Backup at 23:00 UTC avoids the small window and the ETL. It DOES overlap
  the full-retrain weekend window — accepted: pg_dump is minutes-long and
  the training slice is deprioritized, so site + backup keep precedence.
- `Persistent=false` on every timer: after downtime a missed run does NOT
  catch up on boot (a pileup could collide with a training window). A missed
  run surfaces via its missing dead-man ping instead.

### ETL exclusion (no training alongside the ETL), enforced three ways

1. Schedules provably disjoint (table above).
2. `Conflicts=gridiron-train-small.service gridiron-train-full.service` on
   the ETL unit: if training is ever running when the ETL starts (manual
   invocation, clock trouble), the ETL wins and training stops. The reverse
   is intentionally not declared.
3. Window caps (`RuntimeMaxSec`) self-terminate training so it cannot drift
   into Tuesday; overruns surface via missing pings.

## What runs by default vs profiles

| Command | What starts |
|---|---|
| `docker compose up -d` (no profile) | `api` + `db` + `cloudflared`. The steady state. |
| Timers (systemd, never `up -d`) | `etl` (Tue 13:00 UTC), `availability` (Fri 22:00 UTC, not enabled by default; see below), `backup` (daily 23:00 UTC), `training` via `scripts/train_small.sh` / `train_full.sh`. One-shots carry `profiles:` + `restart: "no"`, so plain `up -d` never starts them. |
| `--profile training run --rm training ...` | Interactive refits in the exact full image. |
| `--profile observability up -d prometheus alertmanager otel-collector` | The always-fittable trio (see "Observability fit"). |
| `--profile observability up -d mlflow` (or `jaeger`) | One-at-a-time only, trio down (see "Observability fit"). |

## Memory budget (measured 2026-11-12 — see docs/perf/VPS1_MEASUREMENTS.md)

| Service | `mem_limit` | Notes |
|---|---|---|
| `api` (lean serving) | 1 GB | Serving imports zero heavy libs in artifact-backed mode (measured). Observed 84–142 MB on an empty DB (lower bound — re-measure against the snapshot in the Docker rehearsal); keep the 1 GB cap. |
| `db` (postgres:18) | 700 MB | `shared_buffers` 256 MB + 20 conns + vacuum headroom. |
| `cloudflared` | 128 MB | Single tunnel, one hostname. |
| **Steady site total** | **~1.83 GB** | **≤ 2.5 GB envelope. ✓** |
| `etl` one-shot | 1.5 GB | Training-idle window (Tue 13:00 UTC), draws from host slack. KNOWN TIGHT SPOT: combined site + ETL ≈ 3.3 GB exceeds the site line — fits only because the training slice is idle then. First thing to tighten against the ETL peak-RSS measurement. |
| `availability` one-shot | 512 MB (a guess, not measured) | Fri 22:00 UTC. Fetches one small source and updates status rows per unstarted game; loads no models. Shares `/train_out/.train.lock` with the trainers and waits for them. Record its peak RSS on the first run and set `mem_limit` from it. |
| `backup` one-shot | 512 MB | Daily 23:00 UTC. Site + backup ≈ 2.33 GB ≤ 2.5 GB ✓ (fits even inside the site line). |
| `training` | 10 GB / 5 CPUs / shares 128 | Slice + container caps (see "Training slice"). |
| Observability trio | 448 MB | prometheus 256m + alertmanager 64m + otel 128m. Site + trio ≈ 2.28 GB ≤ 2.5 GB ✓. |
| mlflow / jaeger | 512m each, OFF | Do not fit alongside the site — see "Observability fit". |

Measured verdicts: one lgbm fold peaks ~0.35 GB — FITS the 10 GB training cap
(~3.5%); full tree-OOF retrain projects to ~8–10 h on 5 capped vCores (fits
the weekend window). ETL real RSS unmeasured (snapshot was down) — `etl`
stays capped at 1.5 GB until the Docker rehearsal measures it.
`TRAIN_SMALL_CMD`/`TRAIN_FULL_CMD` are pinned (see "Training slice"); full
TFT/GNN retrains wait for their own fold profile.

## Training slice

- Host unit `systemd/gridiron-training.slice`: `MemoryMax=10G`,
  `CPUQuota=500%` (5 of 8 vCores), `CPUWeight=20` / `IOWeight=20` (default
  100 — site keeps first claim under contention), plus `Nice=10` on the
  training services.
- **Two-layer enforcement (read this before "simplifying"):** a systemd
  `Slice=` on a timer unit caps the `docker compose run` launcher process,
  NOT the training container — Docker nests container cgroups under its own
  hierarchy. The real container enforcement is in compose.yml (`training`:
  `mem_limit: 10g`, `cpus: 5.0`, `cpu_shares: 128`). The slice is the
  host-level envelope and accounting; both layers carry the same numbers.
- **Recipes (pinned 2026-11-14, not yet run on the box):**
  `scripts/train_small.sh` and `scripts/train_full.sh` take `TRAIN_SMALL_CMD`
  / `TRAIN_FULL_CMD` from `.env` (values in `.env.example`) and fail loud when
  unset. Both call `scripts/train_window.sh` inside the container:
  - `full`: the 32-cell causal OOF rebuild with per-row bucket attributions
    (`scripts/retrain_causal_oof.sh`, 20 trials). Resumable: a window killed
    at `RuntimeMaxSec` continues the same rebuild next time (pointer file
    `full_stamp` on the volume).
  - `small`: nightly canary, 3 cells, 5 trials, last 3 kept. Never promoted.
  - Output lives on the `train_out` volume (`/train_out`, owned by
    `appuser`): `oof/rebuild_<STAMP>/`, `logs/`, `last_full_stamp`. Without
    the volume `run --rm` would discard a weekend of training.
  - One trainer at a time (`flock` on `/train_out/.train.lock`, shared with the Friday
    availability job): the small timer fires nightly, including Sat-Mon while the
    full job runs, and waits 2 minutes then skips (exit 75, no ping). The full job
    waits up to 30 minutes (`TRAIN_FULL_LOCK_WAIT_SECONDS`); if it still has no lock it
    logs why, exits 76 and `train_full.sh` pings `HEALTHCHECKS_TRAIN_FULL_URL/fail`.
    Availability holds the lock for seconds at Fri 22:00 and is killed at 20 minutes
    (lock wait 900 s + run 240 s inside `TimeoutStartSec=20min`), so a retrain that
    still holds the lock makes availability fail with no success ping, never the reverse.
  - Promotion is manual: `docs/runbooks/vps4-causal-rerun.md`.
  A full rebuild that cannot fit Fri-night-through-weekend on 5 capped vCores
  is an envelope question for the owner, not a reason to overrun into the ETL.
  The runbook has a wall-time estimate, labeled as one.
- **Release provenance:** `/srv/gridiron` has no `.git`, so set
  `GIT_COMMIT_SHA` in `.env` (`api` and `training` read it through
  `env_file`) to the commit the release was frozen from; it stays pinned
  there until the next re-freeze. The forward producer's code provenance is a
  separate `.env` variable, `CODE_GIT_SHA`: set it to the synced Mac HEAD
  (`git rev-parse HEAD`, from a clean tree) after every code sync and before
  any image build. It is a build arg for `etl` and `training`, baked into
  `/app/CODE_GIT_SHA`; the producer reads that file, not the runtime env, so a
  later `.env` edit cannot change it. A value that is not 40-hex makes the
  producer refuse to write a forward run.
- Every training run pings its dead-man URL on success only
  (`HEALTHCHECKS_TRAIN_SMALL_URL` / `HEALTHCHECKS_TRAIN_FULL_URL`).

## Observability fit (inside the 2.5 GB site envelope)

- The trio (prometheus 256m, alertmanager 64m, otel-collector 128m) may stay
  on with the site: ≈ 2.28 GB combined.
- mlflow and jaeger stay OFF (512m each — neither fits alongside the site).
  One-at-a-time rule: enable at most one with the trio down, e.g.
  `docker compose --profile observability up -d mlflow`, then bring it back
  down. Never steady-state.
- The shared `infra/` configs are reused unchanged (`prometheus.yml`,
  `alerts.yml`, `alertmanager.yml`, `otel-collector-config.yaml`); the
  `backend:8000` scrape target resolves via the `api` network alias. Jaeger
  rides the profile because the shared collector config exports to it.
- First enable needs `CREATE DATABASE mlflow_release` (the shared dev stack
  has the same latent gap — noted, not fixed here).

## Bring-up order (owner, on the box)

```bash
# 1. Bootstrap (as root). Key material comes from the owner's
#    ~/.ssh/ovh_boxes.pub (public-key PATH only — never copied anywhere):
SSH_PUBKEY="$(cat ~/.ssh/ovh_boxes.pub)" bash scripts/bootstrap.sh
#    Then open a NEW ssh session (the VPS-4 host — address in the internal
#    brief, never in this repo) and confirm it works BEFORE closing this one.
#    On the FIRST connect ever, verify the ED25519 host key against the
#    fingerprint in the internal brief; never accept a changed key silently.

# 2. Check out the repo at the pinned commit, standard path /srv/gridiron:
sudo mkdir -p /srv && sudo chown deploy:deploy /srv
git clone <repo-url> /srv/gridiron && cd /srv/gridiron
git checkout <manager-pinned-commit>   # manager names the commit at handoff
cd infra/hosts/vps1
cp .env.example .env && chmod 600 .env   # fill in — names in .env.example

# 3. Tunnel config: replace api.<your-domain> in cloudflared.config.yml
#    (owner: `cloudflared tunnel create gridiron-vps1` + DNS route first).
#    cloudflared needs nothing region-specific — it dials the nearby PoP.

# 4. Image digests are pinned in compose.yml. To bump one: `docker pull <tag>`
#    and copy its digest onto that service's `image:` line.

# 5. Build + start serving:
docker compose build api
docker compose up -d
docker compose exec api curl -f http://localhost:8000/health

# 6. Schema (one-shot from the FULL image):
docker compose build etl   # backend/Dockerfile (see "Decisions")
docker compose --profile jobs run --rm etl \
  python3 /app/scripts/migrate.py upgrade

# 7. Readiness: PRODUCT_MODE=artifact_backed, /integrity green
#    (repo root: make verify-readiness), then restore data per
#    docs/MIGRATION.md §§3–4 (dump/restore, volumes, re-freeze baseline).

# 8. Scheduler (systemd timers, NOT cron — see "Decisions"):
sudo cp systemd/* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gridiron-etl.timer gridiron-backup.timer \
  gridiron-train-small.timer gridiron-train-full.timer
#    The Friday availability refresh is NOT in this list. Enable it by hand
#    (`sudo systemctl enable --now gridiron-availability.timer`) only after the
#    forward producer has run once and served a week. It has no Conflicts= and
#    never stops training: it shares the training lock and has a hard 20 minute
#    limit; train-full starts at 23:30 UTC and waits up to 30 minutes for the lock.
#    Set TRAIN_SMALL_CMD / TRAIN_FULL_CMD (see "Training slice") and the
#    four healthchecks.io URLs in .env; a missing ping after a scheduled
#    run is the alert. Logs: journalctl -u gridiron-<name>.

# 9. Observability trio (optional, fits the envelope):
docker compose exec db psql -U oracle -c 'CREATE DATABASE mlflow_release;'
docker compose --profile observability up -d prometheus alertmanager otel-collector
#    UIs are loopback-bound — reach them over an SSH tunnel.
```

Verify from an outside machine: only 22/tcp is reachable (no 80/443 —
tunnel-only by design).

## Backup / restore — R2 only, no OVH dependency

- **Nightly (Tier 2):** `pg_dump -Fc --no-owner` → gzip → age (when
  `AGE_RECIPIENT` is set) → R2 at
  `<bucket>/gridiron/YYYY/MM/DD-HHMMSS.dump.gz[.age]` → rclone errors fail
  the job → healthchecks ping **on success only**. Retention via R2
  lifecycle rules (14 daily / 8 weekly / 12 monthly — owner-side, still
  [VERIFY] per PLATFORM_SETUP §7.4; never expire the newest verified
  restore).
- **Encryption key (since 2026-11-14).** Backups are age-encrypted to the
  recipient in the box `.env` (`AGE_RECIPIENT`, a public key). The matching
  private key lives **only** in the owner's macOS Keychain as a generic
  password, service `gridiron-backup-age`, account `gridiron`. It is never on
  the box, never in git, never printed. Decrypt on the Mac:
  `security find-generic-password -s gridiron-backup-age -w | age -d -i - FILE.dump.gz.age | gunzip > FILE.dump`,
  then `pg_restore -l FILE.dump` to check it. Losing that Keychain item makes
  every encrypted backup unreadable: keep it in the owner's Keychain backup
  (iCloud Keychain or an exported copy in the password manager). Backups
  before 2026-11-14 16:01 UTC are gzip only.
- **The stack restores from R2 alone.** No OVH backup option is used or
  needed — deliberate, since order-time options on this region are final.
  A restore needs only: a fresh Ubuntu 24.04 box (any region), this repo,
  the `.env` values, read access to the R2 bucket, and the age key from the
  owner's Keychain (decrypt on the Mac, then copy the plain dump over).
- **Restore drill (monthly):** restore into a scratch container, run the SQL
  smoke file (row counts per table, alembic head), compare with the counts
  recorded at backup time, destroy the scratch container. A drill that has
  not passed in 35 days is an open incident.
- **Rollback:** the pre-cutover dump plus the old hosting stay until the
  owner deletes them (docs/MIGRATION.md §6).

## Portability (BHS today, US-East later)

No BHS-specific config exists anywhere in this stack: no region-coupled
endpoints, mirrors, locales, or firewall rules. The region appears only in
this file and in bootstrap.sh comments. A later move to US-East is:
re-bootstrap a box there (~1h — bootstrap.sh is idempotent), restore the
latest R2 dump, re-freeze the baseline, point the tunnel hostname. No
rebuild of these files.

## Decisions (deviations from PLATFORM_SETUP, applied deliberately)

- **Ingress is tunnel-only.** §§4–5 assume Caddy + 80/443. Here ufw allows
  inbound 22/tcp and nothing else; no Caddy, no public port bindings except
  loopback observability UIs. If a public web port is ever needed, that is a
  new decision, not a tweak.
- **Secrets: plain `.env`, mode 0600, accepted for v1.** §8 prescribes
  sops/age per box. Rationale: one shared box, one deploy user — the
  encryption buys little until a second operator appears. The layout is
  sops-ready: secrets enter only via `.env` (never images, compose files, or
  logs), so adopting `sops -d secrets/gridiron.sops.env > .env` later needs
  no compose changes.
- **Third-party images are pinned by digest** (copied from Box B, 2026-11-14).
  Each `image:` line has a comment naming the tag; bump by pulling the tag and
  copying its digest. Locally built images (`gridiron-*`) are not pinned.
- **Scheduler: systemd timers, not cron.** A cron sidecar would sit in RAM
  to sleep until Tuesday, and plain host cron cannot join the training slice
  or express `Conflicts=`/`RuntimeMaxSec`. Timers give slice assignment,
  ETL-exclusion enforcement, window self-termination, and journald logs.
- **ETL image: reuse `backend/Dockerfile` (no separate Dockerfile.etl).**
  The full image is the documented training/refit environment; forking a
  second full-image Dockerfile would drift. Only `Dockerfile.serving` and
  `Dockerfile.backup` are new.
- **Observability reuses the shared `infra/` configs** unchanged (see
  "Observability fit").
- **`user:` omitted for `cloudflared`, `prometheus`, `db`.** Upstream
  run-as uids unverified from here; a wrong uid breaks tunnel credentials,
  TSDB chown, or PGDATA ownership on first start. Verify uids on the
  scratch box, then harden. `no-new-privileges` is set everywhere.
- **`api` is `read_only: true`.** The entrypoint only writes when
  `SETTINGS_FILE` points at a missing path (unset here → no writes), and
  `PYTHONDONTWRITEBYTECODE=1` stops `.pyc` writes. `etl`/`training` stay
  writable (staging scratch) — they are ephemeral one-shots.
- **Shared box, one Postgres per project.** Gridiron's `db` is never shared
  with ATS; cross-project coordination is envelopes + schedule only.

## What waits (not on this box as drawn)

- WAL archiving / PITR (Tier-2 nightly dumps stand; the R2-only path covers
  the region constraint).
- Caddy + public 80/443, if direct ingress is ever wanted.
- Loki / searchable logs (today: `docker logs` + journald with rotation).
- sops/age per-box keys (see "Decisions").
- Second-copy backups (B2) and the off-provider whole-service drill (gate G3).
