# Runbook: shared VPS-4 first connection + bring-up

Dated 2026-11-12. For the bare Ubuntu 24.04 box with only the owner's key on
it. Steps marked **OWNER** are his; the rest run in this order. The box
address and host-key fingerprint live ONLY in the internal brief
(`_internal-docs/gridiron/briefs/ovh-vps1.md`, host record) — this repo is
public, so they never appear here.

## 0. First connect (manager, from the Mac)

```bash
ssh -i ~/.ssh/ovh_boxes ubuntu@<the VPS-4 host — address in the internal brief>
```

On the FIRST connect, compare the offered ED25519 key against the
`SHA256:` fingerprint in the brief BEFORE typing yes. A mismatch means stop
and report — never accept a changed key silently. (The owner has already
logged in, so any mismatch is real.)

## 1. Bootstrap (as root on the box)

The checkout is not on the box yet — copy `scripts/bootstrap.sh` over
(`infra/hosts/vps1/scripts/bootstrap.sh` in this repo) and run it with the
owner's PUBLIC key only:

```bash
sudo apt-get update && sudo apt-get install -y git
SSH_PUBKEY="$(cat ~/.ssh/ovh_boxes.pub)" sudo -E bash bootstrap.sh
```

What it does: `deploy` user + sudo, SSH hardening (`AllowUsers deploy`,
password/root off, lockout-guarded), ufw 22/tcp ONLY, fail2ban,
unattended-upgrades (reboot off), UTC timezone, 4 GB swap (swappiness 10),
Docker Engine + compose plugin, daemon log rotation + live-restore.

Then: **open a NEW `ssh -i ~/.ssh/ovh_boxes deploy@<host>` session and
confirm login BEFORE closing the `ubuntu` session.** After hardening, login
is `deploy` only — `ubuntu`/root/password are off.

Acceptance: from an outside machine only 22/tcp is reachable.

## 2. Check out + configure (as `deploy`)

```bash
sudo mkdir -p /srv && sudo chown deploy:deploy /srv
git clone <repo-url> /srv/gridiron && cd /srv/gridiron
git checkout <gated commit — manager names it in the handoff>
cd infra/hosts/vps1
cp .env.example .env && chmod 600 .env
```

Fill `.env` per `.env.example` (secret VALUES: owner only — see §4).

## 3. Rehearse on the box (as `deploy`)

```bash
docker compose build api etl
docker compose up -d                      # steady state: api + db + cloudflared
curl -f http://localhost:8000/health
docker compose --profile jobs run --rm etl \
  python3 /app/scripts/migrate.py upgrade  # schema to head
# restore the own snapshot (only production-data write allowed), verify counts
docker compose --profile jobs run --rm etl \
  python3 /app/scripts/weekly_refresh.py --dry-run   # measure real peak RSS here
docker compose down && docker compose up -d
docker pull <each image>  # fill TODO(deploy) digest comments with observed digests
```

## 4. Scheduler (as `deploy` + root for install)

```bash
sudo cp systemd/* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gridiron-etl.timer gridiron-backup.timer \
  gridiron-train-small.timer gridiron-train-full.timer
```

Leave `TRAIN_SMALL_CMD`/`TRAIN_FULL_CMD` UNSET (fail-loud) until the recipes
are pinned on the box. Dead-man pings: a missing ping after a scheduled run
is the alert. Logs: `journalctl -u gridiron-<name>`.

## 5. Secrets — OWNER runs these himself (exact commands)

On the box (values are his, never shown to an agent):

```bash
cd /srv/gridiron/infra/hosts/vps1
python3 -c "import secrets; print(secrets.token_hex(32))"  # twice: API_KEY, then ADMIN_API_KEY
# fill .env: API_KEY, ADMIN_API_KEY, ODDS_API_KEY, OPENWEATHER_API_KEY,
# R2_* (least-privilege token, backups bucket), TUNNEL_TOKEN, HEALTHCHECKS_*_URL
```

Pages (from any machine with wrangler):

```bash
CLOUDFLARE_ACCOUNT_ID=72b23948192d06b3cc23f218e381fea9 wrangler pages secret put BACKEND_API_KEY --project-name gridiron-oracle
CLOUDFLARE_ACCOUNT_ID=72b23948192d06b3cc23f218e381fea9 wrangler pages secret put ODDS_API_KEY --project-name gridiron-oracle  # optional
```

Tell the manager the moment the backend waits only on secrets.

## 6. Tunnel + cutover (manager)

Create/route the tunnel hostname (manager-side, `cloudflare-personal`
token — never printed), set `BACKEND_URL` in `frontend/wrangler.toml`
(no trailing slash), deploy the frontend from `main` ONLY with the
calm-state port in, then verify: Pages render, calm state only when backend
down, settings PUT → 403, `/api/docs` → 404, `/api` reads with real data.

## 7. Host ready (manager)

Write `/srv/HOST.md` on the box (envelopes, owners, schedule, ATS note),
notify the ATS worker (Herdr `wR:p1C`), report checkpoints to the
coordinator: bootstrap done / rehearsal done / waiting-on-secrets /
tunnel live / frontend deployed.
