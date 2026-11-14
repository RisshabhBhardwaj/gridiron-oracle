#!/bin/sh
# Small-window training launcher (01:00–07:00 UTC daily, via
# gridiron-train-small.timer). Host-side: runs the full-image `training`
# service, then pings the dead-man URL on success ONLY.
#
# Recipe (pinned 2026-11-14): TRAIN_SMALL_CMD="bash scripts/train_window.sh small"
# in .env. It runs inside /app: a 3-cell canary rebuild (5 trials) written to
# the `train_out` volume, never promoted. Exit 75 means another trainer held
# the lock (no ping). Unset = fail loud before touching anything; the missing
# ping is the alert.
set -eu

COMPOSE_DIR="${COMPOSE_DIR:-/srv/gridiron/infra/hosts/vps1}"

: "${TRAIN_SMALL_CMD:?TRAIN_SMALL_CMD is unset — set the small-retrain recipe (see README 'Training slice')}"

# shellcheck disable=SC2086
/usr/bin/docker compose --project-directory "$COMPOSE_DIR" --profile training \
  run --rm training $TRAIN_SMALL_CMD

if [ -n "${HEALTHCHECKS_TRAIN_SMALL_URL:-}" ]; then
  curl -fsS -m 10 --retry 3 "$HEALTHCHECKS_TRAIN_SMALL_URL" -o /dev/null
fi
