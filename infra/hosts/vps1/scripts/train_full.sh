#!/bin/sh
# Full-window training launcher (Fri 22:00–Mon 06:00 UTC, via
# gridiron-train-full.timer). Host-side: runs the full-image `training`
# service, then pings the dead-man URL on success ONLY.
#
# Recipe (pinned 2026-11-14): TRAIN_FULL_CMD="bash scripts/train_window.sh full"
# in .env. It runs inside /app: the resumable 32-cell causal OOF rebuild with
# bucket attributions (scripts/retrain_causal_oof.sh) into the `train_out`
# volume. A window killed at RuntimeMaxSec resumes the same rebuild next time.
# Promotion is a manager step (docs/runbooks/vps4-causal-rerun.md). Unset =
# fail loud; the missing ping is the alert.
set -eu

COMPOSE_DIR="${COMPOSE_DIR:-/srv/gridiron/infra/hosts/vps1}"

: "${TRAIN_FULL_CMD:?TRAIN_FULL_CMD is unset — set the full-retrain recipe (see README 'Training slice')}"

# shellcheck disable=SC2086
/usr/bin/docker compose --project-directory "$COMPOSE_DIR" --profile training \
  run --rm training $TRAIN_FULL_CMD

if [ -n "${HEALTHCHECKS_TRAIN_FULL_URL:-}" ]; then
  curl -fsS -m 10 --retry 3 "$HEALTHCHECKS_TRAIN_FULL_URL" -o /dev/null
fi
