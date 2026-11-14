#!/bin/sh
# Full-window training launcher (Fri 23:30–Mon 06:00 UTC, via
# gridiron-train-full.timer). Host-side: runs the full-image `training`
# service, then pings the dead-man URL on success. If the recipe could not get
# the shared training lock within its wait (exit 76 from scripts/train_window.sh,
# 30 min by default; the Friday availability job holds it for a few seconds at
# 22:00), it pings `<URL>/fail` and exits non-zero: an alert, never a quiet skip.
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

DOCKER="${DOCKER:-/usr/bin/docker}"
LOCK_TIMEOUT_RC=76

rc=0
# shellcheck disable=SC2086
"$DOCKER" compose --project-directory "$COMPOSE_DIR" --profile training \
  run --rm training $TRAIN_FULL_CMD || rc=$?

if [ "$rc" -ne 0 ]; then
  if [ "$rc" -eq "$LOCK_TIMEOUT_RC" ]; then
    echo "train-full: could not get the training lock; alerting and failing" >&2
    if [ -n "${HEALTHCHECKS_TRAIN_FULL_URL:-}" ]; then
      curl -fsS -m 10 --retry 3 "${HEALTHCHECKS_TRAIN_FULL_URL%/}/fail" -o /dev/null || true
    fi
  fi
  exit "$rc"
fi

if [ -n "${HEALTHCHECKS_TRAIN_FULL_URL:-}" ]; then
  curl -fsS -m 10 --retry 3 "$HEALTHCHECKS_TRAIN_FULL_URL" -o /dev/null
fi
