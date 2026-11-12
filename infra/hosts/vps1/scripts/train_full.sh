#!/bin/sh
# Full-window training launcher (Fri 22:00–Mon 06:00 UTC, via
# gridiron-train-full.timer). Host-side: runs the full-image `training`
# service, then pings the dead-man URL on success ONLY.
#
# RECIPE IS PROVISIONAL (set by the measurement handoff, not invented here):
# export TRAIN_FULL_CMD with the exact full-retrain invocation (full TFT/GNN
# torch-CPU retrains, e.g. around scripts/retrain_causal_oof.sh, plus SHAP),
# as confirmed by the one-fold profile projecting onto 5 capped vCores over
# the 56-hour window. Unset = fail loud; the missing ping is the alert.
set -eu

COMPOSE_DIR="${COMPOSE_DIR:-/srv/gridiron/infra/hosts/vps1}"

: "${TRAIN_FULL_CMD:?TRAIN_FULL_CMD is unset — set the full-retrain recipe (see README 'Training slice')}"

# shellcheck disable=SC2086
/usr/bin/docker compose --project-directory "$COMPOSE_DIR" --profile training \
  run --rm training $TRAIN_FULL_CMD

if [ -n "${HEALTHCHECKS_TRAIN_FULL_URL:-}" ]; then
  curl -fsS -m 10 --retry 3 "$HEALTHCHECKS_TRAIN_FULL_URL" -o /dev/null
fi
