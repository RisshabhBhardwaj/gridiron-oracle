#!/bin/sh
# Small-window training launcher (01:00–07:00 UTC daily, via
# gridiron-train-small.timer). Host-side: runs the full-image `training`
# service, then pings the dead-man URL on success ONLY.
#
# RECIPE IS PROVISIONAL (set by the measurement handoff, not invented here):
# export TRAIN_SMALL_CMD with the exact small-retrain invocation, e.g. one of
#   scripts/train_volume_oof.sh | scripts/train_yardage_oof.sh |
#   scripts/train_passing_yards_qb.sh | scripts/train_fantasy_ppr_collapsed.sh
# plus SHAP/backfill steps, as confirmed by the one-fold profile. The command
# runs inside /app (repo root layout, same as the etl service). Unset =
# fail loud before touching anything; the missing ping is the alert.
set -eu

COMPOSE_DIR="${COMPOSE_DIR:-/srv/gridiron/infra/hosts/vps1}"

: "${TRAIN_SMALL_CMD:?TRAIN_SMALL_CMD is unset — set the small-retrain recipe (see README 'Training slice')}"

# shellcheck disable=SC2086
/usr/bin/docker compose --project-directory "$COMPOSE_DIR" --profile training \
  run --rm training $TRAIN_SMALL_CMD

if [ -n "${HEALTHCHECKS_TRAIN_SMALL_URL:-}" ]; then
  curl -fsS -m 10 --retry 3 "$HEALTHCHECKS_TRAIN_SMALL_URL" -o /dev/null
fi
