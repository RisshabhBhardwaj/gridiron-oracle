#!/bin/sh
# Container entrypoint.
#
# SETTINGS_FILE points at the JSON the /settings endpoint reads and writes. On
# Railway it lives on a mounted volume so PUT /settings survives redeploys; a
# freshly provisioned volume is empty, and backend.app.api.settings._load_config
# silently falls back to _DEFAULT_CONFIG when the file is missing — which would
# quietly discard the tuned weights baked into the image. Seed it once.
#
# No-ops for docker-compose, where SETTINGS_FILE is the relative
# "user_config.json" that already exists in the working directory.
set -e

if [ -n "$SETTINGS_FILE" ] && [ ! -e "$SETTINGS_FILE" ]; then
    settings_dir=$(dirname "$SETTINGS_FILE")
    if [ -d "$settings_dir" ] && [ -w "$settings_dir" ]; then
        # Never fatal: a failed seed costs persisted settings, not the service.
        # Without the guard, `set -e` would turn a full disk into a crash loop.
        if cp /app/user_config.json "$SETTINGS_FILE"; then
            echo "entrypoint: seeded $SETTINGS_FILE from image default"
        else
            echo "entrypoint: could not seed $SETTINGS_FILE — starting anyway" >&2
        fi
    else
        echo "entrypoint: $settings_dir is not a writable directory —" \
             "settings changes will not persist" >&2
    fi
fi

exec python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
