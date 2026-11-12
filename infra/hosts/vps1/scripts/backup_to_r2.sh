#!/bin/sh
# Nightly Tier-2 backup: pg_dump (custom format) → gzip → age (optional) →
# rclone to Cloudflare R2 → dead-man ping on success ONLY.
#
# Runs as the `backup` compose service (profile `jobs`, restart "no"),
# invoked by host cron — never `up -d`. Fails LOUD: any error skips the
# healthchecks ping, and the missing ping is the alert (see §11).
#
# Env (names match .env.example; values never in git):
#   PGHOST PGUSER PGDATABASE PGPASSWORD   postgres over the internal net
#   R2_ENDPOINT R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET
#   AGE_RECIPIENT        optional; when empty the dump is gzip-only.
#                        Accepted v1 — set it (see README "Secrets").
#   HEALTHCHECKS_BACKUP_URL  optional; when empty no ping is sent.
#   BACKUP_PREFIX        optional object prefix, default "gridiron".
set -eu

PREFIX="${BACKUP_PREFIX:-gridiron}"
STAMP="$(date -u +%Y/%m/%d-%H%M%S)"
TMPDIR="${TMPDIR:-/tmp}"
DUMP="$TMPDIR/gridiron.dump"
GZ="$TMPDIR/gridiron.dump.gz"
FINAL="$GZ"
SUFFIX=".dump.gz"

echo "backup: pg_dump $PGUSER@$PGHOST/$PGDATABASE ..."
pg_dump -h "$PGHOST" -U "$PGUSER" -d "$PGDATABASE" --no-owner -Fc -f "$DUMP"
gzip -c "$DUMP" > "$GZ"
rm -f "$DUMP"

if [ -n "${AGE_RECIPIENT:-}" ]; then
  ENC="$TMPDIR/gridiron.dump.gz.age"
  echo "backup: age-encrypting ..."
  age -r "$AGE_RECIPIENT" -o "$ENC" "$GZ"
  rm -f "$GZ"
  FINAL="$ENC"
  SUFFIX=".dump.gz.age"
else
  echo "backup: WARNING: AGE_RECIPIENT unset — uploading unencrypted (v1 accepted, set it)" >&2
fi

# On-the-fly `:s3` backend (no rclone.conf on disk, no secret in a file):
# endpoint/keys travel in argv of this already-ephemeral container.
# (R2 keys are alphanumeric — no comma-escaping needed in the backend string.)
DEST=":s3,provider=Cloudflare,endpoint=${R2_ENDPOINT},access_key_id=${R2_ACCESS_KEY_ID},secret_access_key=${R2_SECRET_ACCESS_KEY},region=auto:${R2_BUCKET}/${PREFIX}/${STAMP}${SUFFIX}"
echo "backup: uploading ..."
rclone copyto "$FINAL" "$DEST"
rm -f "$FINAL"

# Retention is via R2 lifecycle rules on the bucket (14 daily / 8 weekly /
# 12 monthly — set owner-side, still [VERIFY] per PLATFORM_SETUP §7.4).
# Never delete the last N verified restores: lifecycle expiry must keep at
# least the newest object (configure the rule's prefix + noncurrent handling
# accordingly, or prune with an explicit keep-newest step, not shown here).

if [ -n "${HEALTHCHECKS_BACKUP_URL:-}" ]; then
  echo "backup: pinging dead-man check ..."
  curl -fsS -m 10 --retry 3 "$HEALTHCHECKS_BACKUP_URL" -o /dev/null
fi

echo "backup: done."
