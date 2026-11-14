#!/usr/bin/env bash
# Print "<sha256>  <path>" for each NUL-separated path on stdin (xargs -0 style).
# Uses sha256sum when present (Linux images, the box's gridiron-etl:full) and
# falls back to `shasum -a 256` (macOS has only shasum). Both print the same
# format, so SHA256SUMS files are interchangeable between the two.
set -euo pipefail
if command -v sha256sum >/dev/null 2>&1; then
  xargs -0 sha256sum
elif command -v shasum >/dev/null 2>&1; then
  xargs -0 shasum -a 256
else
  echo "neither sha256sum nor shasum is installed" >&2
  exit 127
fi
