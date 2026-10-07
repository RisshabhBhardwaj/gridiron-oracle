# 0002: Serve only pinned artifacts; fail closed

Type: Decision
Status: Accepted
Last verified against code: 2026-10-07
Date: recorded 2026-10-07 (behaviour found in code)

## Context
The API must never show a number that no real model produced.

## Options
1. Serve whatever is in the database and fall back silently.
2. Pin each served model cell by SHA-256 in `releases/current_baseline.json`; refuse to serve on a missing entry or digest mismatch.

## Choice
Option 2 is the default (`PRODUCT_MODE=artifact_backed`, `backend/app/core/config.py`; `ml/artifact_manifest.py`). `graceful_fallback` exists for local use and must label fallback output. A player with no approved forecast returns a structured 404 (`forecast_available: false`), not a degraded number.

## Why
A wrong-but-plausible projection is the worst failure for a product whose value is trust. Caveat: evidence entries in `releases/artifacts/MANIFEST.json` are all still "provisional".
