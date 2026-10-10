# 0007: Public Settings are read-only

Type: Decision
Status: Accepted (owner approved, 2026-10-10)
Last verified against code: 2026-10-10
Date: recorded 2026-10-10

## Context
The Vercel proxy (`frontend/api/proxy.js`) attached the backend admin and API keys to every request. Any visitor could write `/api/settings`. Rate limits could be bypassed through `X-Forwarded-For`.

## Choice
- The proxy never attaches the admin key. `frontend/api/proxy-policy.js` returns 403 for any non-GET request to `settings`. It returns 404 for docs, openapi, integrity and metrics.
- The frontend shows Settings view-only. It becomes editable only when `VITE_SETTINGS_EDITABLE=true` (`frontend/src/lib/config.ts`). The public build must not set it.
- Local development can still edit settings.

## Why
A settings write changes the app's behaviour for every visitor, and the public site has no login. Read-only is the only safe default until there is an owner-only write path. The local settings file is written with an atomic write (`backend/app/api/settings.py`).

## Open
The saved weights have no consumer in the projection code, so the settings page shows values that change nothing. See the owner batch in `.agent-work/gridiron/STATUS.md`.
