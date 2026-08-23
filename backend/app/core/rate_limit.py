"""
backend/app/core/rate_limit.py

Shared slowapi Limiter singleton.

Key function: client IP address. `slowapi.util.get_remote_address` reads
only `request.client.host`, which behind a reverse proxy is the proxy's own
address — every caller would then share a single bucket. `_client_ip` prefers
the leftmost `X-Forwarded-For` entry so limits stay per-caller.

Trusting that header is safe only because the sole route to this service is
the Vercel proxy function, which holds the API key; direct callers cannot get
past the `X-API-Key` guard to spoof it. If the service is ever exposed without
that guard, revert to `get_remote_address`.

Limits (per-endpoint):
  /predict                  — 60/minute  (single-player lookup)
  /projections/week/{n}     — 30/minute  (batch endpoint)
  /projections/season/{n}   — 20/minute  (full-season batch)
  /backtest                 — 10/minute  (expensive walk-forward replay)

Usage in route modules:
    from backend.app.core.rate_limit import limiter
    from fastapi import Request

    @router.get("/my-endpoint")
    @limiter.limit("60/minute")
    def my_handler(request: Request, ...):
        ...

Note: `request: Request` MUST be the first parameter of every rate-limited
handler — slowapi inspects the signature to extract it.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


def _client_ip(request: Request) -> str:
    """Return the originating client IP, preferring X-Forwarded-For."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return get_remote_address(request)


limiter = Limiter(key_func=_client_ip)
