"""
backend/app/core/rate_limit.py

Shared slowapi Limiter singleton.

Key function: client IP address (X-Forwarded-For honoured by slowapi when
behind a reverse proxy; falls back to direct connection address).

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

limiter = Limiter(key_func=get_remote_address)
