"""
backend/app/api/odds.py

GET /odds?markets=<market>: server-side proxy for The Odds API (NFL player
prop lines). It replaced frontend/api/odds-handler.js (retired; the Vercel
/api/odds rewrite now falls through to the authenticated backend proxy) and
keeps that handler's SPORT, ALLOWED_MARKETS, response shapes, status codes
and cache header, plus a monthly quota guard (see below).

Why it exists: the key must not reach the browser bundle, and the upstream
quota (500 requests a month on the free tier) must not be spendable by any
caller who names any market.

Responses:
  no ODDS_API_KEY            200 {"configured": false}, no upstream call
  market missing or unknown  400 {"detail": "Unsupported markets value: <value>"}
  success                    200 {"configured": true, "events": [...]}
                             plus Cache-Control: s-maxage=300, stale-while-revalidate=600
  upstream non-2xx           same status, {"detail": "Odds API error: <status>"};
                             the upstream body is never returned
  upstream unreachable       502 {"detail": "Odds API unreachable: <reason>"}
  quota floor tripped        429 {"detail": "Odds quota low: ..."}; no upstream
                             call. A warm cache is still served (no spend).

Quota guard: the free tier is 500 requests a month, and every upstream answer
carries an x-requests-remaining header. The last seen value is kept in-process
(alongside the cache, same single-replica caveat). Before spending an upstream
call, the remaining value is compared against settings.odds_min_remaining
(ODDS_MIN_REMAINING, default 50): below the floor, the call is refused with
429. Until one upstream answer has been seen, nothing is known and calls go
through; a missing or unparseable header likewise leaves the guard open rather
than blocking on unknown data.

Cache: an in-process TTL cache of 300 seconds, keyed by market. It is per
process. That is acceptable for a single replica. With several replicas each
one spends its own upstream quota, so a shared cache is needed before scaling
out. Only successful responses are cached. Concurrent cold requests for the
same market share one upstream call (single-flight).

The key: the upstream URL carries it as a query parameter, and httpx logs
every request URL at INFO. The httpx logger is kept at WARNING below so the key
never reaches the logs. Error messages are also scrubbed of the key.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Response

from backend.app.core.config import settings

router = APIRouter(prefix="", tags=["odds"])

_log = logging.getLogger(__name__)

# See the module docstring: the upstream URL carries the API key.
logging.getLogger("httpx").setLevel(logging.WARNING)

SPORT = "americanfootball_nfl"
UPSTREAM_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds/"
UPSTREAM_TIMEOUT_S = 10.0
CACHE_TTL_S = 300
CACHE_CONTROL = "s-maxage=300, stale-while-revalidate=600"

# Only the markets the UI renders. An open passthrough would let any caller
# drive spend against the upstream quota.
ALLOWED_MARKETS = frozenset({
    "player_reception_yards",
    "player_rushing_yards",
    "player_passing_yards",
    "player_receptions",
    "player_passing_tds",
    "player_rushing_tds",
    "player_receiving_tds",
})

# Monotonic clock for the TTL. Tests replace it to exercise expiry.
_clock = time.monotonic

_MISS = object()
_cache: dict[str, tuple[float, Any]] = {}
_cache_guard = threading.Lock()
# Fixed set of markets, so one lock per market is bounded and never grows.
_market_locks: dict[str, threading.Lock] = {m: threading.Lock() for m in ALLOWED_MARKETS}

# Last seen x-requests-remaining from the upstream API. None until one
# upstream answer has been observed. Tests reset it alongside _cache.
REMAINING_HEADER = "x-requests-remaining"
_remaining: int | None = None
_remaining_guard = threading.Lock()


def _note_remaining(value: str | None) -> None:
    """Record the upstream quota header. Absent or unparseable values leave
    the previous reading (or the unknown state) untouched."""
    global _remaining
    if value is None:
        return
    try:
        seen = int(value)
    except ValueError:
        return
    if seen < 0:
        return
    with _remaining_guard:
        _remaining = seen


def _quota_tripped() -> int | None:
    """Return the remaining count when it is below the configured floor,
    else None. Unknown (never seen) never trips."""
    with _remaining_guard:
        seen = _remaining
    if seen is None:
        return None
    floor = settings.odds_min_remaining
    return seen if seen < floor else None


class OddsFetchError(Exception):
    """The upstream call failed. status_code is the status returned to our caller."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "[redacted]") if secret else text


def fetch_odds(market: str, api_key: str) -> Any:
    """Fetch one market from The Odds API and return the decoded JSON.

    Kept as a single function so tests can replace it. Raises OddsFetchError.
    Error text never contains the upstream body, and the key is scrubbed from
    transport error messages.
    """
    params = {
        "apiKey": api_key,
        "markets": market,
        "regions": "us",
        "oddsFormat": "american",
    }
    try:
        with httpx.Client(timeout=UPSTREAM_TIMEOUT_S) as client:
            upstream = client.get(UPSTREAM_URL, params=params)
    except httpx.HTTPError as exc:
        _log.warning("odds upstream unreachable: %s", type(exc).__name__)
        reason = _redact(str(exc), api_key)
        raise OddsFetchError(502, f"Odds API unreachable: {reason}") from None

    # Record the quota header on every answer, success or failure: a 429
    # carries it too, and it is the only quota signal the guard ever sees.
    _note_remaining(upstream.headers.get(REMAINING_HEADER))

    if not upstream.is_success:
        _log.warning("odds upstream returned %s", upstream.status_code)
        raise OddsFetchError(upstream.status_code, f"Odds API error: {upstream.status_code}")

    try:
        return upstream.json()
    except ValueError:
        _log.warning("odds upstream returned a non-JSON body")
        raise OddsFetchError(502, "Odds API error: invalid JSON") from None


def _cached_events(market: str) -> Any:
    with _cache_guard:
        entry = _cache.get(market)
    if entry is not None and _clock() - entry[0] < CACHE_TTL_S:
        return entry[1]
    return _MISS


def _events_for(market: str, api_key: str) -> Any:
    hit = _cached_events(market)
    if hit is not _MISS:
        return hit
    with _market_locks[market]:
        # Another request may have filled the cache while this one waited.
        # Cache hits cost nothing upstream, so the quota guard below only
        # gates the fetch, never a served entry.
        hit = _cached_events(market)
        if hit is not _MISS:
            return hit
        tripped = _quota_tripped()
        if tripped is not None:
            floor = settings.odds_min_remaining
            _log.warning("odds quota floor tripped: %s remaining (floor %s)", tripped, floor)
            raise OddsFetchError(
                429,
                f"Odds quota low: {tripped} requests remaining "
                f"(floor {floor}); upstream call skipped",
            )
        events = fetch_odds(market, api_key)
        with _cache_guard:
            _cache[market] = (_clock(), events)
        return events


@router.get("/odds")
def get_odds(
    response: Response,
    markets: list[str] | None = Query(default=None),
) -> dict:
    """Return American player-prop odds for one allowed market, or configured=false."""
    api_key = settings.odds_api_key
    if not api_key:
        return {"configured": False}

    # Like the Vercel handler, use the first value when the parameter repeats.
    requested = markets[0] if markets else None
    if requested is None:
        raise HTTPException(status_code=400, detail="Unsupported markets value: (none)")
    if requested not in ALLOWED_MARKETS:
        raise HTTPException(status_code=400, detail=f"Unsupported markets value: {requested}")

    try:
        events = _events_for(requested, api_key)
    except OddsFetchError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from None

    response.headers["Cache-Control"] = CACHE_CONTROL
    return {"configured": True, "events": events}
