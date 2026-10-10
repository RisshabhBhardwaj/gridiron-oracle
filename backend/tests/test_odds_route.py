"""
backend/tests/test_odds_route.py

Tests for GET /odds, the server-side proxy for The Odds API.

Offline by construction: an autouse fixture makes every real httpx socket
transport raise, so a test that reaches the network fails instead of calling out.
Upstream behaviour is simulated either by patching fetch_odds (route tests) or
by replacing httpx's transport handler (so the real fetch_odds code runs).
"""

from __future__ import annotations

import logging
import threading
import time
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.api import odds as odds_module
from backend.app.core.config import settings
from backend.app.main import app

client = TestClient(app, raise_server_exceptions=True)

TEST_KEY = "odds-test-key-7f3a9c"
MARKET = "player_passing_yards"
EVENTS = [{"id": "evt-1", "bookmakers": [{"key": "draftkings"}]}]


@pytest.fixture(autouse=True)
def _offline_and_clean(monkeypatch):
    """No real network, no leftover cache, no API_KEY middleware, no odds key."""

    def _blocked(self, request):
        raise AssertionError(f"test attempted a real network call to {request.url.host}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _blocked)
    monkeypatch.setattr(settings, "api_key", "")
    monkeypatch.setattr(settings, "odds_api_key", TEST_KEY)
    odds_module._cache.clear()
    yield
    odds_module._cache.clear()


def _fake_upstream(monkeypatch, handler):
    """Route httpx's real transport through `handler(request) -> httpx.Response`."""

    def _handle(self, request):
        return handler(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _handle)


# ---------------------------------------------------------------------------
# (a) No key configured
# ---------------------------------------------------------------------------


def test_no_key_returns_configured_false_and_never_calls_upstream(monkeypatch):
    monkeypatch.setattr(settings, "odds_api_key", "")
    fake = Mock(return_value=EVENTS)
    monkeypatch.setattr(odds_module, "fetch_odds", fake)

    r = client.get("/odds", params={"markets": MARKET})

    assert r.status_code == 200
    assert r.json() == {"configured": False}
    fake.assert_not_called()


# ---------------------------------------------------------------------------
# (b) Market missing or outside the allow-list
# ---------------------------------------------------------------------------


def test_missing_market_returns_400_with_none_detail(monkeypatch):
    fake = Mock(return_value=EVENTS)
    monkeypatch.setattr(odds_module, "fetch_odds", fake)

    r = client.get("/odds")

    assert r.status_code == 400
    assert r.json() == {"detail": "Unsupported markets value: (none)"}
    fake.assert_not_called()


@pytest.mark.parametrize("market", ["player_bogus", "h2h", "player_passing_yards,player_rushing_yards"])
def test_unlisted_market_returns_400_with_exact_detail(monkeypatch, market):
    fake = Mock(return_value=EVENTS)
    monkeypatch.setattr(odds_module, "fetch_odds", fake)

    r = client.get("/odds", params={"markets": market})

    assert r.status_code == 400
    assert r.json() == {"detail": f"Unsupported markets value: {market}"}
    fake.assert_not_called()


# ---------------------------------------------------------------------------
# (c) Success
# ---------------------------------------------------------------------------


def test_success_returns_configured_true_with_upstream_events(monkeypatch):
    fake = Mock(return_value=EVENTS)
    monkeypatch.setattr(odds_module, "fetch_odds", fake)

    r = client.get("/odds", params={"markets": MARKET})

    assert r.status_code == 200
    assert r.json() == {"configured": True, "events": EVENTS}
    assert r.headers["Cache-Control"] == "s-maxage=300, stale-while-revalidate=600"
    fake.assert_called_once_with(MARKET, TEST_KEY)


# ---------------------------------------------------------------------------
# (d) Cache
# ---------------------------------------------------------------------------


def test_identical_requests_hit_upstream_once(monkeypatch):
    fake = Mock(return_value=EVENTS)
    monkeypatch.setattr(odds_module, "fetch_odds", fake)

    first = client.get("/odds", params={"markets": MARKET})
    second = client.get("/odds", params={"markets": MARKET})

    assert first.json() == second.json() == {"configured": True, "events": EVENTS}
    fake.assert_called_once()


def test_cache_is_keyed_by_market(monkeypatch):
    fake = Mock(return_value=EVENTS)
    monkeypatch.setattr(odds_module, "fetch_odds", fake)

    client.get("/odds", params={"markets": MARKET})
    client.get("/odds", params={"markets": "player_rushing_yards"})

    assert fake.call_count == 2


def test_concurrent_cold_requests_share_one_upstream_call(monkeypatch):
    """Single-flight: a burst on a cold market costs one upstream call, not N."""
    started = threading.Event()
    release = threading.Event()
    calls = []

    def _slow_fetch(market, api_key):
        calls.append(market)
        started.set()
        release.wait(timeout=5)
        return EVENTS

    monkeypatch.setattr(odds_module, "fetch_odds", _slow_fetch)
    results = []
    first = threading.Thread(target=lambda: results.append(odds_module._events_for(MARKET, TEST_KEY)))
    second = threading.Thread(target=lambda: results.append(odds_module._events_for(MARKET, TEST_KEY)))

    first.start()
    assert started.wait(timeout=5)
    second.start()
    time.sleep(0.2)  # let the second caller reach the market lock
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert calls == [MARKET]
    assert results == [EVENTS, EVENTS]


def test_cache_entry_expires_after_300_seconds(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(odds_module, "_clock", lambda: now["t"])
    fake = Mock(return_value=EVENTS)
    monkeypatch.setattr(odds_module, "fetch_odds", fake)

    client.get("/odds", params={"markets": MARKET})
    now["t"] += 299
    client.get("/odds", params={"markets": MARKET})
    assert fake.call_count == 1

    now["t"] += 2  # 301 s after the first fetch
    client.get("/odds", params={"markets": MARKET})
    assert fake.call_count == 2


def test_upstream_errors_are_not_cached(monkeypatch):
    calls = []

    def _handler(request):
        calls.append(request)
        return httpx.Response(503, json={}, request=request)

    _fake_upstream(monkeypatch, _handler)

    client.get("/odds", params={"markets": MARKET})
    client.get("/odds", params={"markets": MARKET})

    assert len(calls) == 2


# ---------------------------------------------------------------------------
# (e) Upstream non-2xx: same status, no upstream body echoed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("upstream_status", [401, 429, 500])
def test_upstream_error_status_is_passed_through_without_body(monkeypatch, upstream_status):
    def _handler(request):
        # Mirror a real failure: the body quotes the URL, which carries the key.
        return httpx.Response(
            upstream_status,
            json={"message": f"invalid request for {request.url}"},
            request=request,
        )

    _fake_upstream(monkeypatch, _handler)

    r = client.get("/odds", params={"markets": MARKET})

    assert r.status_code == upstream_status
    assert r.json() == {"detail": f"Odds API error: {upstream_status}"}
    assert TEST_KEY not in r.text
    assert "invalid request" not in r.text


# ---------------------------------------------------------------------------
# (f) Transport failure -> 502
# ---------------------------------------------------------------------------


def test_transport_exception_returns_502(monkeypatch):
    def _handler(request):
        raise httpx.ConnectError("connection refused")

    _fake_upstream(monkeypatch, _handler)

    r = client.get("/odds", params={"markets": MARKET})

    assert r.status_code == 502
    assert r.json() == {"detail": "Odds API unreachable: connection refused"}


# ---------------------------------------------------------------------------
# (g) The key never appears in a response body, a header, or a log record
# ---------------------------------------------------------------------------


def _assert_no_key(response):
    assert TEST_KEY not in response.text
    for name, value in response.headers.items():
        assert TEST_KEY not in value, f"key leaked in header {name}"


def test_key_absent_from_success_response_and_logs(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)

    def _handler(request):
        assert TEST_KEY in str(request.url)  # the key must reach the upstream call
        return httpx.Response(200, json=EVENTS, request=request)

    _fake_upstream(monkeypatch, _handler)

    r = client.get("/odds", params={"markets": MARKET})

    assert r.status_code == 200
    _assert_no_key(r)
    assert TEST_KEY not in caplog.text


def test_key_absent_from_upstream_error_response_and_logs(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    _fake_upstream(
        monkeypatch,
        lambda request: httpx.Response(403, json={"message": str(request.url)}, request=request),
    )

    r = client.get("/odds", params={"markets": MARKET})

    assert r.status_code == 403
    _assert_no_key(r)
    assert TEST_KEY not in caplog.text


def test_key_absent_when_transport_error_message_quotes_the_url(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)

    def _handler(request):
        raise httpx.ReadTimeout(f"timed out fetching {request.url}")

    _fake_upstream(monkeypatch, _handler)

    r = client.get("/odds", params={"markets": MARKET})

    assert r.status_code == 502
    assert r.json()["detail"].startswith("Odds API unreachable: ")
    _assert_no_key(r)
    assert TEST_KEY not in caplog.text


# ---------------------------------------------------------------------------
# (h) The route sits behind APIKeyMiddleware
# ---------------------------------------------------------------------------


def test_request_without_x_api_key_is_rejected_when_api_key_is_set(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "server-side-key")
    fake = Mock(return_value=EVENTS)
    monkeypatch.setattr(odds_module, "fetch_odds", fake)

    missing = client.get("/odds", params={"markets": MARKET})
    wrong = client.get("/odds", params={"markets": MARKET}, headers={"X-API-Key": "nope"})

    assert missing.status_code == 401
    assert wrong.status_code == 401
    fake.assert_not_called()


def test_request_with_x_api_key_reaches_the_route_when_api_key_is_set(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "server-side-key")
    monkeypatch.setattr(settings, "odds_api_key", "")

    r = client.get("/odds", params={"markets": MARKET}, headers={"X-API-Key": "server-side-key"})

    assert r.status_code == 200
    assert r.json() == {"configured": False}
