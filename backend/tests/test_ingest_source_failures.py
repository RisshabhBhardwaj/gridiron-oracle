"""A failed or empty required source fails the ingest (H-4).

Before this change, a load that failed after retries was logged and skipped, so
the season contributed zero rows and the ingest reported success. The weekly job
then went on to build on stale data. These tests run the real run_full_ingest
with patched nflreadpy loaders. No network, no database.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import nflreadpy
import pytest

import scraper.adapters.nflreadpy_adapter as adapter_mod
from scraper.adapters.nflreadpy_adapter import (
    IngestIncompleteError,
    NFLReadPyAdapter,
    SOURCE_FTN_CHARTING,
    SOURCE_PLAYER_STATS,
)


class _Frame:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = [dict(r) for r in rows]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._rows]

    def __len__(self) -> int:
        return len(self._rows)


def _valid_stat_row(pid: str = "00-0000001") -> dict[str, Any]:
    return {
        "player_id": pid, "season": 2026, "week": 1, "team": "KC",
        "game_id": "2026_01_KC_BAL", "position": "WR",
    }


@pytest.fixture
def dead_letters(monkeypatch) -> list[tuple]:
    written: list[tuple] = []
    monkeypatch.setattr(
        NFLReadPyAdapter, "_write_dead_letter",
        lambda self, source, error, raw: written.append((source, error, raw)),
    )
    return written


@pytest.fixture
def live_adapter(monkeypatch, dead_letters):
    """An adapter in live mode: it has a connection, and staging writes are recorded only."""
    staged: list = []
    monkeypatch.setattr(NFLReadPyAdapter, "_flush_staging", lambda self, batch: staged.append(list(batch)))
    adapter = NFLReadPyAdapter("")
    adapter._conn = MagicMock()
    adapter.staged = staged  # type: ignore[attr-defined]
    return adapter


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Keep retry backoff and the between-source delay out of wall-clock time."""
    monkeypatch.setattr(adapter_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _s: None)


# ── Retry ─────────────────────────────────────────────────────────────────────

def test_load_with_retry_retries_a_transient_failure_until_it_succeeds():
    attempts = {"n": 0}

    def flaky(**_kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("429 rate limited")
        return "frame"

    adapter = NFLReadPyAdapter("")
    assert adapter._load_with_retry(flaky, seasons=2026) == "frame"
    assert attempts["n"] == 3


def test_load_with_retry_gives_up_after_three_attempts_and_raises_the_last_error():
    attempts = {"n": 0}

    def always_down(**_kwargs):
        attempts["n"] += 1
        raise ConnectionError(f"down {attempts['n']}")

    adapter = NFLReadPyAdapter("")
    with pytest.raises(ConnectionError, match="down 3"):
        adapter._load_with_retry(always_down, seasons=2026)
    assert attempts["n"] == 3


# ── Failed loads ──────────────────────────────────────────────────────────────

def test_a_load_that_fails_after_retries_fails_the_ingest(monkeypatch, live_adapter, dead_letters):
    def down(**_kwargs):
        raise ConnectionError("nflverse unreachable")

    monkeypatch.setattr(nflreadpy, "load_player_stats", down)
    adapter = live_adapter

    with pytest.raises(IngestIncompleteError) as excinfo:
        adapter.run_full_ingest(seasons=[2026], sources=[SOURCE_PLAYER_STATS], dry_run=False)

    message = str(excinfo.value)
    assert "player_stats" in message
    assert "[2026]" in message
    assert "nflverse unreachable" in message
    assert [(src, season_error) for src, season_error, _ in dead_letters] == [
        ("nflreadpy.load.player_stats", "ConnectionError: nflverse unreachable"),
    ]


def test_a_failed_season_fails_the_ingest_even_when_another_season_loads(monkeypatch, live_adapter):
    def load(seasons):
        if seasons == 2025:
            raise ConnectionError("2025 unavailable")
        return _Frame([_valid_stat_row()])

    monkeypatch.setattr(nflreadpy, "load_player_stats", load)

    with pytest.raises(IngestIncompleteError, match=r"load failed for season\(s\) \[2025\]"):
        live_adapter.run_full_ingest(seasons=[2025, 2026], sources=[SOURCE_PLAYER_STATS], dry_run=False)
    assert len(live_adapter.staged) == 1, "the healthy season is still staged"


# ── Empty required sources ────────────────────────────────────────────────────

def test_a_required_source_that_stages_zero_rows_fails_the_ingest(monkeypatch, live_adapter):
    monkeypatch.setattr(nflreadpy, "load_player_stats", lambda seasons: _Frame([]))

    with pytest.raises(IngestIncompleteError, match="player_stats: loaded from nflreadpy but staged zero valid rows"):
        live_adapter.run_full_ingest(seasons=[2026], sources=[SOURCE_PLAYER_STATS], dry_run=False)
    assert live_adapter.staged == []


def test_a_source_whose_rows_all_fail_validation_counts_as_zero_rows(monkeypatch, live_adapter, dead_letters):
    monkeypatch.setattr(nflreadpy, "load_player_stats", lambda seasons: _Frame([{"not": "a stat row"}]))

    with pytest.raises(IngestIncompleteError, match="staged zero valid rows"):
        live_adapter.run_full_ingest(seasons=[2026], sources=[SOURCE_PLAYER_STATS], dry_run=False)
    assert len(dead_letters) == 1, "each rejected row is still dead-lettered"


def test_a_source_with_no_attempted_loads_is_not_a_failure(monkeypatch):
    """FTN charting skips seasons before 2022. Nothing is attempted, so nothing is missing."""
    monkeypatch.setattr(nflreadpy, "load_ftn_charting", lambda seasons: pytest.fail("must not load 2019"))
    adapter = NFLReadPyAdapter("")

    results = adapter.run_full_ingest(seasons=[2019], sources=[SOURCE_FTN_CHARTING], dry_run=True)

    assert results == {SOURCE_FTN_CHARTING: (0, 0)}


def test_a_complete_ingest_returns_its_counts(monkeypatch):
    monkeypatch.setattr(nflreadpy, "load_player_stats", lambda seasons: _Frame([_valid_stat_row()]))
    adapter = NFLReadPyAdapter("")

    results = adapter.run_full_ingest(seasons=[2026], sources=[SOURCE_PLAYER_STATS], dry_run=True)

    assert results == {SOURCE_PLAYER_STATS: (1, 0)}


def test_dry_run_reports_an_incomplete_source_without_raising(monkeypatch, caplog):
    monkeypatch.setattr(nflreadpy, "load_player_stats", lambda seasons: _Frame([]))
    adapter = NFLReadPyAdapter("")

    with caplog.at_level("WARNING"):
        results = adapter.run_full_ingest(seasons=[2026], sources=[SOURCE_PLAYER_STATS], dry_run=True)

    assert results == {SOURCE_PLAYER_STATS: (0, 0)}
    assert "Ingest incomplete (dry run, not raised)" in caplog.text


def test_problem_state_resets_between_runs(monkeypatch, live_adapter):
    """A failure in one run must not leak into the next run's check."""
    state = {"down": True}

    def load(seasons):
        if state["down"]:
            raise ConnectionError("first run only")
        return _Frame([_valid_stat_row()])

    monkeypatch.setattr(nflreadpy, "load_player_stats", load)
    with pytest.raises(IngestIncompleteError):
        live_adapter.run_full_ingest(seasons=[2026], sources=[SOURCE_PLAYER_STATS], dry_run=False)

    state["down"] = False
    results = live_adapter.run_full_ingest(seasons=[2026], sources=[SOURCE_PLAYER_STATS], dry_run=False)
    assert results == {SOURCE_PLAYER_STATS: (1, 0)}
