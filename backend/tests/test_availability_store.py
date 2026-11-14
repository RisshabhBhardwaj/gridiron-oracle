"""Availability snapshots: stored with their fetch time, read back as of a moment.

A later run must be able to show exactly what was known when a forward row was
written, so each fetch is kept (decision 0012 D3, addendum 2026-11-14).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from backend.tests._fake_db import FakeConn, install_batch_recorder
from pipeline.availability_store import (
    AvailabilityFetchError,
    NoAvailabilitySnapshot,
    capture_snapshot,
    load_availability,
)

NOW = datetime(2026, 11, 17, 9, 30, tzinfo=timezone.utc)


def _injuries() -> pd.DataFrame:
    return pd.DataFrame({
        "season": [2026, 2026, 2026],
        "week": [11, 11, 10],
        "gsis_id": ["00-0000001", "00-0000002", "00-0000003"],
        "team": ["KC", "KC", "KC"],
        "report_status": ["Out", "Questionable", "Out"],
        "practice_status": ["Did Not Participate In Practice", "Limited Participation in Practice", None],
        "date_modified": ["2026-11-14T20:00:00Z", "2026-11-14T20:00:00Z", "2026-11-07T20:00:00Z"],
    })


def _roster() -> pd.DataFrame:
    return pd.DataFrame({
        "season": [2026, 2026, 2026, 2026],
        "week": [11, 11, 11, 10],
        "gsis_id": ["00-0000004", "00-0000005", "00-0000006", "00-0000007"],
        "team": ["KC", "KC", "KC", "KC"],
        "status": ["RES", "ACT", "DEV", "RES"],
    })


def _fetch(season: int, week: int):
    return _injuries(), _roster()


def test_capture_stores_snapshot_and_only_the_target_week(monkeypatch) -> None:
    install_batch_recorder(monkeypatch)
    conn = FakeConn()
    snap = capture_snapshot(conn, 2026, 11, fetch=_fetch, now=lambda: NOW)
    assert snap["fetched_at"] == NOW
    assert snap["snapshot_id"].startswith("avail_2026_w11_")
    assert conn.commits == 1
    inserts = [w for w in conn.writes if w[0] == "execute" and "availability_snapshots" in w[1]]
    assert len(inserts) == 1
    batch = next(w for w in conn.writes if w[0] == "batch")
    stored = {r["player_id"]: r for r in batch[2]}
    # week 11 injuries, plus roster rows that carry information (RES), not ACT/DEV; week 10 is ignored.
    assert set(stored) == {"00-0000001", "00-0000002", "00-0000004"}
    assert stored["00-0000001"]["report_status"] == "Out"
    assert stored["00-0000004"]["roster_status"] == "RES"
    assert snap["n_reports"] == 3


def test_fetch_failure_writes_nothing(monkeypatch) -> None:
    install_batch_recorder(monkeypatch)
    conn = FakeConn()

    def boom(season: int, week: int):
        raise OSError("network down")

    with pytest.raises(AvailabilityFetchError, match="network down"):
        capture_snapshot(conn, 2026, 11, fetch=boom, now=lambda: NOW)
    assert conn.writes == []
    assert conn.commits == 0


def test_empty_source_is_an_error_not_an_all_clear(monkeypatch) -> None:
    install_batch_recorder(monkeypatch)
    conn = FakeConn()
    with pytest.raises(AvailabilityFetchError, match="no injury or roster rows"):
        capture_snapshot(
            conn, 2026, 11,
            fetch=lambda s, w: (_injuries().iloc[0:0], _roster().iloc[0:0]),
            now=lambda: NOW,
        )
    assert conn.writes == []


def _routes(snapshots, reports):
    """``snapshots`` rows are (snapshot_id, fetched_at) or (snapshot_id, fetched_at, n_injury_rows)."""
    snapshots = [s if len(s) == 3 else (*s, 10) for s in snapshots]
    return [
        ("FROM availability_snapshots", (["snapshot_id", "fetched_at", "n_injury_rows"], snapshots)),
        ("FROM availability_reports", (
            ["player_id", "report_status", "practice_status", "roster_status", "date_modified"], reports)),
    ]


def test_load_uses_newest_snapshot_known_at_as_of() -> None:
    as_of = NOW
    conn = FakeConn(_routes(
        [("old", as_of - timedelta(days=1)), ("future", as_of + timedelta(minutes=5))],
        [("00-1", "Out", None, None, None)],
    ))
    result = load_availability(conn, 2026, 11, ["00-1", "00-9"], as_of)
    assert result.snapshot_id == "old"
    assert result.statuses == {"00-1": "Out", "00-9": None}
    # the report query asked for the snapshot that was known, not the future one
    report_query = [e for e in conn.log if e[0] == "execute" and "availability_reports" in e[1]][0]
    assert report_query[2] == ("old",)
    assert result.newest_report_age == timedelta(days=1)


def test_load_refuses_when_nothing_was_known_at_as_of() -> None:
    conn = FakeConn(_routes([("future", NOW + timedelta(minutes=5))], []))
    with pytest.raises(NoAvailabilitySnapshot, match="no availability snapshot"):
        load_availability(conn, 2026, 11, ["00-1"], NOW)


# ── a snapshot with no injury rows is never selected (review N8, fix 19) ──────────────────
def test_capture_records_how_many_injury_rows_the_snapshot_holds(monkeypatch) -> None:
    install_batch_recorder(monkeypatch)
    conn = FakeConn()
    snap = capture_snapshot(conn, 2026, 11, fetch=_fetch, now=lambda: NOW)
    insert = next(w for w in conn.writes if w[0] == "execute" and "INSERT INTO availability_snapshots" in w[1])
    assert "n_injury_rows" in insert[1]
    assert insert[2][-1] == 2  # 00-0000001 and 00-0000002 carry a report or practice status; the roster row does not
    assert snap["n_injury_rows"] == 2


def test_a_roster_only_capture_is_stored_but_recorded_as_having_no_injury_rows(monkeypatch) -> None:
    install_batch_recorder(monkeypatch)
    conn = FakeConn()
    snap = capture_snapshot(conn, 2026, 11, fetch=lambda s, w: (_injuries().iloc[0:0], _roster()), now=lambda: NOW)
    assert snap["n_injury_rows"] == 0 and snap["n_reports"] == 1


def test_a_newer_roster_only_snapshot_is_skipped_for_the_newest_accepted_one() -> None:
    conn = FakeConn(_routes(
        [("good", NOW - timedelta(days=1), 12), ("roster_only", NOW - timedelta(minutes=5), 0)],
        [("00-1", "Out", None, None, None)],
    ))
    result = load_availability(conn, 2026, 11, ["00-1"], NOW)
    assert result.snapshot_id == "good"
    assert result.statuses == {"00-1": "Out"}  # the roster-only snapshot would have cleared it
    report_query = [e for e in conn.log if e[0] == "execute" and "availability_reports" in e[1]][0]
    assert report_query[2] == ("good",)


def test_only_roster_only_snapshots_means_no_snapshot_at_all() -> None:
    conn = FakeConn(_routes([("roster_only", NOW - timedelta(minutes=5), 0)], []))
    with pytest.raises(NoAvailabilitySnapshot, match="injury"):
        load_availability(conn, 2026, 11, ["00-1"], NOW)


def test_naming_a_roster_only_snapshot_does_not_make_it_acceptable() -> None:
    conn = FakeConn(_routes([("good", NOW - timedelta(days=1), 12), ("roster_only", NOW - timedelta(minutes=5), 0)], []))
    with pytest.raises(NoAvailabilitySnapshot):
        load_availability(conn, 2026, 11, ["00-1"], NOW, snapshot_id="roster_only")
