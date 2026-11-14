"""capture_snapshot validates the source before storing it (security review M2).

The parser used to fail open: any unexpected value mapped to "no designation", which is served.
Now a bad file is refused whole and nothing is stored, so neither the Tuesday producer nor the
Friday refresh can turn schema drift into an all-clear.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from backend.tests._fake_db import FakeConn, install_batch_recorder
from pipeline.availability_store import AvailabilityFetchError, build_report_rows, capture_snapshot

NOW = datetime(2026, 11, 20, 22, 0, tzinfo=timezone.utc)
A, B, C = "00-0000001", "00-0000002", "00-0000003"


def injuries(**overrides) -> pd.DataFrame:
    data = {
        "season": [2026, 2026], "week": [12, 12], "gsis_id": [A, B], "team": ["KC", "KC"],
        "report_status": ["Out", "Questionable"],
        "practice_status": ["Did Not Participate In Practice", "Limited Participation in Practice"],
        "date_modified": ["2026-11-19T20:00:00Z", "2026-11-19T20:00:00Z"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def roster(**overrides) -> pd.DataFrame:
    data = {"season": [2026, 2026], "week": [12, 12], "gsis_id": [C, B], "team": ["KC", "KC"], "status": ["RES", "ACT"]}
    data.update(overrides)
    return pd.DataFrame(data)


def capture(inj=None, ros=None):
    install = pytest.MonkeyPatch()
    install_batch_recorder(install)
    conn = FakeConn()
    try:
        snap = capture_snapshot(conn, 2026, 12, fetch=lambda s, w: (inj if inj is not None else injuries(),
                                                                    ros if ros is not None else roster()), now=lambda: NOW)
    finally:
        install.undo()
    return snap, conn


def refused(inj=None, ros=None, match=""):
    install = pytest.MonkeyPatch()
    install_batch_recorder(install)
    conn = FakeConn()
    try:
        with pytest.raises(AvailabilityFetchError, match=match):
            capture_snapshot(conn, 2026, 12, fetch=lambda s, w: (inj if inj is not None else injuries(),
                                                                 ros if ros is not None else roster()), now=lambda: NOW)
    finally:
        install.undo()
    assert conn.writes == [] and conn.commits == 0, "a refused snapshot must store nothing"


def test_a_valid_source_is_stored() -> None:
    snap, conn = capture()
    assert snap["n_reports"] == 3 and conn.commits == 1


# 1. required columns
@pytest.mark.parametrize("column", ["gsis_id", "week", "season", "report_status", "date_modified"])
def test_a_missing_injury_column_is_refused(column) -> None:
    refused(inj=injuries().drop(columns=[column]), match=column)


@pytest.mark.parametrize("column", ["gsis_id", "status"])
def test_a_missing_roster_column_is_refused(column) -> None:
    refused(ros=roster().drop(columns=[column]), match=column)


def test_a_roster_frame_without_a_week_column_is_refused_not_collapsed() -> None:
    refused(ros=roster().drop(columns=["week"]), match="week")


# 2. label allow-lists
@pytest.mark.parametrize("label", ["O", "IR", "PUP", "Injured Reserve", "Out - Doubtful", "Suspended"])
def test_an_unknown_report_status_refuses_instead_of_being_served(label) -> None:
    refused(inj=injuries(report_status=["Out", label]), match=label)


@pytest.mark.parametrize("label", ["NFI", "IR", "Reserve", "XYZ"])
def test_an_unknown_roster_code_refuses(label) -> None:
    refused(ros=roster(status=["RES", label]), match=label)


def test_an_unknown_practice_status_refuses() -> None:
    refused(inj=injuries(practice_status=["Did Not Participate In Practice", "Maybe"]), match="Maybe")


@pytest.mark.parametrize("code", ["ACT", "DEV", "TRC", "TRD", "RES", "INA", "SUS", "CUT", "RET", "EXE"])
def test_known_roster_codes_are_accepted(code) -> None:
    capture(ros=roster(status=["RES", code]))


def test_missing_labels_are_not_unknown_labels() -> None:
    capture(inj=injuries(report_status=["Out", None], practice_status=[None, None]))


def test_an_unknown_label_is_logged(caplog) -> None:
    caplog.set_level("WARNING")
    refused(inj=injuries(report_status=["Out", "IR"]), match="IR")
    assert any("IR" in r.getMessage() for r in caplog.records)


# 3. date_modified <= fetched_at
def test_a_report_dated_after_the_fetch_is_refused() -> None:
    refused(inj=injuries(date_modified=["2026-11-19T20:00:00Z", "2026-11-21T00:00:00Z"]), match="date_modified")


def test_an_unreadable_report_date_is_refused() -> None:
    refused(inj=injuries(date_modified=["2026-11-19T20:00:00Z", "not a date"]), match="date_modified")


def test_a_report_dated_exactly_at_the_fetch_is_accepted() -> None:
    capture(inj=injuries(date_modified=["2026-11-20T22:00:00Z", "2026-11-19T20:00:00Z"]))


# 4. duplicates: most restrictive wins, count logged
def test_duplicate_players_keep_the_most_restrictive_status_not_the_last(caplog) -> None:
    caplog.set_level("INFO")
    inj = injuries(
        gsis_id=[A, A], report_status=["Out", "Questionable"],
        practice_status=["Did Not Participate In Practice", "Limited Participation in Practice"],
    )
    stats: dict = {}
    rows = build_report_rows(inj, roster(status=["ACT", "ACT"]).iloc[0:0], 2026, 12, stats=stats)
    assert [r["report_status"] for r in rows] == ["Out"]
    assert rows[0]["practice_status"] == "Did Not Participate In Practice"
    assert stats["duplicates"] == 1
    reversed_rows = build_report_rows(inj.iloc[::-1], roster().iloc[0:0], 2026, 12)
    assert [r["report_status"] for r in reversed_rows] == ["Out"], "order must not matter"


def test_the_duplicate_count_is_logged_at_capture(caplog) -> None:
    caplog.set_level("INFO")
    capture(inj=injuries(gsis_id=[A, A], report_status=["Out", "Questionable"]))
    assert any("duplicate" in r.getMessage().lower() and "1" in r.getMessage() for r in caplog.records)


def test_duplicate_roster_rows_keep_the_most_restrictive_code() -> None:
    ros = roster(gsis_id=[C, C], status=["INA", "RES"])
    rows = build_report_rows(injuries().iloc[0:0], ros, 2026, 12)
    assert [r["roster_status"] for r in rows] == ["RES"]
    rows = build_report_rows(injuries().iloc[0:0], ros.iloc[::-1], 2026, 12)
    assert [r["roster_status"] for r in rows] == ["RES"]


# 5. gsis id format
@pytest.mark.parametrize("bad", ["1234567", "00-123", "A-0000001", "00-00000012", "player_1"])
def test_ids_not_in_gsis_format_are_refused(bad) -> None:
    refused(inj=injuries(gsis_id=[A, bad]), match="gsis")


def test_a_roster_id_not_in_gsis_format_is_refused() -> None:
    refused(ros=roster(gsis_id=[C, "bogus"]), match="gsis")


# row cap
def test_an_oversized_source_is_refused() -> None:
    n = 6000
    big = pd.DataFrame({
        "season": [2026] * n, "week": [12] * n, "gsis_id": [f"00-{i:07d}" for i in range(n)], "team": ["KC"] * n,
        "report_status": ["Out"] * n, "practice_status": [None] * n, "date_modified": ["2026-11-19T20:00:00Z"] * n,
    })
    refused(inj=big, match="rows")
