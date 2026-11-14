"""Who is expected to play (decision 0012 D3, addendum 2026-11-14).

Status rules and the causal cut-off, with no database: a status is only ever
taken from a report known at or before ``as_of``, a player with no report is
expected to play, and nothing is invented for a value we do not recognise.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from pipeline.availability import (
    EXCLUDED_STATUSES,
    STORED_STATUSES,
    is_served,
    normalize_report_status,
    normalize_roster_status,
    resolve_statuses,
    select_snapshot,
    served_predicate,
)

AS_OF = datetime(2026, 11, 17, 14, 0, tzinfo=timezone.utc)


def _reports(rows: list[dict]) -> pd.DataFrame:
    cols = ["player_id", "report_status", "roster_status", "date_modified"]
    return pd.DataFrame(rows, columns=cols)


@pytest.mark.parametrize(
    "raw,stored",
    [("Out", "Out"), ("out", "Out"), ("Doubtful", "Doubtful"), (" Questionable ", "Questionable")],
)
def test_report_status_names_normalise(raw: str, stored: str) -> None:
    assert normalize_report_status(raw) == stored


@pytest.mark.parametrize("raw", [None, "", "Full", "Limited", "DNP", "nonsense"])
def test_unrecognised_report_status_is_not_invented(raw) -> None:
    assert normalize_report_status(raw) is None


@pytest.mark.parametrize(
    "raw,stored",
    [("RES", "Reserve"), ("INA", "Inactive"), ("SUS", "Suspended"), ("CUT", "Inactive")],
)
def test_roster_status_maps_to_excluded_statuses(raw: str, stored: str) -> None:
    assert normalize_roster_status(raw) == stored


@pytest.mark.parametrize("raw", [None, "ACT", "DEV", "TRD", "nonsense"])
def test_active_or_unknown_roster_status_is_none(raw) -> None:
    assert normalize_roster_status(raw) is None


def test_excluded_statuses_are_never_served() -> None:
    assert EXCLUDED_STATUSES == {"Out", "Reserve", "Suspended", "Inactive"}
    for status in EXCLUDED_STATUSES:
        assert not is_served(status)


def test_questionable_and_doubtful_are_served_and_kept() -> None:
    assert {"Questionable", "Doubtful"} <= set(STORED_STATUSES)
    assert is_served("Questionable")
    assert is_served("Doubtful")
    result = resolve_statuses(
        ["a", "b"],
        _reports([
            {"player_id": "a", "report_status": "Questionable", "roster_status": "ACT", "date_modified": None},
            {"player_id": "b", "report_status": "Doubtful", "roster_status": None, "date_modified": None},
        ]),
        AS_OF,
    )
    assert result.statuses == {"a": "Questionable", "b": "Doubtful"}


def test_player_with_no_report_is_expected_to_play() -> None:
    result = resolve_statuses(["a", "nobody"], _reports([
        {"player_id": "a", "report_status": "Out", "roster_status": None, "date_modified": None},
    ]), AS_OF)
    assert result.statuses["nobody"] is None
    assert is_served(result.statuses["nobody"])


def test_report_dated_after_as_of_is_ignored() -> None:
    later = AS_OF + timedelta(minutes=1)
    earlier = AS_OF - timedelta(hours=3)
    result = resolve_statuses(["late", "early"], _reports([
        {"player_id": "late", "report_status": "Out", "roster_status": None, "date_modified": later},
        {"player_id": "early", "report_status": "Out", "roster_status": None, "date_modified": earlier},
    ]), AS_OF)
    assert result.statuses == {"late": None, "early": "Out"}
    assert result.ignored_after_as_of == 1
    assert result.newest_report_at == earlier


def test_report_dated_exactly_at_as_of_is_known() -> None:
    result = resolve_statuses(["a"], _reports([
        {"player_id": "a", "report_status": "Out", "roster_status": None, "date_modified": AS_OF},
    ]), AS_OF)
    assert result.statuses["a"] == "Out"


def test_report_without_a_timestamp_is_bounded_by_its_snapshot() -> None:
    # nflverse's injury file has no per-row timestamp; the snapshot's fetch time is the bound.
    result = resolve_statuses(["a"], _reports([
        {"player_id": "a", "report_status": "Out", "roster_status": None, "date_modified": None},
    ]), AS_OF, snapshot_fetched_at=AS_OF - timedelta(hours=5))
    assert result.statuses["a"] == "Out"
    assert result.newest_report_at == AS_OF - timedelta(hours=5)
    assert result.newest_report_age == timedelta(hours=5)


def test_most_restrictive_status_wins() -> None:
    result = resolve_statuses(["a"], _reports([
        {"player_id": "a", "report_status": "Questionable", "roster_status": "RES", "date_modified": None},
    ]), AS_OF)
    assert result.statuses["a"] == "Reserve"


def test_snapshot_fetched_after_as_of_is_not_used() -> None:
    snaps = [
        {"snapshot_id": "s1", "fetched_at": AS_OF - timedelta(days=2)},
        {"snapshot_id": "s2", "fetched_at": AS_OF - timedelta(hours=1)},
        {"snapshot_id": "s3", "fetched_at": AS_OF + timedelta(seconds=1)},
    ]
    assert select_snapshot(snaps, AS_OF)["snapshot_id"] == "s2"
    assert select_snapshot(snaps[2:], AS_OF) is None


def test_naive_as_of_is_refused() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_statuses(["a"], _reports([]), datetime(2026, 11, 17, 14, 0))
    with pytest.raises(ValueError, match="timezone-aware"):
        select_snapshot([], datetime(2026, 11, 17, 14, 0))


def test_served_predicate_lists_every_excluded_status() -> None:
    sql = served_predicate("p.availability_status")
    assert sql.startswith("(p.availability_status IS NULL OR p.availability_status NOT IN (")
    for status in EXCLUDED_STATUSES:
        assert f"'{status}'" in sql
