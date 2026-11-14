"""Friday availability-only refresh (decision 0012 addendum, item 8).

Updates one column, ``availability_status``, on the target week's forward rows, in one
transaction, and records itself in ``forecast_runs`` under its own key. It never touches a
prediction column, a history row or a played week.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

from pipeline.availability import AvailabilityResult, is_served
from pipeline.availability_refresh import AvailabilityRefreshRefused, refresh_availability
from pipeline.availability_store import AvailabilityFetchError
from backend.tests._fake_db import FakeConn, install_batch_recorder

NOW = datetime(2026, 11, 20, 22, 0, tzinfo=timezone.utc)  # Friday
SUNDAY = NOW + timedelta(days=2)
SEASON, WEEK = 2026, 11
COLS = ["player_id", "availability_status"]


def _conn(*, kickoffs=(SUNDAY,), scores=(None,), rows=(("p1", "Questionable"), ("p2", None), ("p3", "Out"))):
    games = [(f"g{i}", k, s, s) for i, (k, s) in enumerate(zip(kickoffs, scores * len(kickoffs) if len(scores) == 1 else scores))]
    return FakeConn(routes=[
        ("-- av:games", (["id", "kickoff_at", "home_score", "away_score"], games)),
        ("-- av:rows", (COLS, list(rows))),
    ])


def _loader(statuses):
    def load(conn, season, week, player_ids, as_of, **kw):
        result = AvailabilityResult(statuses={p: statuses.get(p) for p in player_ids}, as_of=as_of)
        result.snapshot_id = "avail_new"
        result.counts = {s: 1 for s in statuses.values() if s}
        return result
    return load


def _run(conn, statuses, **kw):
    calls = {"capture": 0}

    def capture(c, season, week):
        calls["capture"] += 1
        return {"snapshot_id": "avail_new"}

    result = refresh_availability(
        conn, SEASON, WEEK, now=lambda: NOW, capture=kw.pop("capture", capture), loader=_loader(statuses),
        run_ids=["weekly_auto_2026", "weekly_auto_2026_pending"], **kw,
    )
    return result, calls


def _updates(conn):
    batches = [e for e in conn.log if e[0] == "batch" and "UPDATE projections" in e[1]]
    return [row for b in batches for row in b[2]]


@pytest.fixture(autouse=True)
def _batches(monkeypatch):
    install_batch_recorder(monkeypatch)


def test_a_status_change_flips_what_is_served():
    conn = _conn()
    result, _ = _run(conn, {"p1": "Out", "p2": None, "p3": "Out"})  # p1 Questionable -> Out
    assert [(r["player_id"], r["status"]) for r in _updates(conn)] == [("p1", "Out")]
    assert is_served("Questionable") and not is_served("Out")
    assert result["changed"] == 1 and result["unchanged"] == 2
    assert conn.commits >= 1


def test_a_cleared_player_is_served_again():
    conn = _conn()
    _run(conn, {"p1": "Questionable", "p2": None, "p3": None})  # p3 Out -> cleared
    assert [(r["player_id"], r["status"]) for r in _updates(conn)] == [("p3", None)]
    assert is_served(None)


def test_only_the_status_column_is_ever_written():
    conn = _conn()
    _run(conn, {"p1": "Out", "p2": "Doubtful", "p3": None})
    writes = [e for e in conn.writes]
    for entry in writes:
        sql = entry[1]
        if "UPDATE projections" in sql:
            set_clause = re.search(r"SET (.*?) WHERE", sql, re.S | re.I).group(1)
            assert [c.split("=")[0].strip() for c in set_clause.split(",")] == ["availability_status"]
        else:
            assert sql.lstrip().upper().startswith("INSERT INTO FORECAST_RUNS"), sql
    assert not any("DELETE" in e[1].upper() for e in writes)


def test_updates_are_scoped_to_the_week_and_the_forward_run_ids():
    conn = _conn()
    _run(conn, {"p1": "Out", "p2": None, "p3": "Out"})
    row = _updates(conn)[0]
    assert (row["season"], row["week"]) == (SEASON, WEEK)
    assert row["run_ids"] == ["weekly_auto_2026", "weekly_auto_2026_pending"]
    sql = next(e[1] for e in conn.log if e[0] == "batch")
    assert "season = %(season)s" in sql and "week = %(week)s" in sql and "pipeline_run_id = ANY" in sql


def test_the_refresh_records_itself_under_its_own_forecast_runs_key():
    conn = _conn()
    result, _ = _run(conn, {"p1": "Out", "p2": None, "p3": "Out"})
    insert = next(e for e in conn.log if e[0] == "execute" and "INSERT INTO forecast_runs" in e[1])
    key = insert[2][0]
    assert key == "availability_refresh_2026_w11_20261120T220000Z" == result["forecast_run_key"]
    assert not key.startswith("forward_projections_") and not key.startswith("forward_features_")
    assert '"kind": "availability_refresh"' in insert[2][5]
    assert '"availability_snapshot_id": "avail_new"' in insert[2][5]


def test_unchanged_statuses_write_no_updates_but_are_still_recorded():
    conn = _conn()
    result, _ = _run(conn, {"p1": "Questionable", "p2": None, "p3": "Out"})
    assert _updates(conn) == []
    assert result["changed"] == 0
    assert any(e[0] == "execute" and "INSERT INTO forecast_runs" in e[1] for e in conn.log)


def test_a_week_with_a_passed_kickoff_is_refused_before_anything_is_fetched_or_written():
    conn = _conn(kickoffs=(NOW - timedelta(hours=1), SUNDAY))
    with pytest.raises(AvailabilityRefreshRefused, match="kickoff"):
        _run(conn, {"p1": "Out"})
    assert conn.writes == []


def test_a_kickoff_exactly_at_as_of_is_refused():
    conn = _conn(kickoffs=(NOW,))
    with pytest.raises(AvailabilityRefreshRefused):
        _run(conn, {})
    assert conn.writes == []


def test_a_kickoff_that_passes_during_the_fetch_is_caught_by_the_second_check():
    """The first check passes; the capture takes long enough that the clock crosses kickoff."""
    clock = iter([NOW, SUNDAY + timedelta(minutes=1), SUNDAY + timedelta(minutes=1)])
    conn = _conn(kickoffs=(SUNDAY,))
    with pytest.raises(AvailabilityRefreshRefused, match="kickoff"):
        refresh_availability(conn, SEASON, WEEK, now=lambda: next(clock),
                             capture=lambda *a: {"snapshot_id": "avail_new"},
                             loader=_loader({"p1": "Out"}), run_ids=["weekly_auto_2026"])
    assert conn.writes == []


def test_a_played_week_is_refused():
    conn = _conn(scores=(24,))
    with pytest.raises(AvailabilityRefreshRefused, match="scores"):
        _run(conn, {})
    assert conn.writes == []


def test_a_week_with_no_games_is_refused():
    conn = FakeConn(routes=[("-- av:games", (["id", "kickoff_at", "home_score", "away_score"], []))])
    with pytest.raises(AvailabilityRefreshRefused, match="no games"):
        _run(conn, {})


def test_a_week_with_no_forward_rows_is_refused_and_nothing_is_written():
    conn = _conn(rows=())
    with pytest.raises(AvailabilityRefreshRefused, match="no forward projection rows"):
        _run(conn, {})
    assert conn.writes == []


def test_a_failed_fetch_writes_nothing():
    def boom(*a, **k):
        raise AvailabilityFetchError("source unavailable")

    conn = _conn()
    with pytest.raises(AvailabilityFetchError):
        _run(conn, {}, capture=boom)
    assert conn.writes == []


def test_a_failure_during_the_update_rolls_back_the_whole_refresh(monkeypatch):
    import psycopg2.extras

    def explode(cur, sql, args, page_size=100):
        raise RuntimeError("disk full")

    monkeypatch.setattr(psycopg2.extras, "execute_batch", explode)
    conn = _conn()
    with pytest.raises(RuntimeError, match="disk full"):
        _run(conn, {"p1": "Out", "p2": None, "p3": "Out"})
    assert ("rollback",) in conn.log
    assert not any(e[0] == "execute" and "INSERT INTO forecast_runs" in e[1] for e in conn.log)


# ── target week and the CLI ──────────────────────────────────────────────────

def test_the_target_week_is_the_week_of_the_next_kickoff():
    from pipeline.availability_refresh import resolve_target_week

    conn = FakeConn(routes=[("-- av:next_week", (["week"], [(11,)]))])
    assert resolve_target_week(conn, SEASON, NOW) == 11
    params = next(e[2] for e in conn.log if e[0] == "execute")
    assert params == (SEASON, NOW)


def test_no_upcoming_kickoff_means_nothing_to_refresh():
    from pipeline.availability_refresh import resolve_target_week

    conn = FakeConn(routes=[("-- av:next_week", (["week"], []))])
    with pytest.raises(AvailabilityRefreshRefused, match="no upcoming kickoff"):
        resolve_target_week(conn, SEASON, NOW)


def _cli(monkeypatch, argv, *, report=None, error=None):
    from scripts import refresh_availability as cli

    seen = {}

    class _Conn:
        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(cli.psycopg2, "connect", lambda dsn: _Conn())
    monkeypatch.setattr(cli, "resolve_target_week", lambda conn, season, now: 11)

    def fake_refresh(conn, season, week, **kw):
        seen["args"] = (season, week)
        if error:
            raise error
        return report or {"changed": 2}

    monkeypatch.setattr(cli, "refresh_availability", fake_refresh)
    return cli.main(argv), seen


def test_cli_refreshes_the_resolved_week_and_prints_the_report(monkeypatch, capsys):
    rc, seen = _cli(monkeypatch, ["--season", "2026", "--database-url", "postgres://x"])
    assert rc == 0 and seen["args"] == (2026, 11) and seen["closed"]
    assert '"changed": 2' in capsys.readouterr().out


def test_cli_returns_1_and_says_nothing_was_written_on_a_refusal(monkeypatch, capsys):
    rc, _ = _cli(monkeypatch, ["--season", "2026", "--database-url", "postgres://x"],
                 error=AvailabilityRefreshRefused("kickoff passed"))
    assert rc == 1
    assert "Refused, nothing written: kickoff passed" in capsys.readouterr().err


def test_cli_returns_1_when_the_source_is_unavailable(monkeypatch, capsys):
    rc, _ = _cli(monkeypatch, ["--season", "2026", "--database-url", "postgres://x"],
                 error=AvailabilityFetchError("source unavailable"))
    assert rc == 1
    assert "source unavailable" in capsys.readouterr().err


def test_cli_week_flag_overrides_the_resolved_week(monkeypatch):
    rc, seen = _cli(monkeypatch, ["--season", "2026", "--week", "9", "--database-url", "postgres://x"])
    assert rc == 0 and seen["args"] == (2026, 9)


# ── deployment files ─────────────────────────────────────────────────────────

from pathlib import Path  # noqa: E402

UNITS = Path(__file__).resolve().parents[2] / "infra/hosts/vps1"


def test_the_systemd_timer_is_friday_2200_utc_and_does_not_catch_up():
    text = (UNITS / "systemd/gridiron-availability.timer").read_text()
    assert "OnCalendar=Fri *-*-* 22:00:00" in text
    assert "Persistent=false" in text
    assert "Unit=gridiron-availability.service" in text


def test_the_service_is_a_oneshot_that_never_stops_training():
    text = (UNITS / "systemd/gridiron-availability.service").read_text()
    assert "Type=oneshot" in text
    assert "--profile jobs run --rm availability" in text
    # Fri 22:00 is also the full retrain's start. Conflicts= would kill it.
    assert not any(line.startswith("Conflicts=") for line in text.splitlines())


def test_compose_has_a_jobs_profile_one_shot_that_runs_the_script():
    text = (UNITS / "compose.yml").read_text()
    block = text[text.index("\n  availability:"):]
    block = block[: block.index("\n\n")]
    assert 'profiles: ["jobs"]' in block and 'restart: "no"' in block
    assert "scripts/refresh_availability.py" in block
    assert "PRODUCT_MODE: artifact_backed" in block
