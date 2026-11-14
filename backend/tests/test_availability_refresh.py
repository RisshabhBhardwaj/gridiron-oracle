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
from pipeline.availability_refresh import AvailabilityRefreshRefused, refresh_availability, reapply_latest_availability
from pipeline.availability_store import AvailabilityFetchError
from backend.tests._fake_db import FakeConn, install_batch_recorder

NOW = datetime(2026, 11, 20, 22, 0, tzinfo=timezone.utc)  # Friday
THURSDAY = NOW - timedelta(days=1)
SUNDAY = NOW + timedelta(days=2)
MONDAY = NOW + timedelta(days=3)
SEASON, WEEK = 2026, 11
COLS = ["player_id", "availability_status"]
# A normal week: the Thursday game is over, Sunday and Monday are still ahead.
NORMAL = (("g_thu", THURSDAY, 24, 17), ("g_sun1", SUNDAY, None, None), ("g_sun2", SUNDAY, None, None),
          ("g_mon", MONDAY, None, None))


def _conn(*, games=NORMAL, rows=(("p1", "Questionable"), ("p2", None), ("p3", "Out")), injury_rows=40, lock=True):
    return FakeConn(routes=[
        ("-- wk:lock", (["locked"], [(lock,)])),
        ("-- av:games", (["id", "kickoff_at", "home_score", "away_score"], list(games))),
        ("-- av:rows", (COLS, list(rows))),
        ("-- av:injury_rows", (["n"], [(injury_rows,)])),
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
    assert row["game_ids"]
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


def test_a_week_with_thursday_sunday_and_monday_games_updates_only_the_games_not_started():
    """Coordinator answer 6: freeze games that kicked off, update the rest."""
    conn = _conn()
    result, _ = _run(conn, {"p1": "Out", "p2": None, "p3": "Out"})
    row = _updates(conn)[0]
    assert row["game_ids"] == ["g_sun1", "g_sun2", "g_mon"]
    assert "g_thu" not in row["game_ids"]
    sql = next(e[1] for e in conn.log if e[0] == "batch")
    assert "game_id = ANY(%(game_ids)s)" in sql
    assert result["games_updated"] == 3 and result["games_frozen"] == 1
    rows_query = next(e for e in conn.log if e[0] == "execute" and "-- av:rows" in e[1])
    assert "game_id = ANY(%s)" in rows_query[1] and rows_query[2][3] == ["g_sun1", "g_sun2", "g_mon"]


def test_a_game_that_kicked_off_without_a_score_yet_is_frozen_too():
    conn = _conn(games=(("g_thu", THURSDAY, None, None), ("g_sun", SUNDAY, None, None)))
    _run(conn, {"p1": "Out", "p2": None, "p3": "Out"})
    assert _updates(conn)[0]["game_ids"] == ["g_sun"]


def test_a_kickoff_exactly_at_as_of_is_frozen():
    conn = _conn(games=(("g_now", NOW, None, None), ("g_sun", SUNDAY, None, None)))
    _run(conn, {"p1": "Out", "p2": None, "p3": "Out"})
    assert _updates(conn)[0]["game_ids"] == ["g_sun"]


def test_a_game_with_no_kickoff_time_is_frozen_not_guessed():
    conn = _conn(games=(("g_tbd", None, None, None), ("g_sun", SUNDAY, None, None)))
    _run(conn, {"p1": "Out", "p2": None, "p3": "Out"})
    assert _updates(conn)[0]["game_ids"] == ["g_sun"]


def test_a_week_where_every_game_has_started_is_refused_before_anything_is_fetched():
    conn = _conn(games=(("g_thu", THURSDAY, 24, 17), ("g_sun", NOW - timedelta(hours=1), None, None)))
    with pytest.raises(AvailabilityRefreshRefused, match="every game"):
        _run(conn, {"p1": "Out"})
    assert conn.writes == []


def test_a_game_that_kicks_off_during_the_fetch_is_frozen_by_the_second_check():
    """The first check sees two open games; the clock crosses Sunday's kickoff during the fetch."""
    clock = iter([NOW, SUNDAY + timedelta(minutes=1), SUNDAY + timedelta(minutes=1)])
    conn = _conn(games=(("g_sun", SUNDAY, None, None), ("g_mon", MONDAY, None, None)))
    refresh_availability(conn, SEASON, WEEK, now=lambda: next(clock),
                         capture=lambda *a: {"snapshot_id": "avail_new"},
                         loader=_loader({"p1": "Out", "p2": None, "p3": "Out"}), run_ids=["weekly_auto_2026"])
    assert _updates(conn)[0]["game_ids"] == ["g_mon"]


def test_if_every_game_starts_during_the_fetch_nothing_is_written():
    clock = iter([NOW, MONDAY + timedelta(hours=5), MONDAY + timedelta(hours=5)])
    conn = _conn(games=(("g_sun", SUNDAY, None, None), ("g_mon", MONDAY, None, None)))
    with pytest.raises(AvailabilityRefreshRefused, match="every game"):
        refresh_availability(conn, SEASON, WEEK, now=lambda: next(clock),
                             capture=lambda *a: {"snapshot_id": "avail_new"},
                             loader=_loader({}), run_ids=["weekly_auto_2026"])
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


def test_availability_and_training_share_one_lock_and_availability_waits_with_a_timeout():
    """Coordinator answer 5: wait for the retrain (timeout, then an alert); never stop it."""
    compose = (UNITS / "compose.yml").read_text()
    block = compose[compose.index("\n  availability:"):]
    block = block[: block.index("\n\n")]
    assert "train_out:/train_out" in block
    assert "flock -w" in block and "-E 75" in block and "/train_out/.train.lock" in block
    assert "AVAILABILITY_LOCK_WAIT_SECONDS" in block
    assert "HEALTHCHECKS_AVAILABILITY_URL" in block  # a missing ping after a timeout is the alert
    script = (UNITS.parents[2] / "scripts/train_window.sh").read_text()
    assert ".train.lock" in script  # the trainers take the same file
    assert "flock -w" in script and "TRAIN_LOCK_WAIT_SECONDS" in script  # and wait briefly, not skip a window


def test_nothing_in_the_availability_path_stops_a_training_unit():
    for name in ("systemd/gridiron-availability.service", "compose.yml"):
        text = (UNITS / name).read_text()
        block = text[text.index("availability"):] if name == "compose.yml" else text
        assert "systemctl stop" not in block and "docker stop" not in block and "docker kill" not in block


def test_a_snapshot_with_no_injury_report_rows_is_refused_and_changes_nothing():
    """Roster rows alone must not clear every Out/Doubtful designation (manager review, item 2)."""
    conn = _conn(injury_rows=0)
    with pytest.raises(AvailabilityRefreshRefused, match="injury"):
        _run(conn, {"p1": None, "p2": None, "p3": None})  # the loader would clear p3 (Out)
    assert _updates(conn) == []
    assert not any(e[0] == "execute" and "INSERT INTO forecast_runs" in e[1] for e in conn.log)


def test_the_injury_row_check_reads_the_count_recorded_at_capture_for_this_snapshot():
    conn = _conn()
    _run(conn, {"p1": "Out", "p2": None, "p3": "Out"})
    q = next(e for e in conn.log if e[0] == "execute" and "-- av:injury_rows" in e[1])
    assert "n_injury_rows" in q[1] and "availability_snapshots" in q[1] and q[2] == ("avail_new",)


def test_the_refresh_takes_the_week_lock_before_reading_the_rows_it_will_update():
    conn = _conn()
    _run(conn, {"p1": "Out", "p2": None, "p3": "Out"})
    statements = [e[1] for e in conn.log if e[0] == "execute"]
    lock_at = next(i for i, q in enumerate(statements) if "-- wk:lock" in q)
    rows_at = max(i for i, q in enumerate(statements) if "-- av:rows" in q)
    assert lock_at < rows_at
    params = next(e[2] for e in conn.log if e[0] == "execute" and "-- wk:lock" in e[1])
    assert params[1] == SEASON * 100 + WEEK  # the same key the producer takes


def test_a_refresh_that_cannot_get_the_week_lock_refuses_and_changes_nothing():
    conn = _conn(lock=False)
    with pytest.raises(AvailabilityRefreshRefused, match="another writer"):
        _run(conn, {"p1": None, "p2": None, "p3": None})
    assert _updates(conn) == []
    assert not any(e[0] == "execute" and "INSERT INTO forecast_runs" in e[1] for e in conn.log)


def test_the_default_run_ids_are_the_approved_ids_and_never_a_pending_run_not_named_by_the_caller():
    from pipeline.availability_refresh import default_run_ids

    ids = default_run_ids(2026)
    assert "weekly_auto_2026" in ids
    assert not [i for i in ids if "pending" in i]  # a pending id belongs to one in-flight run
    own = "weekly_auto_2026_pending_" + "a" * 32
    assert default_run_ids(2026, pending_id=own) == ids + [own]


# ── re-applying the newest snapshot after the producer or a restore (review N2) ───────────
def _reapply(conn, statuses, **kw):
    return reapply_latest_availability(
        conn, SEASON, WEEK, now=lambda: NOW, loader=_loader(statuses),
        run_ids=kw.pop("run_ids", ["weekly_auto_2026", "weekly_auto_2026_pending_x"]), sleep=lambda s: None, **kw,
    )


def test_a_newer_snapshot_than_the_producers_wins_for_the_games_not_started():
    """The producer stored p1 Questionable from its own as_of; a Friday snapshot says Out."""
    conn = _conn()
    result = _reapply(conn, {"p1": "Out", "p2": None, "p3": "Out"})
    assert [(r["player_id"], r["status"]) for r in _updates(conn)] == [("p1", "Out")]
    assert "g_thu" not in _updates(conn)[0]["game_ids"]  # a game that has started stays frozen
    assert result["changed"] == 1 and result["availability_snapshot_id"] == "avail_new"
    assert conn.commits == 1


def test_a_reapply_fetches_nothing_and_asks_the_loader_for_the_snapshot_known_now():
    seen = {}

    def loader(conn, season, week, player_ids, as_of, **kw):
        seen["as_of"] = as_of
        return _loader({})(conn, season, week, player_ids, as_of)

    conn = _conn()
    reapply_latest_availability(conn, SEASON, WEEK, now=lambda: NOW, loader=loader, run_ids=["weekly_auto_2026"])
    assert seen["as_of"] == NOW


def test_a_reapply_that_changes_nothing_writes_nothing():
    conn = _conn()
    result = _reapply(conn, {"p1": "Questionable", "p2": None, "p3": "Out"})
    assert result["changed"] == 0
    assert _updates(conn) == []
    assert not any("INSERT INTO forecast_runs" in e[1] for e in conn.log if e[0] == "execute")


def test_a_reapply_that_changes_something_records_why_in_forecast_runs():
    conn = _conn()
    _reapply(conn, {"p1": "Out", "p2": None, "p3": "Out"}, reason="after_producer")
    insert = next(e for e in conn.log if e[0] == "execute" and "INSERT INTO forecast_runs" in e[1])
    assert insert[2][0].startswith("availability_refresh_2026_w11_")
    assert '"kind": "availability_refresh"' in insert[2][5] and '"reason": "after_producer"' in insert[2][5]


def test_a_reapply_in_a_week_whose_games_have_all_started_is_a_quiet_skip():
    started = (("g1", THURSDAY, 24, 17), ("g2", THURSDAY, 10, 3))
    conn = _conn(games=started)
    result = _reapply(conn, {"p1": "Out"})
    assert result["skipped"] and result["changed"] == 0
    assert conn.writes == []


def test_a_reapply_waits_for_the_week_lock_then_gives_up_loudly():
    conn = _conn(lock=False)
    sleeps = []
    with pytest.raises(AvailabilityRefreshRefused, match="lock"):
        reapply_latest_availability(conn, SEASON, WEEK, now=lambda: NOW, loader=_loader({}),
                                    run_ids=["weekly_auto_2026"], attempts=3, delay=1.0, sleep=sleeps.append)
    assert sleeps == [1.0, 2.0]
    assert conn.writes == []


# ── the reapply and the restore read only accepted snapshots (review N8, fix 19) ──────────
def _with_snapshots(conn, snapshots, reports):
    conn.routes.extend([
        ("FROM availability_snapshots", (["snapshot_id", "fetched_at", "n_injury_rows"], snapshots)),
        ("FROM availability_reports", (
            ["player_id", "report_status", "practice_status", "roster_status", "date_modified"], reports)),
    ])
    return conn


def _report_query_ids(conn):
    return [e[2] for e in conn.log if e[0] == "execute" and "FROM availability_reports" in e[1]]


def test_a_reapply_skips_a_newer_roster_only_snapshot_so_out_stays_out():
    from pipeline.availability_store import load_availability

    conn = _with_snapshots(
        _conn(), [("good", NOW - timedelta(days=1), 30), ("roster_only", NOW - timedelta(minutes=2), 0)],
        [("p3", "Out", None, None, None)],
    )
    result = reapply_latest_availability(
        conn, SEASON, WEEK, now=lambda: NOW, loader=load_availability, run_ids=["weekly_auto_2026"],
        sleep=lambda s: None,
    )
    assert _report_query_ids(conn) == [("good",)]
    assert result["availability_snapshot_id"] == "good"
    assert all(u["player_id"] != "p3" for u in _updates(conn))  # p3 was Out and is still Out


def test_a_restore_reapplies_only_an_accepted_snapshot():
    from pipeline.week_archive import restore_week

    future = datetime.now(timezone.utc) + timedelta(days=2)
    now = datetime.now(timezone.utc)
    conn = _conn(games=(("g_sun", future, None, None),))
    conn.routes.extend([
        ("-- wa:columns", (["column_name"], [("id",), ("player_id",), ("game_id",), ("season",), ("week",)])),
        ("-- wa:existing", (["n"], [(5,)])),
    ])
    _with_snapshots(
        conn, [("good", now - timedelta(days=1), 30), ("roster_only", now - timedelta(minutes=2), 0)],
        [("p3", "Out", None, None, None)],
    )
    assert restore_week(conn, SEASON, WEEK) == 5
    assert _report_query_ids(conn) == [("good",)]
