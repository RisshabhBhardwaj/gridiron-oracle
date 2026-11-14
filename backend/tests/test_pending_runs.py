"""Per-run pending ids, the start-of-refresh cleanup and the promotion guard (review N1)."""
from __future__ import annotations

import logging

import pytest

from backend.tests._sqlite_pg import SqlitePg
from pipeline.pending_runs import (
    LeftoverRunsError, RunNotProducedError, clear_leftovers, new_pending_run_id, pending_prefix,
    require_run_intact, require_run_producer,
)

APPROVED = "weekly_auto_2026"
OLD = "weekly_auto_2026_pending_" + "a" * 32
NEW = "weekly_auto_2026_pending_" + "b" * 32


def _no_sleep(_seconds: float) -> None:
    return None


def test_a_new_pending_id_carries_the_season_prefix_and_is_unique():
    first, second = new_pending_run_id(2026), new_pending_run_id(2026)
    assert first != second
    for rid in (first, second):
        assert rid.startswith(pending_prefix(2026) + "_")
        assert rid != APPROVED and not rid.startswith(APPROVED + "_pending_x")
    assert pending_prefix(2026) == "weekly_auto_2026_pending"
    assert new_pending_run_id(2027).startswith("weekly_auto_2027_pending_")


def _leftovers(db: SqlitePg) -> None:
    for pid in (OLD, "weekly_auto_2026_pending"):  # a per-run id and the shared name older code wrote
        db.add_projection(f"P-{pid[-4:]}", pid, week=1)
        db.add_projection(f"Q-{pid[-4:]}", pid, week=3)
        db.run("INSERT INTO season_simulation_weeks VALUES (2026, 3, ?, 'x')", (pid,))
        db.run("INSERT INTO season_simulations VALUES (2026, 3, ?, 'x')", (pid,))
        db.run("INSERT INTO season_team_wins VALUES (2026, 3, ?, 'x')", (pid,))
        db.run("INSERT OR REPLACE INTO team_game_predictions VALUES (?, 2026, 3, ?)", (f"T-{pid[-4:]}", pid))
    # Rows that must survive: served, another season's pending, and a name that merely looks similar.
    db.add_projection("SERVED", APPROVED, week=2)
    db.add_projection("OTHER-SEASON", "weekly_auto_2025_pending_" + "c" * 32, week=3, season=2025)
    db.run("INSERT INTO season_simulations VALUES (2026, 3, ?, 'keep')", (APPROVED,))
    db.run("INSERT OR REPLACE INTO team_game_predictions VALUES ('KEEP', 2026, 3, ?)", (APPROVED,))


def test_the_cleanup_deletes_only_this_seasons_pending_rows_in_every_table():
    db = SqlitePg()
    _leftovers(db)

    report = clear_leftovers(db, 2026, sleep=_no_sleep)

    assert db.run("SELECT player_id FROM projections ORDER BY player_id") == [("OTHER-SEASON",), ("SERVED",)]
    assert db.run("SELECT k FROM season_simulations") == [("keep",)]
    assert db.run("SELECT team FROM team_game_predictions") == [("KEEP",)]
    assert db.run("SELECT count(*) FROM season_simulation_weeks")[0][0] == 0
    assert db.run("SELECT count(*) FROM season_team_wins")[0][0] == 0
    assert report["deleted"] == {
        "projections": 4, "season_simulation_weeks": 2, "season_simulations": 2,
        "season_team_wins": 2, "team_game_predictions": 2,
    }


def test_the_cleanup_logs_what_it_deleted(caplog):
    db = SqlitePg()
    _leftovers(db)
    with caplog.at_level(logging.WARNING, logger="pipeline.pending_runs"):
        clear_leftovers(db, 2026, sleep=_no_sleep)
    assert any("4" in r.getMessage() and "projections" in r.getMessage() for r in caplog.records)


def test_the_cleanup_is_quiet_and_writes_nothing_when_there_is_nothing_to_clear():
    db = SqlitePg()
    db.add_projection("SERVED", APPROVED, week=2)
    report = clear_leftovers(db, 2026, sleep=_no_sleep)
    assert report["deleted"] == {} and report["restored"] == {}
    assert not any(s.lstrip().upper().startswith("DELETE") for s in db.statements)


def test_the_cleanup_takes_the_week_lock_of_every_week_it_clears():
    db = SqlitePg()
    _leftovers(db)
    clear_leftovers(db, 2026, sleep=_no_sleep)
    assert [k for _ns, k in db.lock_attempts] == [2026 * 100 + 1, 2026 * 100 + 3]


def test_a_week_held_by_another_writer_is_retried_then_refused_with_nothing_deleted():
    db = SqlitePg()
    _leftovers(db)
    db.held_elsewhere = {(0x474F4657, 2026 * 100 + 3)}
    sleeps: list[float] = []
    with pytest.raises(LeftoverRunsError, match="week lock"):
        clear_leftovers(db, 2026, attempts=3, delay=1.0, sleep=sleeps.append)
    assert sleeps == [1.0, 2.0]
    assert db.run("SELECT count(*) FROM projections")[0][0] == 6  # nothing deleted


def test_the_cleanup_succeeds_when_the_lock_frees_up_between_tries():
    db = SqlitePg()
    _leftovers(db)
    db.held_elsewhere = {(0x474F4657, 2026 * 100 + 3)}

    def free_it(_s: float) -> None:
        db.held_elsewhere.clear()

    report = clear_leftovers(db, 2026, attempts=3, delay=1.0, sleep=free_it)
    assert report["deleted"]["projections"] == 4


def test_archives_are_restored_before_anything_is_deleted():
    db = SqlitePg()
    _leftovers(db)
    db.run("INSERT INTO projections_replaced (player_id, game_id, stat, season, week, pipeline_run_id) "
           "VALUES ('OLD', 'g', 's', 2026, 3, ?)", (APPROVED,))
    order: list[tuple] = []

    def restore(conn, season, week, **_kw) -> int:
        order.append(("restore", season, week, db.run("SELECT count(*) FROM projections")[0][0]))
        return 7

    report = clear_leftovers(db, 2026, restore=restore, sleep=_no_sleep)

    assert order == [("restore", 2026, 3, 6)]  # every leftover and served row was still there when it ran
    assert report["restored"] == {3: 7}
    assert db.run("SELECT count(*) FROM projections")[0][0] == 2  # then the pending rows went


def test_a_failed_restore_refuses_the_refresh_and_points_at_the_restore_script():
    db = SqlitePg()
    _leftovers(db)
    db.run("INSERT INTO projections_replaced (player_id, game_id, stat, season, week, pipeline_run_id) "
           "VALUES ('OLD', 'g', 's', 2026, 3, ?)", (APPROVED,))

    def broken(conn, season, week, **_kw) -> int:
        raise RuntimeError("another writer holds the lock")

    with pytest.raises(LeftoverRunsError) as err:
        clear_leftovers(db, 2026, restore=broken, sleep=_no_sleep)
    assert "scripts/restore_replaced_week.py --season 2026 --week 3" in str(err.value)
    assert db.run("SELECT count(*) FROM projections")[0][0] == 6  # nothing was deleted


# ── the promotion guard ───────────────────────────────────────────────────────────────────
def test_the_guard_passes_for_a_run_whose_producer_wrote_its_forecast_row():
    db = SqlitePg()
    db.add_forecast_run("forward_projections_2026_w03_x", NEW)
    cur = db.cursor()
    require_run_producer(cur, 2026, 3, NEW)


def test_the_guard_refuses_a_run_the_producer_never_wrote():
    db = SqlitePg()
    db.add_forecast_run("forward_projections_2026_w03_x", OLD)  # another run's row does not count
    with pytest.raises(RunNotProducedError, match=NEW):
        require_run_producer(db.cursor(), 2026, 3, NEW)


@pytest.mark.parametrize("season,week,kind", [(2026, 4, "forward_projections"), (2025, 3, "forward_projections"),
                                              (2026, 3, "availability_refresh")])
def test_the_guard_needs_the_forward_producer_row_for_this_season_and_week(season, week, kind):
    db = SqlitePg()
    db.add_forecast_run("k", NEW, season=season, week=week, kind=kind)
    with pytest.raises(RunNotProducedError):
        require_run_producer(db.cursor(), 2026, 3, NEW)


# ── the promotion guard also checks the rows are still there (review N8) ─────────────────
def _produced(db: SqlitePg, *, recorded: int, present: int, skip: tuple[str, ...] = ()) -> None:
    db.add_forecast_run("forward_projections_2026_w03_x", NEW, rows=recorded)
    for i in range(present):
        db.add_projection(f"P{i}", NEW, week=3)
    for table in ("season_simulation_weeks", "season_simulations", "season_team_wins"):
        if table not in skip:
            db.run(f"INSERT INTO {table} VALUES (2026, 3, ?, 'x')", (NEW,))
    if "team_game_predictions" not in skip:
        db.run("INSERT INTO team_game_predictions VALUES ('T', 2026, 3, ?)", (NEW,))


def test_the_intact_guard_passes_when_the_rows_match_what_the_producer_recorded():
    db = SqlitePg()
    _produced(db, recorded=3, present=3)
    require_run_intact(db.cursor(), 2026, 3, NEW)


def test_the_intact_guard_refuses_a_run_whose_rows_were_deleted():
    db = SqlitePg()
    _produced(db, recorded=3, present=0)
    with pytest.raises(RunNotProducedError, match="0 .*3|3 .*0"):
        require_run_intact(db.cursor(), 2026, 3, NEW)


@pytest.mark.parametrize("present", [1, 5])
def test_the_intact_guard_refuses_a_count_that_differs_from_the_recorded_one(present):
    db = SqlitePg()
    _produced(db, recorded=3, present=present)
    with pytest.raises(RunNotProducedError, match=NEW):
        require_run_intact(db.cursor(), 2026, 3, NEW)


def test_the_intact_guard_counts_only_this_runs_rows_for_this_week():
    db = SqlitePg()
    _produced(db, recorded=2, present=2)
    db.add_projection("OTHER-RUN", OLD, week=3)
    db.add_projection("OTHER-WEEK", NEW, week=2)
    require_run_intact(db.cursor(), 2026, 3, NEW)


def test_the_intact_guard_refuses_a_producer_row_that_recorded_no_count():
    db = SqlitePg()
    db.add_forecast_run("forward_projections_2026_w03_x", NEW, rows=None)
    db.add_projection("P0", NEW, week=3)
    with pytest.raises(RunNotProducedError, match="count"):
        require_run_intact(db.cursor(), 2026, 3, NEW)


def test_the_intact_guard_still_needs_the_producer_row():
    db = SqlitePg()
    db.add_projection("P0", NEW, week=3)
    with pytest.raises(RunNotProducedError, match="forward_projections"):
        require_run_intact(db.cursor(), 2026, 3, NEW)


@pytest.mark.parametrize("table", ["season_simulation_weeks", "season_simulations", "season_team_wins",
                                   "team_game_predictions"])
def test_the_intact_guard_refuses_a_run_whose_table_of_this_run_is_empty(table):
    db = SqlitePg()
    _produced(db, recorded=3, present=3, skip=(table,))
    with pytest.raises(RunNotProducedError, match=table):
        require_run_intact(db.cursor(), 2026, 3, NEW)


def test_the_intact_guard_counts_the_season_tables_for_this_run_and_start_week_only():
    db = SqlitePg()
    _produced(db, recorded=3, present=3, skip=("season_simulations",))
    db.run("INSERT INTO season_simulations VALUES (2026, 4, ?, 'other week')", (NEW,))
    db.run("INSERT INTO season_simulations VALUES (2026, 3, ?, 'other run')", (OLD,))
    with pytest.raises(RunNotProducedError, match="season_simulations"):
        require_run_intact(db.cursor(), 2026, 3, NEW)
