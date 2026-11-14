"""A killed refresh leaves pending rows; a later good refresh must never serve them (review N1).

Promotion relabels pending rows to the approved id. With one shared pending name, run B's promotion
relabelled the rows run A had written before it died, rows no step 7 had checked. These tests run the
real ``run_weekly_refresh`` and the real promotion SQL over an in-memory table (``_sqlite_pg.py``), with
the heavy steps replaced by writers that behave like the real ones: the stack materializer writes history
rows, the producer writes the forward week plus its ``forecast_runs`` row, the simulation and the team-game
step write their tables, and every upsert takes over a row with the same key.
"""
from __future__ import annotations

import sys
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from backend.tests._sqlite_pg import SqlitePg
from backend.tests.test_weekly_refresh import (
    _CLEAN_FEATURES, _CLEAN_INGEST, _CLEAN_NORMALIZE, _summary, _weekly_refresh_with_orchestrator,
)
from pipeline.pending_runs import clear_leftovers, require_run_intact
from pipeline.refresh_lock import REFRESH_LOCK_NAMESPACE, season_refresh_lock
from scripts.weekly_refresh import run_weekly_refresh

APPROVED = "weekly_auto_2026"
SHARED_NAME = "weekly_auto_2026_pending"  # what the producer wrote before it was told the run's own id
TABLES_WITH_RUN_ID = (
    ("projections", "pipeline_run_id"), ("season_simulation_weeks", "pipeline_run_id"),
    ("season_simulations", "pipeline_run_id"), ("season_team_wins", "pipeline_run_id"),
    ("team_game_predictions", "model_run_id"),
)


def _db() -> SqlitePg:
    db = SqlitePg()
    db.run("INSERT INTO games VALUES (2026, 1, '2026-09-10T00:00:00'), (2026, 2, '2026-09-17T00:00:00'), "
           "(2026, 3, '2026-10-01T00:00:00')")
    # Served before any of this: last week's approved rows. A pending run must not disturb them.
    db.add_projection("OLD-SERVED", APPROVED, week=2)
    return db


def _run(db: SqlitePg, tag: str, *, die_in_simulation: bool = False, cleanup: bool = True,
         after_gate=None, replace_week: bool = False) -> int:
    """One weekly refresh whose heavy steps write the rows the real ones would. Returns the exit code."""

    def stack(*_a, **_k) -> int:
        rid = sys.argv[sys.argv.index("--pipeline-run-id") + 1]
        db.add_projection(f"{tag}-hist", rid, week=1)
        db.add_projection("SHARED-hist", rid, week=1)
        return 0

    def forward(conn, season, week, **kw) -> dict:
        rid = kw.get("pending_id") or SHARED_NAME  # a producer that was not given an id uses the shared name
        if kw.get("replace_week"):  # the real producer archives the served week before it writes
            db.run("INSERT INTO projections_replaced (player_id, game_id, stat, season, week, pipeline_run_id) "
                   "VALUES ('OLD-WEEK3', 'g', 's', 2026, ?, ?)", (week, APPROVED))
        db.add_projection(f"{tag}-fwd", rid)
        db.add_projection("SHARED-fwd", rid)
        db.add_forecast_run(f"forward_projections_2026_w03_{tag}", rid, rows=2)
        return {"rows_served": 2}

    def simulation(**kw) -> None:
        if die_in_simulation:
            raise KeyboardInterrupt  # the process is killed: nothing after this line runs, no rollback hooks
        db.run("INSERT INTO season_simulations VALUES (2026, 3, ?, ?)", (kw["pipeline_run_id"], tag))

    def team_game(**kw) -> None:
        for team in (f"{tag}-team", "SHARED-team"):
            db.run("INSERT OR REPLACE INTO team_game_predictions VALUES (?, 2026, 3, ?)",
                   (team, kw["pipeline_run_id"]))

    summary = _summary(_CLEAN_INGEST, _CLEAN_NORMALIZE, _CLEAN_FEATURES)
    with ExitStack() as stack_:
        m = stack_.enter_context(_weekly_refresh_with_orchestrator(summary))
        stack_.enter_context(patch("psycopg2.connect", return_value=db))
        # The harness stubs these two; here the real ones run against the table.
        m["run_intact"].side_effect = require_run_intact
        if cleanup:
            m["clear_leftovers"].side_effect = clear_leftovers
        stack_.enter_context(patch("pipeline.refresh_lock.psycopg2.connect", return_value=db))
        m["refresh_lock"].side_effect = season_refresh_lock  # the real lock, on the same table
        m["stack"].side_effect = stack
        if after_gate is not None:
            m["acceptance"].side_effect = lambda *a, **k: (after_gate(), 0)[1]
        m["forward"].side_effect = forward
        m["simulation"].side_effect = simulation
        m["team_game"].side_effect = team_game
        return run_weekly_refresh("postgres://dummy", season=2026, replace_week=replace_week)


def _ids(db: SqlitePg) -> set[str]:
    out: set[str] = set()
    for table, col in TABLES_WITH_RUN_ID:
        out |= {r[0] for r in db.run(f"SELECT DISTINCT {col} FROM {table}")}
    return out


def test_a_killed_runs_rows_are_not_served_by_the_next_promotion_even_without_the_cleanup():
    db = _db()
    with pytest.raises(KeyboardInterrupt):
        _run(db, "A", die_in_simulation=True)
    assert db.served() == {"OLD-SERVED"}  # the killed run promoted nothing
    left = {r[0] for r in db.run("SELECT player_id FROM projections WHERE player_id LIKE 'A-%'")}
    assert left == {"A-hist", "A-fwd"}  # the kill left unchecked rows behind

    assert _run(db, "B", cleanup=False) == 0

    served = db.served()
    assert not {p for p in served if p.startswith("A-")}, f"the killed run's rows are served: {sorted(served)}"
    assert served == {"OLD-SERVED", "B-hist", "SHARED-hist", "B-fwd", "SHARED-fwd"}
    teams = {r[0] for r in db.run(
        "SELECT team FROM team_game_predictions WHERE model_run_id = ?", (APPROVED,))}
    assert teams == {"B-team", "SHARED-team"}
    simulated = {r[0] for r in db.run("SELECT k FROM season_simulations WHERE pipeline_run_id = ?", (APPROVED,))}
    assert simulated == {"B"}


def test_the_next_refresh_deletes_a_killed_runs_rows_and_serves_only_its_own():
    db = _db()
    with pytest.raises(KeyboardInterrupt):
        _run(db, "A", die_in_simulation=True)

    assert _run(db, "B") == 0

    gone = db.run("SELECT count(*) FROM projections WHERE player_id LIKE 'A-%'")[0][0]
    assert gone == 0, "the killed run's rows are still in projections"
    assert db.served() == {"OLD-SERVED", "B-hist", "SHARED-hist", "B-fwd", "SHARED-fwd"}
    leftovers = {i for i in _ids(db) if i.startswith(SHARED_NAME)}
    assert leftovers == set(), f"pending ids still present after a promoted run: {sorted(leftovers)}"


def test_two_runs_never_share_a_pending_id():
    db = _db()
    assert _run(db, "A") == 0
    assert _run(db, "B") == 0
    summaries = [r[0] for r in db.run("SELECT source_summary FROM forecast_runs")]
    pending_ids = {s.split('"pipeline_run_id": "')[1].split('"')[0] for s in summaries}
    assert len(pending_ids) == 2, f"both runs used {pending_ids}"
    assert all(p.startswith(SHARED_NAME + "_") for p in pending_ids), pending_ids


# ── two refreshes of one season must not overlap (review N8) ──────────────────────────────
def test_a_second_refresh_while_the_first_holds_the_season_lock_refuses_and_deletes_nothing():
    db = _db()
    db.add_projection("A-fwd", "weekly_auto_2026_pending_" + "a" * 32)  # run A's rows, mid-run
    db.session_locks.add((REFRESH_LOCK_NAMESPACE, 2026))                # and A holds the lock

    assert _run(db, "B") == 1

    assert db.run("SELECT count(*) FROM projections WHERE player_id = 'A-fwd'")[0][0] == 1
    assert not any(s.lstrip().upper().startswith("DELETE") for s in db.statements)
    assert db.served() == {"OLD-SERVED"}
    assert db.session_locks == {(REFRESH_LOCK_NAMESPACE, 2026)}  # B did not release A's lock


def test_the_lock_is_released_after_a_failed_run_and_after_a_killed_one():
    db = _db()
    with pytest.raises(KeyboardInterrupt):
        _run(db, "A", die_in_simulation=True)
    assert db.session_locks == set()
    assert _run(db, "B") == 0  # the next refresh is not locked out
    assert db.session_locks == set()


def _delete_pending_rows() -> None:
    raise AssertionError("replaced by the test")


def test_step_8_refuses_when_the_pending_rows_vanished_after_step_7_and_the_archive_survives():
    db = _db()

    def another_cleanup_ran() -> None:  # what a second refresh's clear_leftovers would have done
        db.run("DELETE FROM projections WHERE starts_with(pipeline_run_id, 'weekly_auto_2026_pending')")

    assert _run(db, "A", after_gate=another_cleanup_ran, replace_week=True) == 1

    assert db.run("SELECT player_id FROM projections_replaced") == [("OLD-WEEK3",)]  # not discarded
    assert db.served() == {"OLD-SERVED"}  # and nothing was promoted
