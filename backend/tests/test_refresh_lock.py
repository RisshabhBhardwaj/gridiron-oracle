"""One weekly refresh per season at a time (review N8).

A second refresh's ``clear_leftovers`` deletes the pending rows of the first, so two must never overlap.
The lock is session-level: it lives as long as the connection that took it, so a killed process frees it.
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from backend.tests._sqlite_pg import SqlitePg
from pipeline.refresh_lock import (
    REFRESH_LOCK_NAMESPACE, RefreshAlreadyRunning, RefreshLockLost, season_refresh_lock,
)
from pipeline.week_lock import WEEK_LOCK_NAMESPACE


@contextmanager
def _lock(db: SqlitePg, season: int = 2026):
    with patch("pipeline.refresh_lock.psycopg2.connect", return_value=db), season_refresh_lock("postgres://x/y", season) as held:
        yield held


def test_the_lock_is_held_inside_the_block_and_released_after_it():
    db = SqlitePg()
    with _lock(db):
        assert db.session_locks == {(REFRESH_LOCK_NAMESPACE, 2026)}
    assert db.session_locks == set()


def test_a_second_refresh_of_the_same_season_is_refused_and_changes_nothing():
    db = SqlitePg()
    db.add_projection("P", "weekly_auto_2026_pending_" + "a" * 32)
    with _lock(db):
        before = list(db.statements)
        with pytest.raises(RefreshAlreadyRunning, match="2026"):
            with _lock(db):
                pytest.fail("the second refresh must not get in")
        assert db.session_locks == {(REFRESH_LOCK_NAMESPACE, 2026)}  # the first run still holds it
        assert [s for s in db.statements[len(before):] if "advisory" not in s] == []
    assert db.run("SELECT count(*) FROM projections")[0][0] == 1


def test_a_failed_run_still_releases_the_lock():
    db = SqlitePg()
    with pytest.raises(RuntimeError, match="boom"):
        with _lock(db):
            raise RuntimeError("boom")
    assert db.session_locks == set()
    with _lock(db):  # and the next run gets in
        pass


def test_another_season_is_not_blocked():
    db = SqlitePg()
    with _lock(db, 2026), _lock(db, 2025):
        assert db.session_locks == {(REFRESH_LOCK_NAMESPACE, 2026), (REFRESH_LOCK_NAMESPACE, 2025)}


def test_the_namespace_differs_from_the_week_lock():
    assert REFRESH_LOCK_NAMESPACE != WEEK_LOCK_NAMESPACE


def test_the_lock_connection_is_closed_and_not_left_in_a_transaction():
    class Conn(SqlitePg):
        closed = 0
        def close(self):
            Conn.closed += 1

    db = Conn()
    with _lock(db):
        assert db.autocommit is True  # no idle-in-transaction session for the whole run
    assert Conn.closed == 1


def test_the_lock_connection_has_tcp_keepalives_so_a_dead_link_is_noticed():
    db = SqlitePg()
    with patch("pipeline.refresh_lock.psycopg2.connect", return_value=db) as connect:
        with season_refresh_lock("postgres://x/y", 2026):
            pass
    kwargs = connect.call_args.kwargs
    assert kwargs["keepalives"] == 1
    assert kwargs["keepalives_idle"] > 0 and kwargs["keepalives_interval"] > 0 and kwargs["keepalives_count"] > 0


# ── the holder can prove its connection is still there (review N12) ───────────────────────
def test_ensure_alive_runs_select_1_on_the_lock_connection():
    db = SqlitePg()
    with _lock(db) as held:
        before = len(db.statements)
        held.ensure_alive()
        assert any("SELECT 1" in sql for sql in db.statements[before:])
        assert db.session_locks == {(REFRESH_LOCK_NAMESPACE, 2026)}  # looking does not release it


def test_ensure_alive_raises_when_the_connection_is_gone():
    db = SqlitePg()
    with _lock(db) as held:
        def gone():
            raise RuntimeError("server closed the connection unexpectedly")

        db.cursor = gone  # type: ignore[method-assign]
        with pytest.raises(RefreshLockLost, match="lock connection"):
            held.ensure_alive()
        del db.cursor  # let the block release the lock normally
