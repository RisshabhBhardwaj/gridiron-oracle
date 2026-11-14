"""The archive that makes ``--replace-week`` recoverable (review M1).

``projections`` is unique on (player_id, game_id, stat), without the run id, so new rows cannot
sit beside the served ones until promotion. The replace therefore copies the old week to
``projections_replaced`` inside the producer's transaction, and either the promotion discards
the copy or a failure puts it back. Restoring must never touch a week that was not archived.
"""

from __future__ import annotations

import pytest

from backend.tests._fake_db import FakeConn
from pipeline.week_archive import archive_week, archived_rows, discard_week, restore_week

SEASON, WEEK = 2026, 11


def _conn(*, archived: int = 5, lock: bool = True, columns=("id", "player_id", "game_id", "stat", "season", "week")):
    return FakeConn(routes=[
        ("-- wk:lock", (["locked"], [(lock,)])),
        ("-- wa:columns", (["column_name"], [(c,) for c in columns])),
        ("-- wa:existing", (["n"], [(archived,)])),
    ])


def _sql(conn):
    return [e[1] for e in conn.log if e[0] == "execute"]


def test_archive_copies_every_column_then_deletes_the_week():
    conn = _conn()
    with conn.cursor() as cur:
        archive_week(cur, SEASON, WEEK)
    copy, delete = [q for q in _sql(conn) if q.lstrip().upper().startswith(("INSERT", "DELETE"))]
    assert copy.lstrip().startswith("INSERT INTO projections_replaced")
    assert '"id", "player_id", "game_id", "stat", "season", "week"' in copy  # explicit, so drift fails loudly
    assert "FROM projections WHERE season = %s AND week = %s" in copy
    assert delete.lstrip().startswith("DELETE FROM projections WHERE")
    assert [e[2] for e in conn.log if e[0] == "execute" and e[1].lstrip().upper().startswith(("INSERT", "DELETE"))] == [
        (SEASON, WEEK), (SEASON, WEEK)]


def test_archive_refuses_a_table_with_no_columns():
    conn = _conn(columns=())
    with conn.cursor() as cur, pytest.raises(RuntimeError, match="no columns"):
        archive_week(cur, SEASON, WEEK)
    assert conn.writes == []


def test_archived_rows_counts_the_weeks_leftover():
    conn = _conn(archived=7)
    with conn.cursor() as cur:
        assert archived_rows(cur, SEASON, WEEK) == 7


def test_restore_replaces_the_week_with_the_archive_in_one_transaction():
    conn = _conn(archived=5)
    assert restore_week(conn, SEASON, WEEK) == 5
    writes = [q.lstrip() for q in _sql(conn) if q.lstrip().upper().startswith(("INSERT", "DELETE"))]
    assert writes[0].startswith("DELETE FROM projections WHERE")
    assert writes[1].startswith("INSERT INTO projections (")
    assert "FROM projections_replaced" in writes[1]
    assert writes[2].startswith("DELETE FROM projections_replaced")
    assert conn.commits == 1 and ("rollback",) not in conn.log


def test_restore_with_no_archive_touches_nothing():
    conn = _conn(archived=0)
    assert restore_week(conn, SEASON, WEEK) == 0
    assert conn.writes == []  # the served rows of an un-archived week are never deleted
    assert conn.commits == 0


def test_restore_refuses_while_another_writer_holds_the_week():
    conn = _conn(lock=False)
    with pytest.raises(RuntimeError, match="another writer"):
        restore_week(conn, SEASON, WEEK)
    assert conn.writes == []


def test_a_failed_restore_rolls_back_and_raises():
    def boom(sql, params):
        raise RuntimeError("disk full")

    conn = _conn(archived=5)
    conn.routes.insert(0, ("INSERT INTO projections (", boom))
    with pytest.raises(RuntimeError, match="disk full"):
        restore_week(conn, SEASON, WEEK)
    assert ("rollback",) in conn.log and conn.commits == 0


def test_discard_deletes_only_the_weeks_archive():
    conn = _conn()
    with conn.cursor() as cur:
        discard_week(cur, SEASON, WEEK)
    delete = next(e for e in conn.log if e[0] == "execute" and e[1].lstrip().startswith("DELETE"))
    assert "FROM projections_replaced WHERE season = %s AND week = %s" in delete[1]
    assert delete[2] == (SEASON, WEEK)


def test_the_restore_script_restores_through_the_library(monkeypatch):
    import scripts.restore_replaced_week as script

    seen = {}
    monkeypatch.setattr(script.psycopg2, "connect", lambda dsn: FakeConn())
    monkeypatch.setattr(script, "restore_week", lambda conn, season, week: seen.setdefault("args", (season, week)) and 7)
    assert script.main(["--season", "2026", "--week", "11", "--database-url", "postgres://x/y"]) == 0
    assert seen["args"] == (2026, 11)
