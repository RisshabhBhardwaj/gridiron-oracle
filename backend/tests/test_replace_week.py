"""``--replace-week`` of the weekly refresh (decision 0012 D4, commit 7).

The producer never overwrites rows an approved run already serves. Replacing a served
week is therefore a manual, flagged action taken after a database dump: it deletes the
target week's projection rows so step 4b can write them again. It must refuse without
a dump, refuse a week that has been played or has kicked off, and never run from a timer.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts import weekly_refresh
from scripts.weekly_refresh import replace_week_rows

ROOT = Path(__file__).resolve().parents[2]


def _conn(blocking_games: int, deleted: int = 41) -> MagicMock:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (blocking_games,)
    cur.rowcount = deleted
    return conn


def _sql(conn: MagicMock) -> list[str]:
    cur = conn.cursor.return_value.__enter__.return_value
    return [c.args[0] for c in cur.execute.call_args_list]


def test_replace_deletes_only_the_target_weeks_projection_rows_and_commits():
    conn = _conn(blocking_games=0, deleted=41)
    assert replace_week_rows(conn, 2026, 11) == 41

    delete = [q for q in _sql(conn) if q.lstrip().upper().startswith("DELETE")]
    assert len(delete) == 1
    assert "FROM projections" in delete[0] and "season = %s AND week = %s" in delete[0]
    cur = conn.cursor.return_value.__enter__.return_value
    assert cur.execute.call_args_list[-1].args[1] == (2026, 11)
    conn.commit.assert_called_once()


@pytest.mark.parametrize("blocking", [1, 3])
def test_replace_refuses_a_week_that_has_scores_or_has_kicked_off(blocking):
    conn = _conn(blocking_games=blocking)
    with pytest.raises(RuntimeError, match="played or has kicked off"):
        replace_week_rows(conn, 2026, 11)
    assert not any(q.lstrip().upper().startswith("DELETE") for q in _sql(conn))
    conn.commit.assert_not_called()


def test_the_blocking_check_reads_scores_and_kickoff_for_the_same_week():
    conn = _conn(blocking_games=0)
    replace_week_rows(conn, 2026, 11)
    check = _sql(conn)[0]
    assert "FROM games" in check and "kickoff_at" in check and "home_score" in check


def _main(argv):
    with patch.object(weekly_refresh, "run_weekly_refresh", return_value=0) as run:
        try:
            rc = weekly_refresh.main(argv)
        except SystemExit as exc:
            rc = exc.code
    return rc, run


def test_replace_week_without_a_dump_path_is_refused_before_anything_runs():
    rc, run = _main(["--database-url", "postgres://x", "--replace-week"])
    assert rc == 2
    run.assert_not_called()


def test_replace_week_with_a_missing_dump_file_is_refused(tmp_path):
    rc, run = _main(["--database-url", "postgres://x", "--replace-week", "--dump-path", str(tmp_path / "no.dump")])
    assert rc == 2
    run.assert_not_called()


def test_replace_week_with_an_empty_dump_file_is_refused(tmp_path):
    dump = tmp_path / "empty.dump"
    dump.write_bytes(b"")
    rc, run = _main(["--database-url", "postgres://x", "--replace-week", "--dump-path", str(dump)])
    assert rc == 2
    run.assert_not_called()


def test_a_dump_path_without_replace_week_is_refused(tmp_path):
    dump = tmp_path / "d.dump"
    dump.write_bytes(b"PGDMP")
    rc, run = _main(["--database-url", "postgres://x", "--dump-path", str(dump)])
    assert rc == 2
    run.assert_not_called()


def test_replace_week_with_a_real_dump_passes_the_flag_through(tmp_path):
    dump = tmp_path / "d.dump"
    dump.write_bytes(b"PGDMP")
    rc, run = _main(["--database-url", "postgres://x", "--replace-week", "--dump-path", str(dump)])
    assert rc == 0
    assert run.call_args.kwargs["replace_week"] is True


def test_the_default_run_does_not_replace_anything():
    rc, run = _main(["--database-url", "postgres://x"])
    assert rc == 0
    assert run.call_args.kwargs["replace_week"] is False


def test_no_systemd_unit_or_compose_file_ever_passes_replace_week():
    offenders = [
        str(p.relative_to(ROOT))
        for pattern in ("infra/**/*.service", "infra/**/*.timer", "infra/**/*.yml", "docker-compose*.yml", "Makefile")
        for p in ROOT.glob(pattern)
        if "--replace-week" in p.read_text()
    ]
    assert offenders == []
