"""Forward-week structural acceptance gate (audit C-1). DB-off tests.

The statistical gate scores the pending candidate against realized points, which
do not exist before kickoff — for a forward week its join is always empty, so it
can never pass and would block every promotion. The structural gate asks the
defined question for that case instead: did the pending run produce rows for the
target week on every surface steps 4-6 write (projections, season_simulations,
team_game_predictions)? The weekly refresh runs it when the window has no
realized game_logs, and the statistical gate otherwise.

Commit 7 of decision 0012 makes the projection check week-scoped and per cell: rows
for the target week only, for every pinned (stat, position) cell and every position,
under the pending id or (for a re-run of a promoted week) the approved id.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scripts.verify_season_acceptance import verify_forward_week_structural


CELLS = [("targets", "TE"), ("receiving_yards", "WR"), ("passing_yards", "QB")]
PENDING, APPROVED = "weekly_auto_2026_pending", "weekly_auto_2026"


def _rows(run_id=PENDING, cells=CELLS, n=30):
    return [(stat, pos, run_id, n) for stat, pos in cells]


def _conn(proj_rows, sim: int = 96, team: int = 16) -> MagicMock:
    conn = MagicMock()
    cur = conn.__enter__.return_value.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = proj_rows
    cur.fetchone.side_effect = [(sim,), (team,)]
    return conn


def _structural(conn: MagicMock, **kwargs) -> dict:
    with patch("psycopg2.connect", return_value=conn):
        return verify_forward_week_structural(
            "postgres://dummy", season=2026, start_week=3,
            pipeline_run_id=PENDING, cells=CELLS, **kwargs,
        )


def test_passes_when_every_surface_and_every_pinned_cell_has_week_rows():
    payload = _structural(_conn(_rows()))

    assert payload["mode"] == "structural"
    assert payload["row_counts"] == {
        "projections": 90, "season_simulations": 96, "team_game_predictions": 16,
    }
    assert payload["acceptance"] == {"every_surface_has_target_week_rows": True}
    assert payload["missing_surfaces"] == []
    assert payload["missing_cells"] == []
    assert payload["missing_positions"] == []
    assert payload["already_current"] is False


def test_season_level_pending_rows_alone_do_not_pass():
    """The old gate counted the whole season. Rows for other weeks must not satisfy week 3."""
    conn = _conn([])  # the week-scoped query finds nothing, whatever other weeks hold
    payload = _structural(conn)

    assert payload["acceptance"] == {"every_surface_has_target_week_rows": False}
    assert "projections" in payload["missing_surfaces"]
    assert sorted(payload["missing_positions"]) == ["QB", "TE", "WR"]
    cur = conn.__enter__.return_value.cursor.return_value.__enter__.return_value
    sql, params = cur.execute.call_args_list[0].args
    assert "week = %s" in sql and 3 in params


def test_a_missing_position_fails_and_is_named():
    payload = _structural(_conn(_rows(cells=[("targets", "TE"), ("receiving_yards", "WR")])))

    assert payload["acceptance"] == {"every_surface_has_target_week_rows": False}
    assert payload["missing_positions"] == ["QB"]
    assert payload["missing_cells"] == [["passing_yards", "QB"]]
    assert "QB" in payload["diagnosis"]


def test_a_missing_cell_in_a_served_position_fails():
    rows = _rows()
    cells_with_two_wr = CELLS + [("receiving_tds", "WR")]
    with patch("psycopg2.connect", return_value=_conn(rows)):
        payload = verify_forward_week_structural(
            "postgres://dummy", season=2026, start_week=3,
            pipeline_run_id=PENDING, cells=cells_with_two_wr,
        )
    assert payload["missing_positions"] == []
    assert payload["missing_cells"] == [["receiving_tds", "WR"]]
    assert payload["acceptance"] == {"every_surface_has_target_week_rows": False}


def test_an_already_promoted_week_is_current_and_passes():
    """A re-run after promotion writes nothing new: the approved id already serves every cell."""
    payload = _structural(_conn(_rows(run_id=APPROVED)), approved_run_id=APPROVED)

    assert payload["acceptance"] == {"every_surface_has_target_week_rows": True}
    assert payload["already_current"] is True


def test_the_approved_id_is_only_read_when_the_caller_passes_it():
    conn = _conn(_rows())
    _structural(conn)
    cur = conn.__enter__.return_value.cursor.return_value.__enter__.return_value
    for call in cur.execute.call_args_list:
        assert APPROVED not in call.args[1][2]

    conn = _conn(_rows())
    _structural(conn, approved_run_id=APPROVED)
    cur = conn.__enter__.return_value.cursor.return_value.__enter__.return_value
    for call in cur.execute.call_args_list:
        assert call.args[1][2] == [PENDING, APPROVED]


def test_projection_count_counts_only_served_rows():
    """A week whose only rows are all Out must not pass: nothing would be shown."""
    conn = _conn(_rows())
    _structural(conn)
    cur = conn.__enter__.return_value.cursor.return_value.__enter__.return_value
    sql = cur.execute.call_args_list[0].args[0]
    assert "availability_status" in sql and "NOT IN" in sql


def test_default_cells_are_the_pinned_manifest_cells():
    from ml.artifact_manifest import get_manifest

    conn = _conn([])
    with patch("psycopg2.connect", return_value=conn):
        payload = verify_forward_week_structural(
            "postgres://dummy", season=2026, start_week=3, pipeline_run_id=PENDING,
        )
    assert len(payload["missing_cells"]) == len(get_manifest().stack_index) == 32
    assert sorted(payload["missing_positions"]) == ["QB", "RB", "TE", "WR"]


def test_failure_names_the_missing_surface_and_the_run():
    payload = _structural(_conn(_rows(), sim=0))

    assert payload["acceptance"] == {"every_surface_has_target_week_rows": False}
    assert payload["missing_surfaces"] == ["season_simulations"]
    assert PENDING in payload["diagnosis"]


def _main(tmp_path, conn, extra=()):
    import sys

    from scripts.verify_season_acceptance import main as verify_main

    out = tmp_path / "structural.json"
    argv = [
        "verify_season_acceptance.py", "--season", "2026", "--start-week", "3",
        "--structural-only", "--pipeline-run-id", PENDING,
        "--database-url", "postgres://dummy", "--out", str(out), *extra,
    ]
    with patch("psycopg2.connect", return_value=conn), patch.object(sys, "argv", argv), \
            patch("scripts.verify_season_acceptance._pinned_cells", return_value=CELLS):
        rc = verify_main()
    return rc, out


def test_main_writes_a_report_and_returns_zero_on_pass(tmp_path):
    rc, out = _main(tmp_path, _conn(_rows()))
    assert rc == 0 and out.exists()


def test_main_returns_nonzero_when_a_surface_is_empty(tmp_path):
    rc, _ = _main(tmp_path, _conn(_rows(), sim=0))
    assert rc == 1


def test_main_passes_the_approved_run_id_through(tmp_path):
    conn = _conn(_rows(run_id=APPROVED))
    rc, out = _main(tmp_path, conn, extra=("--approved-run-id", APPROVED))
    assert rc == 0
    assert '"already_current": true' in out.read_text()
