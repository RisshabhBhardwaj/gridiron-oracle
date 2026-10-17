"""Forward-week structural acceptance gate (audit C-1). DB-off tests.

The statistical gate scores the pending candidate against realized points, which
do not exist before kickoff — for a forward week its join is always empty, so it
can never pass and would block every promotion. The structural gate asks the
defined question for that case instead: did the pending run produce rows for the
target week on every surface steps 4-6 write (projections, season_simulations,
team_game_predictions)? The weekly refresh runs it when the window has no
realized game_logs, and the statistical gate otherwise.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scripts.verify_season_acceptance import verify_forward_week_structural


def _conn_with_counts(proj: int, sim: int, team: int) -> MagicMock:
    conn = MagicMock()
    cur = conn.__enter__.return_value.cursor.return_value.__enter__.return_value
    cur.fetchone.side_effect = [(proj,), (sim,), (team,)]
    return conn


def _structural(conn: MagicMock) -> dict:
    with patch("psycopg2.connect", return_value=conn):
        return verify_forward_week_structural(
            "postgres://dummy", season=2026, start_week=3,
            pipeline_run_id="weekly_auto_2026_pending",
        )


def test_structural_gate_passes_when_every_surface_has_pending_rows():
    payload = _structural(_conn_with_counts(120, 96, 16))

    assert payload["mode"] == "structural"
    assert payload["row_counts"] == {
        "projections": 120, "season_simulations": 96, "team_game_predictions": 16,
    }
    assert payload["acceptance"] == {"every_surface_has_target_week_rows": True}
    assert payload["missing_surfaces"] == []


def test_structural_gate_fails_and_names_the_missing_surface():
    payload = _structural(_conn_with_counts(120, 0, 16))

    assert payload["acceptance"] == {"every_surface_has_target_week_rows": False}
    assert payload["missing_surfaces"] == ["season_simulations"]
    assert "weekly_auto_2026_pending" in payload["diagnosis"]


def test_structural_gate_reads_the_pending_run_id_not_the_approved_surface():
    conn = _conn_with_counts(5, 5, 5)
    _structural(conn)

    cur = conn.__enter__.return_value.cursor.return_value.__enter__.return_value
    executed = [c.args[0] for c in cur.execute.call_args_list]
    assert len(executed) == 3
    assert "projections" in executed[0] and "pipeline_run_id" in executed[0]
    assert "season_simulations" in executed[1] and "start_week" in executed[1]
    assert "team_game_predictions" in executed[2] and "model_run_id" in executed[2]
    for call in cur.execute.call_args_list:
        assert "weekly_auto_2026_pending" in call.args[1]
        assert all(p != "weekly_auto_2026" for p in call.args[1]), \
            "the gate must read the pending run, never the approved id"


def test_structural_main_writes_a_report_and_returns_zero_on_pass(tmp_path):
    import sys

    from scripts.verify_season_acceptance import main as verify_main

    out = tmp_path / "structural.json"
    argv = [
        "verify_season_acceptance.py",
        "--season", "2026",
        "--start-week", "3",
        "--structural-only",
        "--pipeline-run-id", "weekly_auto_2026_pending",
        "--database-url", "postgres://dummy",
        "--out", str(out),
    ]
    with patch("psycopg2.connect", return_value=_conn_with_counts(1, 1, 1)):
        with patch.object(sys, "argv", argv):
            assert verify_main() == 0
    assert out.exists()


def test_structural_main_returns_nonzero_when_a_surface_is_empty(tmp_path):
    import sys

    from scripts.verify_season_acceptance import main as verify_main

    out = tmp_path / "structural.json"
    argv = [
        "verify_season_acceptance.py",
        "--season", "2026",
        "--start-week", "3",
        "--structural-only",
        "--pipeline-run-id", "weekly_auto_2026_pending",
        "--database-url", "postgres://dummy",
        "--out", str(out),
    ]
    with patch("psycopg2.connect", return_value=_conn_with_counts(1, 0, 1)):
        with patch.object(sys, "argv", argv):
            assert verify_main() == 1
