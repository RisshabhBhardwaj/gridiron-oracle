"""
Unit tests for the `--pipeline-run-id` CLI override added to
scripts/materialize_season_simulation.py and
scripts/materialize_team_game_predictions.py (Task 10 of the promotion-
mechanism plan) — the same override pattern
scripts/materialize_stack_projections.py already has for its
`--pipeline-run-id` flag (its `main()` sets the module-level
PIPELINE_RUN_ID from args.pipeline_run_id when passed).

No live DB / network: `materialize()` is monkeypatched in each module so
these only exercise `main()`'s argparse wiring, matching this repo's
existing convention for pipeline-run-id CLI unit tests (see
test_prune_stale_pipeline_runs.py's TestLoadKeepRunIds, which also avoids
a real DB for pure argument/config-plumbing checks).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.materialize_season_simulation as season_sim_mod  # noqa: E402
import scripts.materialize_team_game_predictions as team_game_mod  # noqa: E402


class TestSeasonSimulationPipelineRunIdOverride:
    def test_flag_passed_through_to_materialize(self, monkeypatch) -> None:
        mock_materialize = MagicMock(return_value=42)
        monkeypatch.setattr(season_sim_mod, "materialize", mock_materialize)
        monkeypatch.setattr(
            sys, "argv",
            [
                "materialize_season_simulation.py",
                "--season", "2026", "--start-week", "1",
                "--pipeline-run-id", "weekly_auto_2026",
            ],
        )
        rc = season_sim_mod.main()
        assert rc == 0
        assert mock_materialize.call_count == 1
        _, kwargs = mock_materialize.call_args
        assert kwargs["pipeline_run_id"] == "weekly_auto_2026"

    def test_omitted_flag_passes_none(self, monkeypatch) -> None:
        mock_materialize = MagicMock(return_value=1)
        monkeypatch.setattr(season_sim_mod, "materialize", mock_materialize)
        monkeypatch.setattr(
            sys, "argv",
            ["materialize_season_simulation.py", "--season", "2026", "--start-week", "1"],
        )
        season_sim_mod.main()
        _, kwargs = mock_materialize.call_args
        assert kwargs["pipeline_run_id"] is None

    def test_minted_id_ignores_override_when_none(self) -> None:
        """
        Without an override, materialize() still mints its own timestamped
        id (unchanged legacy behavior) rather than raising or requiring one.
        """
        import inspect

        sig = inspect.signature(season_sim_mod.materialize)
        assert sig.parameters["pipeline_run_id"].default is None


class TestTeamGamePredictionsPipelineRunIdOverride:
    def test_flag_passed_through_to_materialize(self, monkeypatch) -> None:
        mock_materialize = MagicMock(return_value=7)
        monkeypatch.setattr(team_game_mod, "materialize", mock_materialize)
        monkeypatch.setattr(
            sys, "argv",
            [
                "materialize_team_game_predictions.py",
                "--season", "2026", "--week", "1",
                "--pipeline-run-id", "weekly_auto_2026",
            ],
        )
        rc = team_game_mod.main()
        assert rc == 0
        assert mock_materialize.call_count == 1
        _, kwargs = mock_materialize.call_args
        assert kwargs["pipeline_run_id"] == "weekly_auto_2026"

    def test_omitted_flag_passes_none(self, monkeypatch) -> None:
        mock_materialize = MagicMock(return_value=1)
        monkeypatch.setattr(team_game_mod, "materialize", mock_materialize)
        monkeypatch.setattr(
            sys, "argv",
            ["materialize_team_game_predictions.py", "--season", "2026", "--week", "1"],
        )
        team_game_mod.main()
        _, kwargs = mock_materialize.call_args
        assert kwargs["pipeline_run_id"] is None


class TestMintedIdOverrideBehavior:
    def test_season_simulation_materialize_uses_override_over_minted_id(self, monkeypatch) -> None:
        """
        Exercises the actual override line inside materialize() (not just
        argparse plumbing) by monkeypatching just enough of its
        dependencies to reach the `model_run_id = str(pipeline_run_id) if
        pipeline_run_id else (...)` branch and observing which id lands in
        the row tuples handed to the DB layer.
        """
        import types

        mod = season_sim_mod

        roster_rows = [
            {"player_id": "p1", "position": "TE", "team": "KC", "prior_snap_share": 0.5,
             "team_games_played": 3, "depth_rank": 1, "prior_active_games": 3,
             "player_name": "Player One"},
        ]
        monkeypatch.setattr(mod, "_load_roster", lambda *a, **k: roster_rows)
        monkeypatch.setattr(mod, "_load_prior_game_rows", lambda *a, **k: [])
        monkeypatch.setattr(mod, "_load_schedule_gate", lambda *a, **k: object())

        class _FakeResult:
            player_season_totals = {
                "p1": {"fantasy_ppr": {"mean": 10.0, "p10": 5.0, "p50": 10.0, "p90": 15.0}},
            }
            week_by_week: list = []
            team_win_totals: dict = {}

        class _FakeSimulator:
            def __init__(self, *a, **k):
                pass

            def run(self, *a, **k):
                return _FakeResult()

        monkeypatch.setattr(mod, "SeasonSimulator", _FakeSimulator)

        captured_rows = {}

        class _FakeCursor:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, *a, **k):
                pass

        class _FakeConn:
            def cursor(self):
                return _FakeCursor()

            def commit(self):
                pass

            def close(self):
                pass

        monkeypatch.setattr(mod.psycopg2, "connect", lambda *a, **k: _FakeConn())

        def _fake_execute_values(cur, sql, rows):
            if "season_simulations" in sql:
                captured_rows["rows"] = rows

        monkeypatch.setattr(mod.psycopg2.extras, "execute_values", _fake_execute_values)

        # ml.playing_time.p_active_from_priors is imported inline inside
        # materialize(); stub the module in sys.modules so that import
        # resolves without pulling in the real dependency chain.
        fake_playing_time = types.ModuleType("ml.playing_time")
        fake_playing_time.p_active_from_priors = lambda *a, **k: 0.9
        monkeypatch.setitem(sys.modules, "ml.playing_time", fake_playing_time)

        mod.materialize(
            season=2026, start_week=1, end_week=1, n_simulations=1,
            positions=["TE"], database_url="postgresql://unused",
            pipeline_run_id="weekly_auto_2026",
        )

        rows = captured_rows["rows"]
        assert len(rows) == 1
        # model_run_id is the last element of each season_simulations row tuple.
        assert rows[0][-1] == "weekly_auto_2026"
