"""
backend/tests/test_orchestrator.py

Integration test for pipeline/orchestrator.py.

Runs the full 3-step pipeline (ingest → normalize → feature_engineer) in
dry-run mode using the 2025 nflreadpy data (24h filesystem cache).
No database connection required.

Scope: class — the orchestrator runs once and all tests share the result.
This avoids re-fetching nflreadpy data for each test method.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.orchestrator import Orchestrator, OrchestratorSummary, SeasonResult

pytestmark = [pytest.mark.integration, pytest.mark.network]


class TestOrchestratorDryRun:
    """Integration test: full chain dry-run for 2025 season."""

    @pytest.fixture(scope="class")
    def summary(self) -> OrchestratorSummary:
        """
        Run once per test class.
        Fetches from nflreadpy (cached after first call so subsequent runs
        are fast disk reads).
        """
        orch = Orchestrator(db_url="")
        return orch.run(seasons=[2025], dry_run=True)

    # ── Structural checks ─────────────────────────────────────────────────────

    def test_returns_orchestrator_summary(self, summary: OrchestratorSummary) -> None:
        assert isinstance(summary, OrchestratorSummary)

    def test_dry_run_flag_set(self, summary: OrchestratorSummary) -> None:
        assert summary.dry_run is True

    def test_seasons_recorded(self, summary: OrchestratorSummary) -> None:
        assert summary.seasons == [2025]

    def test_one_season_result(self, summary: OrchestratorSummary) -> None:
        assert len(summary.season_results) == 1

    def test_season_result_is_2025(self, summary: OrchestratorSummary) -> None:
        assert summary.season_results[0].season == 2025

    def test_end_time_set(self, summary: OrchestratorSummary) -> None:
        assert summary.end_time is not None

    # ── Step presence and ordering ────────────────────────────────────────────

    def test_exactly_three_steps(self, summary: OrchestratorSummary) -> None:
        assert len(summary.season_results[0].steps) == 3

    def test_step_names_in_order(self, summary: OrchestratorSummary) -> None:
        names = [s.name for s in summary.season_results[0].steps]
        assert names == ["ingest", "normalize", "feature_engineer"]

    # ── All steps succeeded ───────────────────────────────────────────────────

    def test_ingest_step_ok(self, summary: OrchestratorSummary) -> None:
        ingest = summary.season_results[0].steps[0]
        assert ingest.ok, f"ingest failed: {ingest.error}"

    def test_normalize_step_ok(self, summary: OrchestratorSummary) -> None:
        normalize = summary.season_results[0].steps[1]
        assert normalize.ok, f"normalize failed: {normalize.error}"

    def test_feature_engineer_step_ok(self, summary: OrchestratorSummary) -> None:
        fe = summary.season_results[0].steps[2]
        assert fe.ok, f"feature_engineer failed: {fe.error}"

    def test_no_step_has_error_message(self, summary: OrchestratorSummary) -> None:
        for step in summary.season_results[0].steps:
            assert step.error is None, f"Step {step.name!r} has error: {step.error}"

    # ── Steps produced output ─────────────────────────────────────────────────

    def test_ingest_produced_rows(self, summary: OrchestratorSummary) -> None:
        ingest = summary.season_results[0].steps[0]
        assert ingest.rows_out > 0, "ingest produced no validated rows"

    def test_normalize_produced_rows(self, summary: OrchestratorSummary) -> None:
        norm = summary.season_results[0].steps[1]
        assert norm.rows_out > 0, "normalize produced no output rows"

    def test_feature_engineer_produced_rows(self, summary: OrchestratorSummary) -> None:
        fe = summary.season_results[0].steps[2]
        assert fe.rows_out > 0, "feature_engineer produced no feature rows"

    # ── Timing sanity ─────────────────────────────────────────────────────────

    def test_all_steps_have_positive_elapsed(self, summary: OrchestratorSummary) -> None:
        for step in summary.season_results[0].steps:
            assert step.elapsed_s >= 0, f"Step {step.name!r} has negative elapsed"

    # ── Season-level totals ───────────────────────────────────────────────────

    def test_season_result_players_positive(self, summary: OrchestratorSummary) -> None:
        assert summary.season_results[0].n_players > 0

    def test_season_result_games_positive(self, summary: OrchestratorSummary) -> None:
        assert summary.season_results[0].n_games > 0

    def test_season_result_feature_rows_positive(self, summary: OrchestratorSummary) -> None:
        assert summary.season_results[0].n_feature_rows > 0

    # ── Summary-level totals roll up correctly ────────────────────────────────

    def test_total_players_matches_season(self, summary: OrchestratorSummary) -> None:
        assert summary.total_players == summary.season_results[0].n_players

    def test_total_games_matches_season(self, summary: OrchestratorSummary) -> None:
        assert summary.total_games == summary.season_results[0].n_games

    def test_total_feature_rows_matches_season(self, summary: OrchestratorSummary) -> None:
        assert summary.total_feature_rows == summary.season_results[0].n_feature_rows

    # ── Reasonable magnitudes ─────────────────────────────────────────────────

    def test_players_at_least_100(self, summary: OrchestratorSummary) -> None:
        # nflreadpy load_rosters() returns a snapshot (current or end-of-season
        # active roster), not every player who appeared during the year.
        # With gsis_id filter we get ~100–1,700 depending on roster timing.
        assert summary.total_players >= 100

    def test_games_at_least_270(self, summary: OrchestratorSummary) -> None:
        # Regular season: 272 games + playoffs ≥ 270
        assert summary.total_games >= 270

    def test_feature_rows_between_1_and_sample(self, summary: OrchestratorSummary) -> None:
        from pipeline.orchestrator import DRY_RUN_SAMPLE
        # DRY_RUN_SAMPLE unique players × ≥1 game each
        assert 1 <= summary.total_feature_rows
        # Upper bound: sample players × max ~20 games each
        assert summary.total_feature_rows <= DRY_RUN_SAMPLE * 20


class TestGlobalEnrichmentWiring:
    """
    Elo/embedding enrichment and the PBP pipeline write feature_matrix
    columns across all seasons in one pass, not per-season — previously
    only run_full_etl.sh called them, so an orchestrator-only live run
    silently skipped Elo columns and Bucket 11 (PBP) features entirely.
    """

    def test_dry_run_does_not_call_global_enrichment(self) -> None:
        from unittest.mock import patch
        from pipeline.orchestrator import Orchestrator

        with patch("pipeline.orchestrator._run_global_enrichment") as mock_enrich:
            Orchestrator().run(seasons=[2025], dry_run=True)
        mock_enrich.assert_not_called()

    def test_live_run_calls_global_enrichment_after_the_season_loop(self) -> None:
        from unittest.mock import patch
        from pipeline.orchestrator import Orchestrator, SeasonResult

        with patch("pipeline.orchestrator._live_season", return_value=SeasonResult(season=2025, dry_run=False)), \
             patch("pipeline.orchestrator._run_global_enrichment") as mock_enrich:
            Orchestrator(db_url="postgresql://unused").run(seasons=[2025], dry_run=False)
        mock_enrich.assert_called_once_with("postgresql://unused")


class TestOrchestratorSourcesScoping:
    """
    Orchestrator.run(sources=...) must thread through to
    NFLReadPyAdapter.run_full_ingest(sources=...) so a future weekly cron
    can re-ingest only the sources that vary week-to-week instead of doing
    a full, slow re-ingest of every source every run.
    """

    def test_sources_param_threaded_to_run_full_ingest(self) -> None:
        from unittest.mock import MagicMock, patch
        from pipeline.orchestrator import Orchestrator

        mock_adapter = MagicMock()
        mock_adapter.__enter__.return_value = mock_adapter
        mock_adapter.__exit__.return_value = False
        mock_adapter.run_full_ingest.return_value = {}

        with patch("pipeline.orchestrator.NFLReadPyAdapter", return_value=mock_adapter), \
             patch("pipeline.orchestrator.Normalizer") as mock_normalizer_cls, \
             patch("pipeline.orchestrator.FeatureEngineer") as mock_fe_cls, \
             patch("pipeline.orchestrator._run_global_enrichment"):
            mock_norm = MagicMock()
            mock_norm.__enter__.return_value = mock_norm
            mock_norm.__exit__.return_value = False
            mock_norm.run.return_value = MagicMock(
                players_upserted=0, games_upserted=0, game_logs_upserted=0,
            )
            mock_normalizer_cls.return_value = mock_norm

            mock_fe = MagicMock()
            mock_fe.__enter__.return_value = mock_fe
            mock_fe.__exit__.return_value = False
            mock_fe.run.return_value = 0
            mock_fe_cls.return_value = mock_fe

            Orchestrator(db_url="postgresql://unused").run(
                seasons=[2025],
                sources=["player_stats", "rosters"],
            )

        mock_adapter.run_full_ingest.assert_called_once_with(
            seasons=[2025], sources=["player_stats", "rosters"],
        )

    def test_sources_defaults_to_none_unchanged_behavior(self) -> None:
        """
        With no sources argument, run_full_ingest must be called with
        sources=None — identical to pre-change behavior (full ingest).
        """
        from unittest.mock import patch
        from pipeline.orchestrator import Orchestrator, SeasonResult

        with patch(
            "pipeline.orchestrator._live_season",
            return_value=SeasonResult(season=2025, dry_run=False),
        ) as mock_live_season, \
             patch("pipeline.orchestrator._run_global_enrichment"):
            Orchestrator(db_url="postgresql://unused").run(seasons=[2025])

        mock_live_season.assert_called_once_with(
            2025, "postgresql://unused", sources=None,
        )
