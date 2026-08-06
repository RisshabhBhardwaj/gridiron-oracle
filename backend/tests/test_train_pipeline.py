"""
backend/tests/test_train_pipeline.py

Tests for ml/train.py — PipelineRunner full-stack orchestrator.

All tests use dry_run() which:
  - Uses synthetic player data (no DB required)
  - Bypasses MCMC (fast normal-distribution synthetic samples)
  - Skips DB writes and MLflow logging

Test categories:
  A. Structural — columns, row count, dtypes
  B. Position filter — run only requested positions
  C. DB writes — verify not called in dry_run
  D. MLflow — verify not called in dry_run
  E. Layer ordering — Kalman → Stacking → Bayesian → Monte Carlo
"""

from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.app.core.tracing import get_current_span_id, get_current_trace_id, start_span


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _runner(**kwargs):
    from ml.train import PipelineRunner
    return PipelineRunner(**kwargs)


def _default_cols():
    from ml.train import _PROJECTION_COLUMNS
    return _PROJECTION_COLUMNS


# ---------------------------------------------------------------------------
# A. Structural tests
# ---------------------------------------------------------------------------

class TestDryRunStructure:
    """dry_run() returns correct schema and row count."""

    def test_returns_dataframe(self):
        df = _runner().dry_run(2025, 1)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns_present(self):
        df = _runner().dry_run(2025, 1)
        for col in _default_cols():
            assert col in df.columns, f"missing column: {col}"

    def test_no_extra_columns(self):
        df = _runner().dry_run(2025, 1)
        extra = set(df.columns) - set(_default_cols())
        assert extra == set(), f"unexpected columns: {extra}"

    def test_one_row_per_player_stat(self):
        """Each (player_id, stat) pair should appear exactly once."""
        df = _runner().dry_run(2025, 1)
        assert not df.duplicated(subset=["player_id", "stat"]).any()

    def test_row_count_equals_players_x_stats(self):
        """8 synthetic players × 4 default stats = 32 rows."""
        from ml.train import _DRY_RUN_PLAYERS, _DEFAULT_STATS, _DEFAULT_POSITIONS
        n_players = sum(1 for _, pos in _DRY_RUN_PLAYERS if pos in _DEFAULT_POSITIONS)
        n_stats   = len(_DEFAULT_STATS)
        df = _runner().dry_run(2025, 1)
        assert len(df) == n_players * n_stats

    def test_season_and_week_in_output(self):
        df = _runner().dry_run(2025, 3)
        assert (df["season"] == 2025).all()
        assert (df["week"] == 3).all()

    def test_stat_column_values(self):
        from ml.train import _DEFAULT_STATS
        df = _runner().dry_run(2025, 1)
        assert set(df["stat"].unique()) == set(_DEFAULT_STATS)

    def test_position_column_values(self):
        df = _runner().dry_run(2025, 1)
        assert set(df["position"].unique()).issubset({"QB", "RB", "WR", "TE"})

    def test_projection_numeric(self):
        df = _runner().dry_run(2025, 1)
        assert pd.to_numeric(df["projection"], errors="coerce").notna().all()

    def test_floor_lt_projection_lt_ceiling(self):
        df = _runner().dry_run(2025, 1)
        assert (df["floor"] < df["projection"]).all()
        assert (df["projection"] < df["ceiling"]).all()

    def test_fantasy_floor_lt_projection_lt_ceiling(self):
        df = _runner().dry_run(2025, 1)
        # Stats with 0 pts/unit collapse to 0 — check only non-zero fantasy cols
        mask = df["fantasy_projection"] > 0
        if mask.any():
            sub = df[mask]
            assert (sub["fantasy_floor"] < sub["fantasy_projection"]).all()
            assert (sub["fantasy_projection"] < sub["fantasy_ceiling"]).all()

    def test_boom_plus_bust_le_one(self):
        df = _runner().dry_run(2025, 1)
        assert ((df["boom_probability"] + df["bust_probability"]) <= 1.0 + 1e-9).all()

    def test_probabilities_between_zero_and_one(self):
        df = _runner().dry_run(2025, 1)
        for col in ("boom_probability", "bust_probability"):
            assert (df[col] >= 0.0).all()
            assert (df[col] <= 1.0).all()

    def test_n_samples_column_correct(self):
        from ml.train import _DRY_RUN_N_SAMPLES
        df = _runner().dry_run(2025, 1)
        assert (df["n_samples"] == _DRY_RUN_N_SAMPLES).all()

    def test_player_id_column_dtype_is_string(self):
        df = _runner().dry_run(2025, 1)
        assert df["player_id"].dtype == object  # pandas string columns have object dtype


# ---------------------------------------------------------------------------
# B. Position filter
# ---------------------------------------------------------------------------

class TestPositionFilter:
    """PipelineRunner respects the positions argument."""

    def test_wr_only(self):
        from ml.train import _DRY_RUN_PLAYERS, _DEFAULT_STATS
        df = _runner().dry_run(2025, 1, positions=["WR"])
        assert set(df["position"].unique()) == {"WR"}
        n_wr = sum(1 for _, pos in _DRY_RUN_PLAYERS if pos == "WR")
        assert len(df) == n_wr * len(_DEFAULT_STATS)

    def test_qb_only(self):
        df = _runner().dry_run(2025, 1, positions=["QB"])
        assert set(df["position"].unique()) == {"QB"}

    def test_rb_and_wr(self):
        df = _runner().dry_run(2025, 1, positions=["RB", "WR"])
        assert set(df["position"].unique()) == {"RB", "WR"}

    def test_stat_filter(self):
        df = _runner().dry_run(2025, 1, stats=["receiving_yards"])
        assert set(df["stat"].unique()) == {"receiving_yards"}

    def test_single_stat_row_count(self):
        from ml.train import _DRY_RUN_PLAYERS, _DEFAULT_POSITIONS
        df = _runner().dry_run(2025, 1, stats=["rushing_yards"])
        n_players = sum(1 for _, pos in _DRY_RUN_PLAYERS if pos in _DEFAULT_POSITIONS)
        assert len(df) == n_players

    def test_empty_position_list_returns_empty_df(self):
        df = _runner().dry_run(2025, 1, positions=[])
        assert len(df) == 0

    def test_empty_stats_list_returns_empty_df(self):
        df = _runner().dry_run(2025, 1, stats=[])
        assert len(df) == 0


# ---------------------------------------------------------------------------
# C. DB writes skipped in dry_run
# ---------------------------------------------------------------------------

class TestNoDatabaseWriteInDryRun:
    """_write_db must NOT be called during dry_run()."""

    def test_write_db_not_called(self):
        runner = _runner()
        with patch.object(runner, "_write_db") as mock_write:
            runner.dry_run(2025, 1)
        mock_write.assert_not_called()

    def test_db_session_not_used(self):
        """Even if a db_session is provided, dry_run must not write to it."""
        mock_session = MagicMock()
        runner = _runner(db_session=mock_session)
        runner.dry_run(2025, 1)
        mock_session.add.assert_not_called()
        mock_session.commit.assert_not_called()


# ---------------------------------------------------------------------------
# D. MLflow logging skipped in dry_run
# ---------------------------------------------------------------------------

class TestNoMLflowInDryRun:
    """_log_mlflow must NOT be called during dry_run()."""

    def test_log_mlflow_not_called(self):
        runner = _runner(mlflow_tracking_uri="http://localhost:5000")
        with patch.object(runner, "_log_mlflow") as mock_log:
            runner.dry_run(2025, 1)
        mock_log.assert_not_called()

    def test_mlflow_import_not_triggered(self):
        """dry_run should not import mlflow at all (it may not be installed)."""
        runner = _runner(mlflow_tracking_uri="http://localhost:5000")
        with patch.dict("sys.modules", {"mlflow": None}):
            # Should not raise even if mlflow is removed from sys.modules
            try:
                runner.dry_run(2025, 1)
            except TypeError:
                # MLflow may have been imported at module level somewhere else —
                # what matters is _log_mlflow is not invoked in dry_run
                pass


# ---------------------------------------------------------------------------
# E. Layer ordering
# ---------------------------------------------------------------------------

class TestLayerOrdering:
    """
    Kalman must run before stacking, stacking before Bayesian, Bayesian
    before Monte Carlo. Verified by replacing each step method with a
    wrapper that records its name in a shared call_log list.
    """

    def test_kalman_before_stacking_before_bayesian_before_mc(self):
        runner = _runner()
        call_log: list[str] = []

        orig_kalman   = runner._run_kalman_step
        orig_stacking = runner._run_stacking_step
        orig_bayesian = runner._run_bayesian_step
        orig_mc       = runner._run_mc_step

        def wrap_kalman(*a, **k):
            call_log.append("kalman")
            return orig_kalman(*a, **k)

        def wrap_stacking(*a, **k):
            call_log.append("stacking")
            return orig_stacking(*a, **k)

        def wrap_bayesian(*a, **k):
            call_log.append("bayesian")
            return orig_bayesian(*a, **k)

        def wrap_mc(*a, **k):
            call_log.append("mc")
            return orig_mc(*a, **k)

        runner._run_kalman_step   = wrap_kalman
        runner._run_stacking_step = wrap_stacking
        runner._run_bayesian_step = wrap_bayesian
        runner._run_mc_step       = wrap_mc

        runner.dry_run(2025, 1)

        # Kalman called once before any stacking call
        assert "kalman" in call_log, "kalman step was never called"
        assert "stacking" in call_log, "stacking step was never called"
        assert "bayesian" in call_log, "bayesian step was never called"
        assert "mc"       in call_log, "mc step was never called"

        first_kalman   = call_log.index("kalman")
        first_stacking = call_log.index("stacking")
        first_bayesian = call_log.index("bayesian")
        first_mc       = call_log.index("mc")

        assert first_kalman < first_stacking, (
            f"Kalman ({first_kalman}) must run before stacking ({first_stacking})"
        )
        assert first_stacking < first_bayesian, (
            f"Stacking ({first_stacking}) must run before Bayesian ({first_bayesian})"
        )
        assert first_bayesian < first_mc, (
            f"Bayesian ({first_bayesian}) must run before MC ({first_mc})"
        )

    def test_kalman_called_exactly_once(self):
        """Kalman runs once for all players (not per-stat)."""
        runner = _runner()
        call_log: list[str] = []

        orig = runner._run_kalman_step
        def wrap(*a, **k):
            call_log.append("kalman")
            return orig(*a, **k)
        runner._run_kalman_step = wrap

        runner.dry_run(2025, 1)
        assert call_log.count("kalman") == 1

    def test_stacking_called_once_per_stat_position_pair(self):
        """Stacking runs once per (stat, position) combination."""
        from ml.train import _DEFAULT_STATS, _DEFAULT_POSITIONS
        runner = _runner()
        call_log: list[str] = []

        orig = runner._run_stacking_step
        def wrap(*a, **k):
            call_log.append("stacking")
            return orig(*a, **k)
        runner._run_stacking_step = wrap

        runner.dry_run(2025, 1)
        # At most len(stats) * len(positions) calls
        # (fewer if some position has 0 players, but dry_run always has all 4)
        assert call_log.count("stacking") == len(_DEFAULT_STATS) * len(_DEFAULT_POSITIONS)

    def test_mc_called_once_per_stat_position_pair(self):
        from ml.train import _DEFAULT_STATS, _DEFAULT_POSITIONS
        runner = _runner()
        call_log: list[str] = []

        orig = runner._run_mc_step
        def wrap(*a, **k):
            call_log.append("mc")
            return orig(*a, **k)
        runner._run_mc_step = wrap

        runner.dry_run(2025, 1)
        assert call_log.count("mc") == len(_DEFAULT_STATS) * len(_DEFAULT_POSITIONS)


class TestTracingPropagation:
    def test_pipeline_layer_spans_share_trace_id_and_use_distinct_child_spans(self):
        runner = _runner()
        seen: list[tuple[str, str | None, str | None]] = []

        orig_kalman = runner._run_kalman_step
        orig_stacking = runner._run_stacking_step
        orig_bayesian = runner._run_bayesian_step
        orig_mc = runner._run_mc_step

        def wrap(name, fn):
            def inner(*a, **k):
                seen.append((name, get_current_trace_id(), get_current_span_id()))
                return fn(*a, **k)
            return inner

        runner._run_kalman_step = wrap("kalman", orig_kalman)
        runner._run_stacking_step = wrap("stacking", orig_stacking)
        runner._run_bayesian_step = wrap("bayesian", orig_bayesian)
        runner._run_mc_step = wrap("mc", orig_mc)

        with start_span("test.pipeline.root", tracer_name="backend.tests"):
            root_trace_id = get_current_trace_id()
            root_span_id = get_current_span_id()
            runner.dry_run(2025, 1, positions=["WR"], stats=["receiving_yards"])

        assert root_trace_id is not None
        assert root_span_id is not None
        assert [name for name, _, _ in seen] == ["kalman", "stacking", "bayesian", "mc"]
        assert {trace_id for _, trace_id, _ in seen} == {root_trace_id}
        child_span_ids = [span_id for _, _, span_id in seen]
        assert all(span_id is not None for span_id in child_span_ids)
        assert len(set(child_span_ids)) == 4
        assert root_span_id not in set(child_span_ids)


# ---------------------------------------------------------------------------
# F. Output consistency
# ---------------------------------------------------------------------------

class TestOutputConsistency:
    """Validate output values make sense end-to-end."""

    def test_all_player_ids_are_synthetic_dry_run(self):
        """dry_run should only produce rows for synthetic players."""
        df = _runner().dry_run(2025, 1)
        assert df["player_id"].str.startswith("dry_").all()

    def test_game_id_contains_season_and_week(self):
        df = _runner().dry_run(2025, 5)
        # Synthetic game_id contains season and week
        assert df["game_id"].str.contains("2025").all()
        assert df["game_id"].str.contains("05").all()

    def test_projections_are_finite(self):
        df = _runner().dry_run(2025, 1)
        for col in ("projection", "floor", "ceiling"):
            assert np.isfinite(df[col]).all(), f"{col} has non-finite values"

    def test_different_weeks_give_different_game_ids(self):
        df1 = _runner().dry_run(2025, 1)
        df2 = _runner().dry_run(2025, 2)
        # Game IDs should differ between weeks
        assert not (df1["game_id"].values == df2["game_id"].values).all()

    def test_custom_positions_only_in_output(self):
        df = _runner().dry_run(2025, 1, positions=["TE"])
        assert set(df["position"].unique()) == {"TE"}

    def test_custom_stats_only_in_output(self):
        df = _runner().dry_run(2025, 1, stats=["passing_yards", "rushing_yards"])
        assert set(df["stat"].unique()) == {"passing_yards", "rushing_yards"}


# ---------------------------------------------------------------------------
# G. Stacking inference helpers (_build_inference_features, _load_ridge_coefs,
#    _run_stacking_step fallback / MLflow paths)
# ---------------------------------------------------------------------------

class TestStackingInferenceHelpers:
    """Unit tests for the new stacking inference plumbing in PipelineRunner."""

    # ── _build_inference_features ────────────────────────────────────────────

    def _minimal_kalman_df(self) -> "pd.DataFrame":
        """Return a 3-row kalman_df with all kalman_est_* columns."""
        from ml.utils import FEATURE_COLS
        data = {
            "player_id": ["p1", "p2", "p3"],
            "game_id":   ["g1", "g2", "g3"],
            "season":    [2025, 2025, 2025],
            "week":      [1, 1, 1],
            "position":  ["WR", "WR", "WR"],
        }
        # Populate every feature column
        for col in FEATURE_COLS:
            data[col] = [10.0, 20.0, 30.0]
        return pd.DataFrame(data)

    def test_build_inference_features_has_feature_cols(self):
        """Output must contain exactly FEATURE_COLS."""
        from ml.xgb_model import FEATURE_COLS
        runner = _runner()
        X_df = runner._build_inference_features(self._minimal_kalman_df())
        assert list(X_df.columns) == FEATURE_COLS

    def test_build_inference_features_maps_kalman_to_form(self):
        """kalman_est_receiving_yards → kalman_est_receiving_yards testing."""
        runner = _runner()
        kalman_df = self._minimal_kalman_df()
        kalman_df["kalman_est_receiving_yards"] = [11.0, 22.0, 33.0]
        X_df = runner._build_inference_features(kalman_df)
        assert list(X_df["kalman_est_receiving_yards"]) == [11.0, 22.0, 33.0]

    def test_build_inference_features_missing_cols_become_zero(self):
        """Buckets 2-7 columns not in kalman_df default to 0."""
        runner = _runner()
        # Minimal df — no seas_*, opp_*, etc.
        df = pd.DataFrame({
            "player_id": ["p1"],
            "kalman_est_receiving_yards": [50.0],
        })
        X_df = runner._build_inference_features(df)
        assert (X_df["seas_games_played"] == 0.0).all()
        assert (X_df["opp_avg_receiving_yards_allowed"] == 0.0).all()

    def test_build_inference_features_no_nans(self):
        """All values must be finite — no NaN in output."""
        import numpy as np
        runner = _runner()
        X_df = runner._build_inference_features(self._minimal_kalman_df())
        assert not X_df.isnull().any().any()
        assert np.isfinite(X_df.values).all()

    def test_build_inference_features_row_count_preserved(self):
        runner = _runner()
        df = self._minimal_kalman_df()
        X_df = runner._build_inference_features(df)
        assert len(X_df) == len(df)

    # ── _load_ridge_coefs ────────────────────────────────────────────────────

    def test_load_ridge_coefs_returns_none_when_missing(self, tmp_path):
        from ml.train import PipelineRunner
        runner = PipelineRunner(oof_dir=tmp_path)
        assert runner._load_ridge_coefs("receiving_yards") is None

    def test_load_ridge_coefs_reads_and_normalizes(self, tmp_path):
        import json
        from ml.train import PipelineRunner
        coef_path = tmp_path / "ridge_receiving_yards_coefs.json"
        coef_path.write_text(json.dumps({
            "xgb": 2.0, "lgbm": 1.0, "tft": 1.0,
            "intercept": 0.5,
        }))
        runner = PipelineRunner(oof_dir=tmp_path)
        result = runner._load_ridge_coefs("receiving_yards")
        assert result is not None
        coefs, intercept = result
        assert len(coefs) == 3
        assert coefs == [2.0, 1.0, 1.0]
        assert abs(intercept - 0.5) < 1e-9

    def test_load_ridge_coefs_corrupt_file_returns_none(self, tmp_path):
        from ml.train import PipelineRunner
        coef_path = tmp_path / "ridge_rushing_yards_coefs.json"
        coef_path.write_text("not valid json {{{")
        runner = PipelineRunner(oof_dir=tmp_path)
        assert runner._load_ridge_coefs("rushing_yards") is None

    def test_load_ridge_coefs_requires_position_specific_file_when_position_given(self, tmp_path):
        import json
        from ml.train import PipelineRunner
        (tmp_path / "ridge_passing_yards_coefs.json").write_text(json.dumps({
            "xgb": 1.0, "lgbm": 1.0, "catboost": 1.0, "tft": 1.0, "intercept": 0.0,
        }))
        runner = PipelineRunner(oof_dir=tmp_path)
        assert runner._load_ridge_coefs("passing_yards", position="QB") is None

    # ── _run_stacking_step — no MLflow configured ────────────────────────────

    def test_stacking_no_mlflow_returns_kalman_est(self):
        """Default runner (mlflow_tracking_uri='') returns kalman_est_{stat}."""
        runner = _runner()  # mlflow_tracking_uri="" by default
        df = pd.DataFrame({
            "player_id": ["p1", "p2"],
            "kalman_est_receiving_yards": [55.0, 70.0],
        })
        result = runner._run_stacking_step(df, "receiving_yards", "WR")
        assert list(result) == [55.0, 70.0]

    def test_stacking_no_mlflow_missing_col_returns_zeros(self):
        """If kalman_est_{stat} absent and no MLflow → all-zeros."""
        runner = _runner()
        df = pd.DataFrame({"player_id": ["p1", "p2"]})
        result = runner._run_stacking_step(df, "receiving_yards", "WR")
        assert list(result) == [0.0, 0.0]

    def test_stacking_no_mlflow_never_imports_mlflow(self):
        """Ensure MLflow is never imported when tracking URI is empty."""
        import sys
        runner = _runner()
        df = pd.DataFrame({"player_id": ["p1"], "kalman_est_passing_yards": [200.0]})
        # Removing mlflow from sys.modules must not cause any error.
        saved = sys.modules.pop("mlflow", None)
        try:
            result = runner._run_stacking_step(df, "passing_yards", "QB")
            assert result[0] == 200.0
        finally:
            if saved is not None:
                sys.modules["mlflow"] = saved

    # ── _run_stacking_step — MLflow path falls back gracefully ───────────────

    def test_stacking_mlflow_exception_falls_back_to_kalman(self):
        """If _load_and_run_stacking raises, fall back to kalman_est and warn."""
        import logging
        from unittest.mock import patch
        from ml.train import PipelineRunner

        runner = PipelineRunner(mlflow_tracking_uri="http://fake:9999")
        df = pd.DataFrame({
            "player_id": ["p1"],
            "kalman_est_receiving_yards": [42.0],
        })
        # Force _load_and_run_stacking to raise a connection error.
        with patch.object(runner, "_load_and_run_stacking",
                          side_effect=ConnectionError("MLflow unreachable")):
            result = runner._run_stacking_step(df, "receiving_yards", "WR")
        assert result[0] == 42.0

    def test_stacking_mlflow_no_models_raises_then_falls_back(self):
        """RuntimeError from _load_and_run_stacking triggers kalman fallback."""
        from unittest.mock import patch
        from ml.train import PipelineRunner

        runner = PipelineRunner(mlflow_tracking_uri="http://fake:9999")
        df = pd.DataFrame({
            "player_id": ["p1", "p2"],
            "kalman_est_rushing_yards": [15.0, 25.0],
        })
        with patch.object(runner, "_load_and_run_stacking",
                          side_effect=RuntimeError("No XGB or LGB model artifacts")):
            result = runner._run_stacking_step(df, "rushing_yards", "RB")
        assert list(result) == [15.0, 25.0]

    def test_stacking_mlflow_with_mock_models_uses_their_predictions(self):
        """When real models are injected, their predictions flow through."""
        from unittest.mock import MagicMock, patch
        from ml.train import PipelineRunner

        runner = PipelineRunner(mlflow_tracking_uri="http://fake:9999")
        runner._mlflow_reachable = True
        df = pd.DataFrame({
            "player_id": ["p1", "p2"],
            "kalman_est_receiving_yards": [10.0, 20.0],
        })

        mock_xgb = MagicMock()
        mock_xgb.predict.return_value = np.array([100.0, 200.0])
        mock_xgb.n_features_in_ = 2
        mock_xgb.feature_names_in_ = ["1", "2"]
        
        mock_lgbm = MagicMock()
        mock_lgbm.predict.return_value = np.array([90.0, 180.0])
        mock_lgbm.n_features_in_ = 2
        mock_lgbm.feature_names_in_ = ["1", "2"]

        with patch.object(runner, "_load_latest_mlflow_model") as mock_load, \
             patch("mlflow.set_tracking_uri"), \
             patch("mlflow.xgboost", create=True), \
             patch("mlflow.lightgbm", create=True):
            # Return XGB for "xgb", LGB for "lgbm".
            mock_load.side_effect = (
                lambda learner, stat, position=None:
                mock_xgb if learner == "xgb" else (mock_lgbm if learner == "lgbm" else None)
            )
            result = runner._run_stacking_step(df, "receiving_yards", "WR")

        # Mean of [xgb_pred, lgbm_pred, kalman_tft_proxy] = mean([100,90,10], [200,180,20])
        assert abs(result[0] - np.mean([100.0, 90.0, 10.0])) < 0.01
        assert abs(result[1] - np.mean([200.0, 180.0, 20.0])) < 0.01
