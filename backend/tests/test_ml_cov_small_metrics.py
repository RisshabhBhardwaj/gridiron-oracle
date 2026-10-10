"""
backend/tests/test_ml_cov_small_metrics.py

Coverage tests for small pure modules with known-answer expectations:
  - ml/eval_metrics.py          (MAE, pinball, Poisson deviance, CRPS dispatch)
  - ml/consensus_baseline.py    (ADP ranks, top-N overlap, points lost vs oracle)
  - ml/stat_resolution.py       (stat-name resolution and fumble sums)
  - ml/cohort_guards.py         (training cohort sanity checks)
  - ml/bdb_tracking.py          (Big Data Bowl tracking loader, fails closed)

Every expected value is worked out by hand in the comments.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml import bdb_tracking as bdb
from ml import cohort_guards as cg
from ml import consensus_baseline as cb
from ml import eval_metrics as em
from ml import stat_resolution as sr


# ── eval_metrics ──────────────────────────────────────────────────────────────

class TestEvalMetrics:
    def test_mae_is_mean_absolute_error(self) -> None:
        assert em.mae(np.array([3.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)

    def test_pinball_at_median_is_half_mae(self) -> None:
        # deltas [2, 0]: tau=0.5 gives max(1, -1)=1 and 0, mean 0.5 = MAE/2.
        assert em.pinball_loss(np.array([3.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(0.5)

    def test_pinball_asymmetric_quantile(self) -> None:
        # tau=0.9, deltas [2, -1]: max(1.8, -0.2)=1.8 and max(-0.9, 0.1)=0.1, mean 0.95.
        actual = np.array([3.0, 0.0])
        predicted = np.array([1.0, 1.0])
        assert em.pinball_loss(actual, predicted, tau=0.9) == pytest.approx(0.95)

    def test_poisson_deviance_zero_for_perfect_forecast(self) -> None:
        assert em.poisson_deviance(np.array([2.0, 5.0]), np.array([2.0, 5.0])) == pytest.approx(0.0)

    def test_poisson_deviance_zero_count_term(self) -> None:
        # y=0, mu=1: 2*(0 - (0 - 1)) = 2.
        assert em.poisson_deviance(np.array([0.0]), np.array([1.0])) == pytest.approx(2.0)

    def test_poisson_deviance_known_value(self) -> None:
        # y=2, mu=1: 2*(2*ln2 - 1) = 2*(1.386294 - 1) = 0.772589.
        assert em.poisson_deviance(np.array([2.0]), np.array([1.0])) == pytest.approx(0.7725887, rel=1e-6)

    def test_poisson_clips_predictions_away_from_zero(self) -> None:
        value = em.poisson_deviance(np.array([1.0]), np.array([0.0]))
        assert np.isfinite(value)

    def test_crps_dispatches_to_backtest_implementation(self) -> None:
        # Reference from ml/backtest.py docstring: samples 0..4 vs observation 2 -> CRPS 0.4.
        assert em.crps_from_samples(np.arange(5.0), 2.0) == pytest.approx(0.4)

    def test_primary_score_uses_poisson_for_counts(self) -> None:
        name, value = em.primary_score("passing_tds", np.array([1.0]), np.array([1.0]))
        assert name == "poisson_deviance"
        assert value == pytest.approx(0.0)

    def test_primary_score_uses_mae_for_yardage(self) -> None:
        name, value = em.primary_score("receiving_yards", np.array([10.0]), np.array([7.0]))
        assert name == "mae"
        assert value == pytest.approx(3.0)


# ── consensus_baseline ────────────────────────────────────────────────────────

class TestConsensusBaseline:
    def test_adp_ranks_ascending_with_average_ties_and_nan_kept(self) -> None:
        frame = pd.DataFrame({"adp": [10.0, 2.0, np.nan, 5.0, 5.0]})
        ranks = cb.adp_ranks(frame).to_numpy()
        assert ranks[1] == 1.0
        assert ranks[3] == pytest.approx(2.5) and ranks[4] == pytest.approx(2.5)  # tie averaged
        assert ranks[0] == 4.0
        assert np.isnan(ranks[2])

    def test_adp_ranks_accepts_custom_column(self) -> None:
        frame = pd.DataFrame({"market": ["3", "1", "2"]})
        assert list(cb.adp_ranks(frame, adp_col="market")) == [3.0, 1.0, 2.0]

    def test_top_n_identical_rankings_overlap_fully(self) -> None:
        ranks = np.arange(1.0, 31.0)
        assert cb.top_n_hit_rate(ranks, ranks, n=24) == pytest.approx(1.0)

    def test_top_n_partial_overlap(self) -> None:
        # Top-2 model = {0, 1}; top-2 market = {0, 2}; overlap 1 of 2.
        model = [1.0, 2.0, 3.0, 4.0]
        market = [1.0, 4.0, 2.0, 3.0]
        assert cb.top_n_hit_rate(model, market, n=2) == pytest.approx(0.5)

    def test_top_n_too_few_finite_values_is_nan(self) -> None:
        assert np.isnan(cb.top_n_hit_rate([1.0, np.nan], [1.0, 2.0], n=2))

    def test_points_lost_zero_when_projection_orders_realized_points(self) -> None:
        realized = np.array([10.0, 9.0, 1.0])
        assert cb.points_lost_vs_optimal(realized, realized, slots=2) == pytest.approx(0.0)

    def test_points_lost_known_value(self) -> None:
        # Projection picks indices 2 (proj 9) and 1 (proj 5): realized 1 + 9 = 10.
        # Oracle takes the two best realized values: 10 + 9 = 19. Lost = 9.
        projection = [0.0, 5.0, 9.0]
        realized = [10.0, 9.0, 1.0]
        assert cb.points_lost_vs_optimal(projection, realized, slots=2) == pytest.approx(9.0)

    def test_points_lost_nan_when_too_few_players(self) -> None:
        assert np.isnan(cb.points_lost_vs_optimal([1.0], [1.0], slots=2))


# ── stat_resolution ───────────────────────────────────────────────────────────

class TestStatResolution:
    def test_resolve_known_stat_maps_gamelog_attribute(self) -> None:
        resolved = sr.resolve_stat("pass_attempts")
        assert resolved.gamelog_attr == "attempts"
        assert resolved.kalman_attr == "kalman_est_pass_attempts"
        assert resolved.is_multi_source is False

    def test_resolve_fumbles_is_multi_source(self) -> None:
        resolved = sr.resolve_stat("fumbles")
        assert resolved.is_multi_source is True
        assert resolved.gamelog_attr == "_fumbles_combined"

    def test_resolve_unknown_stat_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown model stat 'nonsense'"):
            sr.resolve_stat("nonsense")

    def test_read_attribute_from_object_and_mapping(self) -> None:
        obj = SimpleNamespace(receiving_yards=77)
        assert sr.read_gamelog_stat(obj, "receiving_yards") == pytest.approx(77.0)
        assert sr.read_gamelog_stat({"receiving_yards": 12}, "receiving_yards") == pytest.approx(12.0)
        assert sr.read_gamelog_stat({"receiving_yards": None}, "receiving_yards") is None
        assert sr.read_gamelog_stat(SimpleNamespace(), "receiving_yards") is None

    def test_fumble_total_from_orm_object_skips_missing_components(self) -> None:
        obj = SimpleNamespace(rushing_fumbles=1.0, receiving_fumbles=None, sack_fumbles=2.0)
        assert sr.read_gamelog_stat(obj, "fumbles") == pytest.approx(3.0)

    def test_fumble_total_from_mapping(self) -> None:
        assert sr.read_gamelog_stat({"rushing_fumbles": 0.0, "sack_fumbles": 1.0}, "fumbles") == pytest.approx(1.0)
        assert sr.read_gamelog_stat({"rushing_fumbles": None}, "fumbles") is None

    def test_default_matrix_is_resolvable(self) -> None:
        assert sr.assert_position_stats_resolvable() is None

    def test_unresolvable_entries_are_reported_together(self) -> None:
        with pytest.raises(AssertionError) as exc:
            sr.assert_position_stats_resolvable({"WR": ["receiving_yards", "bogus_stat"]})
        assert "WR/bogus_stat" in str(exc.value)

    def test_invalid_kalman_attribute_is_reported(self, monkeypatch) -> None:
        monkeypatch.setitem(sr.KALMAN_EST_MAP, "targets", "estimate_targets")
        with pytest.raises(AssertionError, match="invalid kalman attr 'estimate_targets'"):
            sr.assert_position_stats_resolvable({"WR": ["targets"]})

    @pytest.mark.parametrize("stat, family", [
        ("receiving_yards", "continuous"),
        ("fantasy_ppr", "continuous"),
        ("receptions", "count"),
        ("interceptions", "count"),
        ("made_up", "continuous"),
    ])
    def test_metric_family(self, stat, family) -> None:
        assert sr.target_metric_family(stat) == family


# ── cohort_guards ─────────────────────────────────────────────────────────────

def _cohort(positions, targets):
    return pd.DataFrame({"position": positions, "actual_passing_yards": targets})


class TestCohortGuards:
    def test_empty_frame_is_refused(self) -> None:
        with pytest.raises(ValueError, match="training dataframe is empty"):
            cg.validate_training_cohort(pd.DataFrame(), target="receiving_yards",
                                        target_col="actual_receiving_yards",
                                        position_filter="WR", source="test")

    def test_strict_target_requires_its_position(self) -> None:
        df = _cohort(["QB"], [250.0])
        with pytest.raises(ValueError, match="must be trained with position_filter=QB"):
            cg.validate_training_cohort(df, target="passing_yards", target_col="actual_passing_yards",
                                        position_filter="WR", source="test")

    def test_strict_target_rejects_contaminated_positions(self) -> None:
        df = _cohort(["QB", "RB"], [250.0, 260.0])
        with pytest.raises(ValueError, match="contaminated positions"):
            cg.validate_training_cohort(df, target="passing_yards", target_col="actual_passing_yards",
                                        position_filter="QB", source="test")

    def test_high_zero_rate_is_refused(self) -> None:
        # 3 of 4 zeros = 75% > the 20% passing_yards threshold.
        df = _cohort(["QB"] * 4, [0.0, 0.0, 0.0, 300.0])
        with pytest.raises(ValueError, match="zero-rate 75.0% exceeds threshold 20.0%"):
            cg.validate_training_cohort(df, target="passing_yards", target_col="actual_passing_yards",
                                        position_filter="QB", source="test")

    def test_acceptable_cohort_passes_and_logs_zero_rate(self, caplog) -> None:
        df = _cohort(["QB"] * 5, [200.0, 210.0, 190.0, 0.0, 220.0])  # 20% zeros, at threshold
        with caplog.at_level(logging.INFO, logger=cg.logger.name):
            cg.validate_training_cohort(df, target="passing_yards", target_col="actual_passing_yards",
                                        position_filter="QB", source="test")
        assert "zero-rate=20.00%" in caplog.text

    def test_unguarded_target_has_no_checks(self) -> None:
        df = _cohort(["RB"], [0.0])
        cg.validate_training_cohort(df, target="receiving_yards", target_col="actual_receiving_yards",
                                    position_filter=None, source="test")


# ── bdb_tracking ──────────────────────────────────────────────────────────────

class TestBdbTracking:
    def test_missing_file_fails_closed(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="BDB tracking file missing"):
            bdb.load_tracking_week(tmp_path / "tracking_week_1.csv")

    def test_missing_columns_are_named(self, tmp_path) -> None:
        path = tmp_path / "tracking_week_1.csv"
        path.write_text("gameId,playId,x\n1,2,3\n")
        with pytest.raises(ValueError, match=r"missing tracking columns \['event', 'nflId', 'y'\]"):
            bdb.load_tracking_week(path)

    def test_valid_file_loads(self, tmp_path) -> None:
        path = tmp_path / "tracking_week_1.csv"
        path.write_text("gameId,playId,nflId,x,y,event\n1,2,99,10.5,20.0,pass_arrived\n")
        frame = bdb.load_tracking_week(path)
        assert len(frame) == 1
        assert frame.loc[0, "nflId"] == 99
        assert frame.loc[0, "event"] == "pass_arrived"

    def test_availability_requires_a_directory(self, tmp_path) -> None:
        assert bdb.tracking_available(tmp_path / "absent") is False
        empty = tmp_path / "empty"
        empty.mkdir()
        assert bdb.tracking_available(empty) is False

    def test_availability_from_weekly_csv_or_parquet(self, tmp_path) -> None:
        weekly = tmp_path / "weekly"
        weekly.mkdir()
        (weekly / "tracking_week_3.csv").write_text("x\n")
        assert bdb.tracking_available(weekly) is True

        parquet = tmp_path / "parquet"
        parquet.mkdir()
        (parquet / "games.parquet").write_bytes(b"PAR1")
        assert bdb.tracking_available(parquet) is True

    def test_default_directory_is_under_data_bdb(self) -> None:
        assert bdb.DEFAULT_BDB_DIR.parts[-2:] == ("data", "bdb")
