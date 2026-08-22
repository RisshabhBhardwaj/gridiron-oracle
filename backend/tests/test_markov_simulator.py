"""Tests for ml.markov_simulator's game-state-aware Markov fit (Phase 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.markov_simulator import (
    N_QUARTER,
    N_SCORE_BUCKETS,
    DriveMarkovModel,
    _score_diff_bucket,
)


def _synthetic_pbp(n_per_cell: int = 200) -> pd.DataFrame:
    """
    Build a synthetic frame where leading-big teams pass 20% of the time
    and trailing-big teams pass 80% of the time, constant across quarters,
    at a single (fp, down, ytg) state. Every other score bucket passes 50%.
    Used to check the fit recovers a known, exact ground truth rather than
    just "looks plausible" on real data.
    """
    rng = np.random.default_rng(0)
    rows = []
    pass_rate_by_bucket = {0: 0.8, 1: 0.6, 2: 0.5, 3: 0.4, 4: 0.2}  # trailing_big..leading_big
    score_by_bucket = {0: -14, 1: -4, 2: 0, 3: 4, 4: 14}
    for quarter in range(1, 5):
        for bucket, p_pass in pass_rate_by_bucket.items():
            is_pass = rng.random(n_per_cell) < p_pass
            for ip in is_pass:
                rows.append({
                    "play_type": "pass" if ip else "run",
                    "down": 1,
                    "ydstogo": 10,
                    "yardline_100": 50,
                    "quarter": quarter,
                    "score_differential": score_by_bucket[bucket],
                    "yards_gained": rng.normal(5.0, 4.0),
                    "interception": 0,
                    "fumble_lost": 0,
                    "penalty": 0,
                    "penalty_yards": 0,
                })
    return pd.DataFrame(rows)


class TestScoreDiffBucket:
    def test_buckets_match_expected_thresholds(self):
        sd = pd.Series([-20, -9, -8, -1, 0, 1, 8, 9, 20])
        buckets = _score_diff_bucket(sd)
        assert list(buckets) == [0, 0, 1, 1, 2, 3, 3, 4, 4]


class TestGameStateFit:
    def test_fitted_p_pass_recovers_known_asymmetry(self):
        pbp = _synthetic_pbp(n_per_cell=500)
        model = DriveMarkovModel().fit(pbp)
        gs = model.transitions_by_game_state
        assert gs is not None

        q4 = gs[gs["quarter_idx"] == 3]
        leading_big = q4[q4["score_diff_bucket"] == 4]["p_pass"].iloc[0]
        trailing_big = q4[q4["score_diff_bucket"] == 0]["p_pass"].iloc[0]
        # True rates are 0.2 / 0.8; with n=500/cell and light shrinkage
        # (k=50) the fit should land close, well inside a wide tolerance.
        assert leading_big < 0.35
        assert trailing_big > 0.65
        assert leading_big < trailing_big

    def test_missing_game_state_columns_yields_none_not_silent_degradation(self):
        pbp = pd.DataFrame({
            "yardline_100": [50] * 10,
            "down": [1] * 10,
            "ydstogo": [10] * 10,
            "yards_gained": [5.0] * 10,
        })
        model = DriveMarkovModel().fit(pbp)
        assert model.transitions is not None  # 3D fit still works
        assert model.transitions_by_game_state is None  # game-state fit correctly bails

    def test_sparse_cell_shrinks_toward_marginal(self):
        """
        A (fp, down, ytg, score, quarter) cell with a single observed play
        (an extreme outcome, e.g. one interception) should land close to
        the (fp, down, ytg) marginal, not take that one play at face value.
        """
        rng = np.random.default_rng(1)
        rows = []
        # Dense marginal at fp=50/down=1/ytg=10: 50% pass, never turns over.
        for quarter in range(1, 5):
            for _ in range(300):
                rows.append({
                    "play_type": "pass" if rng.random() < 0.5 else "run",
                    "down": 1, "ydstogo": 10, "yardline_100": 50,
                    "quarter": quarter, "score_differential": 0,
                    "yards_gained": rng.normal(5.0, 4.0),
                    "interception": 0, "fumble_lost": 0, "penalty": 0, "penalty_yards": 0,
                })
        # One single outlier play in a rare cell: leading_big (score=20) in Q1,
        # an interception on what was a pass.
        rows.append({
            "play_type": "pass", "down": 1, "ydstogo": 10, "yardline_100": 50,
            "quarter": 1, "score_differential": 20,
            "yards_gained": 0.0, "interception": 1, "fumble_lost": 0, "penalty": 0, "penalty_yards": 0,
        })
        pbp = pd.DataFrame(rows)
        model = DriveMarkovModel().fit(pbp)
        gs = model.transitions_by_game_state
        rare_cell = gs[(gs["quarter_idx"] == 0) & (gs["score_diff_bucket"] == 4)]
        assert len(rare_cell) == 1
        # Raw cell p_turnover would be 1.0 (its one play intercepted); shrunk
        # toward a ~0.0 marginal with k=50 vs n=1, it should be nowhere near 1.0.
        assert rare_cell["p_turnover"].iloc[0] < 0.1

    def test_export_transitions_by_game_state_writes_expected_columns(self, tmp_path):
        pbp = _synthetic_pbp(n_per_cell=50)
        model = DriveMarkovModel().fit(pbp)
        out = tmp_path / "transitions_by_game_state.csv"
        model.export_transitions_by_game_state(str(out))
        written = pd.read_csv(out)
        expected_cols = {
            "fp_bucket", "down_idx", "ytg_bucket", "score_diff_bucket", "quarter_idx",
            "mean_gain", "gain_std", "p_turnover",
            "p_penalty_gain", "p_penalty_loss", "p_pass", "count",
        }
        assert expected_cols.issubset(set(written.columns))
        assert (written["score_diff_bucket"].between(0, N_SCORE_BUCKETS - 1)).all()
        assert (written["quarter_idx"].between(0, N_QUARTER - 1)).all()

    def test_export_transitions_by_game_state_without_fit_raises(self):
        model = DriveMarkovModel()
        with pytest.raises(RuntimeError):
            model.export_transitions_by_game_state("/tmp/should_not_be_written.csv")

    def test_original_3d_export_unaffected_by_game_state_extension(self, tmp_path):
        """Backward-compat guard: the original C++-consumed CSV format is untouched."""
        pbp = _synthetic_pbp(n_per_cell=50)
        model = DriveMarkovModel().fit(pbp)
        out = tmp_path / "transitions.csv"
        model.export_transitions(str(out))
        written = pd.read_csv(out)
        assert list(written.columns) == [
            "fp_bucket", "down_idx", "ytg_bucket",
            "mean_gain", "gain_std", "p_turnover",
            "p_penalty_gain", "p_penalty_loss",
        ]
