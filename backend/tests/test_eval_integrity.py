"""Unit tests for causal baselines, cohort filter, and stat resolution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.baselines import attach_baselines, prev_season_mean, trailing_n_mean
from ml.eval_cohort import CohortSpec, filter_cohort_frame, normalize_offense_pct
from ml.eval_metrics import poisson_deviance, primary_score
from ml.eval_causal import _normalize_oof, score_oof_against_baselines
from ml.stat_resolution import (
    VALID_POSITION_STATS,
    assert_position_stats_resolvable,
    read_gamelog_stat,
    resolve_stat,
)


def test_stat_resolution_startup_assert_passes() -> None:
    assert_position_stats_resolvable(VALID_POSITION_STATS)


def test_pass_attempts_maps_to_attempts() -> None:
    resolved = resolve_stat("pass_attempts")
    assert resolved.gamelog_attr == "attempts"
    gl = type("GL", (), {"attempts": 35})()
    assert read_gamelog_stat(gl, "pass_attempts") == 35.0


def test_fantasy_ppr_maps_to_fantasy_points_ppr() -> None:
    resolved = resolve_stat("fantasy_ppr")
    assert resolved.gamelog_attr == "fantasy_points_ppr"


def test_trailing_mean_is_causal_shift_one() -> None:
    history = pd.DataFrame(
        {
            "player_id": ["p"] * 4,
            "season": [2024] * 4,
            "week": [1, 2, 3, 4],
            "yards": [10.0, 20.0, 30.0, 999.0],
        }
    )
    # Evaluating week 4 may only see weeks 1–3.
    val = trailing_n_mean(
        history, player_id="p", season=2024, week=4, stat_col="yards", n=3
    )
    assert val == pytest.approx(20.0)


def test_prev_season_mean() -> None:
    history = pd.DataFrame(
        {
            "player_id": ["p", "p", "p"],
            "season": [2023, 2023, 2024],
            "week": [1, 2, 1],
            "yards": [10.0, 30.0, 100.0],
        }
    )
    assert prev_season_mean(history, player_id="p", season=2024, stat_col="yards") == 20.0


def test_attach_baselines_columns() -> None:
    history = pd.DataFrame(
        {
            "player_id": ["p"] * 5,
            "season": [2023, 2023, 2024, 2024, 2024],
            "week": [1, 2, 1, 2, 3],
            "yards": [10.0, 30.0, 5.0, 15.0, 25.0],
        }
    )
    eval_rows = history[(history["season"] == 2024) & (history["week"] == 3)].copy()
    out = attach_baselines(eval_rows, history, stat_col="yards", trailing_n=2)
    assert out.iloc[0]["naive_baseline"] == pytest.approx(20.0)
    assert out.iloc[0]["rolling_baseline"] == pytest.approx(10.0)


def test_pregame_cohort_does_not_use_target_snap() -> None:
    assert normalize_offense_pct(85.0) == pytest.approx(0.85)
    assert normalize_offense_pct(0.85) == pytest.approx(0.85)
    frame = pd.DataFrame({
        "player_id": ["p", "p", "p"], "season": [2024] * 3,
        "week": [1, 2, 3], "offense_pct": [0.99, 0.01, 0.99],
    })
    cohort = filter_cohort_frame(frame, spec=CohortSpec(min_prior_games=1))
    assert cohort["week"].tolist() == [2, 3]


def test_count_metric_uses_poisson_deviance() -> None:
    name, score = primary_score(
        "receiving_tds",
        np.array([0.0, 1.0, 0.0]),
        np.array([0.1, 0.2, 0.1]),
    )
    assert name == "poisson_deviance"
    assert score == pytest.approx(poisson_deviance(np.array([0.0, 1.0, 0.0]), np.array([0.1, 0.2, 0.1])))


def test_oof_without_signed_training_provenance_is_rejected() -> None:
    oof = pd.DataFrame({
        "player_id": ["p"], "season": [2024], "week": [2],
        "y_true": [10.0], "y_pred": [9.0],
    })
    with pytest.raises(ValueError, match="max_train_season provenance"):
        _normalize_oof(oof, "yards")


def test_eval_scores_every_predictor_on_one_finite_cohort() -> None:
    # Week 1 is excluded by the explicit pregame cohort. Week 2 has no
    # trailing-three baseline, so it must not contribute to model-only MAE.
    # Week 4 is the sole common finite row.
    oof = pd.DataFrame({
        "player_id": ["p"] * 4,
        "season": [2024] * 4,
        "week": [1, 2, 3, 4],
        "position": ["WR"] * 4,
        "y_true": [1.0, 2.0, 3.0, 40.0],
        "y_pred": [100.0, 100.0, 100.0, 41.0],
        "max_train_season": [2023] * 4,
    })
    history = pd.DataFrame({
        "player_id": ["p"] * 5,
        "season": [2023, 2023, 2023, 2024, 2024],
        "week": [1, 2, 3, 3, 4],
        "yards": [10.0, 20.0, 30.0, 3.0, 40.0],
    })
    result = score_oof_against_baselines(
        oof, history, stat="yards", cohort_spec=CohortSpec(min_prior_games=1)
    )
    row = result.iloc[0]
    assert row["n"] == 1
    assert row["n_model"] > row["n"]
    assert row["model_score"] == pytest.approx(1.0)
