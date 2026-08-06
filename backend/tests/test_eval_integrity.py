"""Unit tests for causal baselines, cohort filter, and stat resolution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.baselines import attach_baselines, prev_season_mean, trailing_n_mean
from ml.eval_cohort import normalize_offense_pct, passes_snap_filter
from ml.eval_metrics import poisson_deviance, primary_score
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


def test_snap_unit_normalization() -> None:
    assert normalize_offense_pct(85.0) == pytest.approx(0.85)
    assert normalize_offense_pct(0.85) == pytest.approx(0.85)
    assert passes_snap_filter(85.0) is True
    assert passes_snap_filter(20.0) is False  # 0.20 after /100
    assert passes_snap_filter(0.20) is False
    assert passes_snap_filter(None) is True


def test_count_metric_uses_poisson_deviance() -> None:
    name, score = primary_score(
        "receiving_tds",
        np.array([0.0, 1.0, 0.0]),
        np.array([0.1, 0.2, 0.1]),
    )
    assert name == "poisson_deviance"
    assert score == pytest.approx(poisson_deviance(np.array([0.0, 1.0, 0.0]), np.array([0.1, 0.2, 0.1])))
