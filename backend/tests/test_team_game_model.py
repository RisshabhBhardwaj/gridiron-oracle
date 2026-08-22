"""Regression guards for ml/team_game_model.py (Phase 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.team_game_model import compare_to_vegas


def test_compare_to_vegas_implied_home_score_formula():
    """
    implied_home_score = (total_line - home_spread) / 2, where home_spread
    is negative when the home team is favored. A model that predicts the
    true implied score exactly should show model_mae == vegas_implied_mae.
    """
    # Home team favored by 3, total 45 -> implied home=24, away=21.
    holdout = pd.DataFrame([
        {"is_home": 1, "spread_line": -3.0, "total_line": 45.0, "points": 24},
        {"is_home": 0, "spread_line": -3.0, "total_line": 45.0, "points": 21},
    ])
    y_pred = np.array([24.0, 21.0])
    result = compare_to_vegas(holdout, y_pred)
    assert result is not None
    assert result["n_lined_rows"] == 2
    assert abs(result["vegas_implied_mae"]) < 1e-9
    assert abs(result["model_mae"]) < 1e-9


def test_compare_to_vegas_returns_none_when_no_lines_present():
    holdout = pd.DataFrame([
        {"is_home": 1, "spread_line": None, "total_line": None, "points": 24},
    ])
    assert compare_to_vegas(holdout, np.array([24.0])) is None


def test_compare_to_vegas_skips_unlined_rows_but_scores_lined_ones():
    holdout = pd.DataFrame([
        {"is_home": 1, "spread_line": -3.0, "total_line": 45.0, "points": 24},
        {"is_home": 1, "spread_line": None, "total_line": None, "points": 10},
    ])
    result = compare_to_vegas(holdout, np.array([24.0, 99.0]))
    assert result["n_lined_rows"] == 1
