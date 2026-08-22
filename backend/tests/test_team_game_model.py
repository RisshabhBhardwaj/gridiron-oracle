"""Regression guards for ml/team_game_model.py (Phase 4)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from ml.team_game_model import TARGET_COLS, compare_to_vegas, derive_win_probability, train

_DB_URL = os.environ.get("DATABASE_URL", "postgresql://oracle:oracle@localhost:15439/oracle").replace(
    "postgresql+asyncpg://", "postgresql://"
)


def _skip_if_db_unreachable():
    try:
        import psycopg2

        psycopg2.connect(_DB_URL).close()
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable ({exc})")


def test_compare_to_vegas_implied_home_score_formula():
    """
    implied_home_score = (total_line + home_spread) / 2, where home_spread
    is POSITIVE when the home team is favored — this database's convention,
    the opposite of standard Vegas notation (verified:
    corr(spread_line, home_margin) = +0.51 on 285 lined 2025 games). A model
    that predicts the true implied score exactly should show
    model_mae == vegas_implied_mae.
    """
    # Home team favored by 3, total 45 -> implied home=24, away=21.
    holdout = pd.DataFrame([
        {"is_home": 1, "spread_line": 3.0, "total_line": 45.0, "points": 24},
        {"is_home": 0, "spread_line": 3.0, "total_line": 45.0, "points": 21},
    ])
    y_pred = np.array([24.0, 21.0])
    result = compare_to_vegas(holdout, y_pred, "points")
    assert result is not None
    assert result["n_lined_rows"] == 2
    assert abs(result["vegas_implied_mae"]) < 1e-9
    assert abs(result["model_mae"]) < 1e-9


def test_spread_line_sign_convention_is_positive_means_home_favored():
    """
    Regression guard for the inverted-sign bug: compare_to_vegas's implied
    score formula assumed spread_line follows the standard Vegas convention
    (negative = favorite). This database uses the opposite convention. If
    the DB's convention ever changes (or this check runs against a
    differently-sourced games table), this test should fail loudly rather
    than silently serving an implied score with the wrong sign again.
    """
    _skip_if_db_unreachable()
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(_DB_URL)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT spread_line, home_score, away_score FROM games
                WHERE season = 2025 AND spread_line IS NOT NULL
                  AND home_score IS NOT NULL
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    if len(rows) < 30:
        pytest.skip("not enough lined 2025 games in this database to check sign convention")

    spread = np.array([r["spread_line"] for r in rows], dtype=float)
    margin = np.array([r["home_score"] - r["away_score"] for r in rows], dtype=float)
    corr = np.corrcoef(spread, margin)[0, 1]
    assert corr > 0, (
        f"corr(spread_line, home_margin) = {corr:.3f}; expected positive "
        "(positive spread_line means home favored in this DB) — "
        "compare_to_vegas's formula assumes this sign convention"
    )


def test_compare_to_vegas_returns_none_when_no_lines_present():
    holdout = pd.DataFrame([
        {"is_home": 1, "spread_line": None, "total_line": None, "points": 24},
    ])
    assert compare_to_vegas(holdout, np.array([24.0]), "points") is None


def test_compare_to_vegas_skips_unlined_rows_but_scores_lined_ones():
    holdout = pd.DataFrame([
        {"is_home": 1, "spread_line": -3.0, "total_line": 45.0, "points": 24},
        {"is_home": 1, "spread_line": None, "total_line": None, "points": 10},
    ])
    result = compare_to_vegas(holdout, np.array([24.0, 99.0]), "points")
    assert result["n_lined_rows"] == 1


def test_compare_to_vegas_returns_none_for_non_points_targets():
    """
    The implied-score formula only has meaning for points. Fabricating an
    "implied plays" or "implied yards" from spread/total would be worse
    than not comparing at all.
    """
    holdout = pd.DataFrame([
        {"is_home": 1, "spread_line": -3.0, "total_line": 45.0, "total_plays": 65},
    ])
    assert compare_to_vegas(holdout, np.array([65.0]), "total_plays") is None


def test_all_targets_train_end_to_end_against_real_data():
    """
    Smoke test for every TARGET_COLS entry against the live DB — each must
    produce a finite MAE and, for points, a non-None Vegas comparison.
    Doesn't assert on specific MAE values (those are reported honestly to
    the user directly, not pinned as a regression gate that would just get
    silently loosened when the model changes).
    """
    _skip_if_db_unreachable()
    for target in TARGET_COLS:
        result = train(holdout_season=2025, target=target, database_url=_DB_URL)
        assert np.isfinite(result.holdout_mae)
        assert np.isfinite(result.constant_baseline_mae)
        assert result.n_holdout > 0
        if target == "points":
            assert result.vegas_comparison is not None
        else:
            assert result.vegas_comparison is None


def test_derive_win_probability_produces_valid_probabilities():
    _skip_if_db_unreachable()
    result = derive_win_probability(holdout_season=2025, database_url=_DB_URL)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.brier_score >= 0.0
    assert result.n_holdout_games > 0
    # A coherent points-derived win model shouldn't be worse than a coin
    # flip on a full season of holdout games — if it is, something upstream
    # (the points model, the margin_std conversion) is broken, not just weak.
    assert result.accuracy > 0.5
    assert result.brier_score < 0.25  # 0.25 = always-predict-50% baseline
