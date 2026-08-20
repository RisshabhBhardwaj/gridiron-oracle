"""Regression tests for the C-01 as-of feature contract."""

import pandas as pd
import pytest

from ml.feature_contract import (
    FORBIDDEN_MODEL_FIELDS,
    assert_model_input_columns,
    assert_no_same_game_postgame_equality,
)
from ml.utils import FEATURE_COLS, FEATURE_REGISTRY
from pipeline.feature_engineer import build_feature_row


def test_default_features_are_registered_and_never_target_game_snap_fields():
    assert set(FEATURE_COLS) == set(FEATURE_REGISTRY)
    assert not set(FEATURE_COLS).intersection(FORBIDDEN_MODEL_FIELDS)


def test_raw_snap_aliases_fail_closed_at_model_boundary():
    with pytest.raises(AssertionError, match="snap_pct_off"):
        assert_model_input_columns(["kalman_est_targets", "snap_pct_off"], consumer="test")
    with pytest.raises(AssertionError, match="routes_run_pct"):
        assert_model_input_columns(["routes_run_pct"], consumer="test")
    with pytest.raises(AssertionError, match="rec_fantasy_points_exp"):
        assert_model_input_columns(["kalman_est_targets", "rec_fantasy_points_exp"], consumer="test")
    with pytest.raises(AssertionError, match="targets_exp"):
        assert_model_input_columns(["targets_exp"], consumer="test")


def test_prior_snap_share_uses_only_completed_game():
    target = {
        "player_id": "p", "game_id": "g2", "season": 2024, "week": 2,
        "position": "WR", "team": "MIN", "opponent_team": "GB", "offense_pct": 0.99,
    }
    prior = [{**target, "game_id": "g1", "week": 1, "offense_pct": 0.42}]
    game = {"home_team": "MIN", "away_team": "GB", "roof": "dome", "surface": "turf"}
    row = build_feature_row(target, prior, game, prior)
    assert row.prior_snap_share == pytest.approx(0.42)
    assert row.snap_pct_off is None
    assert row.routes_run_pct is None
    assert row.blitz_exposure is None


def test_same_game_equality_audit_catches_direct_copy():
    fm = pd.DataFrame({"player_id": ["p"], "game_id": ["g"], "offense_pct": [0.8]})
    logs = pd.DataFrame({"player_id": ["p"], "game_id": ["g"], "offense_pct": [0.8]})
    with pytest.raises(AssertionError, match="offense_pct"):
        assert_no_same_game_postgame_equality(fm, logs, feature_columns=["offense_pct"])
