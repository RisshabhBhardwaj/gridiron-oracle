import numpy as np
import pandas as pd

from ml.reliability import build_training_manifest, promotion_gate, split_conformal_interval
from ml.market_evaluation import evaluate_over_strategy, no_vig_probability


def test_training_manifest_is_schema_and_data_sensitive() -> None:
    df = pd.DataFrame({
        "player_id": ["a", "b"], "game_id": ["g1", "g2"], "season": [2024, 2025],
        "week": [1, 1], "actual_passing_yards": [200.0, 300.0], "feature": [1.0, np.nan],
    })
    manifest = build_training_manifest(
        df, target="passing_yards", position="QB", target_col="actual_passing_yards",
        feature_columns=["feature"], source_contract={"walk_forward": True},
    )
    assert manifest.row_count == 2
    assert manifest.target_summary["mean"] == 250.0
    assert manifest.null_rates["feature"] == 0.5
    assert len(manifest.training_data_hash) == 64


def test_promotion_requires_strict_held_out_improvement() -> None:
    assert promotion_gate(candidate_mae=9.9, incumbent_mae=10.0)[0]
    assert not promotion_gate(candidate_mae=10.0, incumbent_mae=10.0)[0]
    assert not promotion_gate(candidate_mae=9.0, incumbent_mae=None)[0]


def test_split_conformal_interval_uses_oof_residual_quantile() -> None:
    actual = np.arange(30, dtype=float) + 2.0
    fitted = np.arange(30, dtype=float)
    interval = split_conformal_interval(actual, fitted, np.array([10.0]), coverage=0.8)
    assert interval.radius == 2.0
    assert interval.lower.tolist() == [8.0]
    assert interval.upper.tolist() == [12.0]


def test_market_evaluation_uses_fixed_policy_and_no_vig_probabilities() -> None:
    assert np.isclose(no_vig_probability(np.array([-110]), np.array([-110]))[0], 0.5)
    metrics = evaluate_over_strategy(
        actual=np.array([110.0, 90.0]), line=np.array([100.0, 100.0]),
        model_probability=np.array([0.60, 0.40]), over_odds=np.array([-110.0, -110.0]),
    )
    assert metrics.n_bets == 1
    assert metrics.brier_score >= 0.0
