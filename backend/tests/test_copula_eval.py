import math

import pandas as pd

from ml.copula_eval import independence_overstates_joint, pairwise_residual_correlation


def test_pairwise_residual_correlation_needs_overlap() -> None:
    frame = pd.DataFrame([
        {"player_id": "a", "game_id": f"g{i}", "y_pred": 10.0, "y_true": 12.0}
        for i in range(3)
    ] + [
        {"player_id": "b", "game_id": f"g{i}", "y_pred": 8.0, "y_true": 9.0}
        for i in range(3)
    ])
    value = pairwise_residual_correlation(frame, player_a="a", player_b="b")
    assert math.isnan(value)


def test_independence_overstates_when_joint_is_far() -> None:
    assert independence_overstates_joint(0.5, 0.5, 0.40) is True
    assert independence_overstates_joint(0.5, 0.5, 0.25) is False
