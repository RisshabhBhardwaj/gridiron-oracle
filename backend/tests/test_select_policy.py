from scripts.select_nonregressing_stacks import _mean_variance_fallback


def test_single_season_loss_does_not_force_identity() -> None:
    seasonal = [
        {"candidate_mae": 6.5, "baseline_mae": 6.4},
        {"candidate_mae": 5.8, "baseline_mae": 6.0},
        {"candidate_mae": 6.0, "baseline_mae": 6.1},
        {"candidate_mae": 6.3, "baseline_mae": 6.3},
        {"candidate_mae": 6.6, "baseline_mae": 6.7},
    ]
    assert _mean_variance_fallback(seasonal) is False


def test_persistently_worse_mean_falls_back() -> None:
    seasonal = [
        {"candidate_mae": 8.0, "baseline_mae": 6.0},
        {"candidate_mae": 8.2, "baseline_mae": 6.1},
        {"candidate_mae": 7.9, "baseline_mae": 6.2},
    ]
    assert _mean_variance_fallback(seasonal) is True
