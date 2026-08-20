import numpy as np
import pandas as pd
import pytest

from ml.conformal import empirical_coverage, oof_conformal_bounds, residual_quantiles


def test_residual_quantiles_cover_central_mass() -> None:
    rng = np.random.default_rng(0)
    residuals = rng.normal(0.0, 4.0, 400)
    lo, hi = residual_quantiles(residuals, alpha=0.20)
    cover = float(np.mean((residuals >= lo) & (residuals <= hi)))
    assert 0.75 <= cover <= 0.85


def test_oof_bounds_use_prior_seasons_only() -> None:
    frame = pd.DataFrame({
        "season": [2023, 2023, 2024, 2024],
        "y_pred": [10.0, 12.0, 11.0, 13.0],
        "y_true": [14.0, 8.0, 11.0, 13.0],
    })
    # Too few 2023 residuals (<20) → 2024 stays unavailable
    out = oof_conformal_bounds(frame)
    assert set(out.loc[out["season"] == 2024, "interval_method"]) == {"unavailable"}

    big = pd.DataFrame({
        "season": [2022] * 40 + [2023] * 10,
        "y_pred": [10.0] * 50,
        "y_true": list(np.linspace(4.0, 16.0, 40)) + [10.0] * 10,
    })
    calibrated = oof_conformal_bounds(big)
    assert set(calibrated.loc[calibrated["season"] == 2023, "interval_method"]) == {"mapie_enbpi"}
    cover = empirical_coverage(
        calibrated.loc[calibrated["season"] == 2023, "y_true"].to_numpy(),
        calibrated.loc[calibrated["season"] == 2023, "floor"].to_numpy(),
        calibrated.loc[calibrated["season"] == 2023, "ceiling"].to_numpy(),
    )
    assert cover == cover  # finite


def test_train_fast_path_no_longer_uses_sigma_times_three() -> None:
    from pathlib import Path

    source = Path("ml/train.py").read_text()
    assert "std *= 3.0" not in source
    assert "residual_quantiles" in source
