"""Calibration residuals come only from seasons strictly before the evaluation season (ml-01, part B).

The OOF loader used to glob every OOF CSV for a stat, including the season being
evaluated and later seasons. That is not causal. These tests use a synthetic OOF
directory with known residuals per season.

Residuals (y_true - y_pred, with y_pred = 0 so residual = y_true):
    xgb: 2020 -> [1, -1]     2021 -> [10, -10]    2022 -> [100, -100]
    lgbm: 2020 -> [2, -2]    2021 -> [20, -20]    2022 -> [200, -200]
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml import bayesian_layer as bayesian_mod
from ml import conformal as conformal_mod
from ml.train import PipelineRunner

XGB_RESIDUALS = {2020: [1.0, -1.0], 2021: [10.0, -10.0], 2022: [100.0, -100.0]}
LGBM_RESIDUALS = {2020: [2.0, -2.0], 2021: [20.0, -20.0], 2022: [200.0, -200.0]}


def _write_oof(oof_dir: Path, learner: str, residuals_by_season: dict[int, list[float]]) -> Path:
    rows = []
    for season, residuals in residuals_by_season.items():
        for i, residual in enumerate(residuals):
            rows.append({
                "player_id": f"p{season}_{i}",
                "game_id": f"{season}_01_X",
                "season": season,
                "week": 1,
                "position": "WR",
                "y_true": residual,
                "y_pred": 0.0,
                "fold_idx": 0,
            })
    path = oof_dir / f"{learner}_receptions_a1b2c3.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def oof_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "oof"
    directory.mkdir()
    _write_oof(directory, "xgb", XGB_RESIDUALS)
    _write_oof(directory, "lgbm", LGBM_RESIDUALS)
    return directory


def test_loader_requires_explicit_season_with_no_default():
    # A None or missing default would mean "all seasons" and reopen the leak.
    param = inspect.signature(PipelineRunner._load_oof_residuals).parameters["season"]
    assert param.default is inspect.Parameter.empty


def test_loader_uses_only_seasons_strictly_before_evaluation(oof_dir):
    runner = PipelineRunner(oof_dir=oof_dir)

    # Evaluating 2021 may use 2020 only. The 2021 and 2022 residuals are excluded.
    prior_to_2021 = np.sort(runner._load_oof_residuals("WR", "receptions", season=2021))
    assert prior_to_2021.tolist() == [-2.0, -1.0, 1.0, 2.0]

    # Evaluating 2022 may use 2020 and 2021, never 2022 itself.
    prior_to_2022 = np.sort(runner._load_oof_residuals("WR", "receptions", season=2022))
    assert prior_to_2022.tolist() == [-20.0, -10.0, -2.0, -1.0, 1.0, 2.0, 10.0, 20.0]


def test_loader_excludes_file_without_season_column(oof_dir):
    # A file that cannot say which season it covers cannot be proven prior-season.
    pd.DataFrame({
        "player_id": ["px"], "position": ["WR"], "y_true": [1000.0], "y_pred": [0.0],
    }).to_csv(oof_dir / "xgb_receptions_nocol.csv", index=False)
    runner = PipelineRunner(oof_dir=oof_dir)

    residuals = np.sort(runner._load_oof_residuals("WR", "receptions", season=2021))

    assert residuals.tolist() == [-2.0, -1.0, 1.0, 2.0]


def test_loader_earliest_season_has_no_prior_residuals_and_uses_fallback(oof_dir):
    runner = PipelineRunner(oof_dir=oof_dir)

    # Nothing precedes 2020, so the position sigma fallback applies (500 draws).
    residuals = runner._load_oof_residuals("WR", "receptions", season=2020)

    assert len(residuals) == 500


def test_fast_pipeline_calibrates_from_prior_seasons_only(oof_dir, monkeypatch):
    seen: list[np.ndarray] = []
    real_quantiles = conformal_mod.residual_quantiles

    def capture(residuals, *, alpha=0.20):
        seen.append(np.sort(np.asarray(residuals, dtype=float)))
        return real_quantiles(residuals, alpha=alpha)

    monkeypatch.setattr(conformal_mod, "residual_quantiles", capture)
    runner = PipelineRunner(oof_dir=oof_dir, fast=True)
    monkeypatch.setattr(
        runner,
        "_run_stacking_step",
        lambda pos_df, stat, position, **_: pos_df["kalman_est_receptions"].to_numpy(dtype=float),
    )
    kalman_df = pd.DataFrame({
        "player_id": ["p1", "p2"],
        "game_id": ["2021_05_A", "2021_05_B"],
        "position": ["WR", "WR"],
        "kalman_est_receptions": [5.0, 6.0],
        "kalman_variance_receptions": [4.0, 4.0],
    })

    runner._run_single_stat_position(
        stat="receptions",
        position="WR",
        kalman_df=kalman_df,
        n_samples=8,
        season=2021,
        week=5,
        dry_run_mode=False,
    )

    assert len(seen) == 1
    assert seen[0].tolist() == [-2.0, -1.0, 1.0, 2.0]


def test_production_calibration_is_refit_per_season(oof_dir, monkeypatch):
    fits: list[np.ndarray] = []

    class FakeProjection:
        def fit(self, position, historical_residuals):
            fits.append(np.sort(np.asarray(historical_residuals, dtype=float)))
            return self

        def posterior_samples(self, stacked_estimate, kalman_variance, n_samples):
            return np.zeros(n_samples)

    monkeypatch.setattr(bayesian_mod, "BayesianProjection", FakeProjection)
    runner = PipelineRunner(oof_dir=oof_dir)  # production path (fast=False)
    call = dict(
        stacked_estimates=np.array([50.0]),
        kalman_variances=np.array([4.0]),
        stat="receptions",
        position="WR",
        player_ids=["p1"],
        n_samples=8,
    )

    runner._run_bayesian_step(season=2022, **call)
    runner._run_bayesian_step(season=2021, **call)
    runner._run_bayesian_step(season=2022, **call)  # same season: reuse the fit

    # A fit cached under one season must never serve a different season.
    assert len(fits) == 2
    assert fits[0].tolist() == [-20.0, -10.0, -2.0, -1.0, 1.0, 2.0, 10.0, 20.0]
    assert fits[1].tolist() == [-2.0, -1.0, 1.0, 2.0]
