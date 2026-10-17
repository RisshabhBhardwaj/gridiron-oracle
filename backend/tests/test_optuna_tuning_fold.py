"""Optuna tunes on the fold before the last, never on the final season.

Validating on the last fold's season lets hyperparameters see the final
evaluation season, so the last OOF fold is not a clean held-out result
(audit C4). Each learner must tune on folds[-2] when it exists. Without
the fix the recorded tuning season is the last one and these tests fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import ml.catboost_model as cbm
import ml.lgbm_model as lgbm_mod
import ml.xgb_model as xgb_mod
from ml.utils import FEATURE_COLS


def _tiny_df(seasons: list[int], seed: int = 21) -> pd.DataFrame:
    frames = []
    for season in seasons:
        rng = np.random.default_rng(seed + season)
        n = 40
        row: dict = {
            "player_id": [f"O{season}_{i:03d}" for i in range(n)],
            "game_id": [f"{season}_01_AA_BB_{i}" for i in range(n)],
            "season": [season] * n,
            "week": rng.integers(1, 18, n).tolist(),
            "position": ["WR"] * n,
        }
        for feat in FEATURE_COLS:
            vals = rng.standard_normal(n) * 10 + 5
            vals[rng.random(n) < 0.1] = np.nan
            row[feat] = vals.tolist()
        row["actual_receiving_yards"] = (40.0 + rng.normal(0, 2, n)).tolist()
        frames.append(pd.DataFrame(row))
    return pd.concat(frames, ignore_index=True)


def _spy_optuna(monkeypatch, module, defaults: dict) -> dict:
    seen: dict = {}
    real = module._run_optuna

    def spy(optuna_train, optuna_val, features, target_col, n_trials):
        seen["tune_val_seasons"] = sorted(optuna_val["season"].unique().tolist())
        seen["tune_train_seasons"] = sorted(optuna_train["season"].unique().tolist())
        assert real is not None  # the real tuner exists; the stub skips its cost
        return dict(defaults)

    monkeypatch.setattr(module, "_run_optuna", spy)
    return seen


def _train_kwargs(tmp_path, out_name: str) -> dict:
    return dict(
        target="receiving_yards",
        n_optuna_trials=2,
        position_filter="WR",
        mlflow_tracking_uri="",
        out_dir=tmp_path / out_name,
    )


class TestOptunaTuningFold:
    def test_catboost_tunes_before_last_season(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setitem(cbm.DEFAULT_CATBOOST_PARAMS, "iterations", 30)
        seen = _spy_optuna(monkeypatch, cbm, cbm.DEFAULT_CATBOOST_PARAMS)
        cbm.train(_tiny_df([2021, 2022, 2023]), [2021, 2022, 2023], **_train_kwargs(tmp_path, "cb"))
        assert seen["tune_val_seasons"] == [2022]
        assert seen["tune_train_seasons"] == [2021]

    def test_lgbm_tunes_before_last_season(self, tmp_path, monkeypatch) -> None:
        seen = _spy_optuna(monkeypatch, lgbm_mod, lgbm_mod.DEFAULT_LGBM_PARAMS)
        lgbm_mod.train(_tiny_df([2021, 2022, 2023]), [2021, 2022, 2023], **_train_kwargs(tmp_path, "lgb"))
        assert seen["tune_val_seasons"] == [2022]
        assert seen["tune_train_seasons"] == [2021]

    def test_xgb_tunes_before_last_season(self, tmp_path, monkeypatch) -> None:
        seen = _spy_optuna(monkeypatch, xgb_mod, xgb_mod.DEFAULT_XGB_PARAMS)
        xgb_mod.train(_tiny_df([2021, 2022, 2023]), [2021, 2022, 2023], **_train_kwargs(tmp_path, "xgb"))
        assert seen["tune_val_seasons"] == [2022]
        assert seen["tune_train_seasons"] == [2021]

    def test_single_fold_falls_back_to_only_season(self, tmp_path, monkeypatch, caplog) -> None:
        import logging

        monkeypatch.setitem(cbm.DEFAULT_CATBOOST_PARAMS, "iterations", 30)
        seen = _spy_optuna(monkeypatch, cbm, cbm.DEFAULT_CATBOOST_PARAMS)
        with caplog.at_level(logging.WARNING, logger=cbm.logger.name):
            cbm.train(_tiny_df([2021, 2022]), [2021, 2022], **_train_kwargs(tmp_path, "cb1"))
        assert seen["tune_val_seasons"] == [2022]
        assert "no earlier season available" in caplog.text
