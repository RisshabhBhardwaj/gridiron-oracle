"""OOF run ids are unique per run, even within the same second.

A time-only pseudo id (``%Y%m%dT%H%M%S``) gave two same-second runs one file
name, so the second run overwrote the first OOF. Every learner's MLflow-off
fallback now mints ``ml.utils.new_pseudo_run_id`` (time-ordered 16 hex
digits with a random suffix), and every OOF saver keeps the leading 16
characters. These tests freeze the clock: without the fix the same-second
pairs below share a file name and the assertions fail.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import ml.lgbm_model as lgbm_mod
import ml.stacking_ensemble as stack_mod
import ml.utils as ml_utils_mod
import ml.xgb_model as xgb_mod
from ml.stacking_ensemble import _save_stack_oof
from ml.utils import RUN_ID_FILE_CHARS, new_pseudo_run_id, save_oof

HEX16 = re.compile(r"[0-9a-f]{16}")


class _FrozenClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now


def _tiny_df(seasons: list[int], seed: int = 11) -> pd.DataFrame:
    frames = []
    for season in seasons:
        rng = np.random.default_rng(seed + season)
        n = 40
        row: dict = {
            "player_id": [f"Q{season}_{i:03d}" for i in range(n)],
            "game_id": [f"{season}_01_AA_BB_{i}" for i in range(n)],
            "season": [season] * n,
            "week": rng.integers(1, 18, n).tolist(),
            "position": ["WR"] * n,
        }
        for feat in ml_utils_mod.FEATURE_COLS:
            vals = rng.standard_normal(n) * 10 + 5
            vals[rng.random(n) < 0.1] = np.nan
            row[feat] = vals.tolist()
        row["actual_receiving_yards"] = (
            40.0 + rng.normal(0, 2, n)
        ).tolist()
        frames.append(pd.DataFrame(row))
    return pd.concat(frames, ignore_index=True)


def _freeze_time(monkeypatch, now: float = 1_780_000_000.0) -> _FrozenClock:
    clock = _FrozenClock(now)
    monkeypatch.setattr(ml_utils_mod, "time", clock, raising=False)
    return clock


class TestNewPseudoRunId:
    def test_same_second_ids_differ(self, monkeypatch) -> None:
        _freeze_time(monkeypatch)
        first, second = new_pseudo_run_id(), new_pseudo_run_id()
        assert first != second
        assert HEX16.fullmatch(first)
        assert HEX16.fullmatch(second)
        assert "_" not in first and "_" not in second

    def test_time_ordered(self, monkeypatch) -> None:
        clock = _freeze_time(monkeypatch)
        first = new_pseudo_run_id()
        clock.now += 60
        assert new_pseudo_run_id() > first

    def test_sorts_after_legacy_stamps(self, monkeypatch) -> None:
        _freeze_time(monkeypatch, now=1_780_000_000.0)
        assert new_pseudo_run_id() > "20261231"
        assert new_pseudo_run_id() > "69b8d880"  # legacy 8-hex catboost stamp


class TestSaveOofWidth:
    def test_stamp_is_16_hex_chars(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "player_id": ["P1"],
                "game_id": ["2021_01_AA_BB"],
                "season": [2021],
                "week": [1],
                "y_true": [10.0],
                "y_pred": [9.0],
                "fold_idx": [0],
            }
        )
        path = save_oof(df, "receiving_yards", "f" * 32, tmp_path, prefix="xgb", position="WR")
        stamp = path.stem.rsplit("_", 1)[-1]
        assert len(stamp) == RUN_ID_FILE_CHARS == 16


class TestLearnerSameSecondRuns:
    def test_lgbm_same_second_runs_keep_separate_files(self, tmp_path: Path, monkeypatch) -> None:
        _freeze_time(monkeypatch)
        kwargs = dict(
            target="receiving_yards",
            n_optuna_trials=0,
            position_filter="WR",
            mlflow_tracking_uri="",
            out_dir=tmp_path / "oof",
        )
        df = _tiny_df([2021, 2022, 2023])
        first = lgbm_mod.train(df, [2021, 2022, 2023], **kwargs)
        second = lgbm_mod.train(df, [2021, 2022, 2023], **kwargs)
        assert first.oof_path is not None and second.oof_path is not None
        assert first.oof_path != second.oof_path
        assert first.oof_path.exists() and second.oof_path.exists()

    def test_xgb_same_second_runs_keep_separate_files(self, tmp_path: Path, monkeypatch) -> None:
        _freeze_time(monkeypatch)
        kwargs = dict(
            target="receiving_yards",
            n_optuna_trials=0,
            position_filter="WR",
            mlflow_tracking_uri="",
            out_dir=tmp_path / "oof",
        )
        df = _tiny_df([2021, 2022, 2023])
        first = xgb_mod.train(df, [2021, 2022, 2023], **kwargs)
        second = xgb_mod.train(df, [2021, 2022, 2023], **kwargs)
        assert first.oof_path is not None and second.oof_path is not None
        assert first.oof_path != second.oof_path
        assert first.oof_path.exists() and second.oof_path.exists()


class TestStackSaver:
    def _stack_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "player_id": ["P1", "P2"],
                "game_id": ["2021_01_AA_BB"] * 2,
                "season": [2021] * 2,
                "week": [1] * 2,
                "y_true": [10.0, 12.0],
                "y_pred": [9.0, 11.0],
                "fold_idx": [0] * 2,
                "max_train_season": [2020] * 2,
                "xgb_pred": [9.5, 11.5],
                "lgbm_pred": [8.5, 10.5],
            }
        )

    def test_same_second_stacks_keep_separate_files(self, tmp_path: Path, monkeypatch) -> None:
        _freeze_time(monkeypatch)
        df = self._stack_df()
        first = _save_stack_oof(df, "receiving_yards", new_pseudo_run_id(), tmp_path, "WR")
        second = _save_stack_oof(df, "receiving_yards", new_pseudo_run_id(), tmp_path, "WR")
        assert first != second
        assert first.exists() and second.exists()
        stamp = first.stem.rsplit("_", 1)[-1]
        assert HEX16.fullmatch(stamp)

    def test_stack_modules_use_shared_generator(self) -> None:
        # Wiring check: the fallback paths must mint ids from the shared
        # generator, not a per-module time-only format.
        assert lgbm_mod.new_pseudo_run_id is new_pseudo_run_id
        assert xgb_mod.new_pseudo_run_id is new_pseudo_run_id
        assert stack_mod.new_pseudo_run_id is new_pseudo_run_id
