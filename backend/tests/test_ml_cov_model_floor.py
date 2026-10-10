"""
backend/tests/test_ml_cov_model_floor.py

Coverage tests for ml/model_floor.py (Ridge floor on a causal holdout).

The feature-matrix loader is replaced with a synthetic frame, so no database
is needed. The synthetic target is an exact linear function of two features,
so the Ridge floor (tiny alpha) must recover it with near-zero error.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import ml.utils as ml_utils
from ml import model_floor as mf
from ml.model_floor import ModelFloorResult, fit_ridge_floor, main


def _linear_frame(seasons: list[int], n_per_season: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for season in seasons:
        x1 = rng.uniform(0, 10, n_per_season)
        x2 = rng.uniform(0, 5, n_per_season)
        frames.append(pd.DataFrame({
            "player_id": [f"{season}_{i}" for i in range(n_per_season)],
            "season": season,
            "position": "WR",
            "seas_avg_fantasy_ppr": x1,
            "seas_games_played": x2,
            "actual_fantasy_ppr": 2.0 * x1 + 3.0 * x2 + 1.0,
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def loader(monkeypatch):
    """Patch the DB loader; record the call and the FEATURE_COLS visible during the call."""
    state = {"calls": [], "frame": None, "feature_cols_seen": None}

    def fake_load(db_url, seasons, position_filter=None, feature_cols=None):
        state["calls"].append({"db_url": db_url, "seasons": list(seasons), "position": position_filter})
        state["feature_cols_seen"] = list(ml_utils.FEATURE_COLS)
        return state["frame"].copy()

    monkeypatch.setattr(mf, "load_feature_matrix", fake_load)
    return state


class TestFitRidgeFloor:
    def test_recovers_exact_linear_target_on_holdout(self, loader) -> None:
        loader["frame"] = _linear_frame([2019, 2020, 2021, 2022])
        result = fit_ridge_floor(
            holdout_season=2022, position="wr", target="fantasy_ppr", alpha=1e-6,
            database_url="postgresql://test",
        )
        assert isinstance(result, ModelFloorResult)
        assert result.holdout_season == 2022
        assert result.position == "WR"  # upper-cased
        assert result.target == "fantasy_ppr"
        assert result.n_train == 3 * 40
        assert result.n_holdout == 40
        assert result.n_features == 2
        assert result.mae == pytest.approx(0.0, abs=1e-3)
        assert result.rmse == pytest.approx(0.0, abs=1e-3)
        assert result.feature_groups == []
        assert loader["calls"][0] == {
            "db_url": "postgresql://test", "seasons": [2019, 2020, 2021, 2022], "position": "WR",
        }

    def test_training_uses_only_seasons_before_holdout(self, loader) -> None:
        # Holdout season has a deliberately shifted target. Training must never see it,
        # so the fitted floor misses the holdout by exactly the shift.
        frame = _linear_frame([2019, 2020, 2021])
        hold = _linear_frame([2022], seed=3)
        hold["actual_fantasy_ppr"] = hold["actual_fantasy_ppr"] + 100.0
        loader["frame"] = pd.concat([frame, hold], ignore_index=True)

        result = fit_ridge_floor(2022, "WR", target="fantasy_ppr", alpha=1e-6)
        assert result.n_train == 3 * 40
        assert result.mae == pytest.approx(100.0, abs=1e-2)

    def test_feature_groups_are_added_then_restored(self, loader) -> None:
        loader["frame"] = _linear_frame([2020, 2021])
        original = list(ml_utils.FEATURE_COLS)
        result = fit_ridge_floor(2021, "WR", target="fantasy_ppr", alpha=1e-6, feature_groups=["pace_script"])
        # The loader saw the pace_script columns appended to FEATURE_COLS...
        assert "team_pace" in loader["feature_cols_seen"]
        # ...and FEATURE_COLS is restored afterwards.
        assert ml_utils.FEATURE_COLS == original
        assert result.feature_groups == ["pace_script"]

    def test_feature_cols_restored_even_when_load_fails(self, monkeypatch) -> None:
        original = list(ml_utils.FEATURE_COLS)

        def boom(*args, **kwargs):
            raise ConnectionError("db down")

        monkeypatch.setattr(mf, "load_feature_matrix", boom)
        with pytest.raises(ConnectionError):
            fit_ridge_floor(2021, "WR", feature_groups=["pace_script"])
        assert ml_utils.FEATURE_COLS == original

    def test_empty_holdout_raises(self, loader) -> None:
        loader["frame"] = _linear_frame([2019, 2020])  # no 2021 rows
        with pytest.raises(RuntimeError, match="Empty train/holdout"):
            fit_ridge_floor(2021, "WR", target="fantasy_ppr")

    def test_unknown_target_raises_key_error(self, loader) -> None:
        loader["frame"] = _linear_frame([2019, 2020])
        with pytest.raises(KeyError, match="Missing target nope"):
            fit_ridge_floor(2020, "WR", target="nope")

    def test_rows_with_missing_target_are_excluded(self, loader) -> None:
        frame = _linear_frame([2019, 2020])
        frame.loc[frame.index[:5], "actual_fantasy_ppr"] = np.nan
        loader["frame"] = frame
        result = fit_ridge_floor(2020, "WR", target="fantasy_ppr", alpha=1e-6)
        assert result.n_train == 40 - 5
        assert result.n_holdout == 40


class TestMain:
    def test_main_writes_json_and_prints_path(self, monkeypatch, tmp_path, capsys) -> None:
        captured: dict = {}

        def fake_fit(**kwargs):
            captured.update(kwargs)
            return ModelFloorResult(
                holdout_season=2024, position="TE", target="fantasy_ppr", alpha=10.0,
                n_train=100, n_holdout=20, mae=4.5, rmse=6.0, feature_groups=["progression"],
                n_features=12, created_at="2026-10-10T00:00:00+00:00",
            )

        monkeypatch.setattr(mf, "fit_ridge_floor", fake_fit)
        monkeypatch.setattr(mf, "_OUT", tmp_path / "floor")

        code = main(["--holdout-season", "2024", "--position", "te", "--groups", "progression"])
        assert code == 0
        assert captured == {
            "holdout_season": 2024, "position": "te", "target": "fantasy_ppr",
            "alpha": 10.0, "feature_groups": ["progression"],
        }

        written = list((tmp_path / "floor").glob("ridge_TE_2024_*.json"))
        assert len(written) == 1
        payload = json.loads(written[0].read_text())
        assert payload["mae"] == 4.5
        assert payload["n_features"] == 12
        assert f"saved={written[0]}" in capsys.readouterr().out

    def test_main_without_groups_passes_none(self, monkeypatch, tmp_path) -> None:
        seen: dict = {}

        def fake_fit(**kwargs):
            seen.update(kwargs)
            return ModelFloorResult(
                holdout_season=2023, position="WR", target="receiving_yards", alpha=2.0,
                n_train=1, n_holdout=1, mae=0.0, rmse=0.0, feature_groups=[], n_features=0,
                created_at="x",
            )

        monkeypatch.setattr(mf, "fit_ridge_floor", fake_fit)
        monkeypatch.setattr(mf, "_OUT", tmp_path)
        assert main(["--holdout-season", "2023", "--target", "receiving_yards", "--alpha", "2"]) == 0
        assert seen["feature_groups"] is None
        assert seen["alpha"] == 2.0
        assert seen["target"] == "receiving_yards"
