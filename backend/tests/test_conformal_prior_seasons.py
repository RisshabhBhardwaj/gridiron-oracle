"""Yardage conformal intervals calibrate on prior OOF seasons only.

Calibrating on the evaluated season's own residuals is in-sample: the
shipped intervals look tighter than they are on unseen games. The runner
must filter the pinned stack OOF to seasons strictly before the evaluated
one — the same causal cut the Bayesian path already applies. Without the
fix the evaluation season's large residuals widen (or reshape) the radius
and the first test below fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml import train as train_mod


def _stub_manifest(monkeypatch, oof: pd.DataFrame):
    def resolve_stack(stat: str, position: str, *, verify: bool = True):
        path = oof_path_holder["dir"] / "stack.csv"
        oof.to_csv(path, index=False)
        return path

    stub = type("StubManifest", (), {"resolve_stack": staticmethod(resolve_stack)})()
    monkeypatch.setattr(train_mod, "get_manifest", lambda: stub)


oof_path_holder: dict = {}


def _oof(prior_resid: float, eval_resid: float, n_each: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n_each):
        rows.append(
            {
                "player_id": f"P{i}",
                "game_id": f"2023_01_AA_BB_{i}",
                "season": 2023,
                "week": 1,
                "y_true": 100.0 + prior_resid,
                "y_pred": 100.0,
                "fold_idx": 0,
            }
        )
    for i in range(n_each):
        rows.append(
            {
                "player_id": f"Q{i}",
                "game_id": f"2024_01_AA_BB_{i}",
                "season": 2024,
                "week": 1,
                "y_true": 100.0 + eval_resid,
                "y_pred": 100.0,
                "fold_idx": 1,
            }
        )
    return pd.DataFrame(rows)


def _mc_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"projection": [100.0, 102.0]}, index=pd.Index(["P1", "P2"], name="player_id")
    )


class TestConformalPriorSeasonsOnly:
    def test_eval_season_residuals_excluded(self, tmp_path, monkeypatch) -> None:
        oof_path_holder["dir"] = tmp_path
        _stub_manifest(monkeypatch, _oof(prior_resid=1.0, eval_resid=50.0))
        runner = train_mod.PipelineRunner()
        out = runner._apply_conformal_yardage_intervals(
            _mc_df(), np.array([100.0, 102.0]),
            stat="receiving_yards", position="WR", season=2024,
        )
        # Prior-only |residual| is 1.0; the eval season's 50.0 must not leak in.
        assert out["conformal_radius"].max() < 5.0
        assert (out["ceiling"] - out["floor"]).max() < 10.0

    def test_no_prior_seasons_fails_closed_by_default(self, tmp_path, monkeypatch) -> None:
        oof_path_holder["dir"] = tmp_path
        _stub_manifest(monkeypatch, _oof(prior_resid=1.0, eval_resid=50.0).query("season == 2024"))
        monkeypatch.delenv("PRODUCT_MODE", raising=False)
        runner = train_mod.PipelineRunner()
        with pytest.raises(RuntimeError, match="No stacked OOF residuals"):
            runner._apply_conformal_yardage_intervals(
                _mc_df(), np.array([100.0, 102.0]),
                stat="receiving_yards", position="WR", season=2024,
            )

    def test_no_prior_seasons_keeps_bayesian_in_fallback(self, tmp_path, monkeypatch) -> None:
        oof_path_holder["dir"] = tmp_path
        _stub_manifest(monkeypatch, _oof(prior_resid=1.0, eval_resid=50.0).query("season == 2024"))
        monkeypatch.setenv("PRODUCT_MODE", "graceful_fallback")
        runner = train_mod.PipelineRunner()
        mc_df = _mc_df()
        assert runner._apply_conformal_yardage_intervals(
            mc_df, np.array([100.0, 102.0]),
            stat="receiving_yards", position="WR", season=2024,
        ) is mc_df

    def test_season_is_required(self, tmp_path, monkeypatch) -> None:
        oof_path_holder["dir"] = tmp_path
        _stub_manifest(monkeypatch, _oof(prior_resid=1.0, eval_resid=50.0))
        runner = train_mod.PipelineRunner()
        with pytest.raises(TypeError):
            runner._apply_conformal_yardage_intervals(
                _mc_df(), np.array([100.0, 102.0]),
                stat="receiving_yards", position="WR",  # type: ignore[call-arg]
            )
