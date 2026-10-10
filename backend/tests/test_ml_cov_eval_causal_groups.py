"""
backend/tests/test_ml_cov_eval_causal_groups.py

Coverage tests for:
  - ml/eval_causal.py      (causal OOF scoring against naive, trailing-3 and ADP baselines)
  - ml/feature_groups.py   (Ridge proxy for an optional feature group, causal holdout)
  - ml/utils.configure_thread_env (thread-cap branch when torch is loaded)

The causality guard in eval_causal is tested directly: an OOF row whose
max_train_season is not strictly before its evaluation season must be refused.
Feature-group loading is replaced by a synthetic frame; no database is used.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

import ml.utils as ml_utils
from ml import eval_causal as ec
from ml import feature_groups as fg
from ml.utils import FEATURE_GROUPS, TARGET_COL_MAP


# ── eval_causal._normalize_oof ────────────────────────────────────────────────

def _oof_frame(**overrides) -> pd.DataFrame:
    base = {
        "player_id": ["P1", "P2"],
        "season": [2024, 2024],
        "week": [2, 2],
        "actual": [90.0, 40.0],
        "predicted": [85.0, 45.0],
        "max_train_season": [2023, 2023],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestNormalizeOof:
    def test_renames_legacy_columns_and_adds_stat(self) -> None:
        legacy = _oof_frame().rename(columns={"actual": "y_true", "predicted": "y_pred"})
        out = ec._normalize_oof(legacy, "receiving_yards")
        assert {"actual", "predicted"} <= set(out.columns)
        assert "y_true" not in out.columns and "y_pred" not in out.columns
        assert set(out["stat"]) == {"receiving_yards"}

    def test_existing_stat_column_is_kept(self) -> None:
        frame = _oof_frame(stat=["targets", "targets"])
        out = ec._normalize_oof(frame, "receiving_yards")
        assert set(out["stat"]) == {"targets"}

    def test_missing_required_columns_are_named(self) -> None:
        frame = _oof_frame().drop(columns=["predicted"])
        with pytest.raises(ValueError, match=r"OOF missing columns \['predicted'\]"):
            ec._normalize_oof(frame, "receiving_yards")

    def test_missing_provenance_is_refused(self) -> None:
        frame = _oof_frame().drop(columns=["max_train_season"])
        with pytest.raises(ValueError, match="missing max_train_season provenance"):
            ec._normalize_oof(frame, "receiving_yards")

    def test_non_numeric_provenance_is_refused(self) -> None:
        frame = _oof_frame(max_train_season=["2023", "unknown"])
        with pytest.raises(ValueError, match="non-numeric max_train_season"):
            ec._normalize_oof(frame, "receiving_yards")

    def test_same_season_training_is_non_causal(self) -> None:
        # Training through the evaluation season leaks the target; the guard must refuse it.
        frame = _oof_frame(max_train_season=[2024, 2023])
        with pytest.raises(ValueError, match="non-causal max_train_season"):
            ec._normalize_oof(frame, "receiving_yards")

    def test_causal_provenance_is_accepted_and_coerced(self) -> None:
        frame = _oof_frame(max_train_season=["2023", "2022"])
        out = ec._normalize_oof(frame, "receiving_yards")
        assert list(out["max_train_season"]) == [2023.0, 2022.0]


# ── eval_causal.score_oof_against_baselines ───────────────────────────────────

def _history_p1_p2() -> pd.DataFrame:
    rows = []
    for week, value in zip((1, 2, 3, 4), (80.0, 90.0, 100.0, 110.0)):
        rows.append({"player_id": "P1", "season": 2024, "week": week, "receiving_yards": value})
    for week, value in zip((1, 2), (50.0, 70.0)):
        rows.append({"player_id": "P1", "season": 2023, "week": week, "receiving_yards": value})
    rows.append({"player_id": "P2", "season": 2024, "week": 1, "receiving_yards": 30.0})
    rows.append({"player_id": "P2", "season": 2024, "week": 2, "receiving_yards": 35.0})
    return pd.DataFrame(rows)


class TestScoreOofAgainstBaselines:
    def test_perfect_model_beats_both_baselines_with_known_scores(self) -> None:
        # P1 weeks 2-4 are scored (week 1 has no prior game). Predictions equal actuals.
        # MAE vs naive (2023 mean 60):           |90-60|, |100-60|, |110-60| -> mean 40.
        # MAE vs trailing-3 (80; 85; 90):        |90-80|, |100-85|, |110-90| -> mean 15.
        # The cohort counts prior games inside the OOF frame, so the full season (week 1 included)
        # must be present. Week 1 is then the excluded first game.
        oof = pd.DataFrame({
            "player_id": ["P1", "P1", "P1", "P1"],
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 3, 4],
            "position": ["WR", "WR", "WR", "WR"],
            "actual": [80.0, 90.0, 100.0, 110.0],
            "predicted": [80.0, 90.0, 100.0, 110.0],
            "max_train_season": [2023, 2023, 2023, 2023],
        })
        table = ec.score_oof_against_baselines(oof, _history_p1_p2(), stat="receiving_yards")

        assert len(table) == 1
        row = table.iloc[0]
        assert row["metric"] == "mae"
        assert row["n"] == 3
        assert row["model_score"] == pytest.approx(0.0, abs=1e-9)
        assert row["naive_score"] == pytest.approx(40.0)
        assert row["trailing3_score"] == pytest.approx(15.0)
        assert bool(row["beats_naive"]) is True
        assert bool(row["beats_trailing3"]) is True
        assert bool(row["max_train_season_ok"]) is True
        assert row["cohort_rule"].startswith("player has at least")

    def test_position_filter_keeps_only_requested_rows(self) -> None:
        oof = pd.DataFrame({
            "player_id": ["P1", "P2", "P1", "P2"], "season": [2024, 2024, 2024, 2024],
            "week": [1, 1, 2, 2], "position": ["WR", "RB", "WR", "RB"],
            "actual": [80.0, 30.0, 90.0, 35.0], "predicted": [80.0, 30.0, 90.0, 35.0],
            "max_train_season": [2023, 2023, 2023, 2023],
        })
        table = ec.score_oof_against_baselines(oof, _history_p1_p2(), stat="receiving_yards", position="WR")
        assert list(table["position"]) == ["WR"]
        assert list(table["n"]) == [1]

    def test_no_eligible_rows_gives_empty_table(self) -> None:
        # Week 1 is the only row, and it has no prior game in the season, so the cohort drops it.
        oof = pd.DataFrame({
            "player_id": ["P1"], "season": [2024], "week": [1],
            "actual": [80.0], "predicted": [80.0], "max_train_season": [2023],
        })
        assert ec.score_oof_against_baselines(oof, _history_p1_p2(), stat="receiving_yards").empty

    def test_history_without_stat_column_is_refused(self) -> None:
        oof = _oof_frame(player_id=["P1", "P1"], week=[2, 3], season=[2024, 2024])
        with pytest.raises(ValueError, match="history must include column 'targets'"):
            ec.score_oof_against_baselines(oof, _history_p1_p2(), stat="targets")

    def test_non_causal_oof_is_refused_before_scoring(self) -> None:
        oof = pd.DataFrame({
            "player_id": ["P1"], "season": [2024], "week": [2],
            "actual": [90.0], "predicted": [90.0], "max_train_season": [2024],
        })
        with pytest.raises(ValueError, match="non-causal"):
            ec.score_oof_against_baselines(oof, _history_p1_p2(), stat="receiving_yards")


class TestEvalCausalCli:
    def test_main_writes_table_from_csv_inputs(self, monkeypatch, tmp_path, capsys) -> None:
        oof_path = tmp_path / "oof.csv"
        history_path = tmp_path / "history.csv"
        out_path = tmp_path / "out" / "table.csv"
        pd.DataFrame({
            "player_id": ["P1", "P1"], "season": [2024, 2024], "week": [1, 2], "position": ["WR", "WR"],
            "actual": [80.0, 90.0], "predicted": [80.0, 90.0], "max_train_season": [2023, 2023],
        }).to_csv(oof_path, index=False)
        _history_p1_p2().to_csv(history_path, index=False)

        monkeypatch.setattr(sys, "argv", [
            "eval_causal", "--oof", str(oof_path), "--stat", "receiving_yards",
            "--history", str(history_path), "--out", str(out_path),
        ])
        assert ec.main() == 0
        saved = pd.read_csv(out_path)
        assert list(saved["stat"]) == ["receiving_yards"]
        assert "model_score" in capsys.readouterr().out


# ── feature_groups.evaluate_feature_group ─────────────────────────────────────

def _group_frame(seasons, n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for season in seasons:
        pace = rng.uniform(20, 30, n)
        base = rng.uniform(5, 15, n)
        frames.append(pd.DataFrame({
            "player_id": [f"{season}_{i}" for i in range(n)],
            "season": season,
            "week": 2,
            "position": "WR",
            "seas_avg_fantasy_ppr": base,
            "team_pace": pace,                       # the group feature carries the signal
            "actual_fantasy_ppr": 1.5 * pace + 0.2 * base + rng.normal(0, 0.1, n),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def group_loader(monkeypatch):
    state = {"calls": [], "frame": None}

    def fake_load(db_url, seasons, position_filter=None, feature_cols=None):
        state["calls"].append({"seasons": list(seasons), "position": position_filter,
                               "cols": list(feature_cols or [])})
        return state["frame"].copy()

    monkeypatch.setattr(fg, "load_feature_matrix", fake_load)
    return state


class TestEvaluateFeatureGroup:
    def test_unknown_group_is_refused(self) -> None:
        with pytest.raises(KeyError, match="Unknown feature group 'nope'"):
            fg.evaluate_feature_group("nope", 2022, "WR")

    def test_group_that_improves_holdout_passes_proxy(self, group_loader) -> None:
        group_loader["frame"] = _group_frame([2020, 2021, 2022])
        result = fg.evaluate_feature_group(
            "pace_script", 2022, "wr", target="fantasy_ppr", database_url="postgresql://t/db",
        )
        assert result.group == "pace_script"
        assert result.position == "WR"
        assert result.feature_cols_added == FEATURE_GROUPS["pace_script"]
        assert result.n_train == 80
        assert result.n_holdout == 40
        assert result.delta_mae == pytest.approx(result.treatment_mae - result.baseline_mae)
        assert result.delta_mae < -0.05  # the pace feature explains the target
        assert result.promote is False and result.promotion_eligible is False
        assert result.evaluation_model == "ridge_proxy"
        assert "proxy_passed=True" in result.limitation
        call = group_loader["calls"][0]
        assert call["seasons"] == [2019, 2020, 2021, 2022]
        assert call["position"] == "WR"
        assert "team_pace" in call["cols"]

    def test_min_delta_controls_proxy_flag(self, group_loader) -> None:
        group_loader["frame"] = _group_frame([2020, 2021, 2022])
        strict = fg.evaluate_feature_group("pace_script", 2022, "WR", min_delta=-1e9)
        lenient = fg.evaluate_feature_group("pace_script", 2022, "WR", min_delta=1e9)
        assert "proxy_passed=False" in strict.limitation
        assert "proxy_passed=True" in lenient.limitation

    def test_explicit_train_seasons_are_used(self, group_loader) -> None:
        group_loader["frame"] = _group_frame([2021, 2022])
        result = fg.evaluate_feature_group("pace_script", 2022, "WR", train_seasons=[2021])
        assert group_loader["calls"][0]["seasons"] == [2021]
        assert result.n_train == 40

    def test_empty_load_is_refused(self, group_loader) -> None:
        group_loader["frame"] = pd.DataFrame()
        with pytest.raises(RuntimeError, match="No rows for position=WR"):
            fg.evaluate_feature_group("pace_script", 2022, "WR")

    def test_missing_target_is_refused(self, group_loader) -> None:
        group_loader["frame"] = _group_frame([2021, 2022]).drop(columns=["actual_fantasy_ppr"])
        with pytest.raises(KeyError, match="actual_fantasy_ppr"):
            fg.evaluate_feature_group("pace_script", 2022, "WR")

    def test_missing_holdout_season_is_refused(self, group_loader) -> None:
        group_loader["frame"] = _group_frame([2019, 2020])
        with pytest.raises(RuntimeError, match="Need train seasons < 2022 and holdout=2022"):
            fg.evaluate_feature_group("pace_script", 2022, "WR")

    def test_target_mapping_is_applied(self) -> None:
        assert TARGET_COL_MAP["fantasy_ppr"] == "actual_fantasy_ppr"


# ── utils.configure_thread_env ────────────────────────────────────────────────

class TestConfigureThreadEnv:
    def test_caps_threads_and_respects_existing_values(self, monkeypatch) -> None:
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
        ml_utils.configure_thread_env()
        assert os.environ["OMP_NUM_THREADS"] == "4"  # setdefault keeps an explicit value
        assert os.environ["KMP_DUPLICATE_LIB_OK"] in {"True", "TRUE"}

    def test_torch_thread_pool_is_capped_when_loaded(self, monkeypatch) -> None:
        calls: list = []
        fake_torch = types.ModuleType("torch")
        fake_torch.set_num_threads = lambda n: calls.append(n)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        ml_utils.configure_thread_env()
        assert calls == [1]

    def test_broken_torch_is_ignored(self, monkeypatch) -> None:
        fake_torch = types.ModuleType("torch")

        def boom(n):
            raise RuntimeError("no BLAS")

        fake_torch.set_num_threads = boom
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        ml_utils.configure_thread_env()  # must not raise
