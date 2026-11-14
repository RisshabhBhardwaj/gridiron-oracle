"""Params persistence (decision 0012, commit 1): trainers write the params they
used next to the OOF, and a recovery script rebuilds the same files from the
`Optuna best RMSE=... params=...` log lines of an earlier rebuild."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.model_params import (
    ParamsError,
    params_filename,
    read_params,
    required_models_from_ridge_dir,
    write_params,
)
from scripts.recover_optuna_params import RecoveryError, parse_log_line, recover


# ── file name and round trip ──────────────────────────────────────────────────

def test_filename_is_learner_stat_position() -> None:
    assert params_filename("lgbm", "passing_yards", "QB") == "params_lgbm_passing_yards_QB.json"


def test_no_position_is_all() -> None:
    assert params_filename("catboost", "fantasy_ppr", None) == "params_catboost_fantasy_ppr_all.json"


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    path = write_params(tmp_path, "lgbm", "targets", "TE", {"num_leaves": 31, "learning_rate": 0.1}, source="trainer")
    assert path.name == "params_lgbm_targets_TE.json"
    got = read_params(path)
    assert got["params"] == {"num_leaves": 31, "learning_rate": 0.1}
    assert (got["learner"], got["stat"], got["position"], got["source"]) == ("lgbm", "targets", "TE", "trainer")


def test_read_rejects_a_file_whose_name_and_content_disagree(tmp_path: Path) -> None:
    path = write_params(tmp_path, "lgbm", "targets", "TE", {"num_leaves": 31}, source="trainer")
    path.rename(tmp_path / "params_lgbm_targets_WR.json")
    with pytest.raises(ParamsError, match="does not match"):
        read_params(tmp_path / "params_lgbm_targets_WR.json")


def test_read_rejects_empty_params(tmp_path: Path) -> None:
    path = tmp_path / "params_lgbm_targets_TE.json"
    path.write_text(json.dumps({"learner": "lgbm", "stat": "targets", "position": "TE", "params": {}, "source": "x"}))
    with pytest.raises(ParamsError, match="no params"):
        read_params(path)


# ── trainers persist what they used ───────────────────────────────────────────

def test_lgbm_train_writes_the_params_it_used(tmp_path: Path) -> None:
    from backend.tests.test_lgbm_model import _make_synthetic_df
    from ml.lgbm_model import train

    df = _make_synthetic_df([2022, 2023, 2024])
    result = train(df, seasons=[2022, 2023, 2024], target="receiving_yards", n_optuna_trials=0,
                   position_filter="WR", mlflow_tracking_uri="", out_dir=tmp_path)
    got = read_params(tmp_path / "params_lgbm_receiving_yards_WR.json")
    assert got["params"] == result.best_params
    assert got["source"] == "trainer_defaults"  # n_optuna_trials=0


def test_catboost_train_writes_the_params_it_used(tmp_path: Path, monkeypatch) -> None:
    from backend.tests.test_ml_cov_catboost import SMALL_ITERATIONS, _synthetic_df
    from ml.catboost_model import DEFAULT_CATBOOST_PARAMS, train

    monkeypatch.setitem(DEFAULT_CATBOOST_PARAMS, "iterations", SMALL_ITERATIONS)
    monkeypatch.chdir(tmp_path)  # CatBoost writes training logs to the cwd
    df = _synthetic_df([2021, 2022, 2023])
    result = train(df, [2021, 2022, 2023], target="receiving_yards", n_optuna_trials=0,
                   position_filter="WR", mlflow_tracking_uri="", out_dir=tmp_path)
    got = read_params(tmp_path / "params_catboost_receiving_yards_WR.json")
    assert got["params"] == result.best_params


def test_optuna_completion_matches_what_the_trainers_train_with() -> None:
    """Recovery must rebuild the exact dict the trainer used after tuning."""
    from ml.catboost_model import complete_optuna_params as cb_complete
    from ml.lgbm_model import complete_optuna_params as lgb_complete

    lgb = lgb_complete({"num_leaves": 20})
    assert lgb["objective"] == "regression_l2" and lgb["random_state"] == 42 and lgb["num_leaves"] == 20
    cb = cb_complete({"depth": 5})
    assert cb["bootstrap_type"] == "Bernoulli" and cb["random_seed"] == 42 and cb["nan_mode"] == "Min"


# ── log line parsing ──────────────────────────────────────────────────────────

LGBM_LINE = (
    "2026-11-14 10:40:01,123 [INFO] __main__: Optuna best RMSE=12.345  "
    "params={'n_estimators': 412, 'num_leaves': 77, 'learning_rate': 0.0213, 'feature_fraction': 0.71, "
    "'bagging_fraction': 0.9, 'bagging_freq': 3, 'min_child_samples': 40, 'lambda_l1': 1.2e-05, 'lambda_l2': 0.5}"
)
CB_LINE = (
    "2026-11-14 10:50:01,123 [INFO] __main__: Optuna best RMSE=3.2  "
    "params={'iterations': 300, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 2.0, "
    "'subsample': 0.8, 'colsample_bylevel': 0.9}"
)


def test_parse_log_line_extracts_rmse_and_params() -> None:
    rmse, params = parse_log_line(LGBM_LINE)
    assert rmse == pytest.approx(12.345)
    assert params["num_leaves"] == 77 and params["lambda_l1"] == pytest.approx(1.2e-05)


def test_parse_log_line_ignores_other_lines() -> None:
    assert parse_log_line("2026-11-14 10:40:01 [INFO] x: Fold 1: n_train=5") is None


def test_parse_log_line_rejects_code_in_the_params_field() -> None:
    with pytest.raises(RecoveryError, match="unparseable"):
        parse_log_line("Optuna best RMSE=1.0  params=__import__('os').system('true')")


# ── recovery over a log directory ─────────────────────────────────────────────

def _release_dir(tmp_path: Path) -> Path:
    """Three cells: both learners, lgbm only, and an identity cell (catboost weight 0)."""
    rel = tmp_path / "release"
    rel.mkdir()
    cells = {
        ("targets", "TE"): {"lgbm": 0.6, "catboost": 0.4},
        ("passing_yards", "QB"): {"lgbm": 1.0},
        ("receiving_tds", "WR"): {"lgbm": 1.0, "catboost": 0.0},
    }
    for (stat, pos), weights in cells.items():
        (rel / f"ridge_{stat}_{pos}_coefs.json").write_text(json.dumps(
            {"learner_order": ["lgbm", "catboost"][: len(weights)] if len(weights) > 1 else list(weights),
             "weights": weights, "intercept": 0.0}))
    return rel


def _logs(tmp_path: Path, lines: dict[str, str]) -> Path:
    logs = tmp_path / "logs"
    logs.mkdir()
    for name, line in lines.items():
        (logs / name).write_text("noise\n" + line + "\nmore noise\n")
    return logs


def test_required_models_skip_zero_weight_learners(tmp_path: Path) -> None:
    assert required_models_from_ridge_dir(_release_dir(tmp_path)) == {
        ("lgbm", "targets", "TE"), ("catboost", "targets", "TE"),
        ("lgbm", "passing_yards", "QB"),
        ("lgbm", "receiving_tds", "WR"),
    }


def test_recover_writes_a_params_file_per_log_and_returns_them(tmp_path: Path) -> None:
    logs = _logs(tmp_path, {
        "lgbm_targets_TE.log": LGBM_LINE, "catboost_targets_TE.log": CB_LINE,
        "lgbm_passing_yards_QB.log": LGBM_LINE, "lgbm_receiving_tds_WR.log": LGBM_LINE,
        "catboost_receiving_tds_WR.log": CB_LINE,   # not needed (weight 0), still recovered
        "stack_targets_TE.log": "no optuna here",
    })
    out = tmp_path / "out"
    written = recover(logs, out, _release_dir(tmp_path))
    assert len(written) == 5
    got = read_params(out / "params_catboost_targets_TE.json")
    assert got["source"] == "recovered_from_log" and got["params"]["depth"] == 6
    assert got["params"]["bootstrap_type"] == "Bernoulli"   # fixed params restored
    assert read_params(out / "params_lgbm_targets_TE.json")["params"]["objective"] == "regression_l2"


def test_recover_fails_when_a_needed_cell_has_no_line(tmp_path: Path) -> None:
    logs = _logs(tmp_path, {"lgbm_targets_TE.log": LGBM_LINE, "lgbm_passing_yards_QB.log": LGBM_LINE,
                            "lgbm_receiving_tds_WR.log": LGBM_LINE})
    with pytest.raises(RecoveryError, match="catboost_targets_TE"):
        recover(logs, tmp_path / "out", _release_dir(tmp_path))
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())  # writes nothing on failure


def test_recover_fails_when_a_cell_has_two_different_lines(tmp_path: Path) -> None:
    logs = _logs(tmp_path, {"lgbm_targets_TE.log": LGBM_LINE, "catboost_targets_TE.log": CB_LINE,
                            "lgbm_passing_yards_QB.log": LGBM_LINE, "lgbm_receiving_tds_WR.log": LGBM_LINE})
    other = LGBM_LINE.replace("'num_leaves': 77", "'num_leaves': 78")
    with (logs / "lgbm_targets_TE.log").open("a") as fh:
        fh.write(other + "\n")
    with pytest.raises(RecoveryError, match="two different"):
        recover(logs, tmp_path / "out", _release_dir(tmp_path))


def test_recover_accepts_an_identical_repeated_line(tmp_path: Path) -> None:
    logs = _logs(tmp_path, {"lgbm_targets_TE.log": LGBM_LINE, "catboost_targets_TE.log": CB_LINE,
                            "lgbm_passing_yards_QB.log": LGBM_LINE, "lgbm_receiving_tds_WR.log": LGBM_LINE})
    with (logs / "lgbm_targets_TE.log").open("a") as fh:
        fh.write(LGBM_LINE + "\n")
    assert len(recover(logs, tmp_path / "out", _release_dir(tmp_path))) == 4


def test_recover_refuses_to_overwrite_a_different_existing_params_file(tmp_path: Path) -> None:
    logs = _logs(tmp_path, {"lgbm_targets_TE.log": LGBM_LINE, "catboost_targets_TE.log": CB_LINE,
                            "lgbm_passing_yards_QB.log": LGBM_LINE, "lgbm_receiving_tds_WR.log": LGBM_LINE})
    out = tmp_path / "out"
    write_params(out, "lgbm", "targets", "TE", {"num_leaves": 3}, source="trainer")
    with pytest.raises(RecoveryError, match="already exists"):
        recover(logs, out, _release_dir(tmp_path))
