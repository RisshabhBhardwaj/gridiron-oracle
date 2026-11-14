"""Final fit and its replay gate (decision 0012, commit 4).

The tiny trainers here are the real ones (``train`` with Optuna off), so the OOF
the gate compares against is produced the way the release's OOF was.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.tests.test_lgbm_model import _make_synthetic_df
from ml import final_fit
from ml.artifact_manifest import load_manifest
from ml.model_params import ParamsError
from ml.season_constants import LAST_COMPLETE_SEASON
from ml.utils import FEATURE_COLS

STAT, POS = "receiving_yards", "WR"
SEASONS = [2022, 2023, 2024]


@pytest.fixture(autouse=True)
def _catboost_small(monkeypatch, tmp_path):
    from ml.catboost_model import DEFAULT_CATBOOST_PARAMS

    monkeypatch.setitem(DEFAULT_CATBOOST_PARAMS, "iterations", 40)
    monkeypatch.chdir(tmp_path)  # CatBoost writes logs to the cwd
    import ml.utils as ml_utils_mod

    monkeypatch.setattr(ml_utils_mod, "_ONNX_DIR", tmp_path / "onnx")


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """Synthetic rows whose target depends on two features, so the models learn something.

    With a pure-noise target a model stops at one tree and predicts a constant,
    and a dropped feature or changed parameter would go unnoticed by the gate.
    """
    df = _make_synthetic_df(SEASONS, n_per_season=120)
    rng = np.random.default_rng(11)
    signal = (3.0 * df["seas_avg_receiving_yards"].fillna(5.0)
              + 2.0 * df[FEATURE_COLS[3]].fillna(5.0))
    df["actual_receiving_yards"] = 40.0 + signal + rng.normal(0, 2.0, len(df))
    return df


def _train(learner: str, df: pd.DataFrame, out_dir: Path):
    if learner == "lgbm":
        from ml.lgbm_model import train
    else:
        from ml.catboost_model import train
    return train(df, seasons=SEASONS, target=STAT, n_optuna_trials=0, position_filter=POS,
                 mlflow_tracking_uri="", out_dir=out_dir)


@pytest.fixture(scope="module", params=["lgbm", "catboost"])
def trained(request, frame, tmp_path_factory):
    """One real trainer run per learner, shared by the module (the OOF is read-only)."""
    from ml.catboost_model import DEFAULT_CATBOOST_PARAMS

    out = tmp_path_factory.mktemp(f"oof_{request.param}")
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(DEFAULT_CATBOOST_PARAMS, "iterations", 40)
        mp.chdir(out)  # CatBoost writes logs to the cwd
        result = _train(request.param, frame, out)
    return request.param, result, out


# ── replay gate ───────────────────────────────────────────────────────────────

def test_replay_gate_passes_for_the_params_that_made_the_oof(trained, frame) -> None:
    learner, result, _ = trained
    replay = final_fit.replay_gate(learner, STAT, POS, frame, result.best_params, result.oof_df)
    assert replay.passed, replay
    assert replay.val_season == 2024
    assert replay.n_rows == int((frame["season"] == 2024).sum())
    assert replay.max_abs_diff <= replay.tolerance_abs
    assert replay.best_iteration >= 1


def test_replay_gate_fails_when_the_params_changed(trained, frame) -> None:
    learner, result, _ = trained
    changed = dict(result.best_params)
    if learner == "lgbm":
        changed["learning_rate"] = 0.2
    else:
        changed["learning_rate"] = 0.3
    replay = final_fit.replay_gate(learner, STAT, POS, frame, changed, result.oof_df)
    assert not replay.passed
    assert replay.max_abs_diff > replay.tolerance_abs


def test_replay_gate_fails_when_a_feature_was_dropped(trained, frame) -> None:
    learner, result, _ = trained
    dropped = frame.drop(columns=["seas_avg_receiving_yards"])
    replay = final_fit.replay_gate(learner, STAT, POS, dropped, result.best_params, result.oof_df)
    assert not replay.passed


def test_replay_gate_fails_when_the_oof_rows_are_not_the_refit_rows(trained, frame) -> None:
    learner, result, _ = trained
    short = result.oof_df.iloc[:-5]
    replay = final_fit.replay_gate(learner, STAT, POS, frame, result.best_params, short)
    assert not replay.passed
    assert "rows" in replay.reason


# ── final fit ─────────────────────────────────────────────────────────────────

def _fit(learner, frame, result, out_dir, **kw):
    return final_fit.fit_cell(
        learner, STAT, POS, df=frame, params=result.best_params, oof=result.oof_df,
        out_dir=out_dir, release_id="causal_test", code_sha="abc123", **kw,
    )


def test_final_fit_uses_every_season_through_the_cap_and_records_it(trained, frame, tmp_path) -> None:
    learner, result, _ = trained
    entry = _fit(learner, frame, result, tmp_path)
    assert entry["train_seasons"] == SEASONS
    assert entry["max_train_season"] == 2024 <= LAST_COMPLETE_SEASON
    assert entry["n_rows"] == len(frame)
    assert entry["features"] == [f for f in FEATURE_COLS if f in frame.columns]
    assert entry["replay_gate"]["passed"] is True
    assert entry["code_git_sha"] == "abc123"
    assert {"python", "numpy", "pandas", learner if learner == "catboost" else "lightgbm"} <= set(entry["library_versions"])
    assert set(entry["null_rates"]) == set(entry["features"])
    assert (tmp_path / entry["file"]).exists()
    assert entry["sha256"] == __import__("hashlib").sha256((tmp_path / entry["file"]).read_bytes()).hexdigest()


def test_final_model_gets_the_tree_count_the_replay_early_stopped_at(trained, frame, tmp_path) -> None:
    learner, result, _ = trained
    entry = _fit(learner, frame, result, tmp_path)
    replay = final_fit.replay_gate(learner, STAT, POS, frame, result.best_params, result.oof_df)
    assert entry["trees"] == replay.best_iteration
    assert entry["trees_rule"] == "replay_best_iteration"
    key = "n_estimators" if learner == "lgbm" else "iterations"
    assert entry["params"][key] == replay.best_iteration


def test_saved_model_round_trips_and_predicts_like_the_fitted_one(trained, frame, tmp_path) -> None:
    learner, result, _ = trained
    entry = _fit(learner, frame, result, tmp_path)
    model_path = tmp_path / entry["file"]
    rows = frame[frame["season"] == 2024].head(10)
    X = rows[entry["features"]].fillna(-9999.0)
    if learner == "lgbm":
        import lightgbm as lgb

        pred = lgb.Booster(model_file=str(model_path)).predict(X)
    else:
        import catboost as cb

        model = cb.CatBoostRegressor()
        model.load_model(str(model_path))
        pred = model.predict(X.values)
    assert np.isfinite(pred).all() and len(pred) == 10
    assert entry["file"] == f"{learner}_{STAT}_{POS}.{'txt' if learner == 'lgbm' else 'cbm'}"


def test_failed_replay_gate_writes_no_model(trained, frame, tmp_path) -> None:
    learner, result, _ = trained
    bad = dict(result.best_params, learning_rate=0.25)
    with pytest.raises(final_fit.ReplayGateError):
        final_fit.fit_cell(learner, STAT, POS, df=frame, params=bad, oof=result.oof_df,
                           out_dir=tmp_path, release_id="causal_test", code_sha="abc123")
    assert [p.name for p in tmp_path.iterdir() if p.name != "catboost_info"] == []


def test_final_fit_window_refuses_any_season_above_the_cap(frame, tmp_path) -> None:
    future = _make_synthetic_df([LAST_COMPLETE_SEASON + 1], n_per_season=20)
    with pytest.raises(ValueError, match="season cap"):
        final_fit.fit_window(pd.concat([frame, future], ignore_index=True))
    assert final_fit.fit_window(frame)["season"].max() == 2024


def test_code_sha_is_required(trained, frame, tmp_path, monkeypatch) -> None:
    learner, result, _ = trained
    monkeypatch.delenv("CODE_GIT_SHA", raising=False)
    monkeypatch.setattr(final_fit, "_git_sha", lambda: None)
    with pytest.raises(final_fit.FinalFitError, match="code sha"):
        final_fit.fit_cell(learner, STAT, POS, df=frame, params=result.best_params, oof=result.oof_df,
                           out_dir=tmp_path, release_id="causal_test", code_sha=None)


# ── inputs come from files, never from Optuna ─────────────────────────────────

def test_inputs_are_read_from_params_files_and_optuna_is_never_called(trained, tmp_path, monkeypatch) -> None:
    learner, result, oof_dir = trained
    import optuna

    def boom(*a, **k):
        raise AssertionError("final fit must never run Optuna")

    monkeypatch.setattr(optuna, "create_study", boom)
    params, oof, oof_path = final_fit.load_cell_inputs(oof_dir, oof_dir, learner, STAT, POS)
    assert params == result.best_params
    assert oof_path.name.startswith(f"{learner}_{STAT}_{POS}_")
    assert len(oof) == len(result.oof_df)


def test_missing_params_file_is_an_error_not_a_default(trained, tmp_path) -> None:
    learner, result, oof_dir = trained
    empty = tmp_path / "no_params"
    empty.mkdir()
    for csv in oof_dir.glob("*.csv"):
        (empty / csv.name).write_text(csv.read_text())
    with pytest.raises(ParamsError, match="cannot read params"):
        final_fit.load_cell_inputs(empty, empty, learner, STAT, POS)


# ── MODELS.json assembly ──────────────────────────────────────────────────────

def _ridge_dir(tmp_path: Path, weights: dict) -> Path:
    d = tmp_path / "release"
    d.mkdir(exist_ok=True)
    (d / f"ridge_{STAT}_{POS}_coefs.json").write_text(json.dumps(
        {"learner_order": ["lgbm", "catboost"], "weights": weights, "intercept": 0.0}))
    return d


def test_assemble_requires_every_model_the_ridge_files_use(frame, tmp_path) -> None:
    out = tmp_path / "models"
    result = _train("lgbm", frame, tmp_path / "oof_l")
    _fit("lgbm", frame, result, out)
    release = _ridge_dir(tmp_path, {"lgbm": 0.5, "catboost": 0.5})
    with pytest.raises(final_fit.FinalFitError, match="catboost"):
        final_fit.assemble_models_json(out, release, "causal_test")
    assert not (out / "MODELS.json").exists()


def test_assemble_refuses_a_model_no_ridge_file_uses(frame, tmp_path) -> None:
    out = tmp_path / "models"
    for learner in ("lgbm", "catboost"):
        _fit(learner, frame, _train(learner, frame, tmp_path / f"oof_{learner}"), out)
    release = _ridge_dir(tmp_path, {"lgbm": 1.0, "catboost": 0.0})
    with pytest.raises(final_fit.FinalFitError, match="no ridge file uses"):
        final_fit.assemble_models_json(out, release, "causal_test")


def test_assembled_models_json_passes_the_manifest_cross_check(frame, tmp_path) -> None:
    """The writer and the loader of commit 3 agree on the format."""
    import hashlib

    root = tmp_path / "repo"
    rel = "releases/candidates/causal_test"
    models = root / rel / "models"
    for learner in ("lgbm", "catboost"):
        _fit(learner, frame, _train(learner, frame, tmp_path / f"oof_{learner}"), models)
    release = _ridge_dir(tmp_path, {"lgbm": 0.4, "catboost": 0.6})
    ridge = root / rel / f"ridge_{STAT}_{POS}_coefs.json"
    ridge.write_text((release / ridge.name).read_text())
    meta_path = final_fit.assemble_models_json(models, root / rel, "causal_test")
    assert meta_path == models / "MODELS.json"

    stack = root / f"ml/oof/stack_{STAT}_{POS}_20261114.csv"
    stack.parent.mkdir(parents=True)
    stack.write_text("x")
    pinned = [p.relative_to(root).as_posix() for p in [*models.glob("*.txt"), *models.glob("*.cbm"), meta_path]]
    digests = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in pinned}
    for p in (stack, ridge):
        digests[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {
        "artifacts": {"stacks": [stack.relative_to(root).as_posix()],
                      "ridge_coefficients": [ridge.relative_to(root).as_posix()], "base_models": pinned},
        "artifact_digests": digests,
    }
    (root / "releases").mkdir(exist_ok=True)
    (root / "releases/current_baseline.json").write_text(json.dumps(manifest))
    loaded = load_manifest(root / "releases/current_baseline.json", root=root)
    assert loaded.resolve_base_model(STAT, POS, "lgbm").exists()
    assert loaded.base_model_meta(STAT, POS, "catboost")["release_id"] == "causal_test"
