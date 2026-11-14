"""Per-row bucket attributions add up to the served OOF value (decision 0008 item 6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml import bucket_attribution as ba
from ml.bucket_weights import BUCKET_KEYS, UNWEIGHTED_KEY
from ml.xgb_model import FEATURE_COLS, _make_walk_forward_folds
from backend.tests.test_lgbm_model import _make_synthetic_df

SEASONS = [2022, 2023, 2024]
TARGET = "actual_receiving_yards"


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _make_synthetic_df(SEASONS, n_per_season=80)


def _collect(trainer_module, params, frame):
    sink: list[pd.DataFrame] = []
    features = [f for f in FEATURE_COLS if f in frame.columns]
    folds = _make_walk_forward_folds(SEASONS)
    _, oof, _ = trainer_module._walk_forward_cv(
        frame, folds, features, TARGET, params,
        on_fold=trainer_module._fold_attribution_collector(features, sink),
    )
    return oof, pd.concat(sink, ignore_index=True)


def test_lgbm_attributions_add_up_to_oof(frame):
    from ml import lgbm_model

    params = {**lgbm_model.DEFAULT_LGBM_PARAMS, "n_estimators": 40}
    oof, attr = _collect(lgbm_model, params, frame)
    assert len(attr) == len(oof)
    assert list(attr["player_id"]) == list(oof["player_id"])
    assert np.allclose(attr["y_pred"], oof["y_pred"])
    assert ba.max_residual(attr) <= ba.ATTRIBUTION_TOLERANCE


def test_catboost_attributions_add_up_to_oof(frame):
    from ml import catboost_model

    params = {**catboost_model.DEFAULT_CATBOOST_PARAMS, "iterations": 40, "nan_mode": "Min"}
    oof, attr = _collect(catboost_model, params, frame)
    assert len(attr) == len(oof)
    assert np.allclose(attr["y_pred"], oof["y_pred"])
    assert ba.max_residual(attr) <= ba.ATTRIBUTION_TOLERANCE


def test_assert_additive_rejects_a_wrong_row():
    row = {c: 0.0 for c in ba.ATTR_COLUMNS}
    row.update(base=10.0, y_pred=10.0)
    good = pd.DataFrame([row])
    assert ba.assert_additive(good, context="t") == 0.0
    bad = good.copy()
    bad.loc[0, BUCKET_KEYS[0]] = 0.5
    with pytest.raises(ba.AttributionError):
        ba.assert_additive(bad, context="t")


def test_group_contributions_rejects_unknown_feature():
    with pytest.raises(KeyError):
        ba.group_contributions(np.zeros((2, 2)), ["not_a_feature"])


def test_group_contributions_rejects_wrong_width():
    with pytest.raises(ba.AttributionError):
        ba.group_contributions(np.zeros((2, 5)), [FEATURE_COLS[0]])


def _learner_frame(base: float, bucket_value: float) -> pd.DataFrame:
    row = {c: 0.0 for c in ba.ATTR_COLUMNS}
    row.update(base=base, **{BUCKET_KEYS[0]: bucket_value})
    row.update(player_id="p1", game_id="g1")
    return pd.DataFrame([row])


def test_blend_is_the_meta_weighted_sum_with_intercept_in_base():
    lg = _learner_frame(base=10.0, bucket_value=4.0)   # lgbm predicts 14
    cb = _learner_frame(base=12.0, bucket_value=2.0)   # catboost predicts 14
    meta = pd.DataFrame([{
        "player_id": "p1", "game_id": "g1", "season": 2024, "week": 1,
        "meta_intercept": 1.0, "meta_w_lgbm": 0.25, "meta_w_catboost": 0.75,
        "y_pred": 1.0 + 0.25 * 14 + 0.75 * 14,
    }])
    out = ba.blend({"lgbm": lg, "catboost": cb}, meta)
    assert out["base"].iloc[0] == pytest.approx(1.0 + 0.25 * 10 + 0.75 * 12)
    assert out[BUCKET_KEYS[0]].iloc[0] == pytest.approx(0.25 * 4 + 0.75 * 2)
    assert out[UNWEIGHTED_KEY].iloc[0] == 0.0
    assert ba.assert_additive(out, context="blend") <= ba.ATTRIBUTION_TOLERANCE


def test_blend_fails_when_a_learner_row_is_missing():
    lg = _learner_frame(10.0, 4.0)
    meta = pd.DataFrame([{
        "player_id": "other", "game_id": "g9", "season": 2024, "week": 1,
        "meta_intercept": 0.0, "meta_w_lgbm": 1.0, "y_pred": 14.0,
    }])
    with pytest.raises(ba.AttributionError):
        ba.blend({"lgbm": lg}, meta)


def test_write_attribution_file_sits_beside_the_oof(tmp_path):
    attr = _learner_frame(1.0, 0.0).assign(season=2024, week=1, y_pred=1.0)
    path = ba.write_attribution_file(attr, tmp_path / "lgbm_x_WR_ab12.csv")
    assert path.name == "attr_lgbm_x_WR_ab12.parquet"
    assert pd.read_parquet(path).shape == attr.shape
    assert not list(tmp_path.glob("*.tmp"))


# ── Stack blend end to end (synthetic base learners) ───────────────────────────

def _write_base_with_attr(oof_path, oof: pd.DataFrame, seed: int) -> None:
    """Write a base OOF CSV and an attribution file that adds up to its y_pred."""
    rng = np.random.default_rng(seed)
    oof.to_csv(oof_path, index=False)
    n = len(oof)
    attr = oof[["player_id", "game_id", "season", "week"]].copy()
    attr["y_pred"] = oof["y_pred"].to_numpy()
    attr["base"] = 20.0 + rng.normal(0, 1, n)
    remaining = attr["y_pred"].to_numpy() - attr["base"].to_numpy()
    for key in BUCKET_KEYS:
        attr[key] = rng.normal(0, 2, n)
        remaining = remaining - attr[key].to_numpy()
    attr[UNWEIGHTED_KEY] = remaining
    ba.write_attribution_file(attr, oof_path)


@pytest.fixture()
def two_learners(tmp_path):
    from backend.tests.test_stacking_ensemble import _make_oof_pair

    catboost, lgbm = _make_oof_pair(n_folds=3, n_per_fold=30, seed=7)
    cb_path = tmp_path / "catboost_receiving_yards_abc12345.csv"
    lg_path = tmp_path / "lgbm_receiving_yards_def67890.csv"
    _write_base_with_attr(cb_path, catboost, seed=1)
    _write_base_with_attr(lg_path, lgbm, seed=2)
    return cb_path, lg_path, tmp_path


def test_stack_attribution_adds_up_to_the_served_value(two_learners):
    from ml.stacking_ensemble import stack

    cb_path, lg_path, out = two_learners
    result = stack([cb_path, lg_path], target="receiving_yards", mlflow_tracking_uri="",
                   out_dir=out, collect_attributions=True)
    attr = pd.read_parquet(result.attr_path)
    stack_csv = pd.read_csv(result.oof_path)
    assert result.attr_path.name == f"attr_{result.oof_path.stem}.parquet"
    assert len(attr) == len(stack_csv)
    merged = stack_csv.merge(attr, on=["player_id", "game_id"], suffixes=("", "_attr"))
    assert np.allclose(merged["y_pred"], merged["y_pred_attr"])
    assert ba.max_residual(attr) <= ba.ATTRIBUTION_TOLERANCE


def test_validated_best_base_attribution_is_that_learners_own(two_learners, monkeypatch):
    from ml import stacking_ensemble as se

    monkeypatch.setattr(se, "promotion_gate", lambda **_: (False, "forced"))
    cb_path, lg_path, out = two_learners
    result = se.stack([cb_path, lg_path], target="receiving_yards", mlflow_tracking_uri="",
                      out_dir=out, collect_attributions=True)
    assert result.selection_mode == "validated_best_base"
    attr = pd.read_parquet(result.attr_path)
    own = pd.read_parquet(ba.attribution_path_for(
        cb_path if result.selected_learner == "catboost" else lg_path))
    own = own.set_index(["player_id", "game_id"]).loc[
        list(zip(attr["player_id"], attr["game_id"]))]
    for column in ba.ATTR_COLUMNS:
        assert np.allclose(attr[column].to_numpy(), own[column].to_numpy())


def test_stack_refuses_without_base_attributions(two_learners):
    from ml.stacking_ensemble import stack

    cb_path, lg_path, out = two_learners
    ba.attribution_path_for(lg_path).unlink()
    with pytest.raises(ba.AttributionError, match="--attributions"):
        stack([cb_path, lg_path], target="receiving_yards", mlflow_tracking_uri="",
              out_dir=out, collect_attributions=True)
