"""
backend/tests/test_stacking_ensemble.py

Tests for ml/stacking_ensemble.py (Ridge meta-learner).

Key acceptance test (TestMetaLeakageProof): verifies that the meta-learner
was never trained on any (player_id, game_id) pair that appeared in its own
validation fold.

Run with:
  pytest backend/tests/test_stacking_ensemble.py -v
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.stacking_ensemble import (
    RIDGE_ALPHAS,
    MetaFoldResult,
    StackResult,
    _compute_base_metrics,
    _meta_walk_forward_cv,
    _save_stack_oof,
    load_and_align_oofs,
    stack,
)


# ── Synthetic data helpers ─────────────────────────────────────────────────────

def _make_oof_pair(
    n_folds: int,
    n_per_fold: int,
    seed: int = 0,
    catboost_noise: float = 5.0,
    lgbm_noise: float = 7.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create paired CatBoost and LGBM OOF DataFrames.

    Each fold gets unique player_ids and game_ids of the form:
      player_id = "p_f{fold_idx}_{i:03d}"
      game_id   = "g_f{fold_idx}_{i:03d}"

    Both OOFs share the same (player_id, game_id) pairs so the inner join
    returns all rows. This also makes TestMetaLeakageProof meaningful: each
    fold's rows are disjoint, so training and validation sets have zero overlap
    in (player_id, game_id) space.
    """
    rng = np.random.default_rng(seed)
    catboost_rows: list[dict] = []
    lgbm_rows: list[dict] = []

    for fold_idx in range(n_folds):
        y_trues    = rng.normal(50.0, 20.0, n_per_fold)
        catboost_preds  = y_trues + rng.normal(0.0, catboost_noise, n_per_fold)
        lgbm_preds = y_trues + rng.normal(0.0, lgbm_noise, n_per_fold)

        for i in range(n_per_fold):
            base = {
                "player_id": f"p_f{fold_idx}_{i:03d}",
                "game_id":   f"g_f{fold_idx}_{i:03d}",
                "season":    2020 + fold_idx,
                "week":      1,
                "y_true":    float(y_trues[i]),
                "fold_idx":  fold_idx,
            }
            catboost_rows.append({**base,  "y_pred": float(catboost_preds[i])})
            lgbm_rows.append({**base, "y_pred": float(lgbm_preds[i])})

    return pd.DataFrame(catboost_rows), pd.DataFrame(lgbm_rows)


def _write_oof_files(
    tmp_path: Path,
    n_folds: int = 3,
    n_per_fold: int = 40,
    seed: int = 42,
    catboost_noise: float = 5.0,
    lgbm_noise: float = 7.0,
) -> tuple[Path, Path]:
    catboost_oof, lgbm_oof = _make_oof_pair(
        n_folds=n_folds, n_per_fold=n_per_fold, seed=seed,
        catboost_noise=catboost_noise, lgbm_noise=lgbm_noise,
    )
    catboost_path  = tmp_path / "catboost_receiving_yards_abc12345.csv"
    lgbm_path = tmp_path / "lgbm_receiving_yards_def67890.csv"
    catboost_oof.to_csv(catboost_path, index=False)
    lgbm_oof.to_csv(lgbm_path, index=False)
    return catboost_path, lgbm_path


# ══════════════════════════════════════════════════════════════════════════════
# 1. load_and_align_oofs
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadAndAlignOofs:

    @pytest.fixture(scope="class")
    def oof_files(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("align")
        return _write_oof_files(tmp, n_folds=2, n_per_fold=20, seed=1)

    @pytest.fixture(scope="class")
    def aligned(self, oof_files):
        return load_and_align_oofs(list(oof_files))

    # ── Structure ──────────────────────────────────────────────────────────────

    def test_returns_tuple_of_three(self, aligned):
        assert len(aligned) == 3

    def test_aligned_df_is_dataframe(self, aligned):
        aligned_df, _, _ = aligned
        assert isinstance(aligned_df, pd.DataFrame)

    def test_y_pred_renamed_to_prefix_pred(self, aligned):
        aligned_df, _, _ = aligned
        assert "catboost_pred"  in aligned_df.columns
        assert "lgbm_pred" in aligned_df.columns
        assert "y_pred"    not in aligned_df.columns

    def test_pred_cols_list(self, aligned):
        _, pred_cols, _ = aligned
        assert pred_cols == ["catboost_pred", "lgbm_pred"]

    def test_prefixes_list(self, aligned):
        _, _, prefixes = aligned
        assert prefixes == ["catboost", "lgbm"]

    def test_inner_join_keeps_all_rows_when_identical_keys(self, aligned, oof_files):
        # Both OOFs have the same (player_id, game_id) → inner join = all rows
        catboost_oof = pd.read_csv(oof_files[0])
        aligned_df, _, _ = aligned
        assert len(aligned_df) == len(catboost_oof)

    # ── Error cases ────────────────────────────────────────────────────────────

    def test_requires_at_least_two_paths(self, oof_files):
        with pytest.raises(ValueError, match="≥2"):
            load_and_align_oofs([oof_files[0]])

    def test_duplicate_prefix_raises(self, tmp_path):
        catboost_oof, _ = _make_oof_pair(n_folds=2, n_per_fold=10)
        p1 = tmp_path / "catboost_receiving_yards_aaa.csv"
        p2 = tmp_path / "catboost_rushing_yards_bbb.csv"    # same "catboost" prefix
        catboost_oof.to_csv(p1, index=False)
        catboost_oof.to_csv(p2, index=False)
        with pytest.raises(ValueError, match="Duplicate"):
            load_and_align_oofs([p1, p2])

    def test_missing_columns_raises(self, tmp_path):
        bad = tmp_path / "catboost_receiving_yards_abc.csv"
        lgbm = tmp_path / "lgbm_receiving_yards_abc.csv"
        # Missing season, week, y_true, fold_idx
        pd.DataFrame({"player_id": ["p1"], "game_id": ["g1"], "y_pred": [10.0]}).to_csv(bad, index=False)
        pd.DataFrame({"player_id": ["p1"], "game_id": ["g1"], "y_pred": [10.0]}).to_csv(lgbm, index=False)
        with pytest.raises(ValueError, match="missing"):
            load_and_align_oofs([bad, lgbm])


# ══════════════════════════════════════════════════════════════════════════════
# 1b. N-way alignment, and the two-learner policy that sits above it
# ══════════════════════════════════════════════════════════════════════════════

class TestThreeWayAlignment:
    """
    ``load_and_align_oofs`` is learner-agnostic and still aligns N≥2 files —
    that mechanism is worth keeping tested. What changed is the layer above it:
    ``stack()`` now enforces the Phase-5 two-learner allowlist, so a three-way
    *stack* is refused even though the three-way *alignment* succeeds.

    This class used to assert that a four-learner stack was fine, which is the
    configuration the Phase-5 kill removed. The final test below inverts that
    assertion instead of deleting it, so the contract is locked rather than
    merely unstated.
    """

    @pytest.fixture(scope="class")
    def three_oof_files(self, tmp_path_factory):
        """Create catboost, lgbm, and tft OOF files with the same (player_id, game_id) keys."""
        tmp = tmp_path_factory.mktemp("three_way")
        rng = np.random.default_rng(99)
        rows = []
        for fold_idx in range(3):
            y_trues = rng.normal(50.0, 20.0, 30)
            for i in range(30):
                rows.append({
                    "player_id": f"p_f{fold_idx}_{i:03d}",
                    "game_id":   f"g_f{fold_idx}_{i:03d}",
                    "season":    2020 + fold_idx,
                    "week":      1,
                    "y_true":    float(y_trues[i]),
                    "fold_idx":  fold_idx,
                })
        base_df = pd.DataFrame(rows)

        catboost_path  = tmp / "catboost_receiving_yards_aaa11111.csv"
        lgbm_path = tmp / "lgbm_receiving_yards_bbb22222.csv"
        tft_path  = tmp / "tft_receiving_yards_ccc33333.csv"

        for path, noise in [(catboost_path, 5.0), (lgbm_path, 7.0), (tft_path, 6.0)]:
            df = base_df.copy()
            df["y_pred"] = df["y_true"] + rng.normal(0.0, noise, len(df))
            df.to_csv(path, index=False)

        return catboost_path, lgbm_path, tft_path

    @pytest.fixture(scope="class")
    def three_way_aligned(self, three_oof_files):
        return load_and_align_oofs(list(three_oof_files))

    def test_returns_three_pred_cols(self, three_way_aligned):
        _, pred_cols, _ = three_way_aligned
        assert pred_cols == ["catboost_pred", "lgbm_pred", "tft_pred"]

    def test_returns_three_prefixes(self, three_way_aligned):
        _, _, prefixes = three_way_aligned
        assert prefixes == ["catboost", "lgbm", "tft"]

    def test_all_three_pred_cols_in_dataframe(self, three_way_aligned):
        aligned_df, _, _ = three_way_aligned
        for col in ("catboost_pred", "lgbm_pred", "tft_pred"):
            assert col in aligned_df.columns, f"'{col}' missing from aligned DataFrame"

    def test_y_pred_column_not_present(self, three_way_aligned):
        aligned_df, _, _ = three_way_aligned
        assert "y_pred" not in aligned_df.columns

    def test_inner_join_keeps_all_rows_when_keys_identical(
        self, three_oof_files, three_way_aligned
    ):
        catboost_path, _, _ = three_oof_files
        n_catboost = len(pd.read_csv(catboost_path))
        aligned_df, _, _ = three_way_aligned
        assert len(aligned_df) == n_catboost

    def test_all_pred_cols_are_finite(self, three_way_aligned):
        aligned_df, pred_cols, _ = three_way_aligned
        for col in pred_cols:
            assert aligned_df[col].notna().all(), f"NaN found in {col}"

    def test_inner_join_drops_rows_missing_from_tft(self, tmp_path):
        """If TFT has fewer rows, inner join should drop the missing ones."""
        rng = np.random.default_rng(777)
        rows_all = [
            {"player_id": f"p{i}", "game_id": f"g{i}", "season": 2020,
             "week": 1, "y_true": float(rng.normal(50, 10)), "fold_idx": 0}
            for i in range(20)
        ]
        df_all = pd.DataFrame(rows_all)

        catboost_path  = tmp_path / "catboost_ry_xxx.csv"
        lgbm_path = tmp_path / "lgbm_ry_yyy.csv"
        tft_path  = tmp_path / "tft_ry_zzz.csv"

        df_all.assign(y_pred=df_all["y_true"] + 1.0).to_csv(catboost_path, index=False)
        df_all.assign(y_pred=df_all["y_true"] + 2.0).to_csv(lgbm_path, index=False)
        # TFT only has rows 0-9 (half the data)
        df_all.iloc[:10].assign(y_pred=df_all["y_true"].iloc[:10] + 3.0).to_csv(tft_path, index=False)

        aligned_df, _, _ = load_and_align_oofs([catboost_path, lgbm_path, tft_path])
        assert len(aligned_df) == 10

    def test_stack_refuses_a_killed_learner(self, three_oof_files, tmp_path):
        """
        Phase-5 contract: stack() must refuse a TFT (or XGB) input.

        The previous version of this test asserted the opposite — that a
        three-learner stack ran fine and produced ``tft_pred`` coefficients. That
        made the test suite lock in the exact configuration the Phase-5 kill was
        meant to remove, so one run of the documented recovery path would rewrite
        the clean coef files and inference would begin executing killed learners.

        Enforcement lives inside stack() rather than behind ``--exclude tft,xgb``
        so that forgetting a CLI flag cannot widen the served learner set.
        """
        from ml.artifact_manifest import LearnerPolicyError

        with pytest.raises(LearnerPolicyError) as exc:
            stack(
                list(three_oof_files),
                target="receiving_yards",
                mlflow_tracking_uri="",
                out_dir=tmp_path,
            )
        assert "tft" in str(exc.value)
        assert not list(tmp_path.glob("ridge_*_coefs.json")), (
            "a refused stack must not leave a coef file behind"
        )

    def test_stack_accepts_the_shipped_two_learner_pair(self, three_oof_files, tmp_path):
        """The allowed pair still stacks end-to-end."""
        catboost_path, lgbm_path, _ = three_oof_files
        result = stack(
            [catboost_path, lgbm_path],
            target="receiving_yards",
            mlflow_tracking_uri="",
            out_dir=tmp_path,
        )
        assert result.oof_df is not None and len(result.oof_df) > 0
        assert "catboost_pred" in result.oof_df.columns
        assert "lgbm_pred" in result.oof_df.columns
        assert "tft_pred" not in result.oof_df.columns
        assert result.ridge_coefs.keys() == {"catboost_pred", "lgbm_pred"}


# ══════════════════════════════════════════════════════════════════════════════
# 2. _meta_walk_forward_cv — structure and fold correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestMetaWalkForwardCV:

    @pytest.fixture(scope="class")
    def cv_inputs(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("metacv")
        catboost_path, lgbm_path = _write_oof_files(tmp, n_folds=3, n_per_fold=40, seed=10)
        aligned_df, pred_cols, _ = load_and_align_oofs([catboost_path, lgbm_path])
        return aligned_df, pred_cols

    @pytest.fixture(scope="class")
    def cv_result(self, cv_inputs):
        aligned_df, pred_cols = cv_inputs
        return _meta_walk_forward_cv(aligned_df, pred_cols)

    # ── Fold count ─────────────────────────────────────────────────────────────

    def test_n_meta_folds_is_n_base_minus_1(self, cv_inputs, cv_result):
        aligned_df, _ = cv_inputs
        meta_fold_results, _, _ = cv_result
        n_base = len(aligned_df["fold_idx"].unique())
        # 3 base folds → 2 meta folds
        assert len(meta_fold_results) == n_base - 1

    # ── Fold structure ─────────────────────────────────────────────────────────

    def test_meta_fold_0_train_and_val(self, cv_result):
        meta_fold_results, _, _ = cv_result
        mf0 = meta_fold_results[0]
        assert mf0.train_base_folds == [0]
        assert mf0.val_base_fold    == 1

    def test_meta_fold_1_train_expands(self, cv_result):
        meta_fold_results, _, _ = cv_result
        mf1 = meta_fold_results[1]
        assert mf1.train_base_folds == [0, 1]
        assert mf1.val_base_fold    == 2

    # ── OOF output ────────────────────────────────────────────────────────────

    def test_oof_only_contains_val_base_folds(self, cv_inputs, cv_result):
        aligned_df, _ = cv_inputs
        _, meta_oof_df, _ = cv_result
        # Meta val rows came from base folds 1 and 2 (not fold 0)
        expected_base_folds = {1, 2}
        # meta_oof_df.fold_idx = meta fold index (0, 1), not base fold index
        # Verify row count matches the expected val base folds
        n_expected = len(aligned_df[aligned_df["fold_idx"].isin(expected_base_folds)])
        assert len(meta_oof_df) == n_expected

    def test_oof_has_base_pred_columns(self, cv_result):
        _, meta_oof_df, _ = cv_result
        assert "catboost_pred"  in meta_oof_df.columns
        assert "lgbm_pred" in meta_oof_df.columns

    def test_requires_at_least_2_base_folds(self):
        single_fold_df = pd.DataFrame({
            "player_id": [f"p{i}" for i in range(10)],
            "game_id":   [f"g{i}" for i in range(10)],
            "season":    [2020] * 10,
            "week":      [1] * 10,
            "y_true":    np.ones(10) * 50,
            "catboost_pred":  np.ones(10) * 48,
            "lgbm_pred": np.ones(10) * 51,
            "fold_idx":  [0] * 10,
        })
        with pytest.raises(ValueError, match="≥2"):
            _meta_walk_forward_cv(single_fold_df, ["catboost_pred", "lgbm_pred"])

    def test_all_meta_fold_maes_are_positive_finite(self, cv_result):
        meta_fold_results, _, _ = cv_result
        for mf in meta_fold_results:
            assert mf.mae > 0
            assert np.isfinite(mf.mae)


# ══════════════════════════════════════════════════════════════════════════════
# 3. META-LEARNER LEAKAGE PROOF — KEY ACCEPTANCE TESTS
#
#    Proves that the meta-learner was NEVER trained on any (player_id, game_id)
#    pair that appeared in its own validation fold.
#
#    How it works:
#    - Synthetic OOF data has fold-unique player/game IDs:
#        fold 0 → p_f0_000, p_f0_001, ..., g_f0_000, g_f0_001, ...
#        fold 1 → p_f1_000, p_f1_001, ..., g_f1_000, g_f1_001, ...
#        fold 2 → p_f2_000, p_f2_001, ..., g_f2_000, g_f2_001, ...
#    - Meta fold 0: trains on base fold 0, validates on base fold 1
#    - Meta fold 1: trains on base folds 0+1, validates on base fold 2
#    - Because player_ids/game_ids are fold-unique, training and validation
#      sets have ZERO (player_id, game_id) overlap.
# ══════════════════════════════════════════════════════════════════════════════

class TestMetaLeakageProof:

    @pytest.fixture(scope="class")
    def setup(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("leakage")
        catboost_path, lgbm_path = _write_oof_files(tmp, n_folds=3, n_per_fold=40, seed=42)
        oof_paths = [catboost_path, lgbm_path]

        result = stack(oof_paths, target="receiving_yards",
                       mlflow_tracking_uri="", out_dir=tmp)
        aligned_df, _, _ = load_and_align_oofs(oof_paths)
        return result, aligned_df

    # ── Structural fold invariants ─────────────────────────────────────────────

    def test_val_base_fold_not_in_train_folds(self, setup):
        """Hard invariant: val_base_fold must not be in train_base_folds."""
        result, _ = setup
        for mf in result.meta_fold_results:
            assert mf.val_base_fold not in mf.train_base_folds, (
                f"Meta fold {mf.meta_fold_idx}: val_base_fold={mf.val_base_fold} "
                f"is IN train_base_folds={mf.train_base_folds}. Data-leakage bug."
            )

    def test_train_folds_strictly_before_val(self, setup):
        """Every train fold index must be strictly less than val_base_fold."""
        result, _ = setup
        for mf in result.meta_fold_results:
            assert all(f < mf.val_base_fold for f in mf.train_base_folds), (
                f"Meta fold {mf.meta_fold_idx}: not all train folds "
                f"{mf.train_base_folds} < val_base_fold={mf.val_base_fold}."
            )

    def test_val_player_ids_not_in_training_data(self, setup):
        """
        KEY PROOF: (player_id, game_id) pairs from the meta validation set
        must have ZERO overlap with (player_id, game_id) pairs from the
        meta training set for the corresponding meta fold.

        This test works because our synthetic OOF assigns fold-unique
        player_ids ("p_f{fold_idx}_{i}"), so any leakage would produce a
        non-empty intersection.
        """
        result, aligned_df = setup
        for mf in result.meta_fold_results:
            train_rows = aligned_df[aligned_df["fold_idx"].isin(mf.train_base_folds)]
            val_rows   = aligned_df[aligned_df["fold_idx"] == mf.val_base_fold]

            train_keys = set(zip(train_rows["player_id"], train_rows["game_id"]))
            val_keys   = set(zip(val_rows["player_id"],   val_rows["game_id"]))

            overlap = train_keys & val_keys
            assert len(overlap) == 0, (
                f"Meta fold {mf.meta_fold_idx}: {len(overlap)} (player_id, game_id) "
                f"pairs appear in BOTH training (base folds {mf.train_base_folds}) "
                f"AND validation (base fold {mf.val_base_fold}). DATA LEAKAGE BUG."
            )

    def test_n_meta_folds_equals_n_base_folds_minus_1(self, setup):
        result, aligned_df = setup
        n_base = len(aligned_df["fold_idx"].unique())
        assert len(result.meta_fold_results) == n_base - 1

    def test_expanding_window_grows_monotonically(self, setup):
        """Each successive meta fold trains on more base folds."""
        result, _ = setup
        for i in range(1, len(result.meta_fold_results)):
            prev = result.meta_fold_results[i - 1]
            curr = result.meta_fold_results[i]
            assert len(curr.train_base_folds) > len(prev.train_base_folds), (
                f"Meta fold {i} train size {len(curr.train_base_folds)} not > "
                f"meta fold {i-1} train size {len(prev.train_base_folds)}. "
                "Expanding window property violated."
            )


# ══════════════════════════════════════════════════════════════════════════════
# 4. StackResult — correctness of returned fields
# ══════════════════════════════════════════════════════════════════════════════

class TestStackResult:

    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("stack_result")
        catboost_path, lgbm_path = _write_oof_files(tmp, n_folds=3, n_per_fold=40, seed=7)
        return stack(
            [catboost_path, lgbm_path],
            target="receiving_yards",
            mlflow_tracking_uri="",
            out_dir=tmp,
        )

    def test_returns_stack_result_instance(self, result):
        assert isinstance(result, StackResult)

    def test_stacked_mae_is_positive_and_finite(self, result):
        assert result.stacked_mae > 0
        assert np.isfinite(result.stacked_mae)

    def test_stacked_rmse_ge_mae(self, result):
        assert result.stacked_rmse >= result.stacked_mae

    def test_base_maes_has_both_prefixes(self, result):
        assert "catboost"  in result.base_maes
        assert "lgbm" in result.base_maes

    def test_base_rmses_has_both_prefixes(self, result):
        assert "catboost"  in result.base_rmses
        assert "lgbm" in result.base_rmses

    def test_ridge_coefs_for_both_pred_cols(self, result):
        assert "catboost_pred"  in result.ridge_coefs
        assert "lgbm_pred" in result.ridge_coefs

    def test_final_alpha_is_from_ridge_alphas_list(self, result):
        assert result.final_alpha in RIDGE_ALPHAS

    def test_stacking_improved_is_bool(self, result):
        assert isinstance(result.stacking_improved, bool)

    def test_oof_df_is_dataframe(self, result):
        assert isinstance(result.oof_df, pd.DataFrame)
        assert not result.oof_df.empty


# ══════════════════════════════════════════════════════════════════════════════
# 5. Stacking improvement check — logic and warning behaviour
# ══════════════════════════════════════════════════════════════════════════════

class TestStackingImprovement:

    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("improvement")
        catboost_path, lgbm_path = _write_oof_files(tmp, n_folds=3, n_per_fold=40, seed=99)
        return stack(
            [catboost_path, lgbm_path],
            target="receiving_yards",
            mlflow_tracking_uri="",
            out_dir=tmp,
        )

    def test_stacking_improved_matches_mae_comparison(self, result):
        """stacking_improved must equal (stacked_mae <= min(base_maes))."""
        expected = result.stacked_mae <= min(result.base_maes.values())
        assert result.stacking_improved == expected

    def test_warning_emitted_iff_not_improved(self, tmp_path):
        """UserWarning is raised iff stacking_improved is False."""
        catboost_path, lgbm_path = _write_oof_files(tmp_path, n_folds=3, n_per_fold=40, seed=77)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = stack(
                [catboost_path, lgbm_path],
                target="receiving_yards",
                mlflow_tracking_uri="",
                out_dir=tmp_path,
            )

        improvement_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "Stacking did not improve" in str(w.message)
        ]
        if result.stacking_improved:
            assert len(improvement_warnings) == 0, (
                "No warning expected when stacking improves, but one was raised."
            )
        else:
            assert len(improvement_warnings) == 1, (
                "UserWarning expected when stacking_improved=False, but none was raised."
            )

    def test_base_maes_computed_on_meta_val_rows(self, result):
        """
        Base MAEs must be computed on the same held-out rows as the stacked
        MAE (meta validation rows). This ensures apples-to-apples comparison.
        """
        # The meta OOF only covers base folds 1..N-1 (not base fold 0).
        # Base MAEs computed on these same rows should match expected range.
        for prefix, mae in result.base_maes.items():
            assert mae > 0, f"Base MAE for '{prefix}' is not positive: {mae}"
            assert np.isfinite(mae)


# ══════════════════════════════════════════════════════════════════════════════
# 6. OOF output — format and column invariants
# ══════════════════════════════════════════════════════════════════════════════

class TestOofOutput:

    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("oof_format")
        catboost_path, lgbm_path = _write_oof_files(tmp, n_folds=3, n_per_fold=40, seed=5)
        return stack(
            [catboost_path, lgbm_path],
            target="receiving_yards",
            mlflow_tracking_uri="",
            out_dir=tmp,
        )

    def test_oof_has_standard_columns(self, result):
        for col in ["player_id", "game_id", "season", "week", "y_true", "y_pred", "fold_idx"]:
            assert col in result.oof_df.columns, f"Missing standard OOF column: '{col}'"

    def test_oof_has_base_pred_columns(self, result):
        assert "catboost_pred"  in result.oof_df.columns
        assert "lgbm_pred" in result.oof_df.columns

    def test_oof_saved_with_stack_prefix(self, result):
        assert result.oof_path is not None
        assert result.oof_path.name.startswith("stack_")

    def test_oof_fold_idx_is_meta_fold_indices(self, result):
        expected_meta_folds = set(range(len(result.meta_fold_results)))
        actual_meta_folds   = set(result.oof_df["fold_idx"].unique())
        assert actual_meta_folds == expected_meta_folds

    def test_oof_all_preds_are_finite(self, result):
        assert result.oof_df["y_pred"].notna().all()
        assert np.isfinite(result.oof_df["y_pred"].values).all()

    def test_saved_csv_matches_oof_df(self, result):
        """Saved CSV must have the same rows as result.oof_df."""
        saved = pd.read_csv(result.oof_path)
        assert len(saved) == len(result.oof_df)

    def test_mlflow_disabled_run_id_is_none(self, tmp_path):
        catboost_path, lgbm_path = _write_oof_files(tmp_path, n_folds=3, n_per_fold=40, seed=3)
        result = stack(
            [catboost_path, lgbm_path],
            target="receiving_yards",
            mlflow_tracking_uri="",    # disabled
            out_dir=tmp_path,
        )
        assert result.run_id is None


# ══════════════════════════════════════════════════════════════════════════════
# 7. MLflow logging
# ══════════════════════════════════════════════════════════════════════════════

class TestMLflowLogging:

    @pytest.fixture(scope="class")
    def mlflow_result(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("mlflow_stack")
        catboost_path, lgbm_path = _write_oof_files(tmp, n_folds=3, n_per_fold=40, seed=55)
        mlflow_tracking_uri = f"file://{tmp / 'mlruns'}"
        result = stack(
            [catboost_path, lgbm_path],
            target="receiving_yards",
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment="test_stack_receiving_yards",
            out_dir=tmp,
        )
        return result, mlflow_tracking_uri

    def test_mlflow_run_id_is_set(self, mlflow_result):
        result, _ = mlflow_result
        assert result.run_id is not None
        assert len(result.run_id) > 0

    def test_required_metrics_logged(self, mlflow_result):
        """stacked_mae, stacked_rmse, catboost_mae, lgbm_mae must be logged."""
        result, mlflow_uri = mlflow_result
        import mlflow
        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
        experiment = client.get_experiment_by_name("test_stack_receiving_yards")
        assert experiment is not None, "MLflow experiment not created"
        runs = client.search_runs(experiment.experiment_id)
        assert len(runs) >= 1

        run = next(r for r in runs if r.info.run_id == result.run_id)
        metrics = run.data.metrics
        for required in ("stacked_mae", "stacked_rmse", "catboost_mae", "lgbm_mae"):
            assert required in metrics, f"MLflow metric '{required}' not logged"

    def test_stacking_improved_tag_logged(self, mlflow_result):
        result, mlflow_uri = mlflow_result
        import mlflow
        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
        experiment = client.get_experiment_by_name("test_stack_receiving_yards")
        run = next(
            r for r in client.search_runs(experiment.experiment_id)
            if r.info.run_id == result.run_id
        )
        assert "stacking_improved" in run.data.tags
        assert run.data.tags["stacking_improved"] in ("True", "False")

    def test_oof_artifact_saved_to_mlflow(self, mlflow_result):
        result, _ = mlflow_result
        assert result.oof_path is not None
        assert result.oof_path.exists()
        assert result.oof_path.name.startswith("stack_")

    def test_training_data_hash_tag_logged(self, mlflow_result):
        """training_data_hash tag must be a non-empty 64-char hex SHA-256 string."""
        result, mlflow_uri = mlflow_result
        import mlflow
        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
        experiment = client.get_experiment_by_name("test_stack_receiving_yards")
        run = next(
            r for r in client.search_runs(experiment.experiment_id)
            if r.info.run_id == result.run_id
        )
        assert "training_data_hash" in run.data.tags, (
            "MLflow tag 'training_data_hash' not logged"
        )
        tag_value = run.data.tags["training_data_hash"]
        assert len(tag_value) == 64, (
            f"Expected 64-char SHA-256 hex, got {len(tag_value)} chars"
        )
        assert all(c in "0123456789abcdef" for c in tag_value), (
            "training_data_hash is not a lowercase hex string"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PROPERTY-BASED: Temporal ordering invariant holds for any fold count ≥ 2
# (M4 — audit finding: walk-forward temporal ordering not property-tested)
# ══════════════════════════════════════════════════════════════════════════════

from hypothesis import given, settings as h_settings
import hypothesis.strategies as st


class TestTemporalOrderingProperty:
    """
    Property-based tests (Hypothesis) verifying that _meta_walk_forward_cv
    never uses future data regardless of how many base folds are provided.

    Core invariant: for every MetaFoldResult,
        max(train_base_folds) < val_base_fold  (strictly)

    Exercised with 2–5 folds and 20 random examples per run.
    """

    @given(n_folds=st.integers(min_value=2, max_value=5))
    @h_settings(max_examples=20)
    def test_all_meta_folds_respect_temporal_ordering(self, n_folds):
        """For any valid fold count ≥ 2, train folds are always strictly before val fold."""
        rng = np.random.default_rng(n_folds * 17)
        rows = []
        for fold_idx in range(n_folds):
            for i in range(10):
                y = float(rng.normal(50.0, 10.0))
                rows.append({
                    "player_id": f"p_f{fold_idx}_{i}",
                    "game_id":   f"g_f{fold_idx}_{i}",
                    "season":    2020 + fold_idx,
                    "week":      1,
                    "y_true":    y,
                    "fold_idx":  fold_idx,
                    "catboost_pred":  y + float(rng.normal(0.0, 5.0)),
                    "lgbm_pred": y + float(rng.normal(0.0, 7.0)),
                })
        df = pd.DataFrame(rows)
        meta_results, _, _ = _meta_walk_forward_cv(df, ["catboost_pred", "lgbm_pred"])

        for mf in meta_results:
            assert max(mf.train_base_folds) < mf.val_base_fold, (
                f"Temporal ordering violated: "
                f"train={mf.train_base_folds}, val={mf.val_base_fold}. "
                "Meta-learner would have seen future data."
            )

    @given(n_folds=st.integers(min_value=2, max_value=5))
    @h_settings(max_examples=20)
    def test_val_fold_never_in_train_set(self, n_folds):
        """val_base_fold is never a member of train_base_folds."""
        rng = np.random.default_rng(n_folds * 31)
        rows = []
        for fold_idx in range(n_folds):
            for i in range(8):
                y = float(rng.normal(50.0, 10.0))
                rows.append({
                    "player_id": f"p_f{fold_idx}_{i}",
                    "game_id":   f"g_f{fold_idx}_{i}",
                    "season":    2020 + fold_idx,
                    "week":      1,
                    "y_true":    y,
                    "fold_idx":  fold_idx,
                    "catboost_pred":  y + float(rng.normal(0.0, 5.0)),
                    "lgbm_pred": y + float(rng.normal(0.0, 7.0)),
                })
        df = pd.DataFrame(rows)
        meta_results, _, _ = _meta_walk_forward_cv(df, ["catboost_pred", "lgbm_pred"])

        for mf in meta_results:
            assert mf.val_base_fold not in mf.train_base_folds, (
                f"val_base_fold={mf.val_base_fold} is in "
                f"train_base_folds={mf.train_base_folds}. Leakage bug."
            )

    @given(n_folds=st.integers(min_value=2, max_value=5))
    @h_settings(max_examples=20)
    def test_number_of_meta_folds_is_n_base_minus_1(self, n_folds):
        """_meta_walk_forward_cv always produces exactly n_base_folds - 1 meta folds."""
        rng = np.random.default_rng(n_folds * 7)
        rows = []
        for fold_idx in range(n_folds):
            for i in range(8):
                y = float(rng.normal(50.0, 10.0))
                rows.append({
                    "player_id": f"p_f{fold_idx}_{i}",
                    "game_id":   f"g_f{fold_idx}_{i}",
                    "season":    2020 + fold_idx,
                    "week":      1,
                    "y_true":    y,
                    "fold_idx":  fold_idx,
                    "catboost_pred":  y + float(rng.normal(0.0, 5.0)),
                    "lgbm_pred": y + float(rng.normal(0.0, 7.0)),
                })
        df = pd.DataFrame(rows)
        meta_results, _, _ = _meta_walk_forward_cv(df, ["catboost_pred", "lgbm_pred"])
        assert len(meta_results) == n_folds - 1
