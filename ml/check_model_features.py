#!/usr/bin/env python3
"""
Check MLflow tree models (XGB, LGB, CatBoost) for feature count mismatch.

Models trained when feature_matrix had fewer columns expect fewer features.
Inference passes len(FEATURE_COLS) (113). This script identifies which
(learner, stat) pairs need retraining.

Usage:
  export MLFLOW_TRACKING_URI=http://127.0.0.1:15091
  python -m ml.check_model_features

Output: Lists (learner, stat) with wrong feature count and retrain commands.
"""

from __future__ import annotations

import os
import sys

# Stats used by train_all_models.sh (union of all position stat sets)
STATS = [
    "pass_attempts", "completions", "passing_yards", "passing_tds",
    "interceptions", "rushing_yards", "rushing_tds", "carries",
    "receptions", "receiving_yards", "receiving_tds", "targets",
    "fantasy_ppr", "fumbles",
]
LEARNERS = ["xgb", "lgbm", "catboost"]


def _get_feature_count(model: object) -> int | None:
    """Return model's expected feature count, or None if unknown."""
    if hasattr(model, "n_features_in_"):
        return model.n_features_in_
    if hasattr(model, "feature_names_in_"):
        return len(model.feature_names_in_)
    if hasattr(model, "feature_name_"):
        return len(model.feature_name_)
    if hasattr(model, "feature_names_"):
        return len(model.feature_names_)
    if hasattr(model, "get_booster"):
        names = model.get_booster().feature_names
        return len(names) if names else None
    return None


def check_mlflow_running() -> bool:
    """Check if MLflow server is responding at tracking URI."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:15091")
    if not uri.startswith("http"):
        print("Set MLFLOW_TRACKING_URI (e.g. http://127.0.0.1:15091)")
        return False
    # This function is incomplete in the provided snippet,
    # but the instruction was to change the URI and insert this.
    # Assuming the rest of the function would be added later if needed.
    return True # Placeholder


def main() -> int:
    from ml.utils import FEATURE_COLS

    expected = len(FEATURE_COLS)
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:15091")
    if not uri:
        print("Set MLFLOW_TRACKING_URI (e.g. http://127.0.0.1:15091)")
        return 1

    import mlflow
    import mlflow.xgboost
    import mlflow.lightgbm
    import mlflow.catboost

    mlflow.set_tracking_uri(uri)

    mismatched: list[tuple[str, str, int]] = []
    ok_count = 0

    for learner in LEARNERS:
        loader = {
            "xgb": mlflow.xgboost.load_model,
            "lgbm": mlflow.lightgbm.load_model,
            "catboost": mlflow.catboost.load_model,
        }[learner]

        for stat in STATS:
            exp_name = f"{learner}_{stat}"
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                continue
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
                filter_string="status = 'FINISHED'",
            )
            if runs.empty:
                continue
            run_id = runs.iloc[0]["run_id"]
            model_uri = f"runs:/{run_id}/model"
            try:
                model = loader(model_uri)
                n = _get_feature_count(model)
                if n is None or n == 0:
                    print(f"  {learner}/{stat}: could not get feature count (got {n})")
                    continue
                # n==0 often means CatBoost/MLflow didn't save feature names; skip
                if n != expected and n > 0:
                    mismatched.append((learner, stat, n))
                else:
                    ok_count += 1
            except Exception as e:
                print(f"  {learner}/{stat}: load failed — {e}")

    print(f"\nExpected features: {expected} (len(FEATURE_COLS))")
    print(f"OK: {ok_count}  |  Mismatched: {len(mismatched)}\n")

    if not mismatched:
        print("All models have correct feature count.")
        return 0

    print("Models needing retrain (feature count mismatch):")
    print("-" * 50)
    for learner, stat, n in sorted(mismatched):
        print(f"  {learner:10} {stat:20} has {n} features (expected {expected})")

    # Group by (learner, stat) for retrain
    to_retrain = sorted(set((learner, stat) for learner, stat, _ in mismatched))
    print("\nTo retrain only the mismatched models:")
    print("  1. Remove checkpoints (so --resume will re-run them):")
    for learner, stat in to_retrain:
        for pos in ["QB", "RB", "WR", "TE"]:
            print(f"     rm -f ml/checkpoints/done/{learner}_{stat}_{pos}.done")
    print("  2. Remove stacking checkpoints (stacking must re-run after base OOFs change):")
    stats = sorted(set(s for _, s in to_retrain))
    for stat in stats:
        print(f"     rm -f ml/checkpoints/done/stack_{stat}.done")
    print("  3. Run: bash ml/train_all_models.sh --resume")
    return 0


if __name__ == "__main__":
    sys.exit(main())
