#!/usr/bin/env python3
"""Freeze an LGBM reference and select only seasonally non-regressing stacks."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.artifact_manifest import REQUIRED_SERVING_CELLS, sha256_of


def _mean_variance_fallback(seasonal: list[dict[str, object]]) -> bool:
    """Fallback to LGBM identity only when mean MAE is worse after a variance penalty.

    Losing any single season is not enough. Mean candidate MAE plus 0.5 × std of
    (candidate − baseline) must exceed mean baseline MAE.
    """
    if not seasonal:
        return True
    deltas = [float(row["candidate_mae"]) - float(row["baseline_mae"]) for row in seasonal]
    mean_cand = sum(float(row["candidate_mae"]) for row in seasonal) / len(seasonal)
    mean_base = sum(float(row["baseline_mae"]) for row in seasonal) / len(seasonal)
    if len(deltas) == 1:
        return mean_cand > mean_base
    mean_delta = sum(deltas) / len(deltas)
    var = sum((d - mean_delta) ** 2 for d in deltas) / (len(deltas) - 1)
    return (mean_cand + 0.5 * (var ** 0.5)) > mean_base


def _one(directory: Path, pattern: str) -> Path:
    paths = sorted(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern}, found {[p.name for p in paths]}")
    return paths[0]


def select(candidate_dir: Path, baseline_path: Path) -> dict[str, object]:
    reference_dir = candidate_dir / "bootstrap_lgbm_reference"
    reference_dir.mkdir(exist_ok=True)
    frozen: dict[str, dict[str, object]] = {}
    decisions: dict[str, object] = {}
    for stat, position in REQUIRED_SERVING_CELLS:
        stack_path = _one(candidate_dir, f"stack_{stat}_{position}_*.csv")
        stack = pd.read_csv(stack_path)
        required = {"player_id", "game_id", "season", "week", "y_true", "y_pred", "lgbm_pred", "max_train_season"}
        missing = required - set(stack.columns)
        if missing:
            raise ValueError(f"{stack_path.name} missing {sorted(missing)}")
        if not (stack["max_train_season"] < stack["season"]).all():
            raise ValueError(f"{stack_path.name} has non-causal walk-forward provenance")
        ref = stack[["player_id", "game_id", "season", "week", "y_true", "lgbm_pred", "max_train_season"]].rename(columns={"lgbm_pred": "y_pred"})
        ref_path = reference_dir / f"lgbm_reference_{stat}_{position}.csv"
        ref.to_csv(ref_path, index=False)
        digest = sha256_of(ref_path)
        seasonal = []
        for season, sub in stack.groupby("season"):
            candidate_mae = float((sub["y_true"] - sub["y_pred"]).abs().mean())
            baseline_mae = float((sub["y_true"] - sub["lgbm_pred"]).abs().mean())
            seasonal.append({"season": int(season), "candidate_mae": candidate_mae, "baseline_mae": baseline_mae, "ok": candidate_mae <= baseline_mae})
            frozen[f"{stat}:{position}:{int(season)}"] = {
                "mae": baseline_mae,
                "oof_sha256": digest,
                "release_id": "bootstrap_causal_lgbm_reference_20260819",
                "reference_kind": "user_authorized_first_causal_reference",
            }
        fallback = _mean_variance_fallback(seasonal)
        if fallback:
            stack["y_pred"] = stack["lgbm_pred"]
            stack.to_csv(stack_path, index=False)
            coef = candidate_dir / f"ridge_{stat}_{position}_coefs.json"
            coef.write_text(json.dumps({"learner_order": ["lgbm", "catboost"], "weights": {"lgbm": 1.0, "catboost": 0.0}, "intercept": 0.0}, indent=2) + "\n")
        decisions[f"{stat}:{position}"] = {"selection": "lgbm_identity" if fallback else "ridge_stack", "before": seasonal}

    baseline = {
        "release_id": "bootstrap_causal_lgbm_reference_20260819",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bootstrap_reference": True,
        "governance": "User-authorized first causal reference after C-01 invalidated historical releases.",
        "frozen_baseline": frozen,
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": "mean_mae_with_variance_penalty_vs_walkforward_lgbm",
        "decisions": decisions,
    }
    (candidate_dir / "CONSTRAINED_STACK_SELECTION.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate_dir if args.candidate_dir.is_absolute() else ROOT / args.candidate_dir
    baseline = args.baseline if args.baseline.is_absolute() else ROOT / args.baseline
    print(json.dumps(select(candidate.resolve(), baseline.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
