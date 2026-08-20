"""
scripts/reprojection_gate.py

Phase 8 — per-season reprojection gate.

Before promoting a train run, re-score holdout seasons with causal constraints
and refuse promotion if MAE regresses vs the frozen baseline (or vs naive).

Usage:
  python scripts/reprojection_gate.py --holdout-season 2024 --position WR --target fantasy_ppr \
      --candidate-oof releases/candidates/causal_20260810/stack_fantasy_ppr_WR_20260819.csv \
      --baseline-manifest releases/baselines/previous_promoted_release.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.artifact_manifest import sha256_of  # noqa: E402
from ml.multiplicity import require_dsr_pass  # noqa: E402

FIRST_RELEASE_OVERRIDE = "no_prior_baseline_requires_human_override"

logger = logging.getLogger(__name__)
_OUT = ROOT / "ml" / "experiments" / "reprojection_gate"


def _load_oof(path: Path) -> pd.DataFrame:
    """Load exactly one candidate OOF; selection by glob is not a gate."""
    if not path.exists():
        raise FileNotFoundError(f"Candidate OOF does not exist: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Candidate OOF is empty: {path}")
    return frame


def _oof_mae(df: pd.DataFrame, holdout_season: int) -> Optional[float]:
    if df is None or df.empty:
        return None
    season_col = "season" if "season" in df.columns else None
    y_col = next((c for c in ("y_true", "actual", "target") if c in df.columns), None)
    p_col = next((c for c in ("y_pred", "pred", "oof_pred", "prediction") if c in df.columns), None)
    if not y_col or not p_col:
        return None
    sub = df
    if season_col:
        sub = df[df[season_col] == holdout_season]
    if sub.empty:
        return None
    err = (pd.to_numeric(sub[y_col], errors="coerce") - pd.to_numeric(sub[p_col], errors="coerce")).abs()
    err = err.dropna()
    return float(err.mean()) if len(err) else None


def _load_baseline_measurement(
    path: Path, position: str, target: str, holdout_season: int
) -> dict:
    """Load a prior-release metric, never the candidate release's own metric."""
    if not path.exists():
        raise FileNotFoundError(f"Frozen baseline manifest does not exist: {path}")
    payload = json.loads(path.read_text())
    gates = payload.get("frozen_baseline")
    if not isinstance(gates, dict):
        raise ValueError(
            f"{path} has no frozen_baseline block. Legacy reprojection_gates are "
            "not eligible because they may be candidate self-comparisons."
        )
    key = f"{target}:{position}:{holdout_season}"
    measurement = gates.get(key)
    if not isinstance(measurement, dict):
        raise ValueError(f"Frozen baseline has no measurement for {key}")
    required = {"mae", "oof_sha256", "release_id"}
    missing = required - set(measurement)
    if missing:
        raise ValueError(f"Frozen baseline {key} missing {sorted(missing)}")
    mae = float(measurement["mae"])
    if not np.isfinite(mae) or mae <= 0:
        raise ValueError(f"Frozen baseline {key} has invalid MAE {mae!r}")
    return {**measurement, "mae": mae}


def run_gate(
    holdout_season: int,
    position: str,
    target: str,
    candidate_oof: Path,
    baseline_manifest: Path,
    max_regression: float = 0.0,
    approval: str | None = None,
    n_trials: int | None = None,
    observed_sharpe: float | None = None,
    n_observations: int | None = None,
) -> dict:
    if max_regression < 0:
        raise ValueError("max_regression cannot be negative")
    if max_regression > 0 and not approval:
        raise ValueError("A non-zero regression allowance requires a recorded approval")
    oof_df = _load_oof(candidate_oof)
    oof_mae = _oof_mae(oof_df, holdout_season)
    if oof_mae is None:
        raise ValueError(f"Candidate OOF has no finite {holdout_season} predictions")
    candidate_sha256 = sha256_of(candidate_oof)

    try:
        baseline = _load_baseline_measurement(baseline_manifest, position, target, holdout_season)
    except (FileNotFoundError, ValueError):
        if approval != FIRST_RELEASE_OVERRIDE:
            raise
        baseline = None

    if baseline is not None and candidate_sha256 == str(baseline["oof_sha256"]):
        raise ValueError("Candidate OOF is byte-identical to the frozen baseline; self-comparison is forbidden")

    if n_trials is not None:
        if observed_sharpe is None or n_observations is None:
            raise ValueError("DSR check requires observed_sharpe and n_observations")
        require_dsr_pass(observed_sharpe, n_trials, n_observations)

    if baseline is None:
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "holdout_season": holdout_season,
            "position": position.upper(),
            "target": target,
            "candidate_source": str(candidate_oof),
            "candidate_mae": oof_mae,
            "oof_mae": oof_mae,
            "baseline_mae": None,
            "baseline_manifest": str(baseline_manifest),
            "approval": approval,
            "promote": True,
            "reasons": ["first release: no frozen baseline; human override recorded"],
            "checks": [{
                "name": "first_release_human_override",
                "ok": True,
                "candidate_mae": oof_mae,
                "candidate_oof_sha256": candidate_sha256,
            }],
        }
        return report

    baseline_mae = float(baseline["mae"])
    promote = oof_mae <= baseline_mae * (1.0 + max_regression)
    reasons = [] if promote else [
        f"candidate MAE {oof_mae:.4f} exceeds frozen baseline {baseline_mae:.4f} "
        f"by more than {max_regression:.0%}"
    ]
    checks = [{
        "name": "vs_frozen_prior_release",
        "baseline_mae": baseline_mae,
        "candidate_mae": oof_mae,
        "baseline_release_id": baseline["release_id"],
        "baseline_oof_sha256": baseline["oof_sha256"],
        "candidate_oof_sha256": candidate_sha256,
        "ok": promote,
    }]

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "holdout_season": holdout_season,
        "position": position.upper(),
        "target": target,
        "candidate_source": str(candidate_oof),
        "candidate_mae": oof_mae,
        "oof_mae": oof_mae,
        "baseline_mae": baseline_mae,
        "baseline_manifest": str(baseline_manifest),
        "approval": approval,
        "promote": promote,
        "reasons": reasons,
        "checks": checks,
    }
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Phase 8 reprojection promotion gate")
    p.add_argument("--holdout-season", type=int, required=True)
    p.add_argument("--position", default="WR")
    p.add_argument("--target", default="fantasy_ppr")
    p.add_argument("--candidate-oof", type=Path, required=True)
    p.add_argument("--baseline-manifest", type=Path, required=True)
    p.add_argument("--max-regression", type=float, default=0.0)
    p.add_argument("--approval", default=None)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--observed-sharpe", type=float, default=None)
    p.add_argument("--n-observations", type=int, default=None)
    args = p.parse_args()

    report = run_gate(
        holdout_season=args.holdout_season,
        position=args.position,
        target=args.target,
        candidate_oof=args.candidate_oof,
        baseline_manifest=args.baseline_manifest,
        max_regression=args.max_regression,
        approval=args.approval,
        n_trials=args.n_trials,
        observed_sharpe=args.observed_sharpe,
        n_observations=args.n_observations,
    )
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / (
        f"gate_{report['target']}_{report['position']}_{report['holdout_season']}.json"
    )
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved={path}")
    return 0 if report["promote"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
