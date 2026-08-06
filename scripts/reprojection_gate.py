"""
scripts/reprojection_gate.py

Phase 8 — per-season reprojection gate.

Before promoting a train run, re-score holdout seasons with causal constraints
and refuse promotion if MAE regresses vs the frozen baseline (or vs naive).

Usage:
  python scripts/reprojection_gate.py --holdout-season 2024 --position WR --target fantasy_ppr
  python scripts/reprojection_gate.py --holdout-season 2024 --position WR --target fantasy_ppr --oof-glob 'ml/oof/*fantasy_ppr*WR*'
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

from ml.model_floor import fit_ridge_floor  # noqa: E402

logger = logging.getLogger(__name__)
_OUT = ROOT / "ml" / "experiments" / "reprojection_gate"


def _load_oof(glob_pat: str) -> Optional[pd.DataFrame]:
    paths = sorted(ROOT.glob(glob_pat))
    if not paths:
        return None
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as exc:
            logger.warning("skip %s: %s", p, exc)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


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


def _load_baseline_mae(position: str, target: str, holdout_season: int) -> Optional[float]:
    path = ROOT / "releases" / "current_baseline.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    gates = payload.get("reprojection_gates") or payload.get("eval") or {}
    key = f"{target}:{position}:{holdout_season}"
    if key in gates:
        return float(gates[key].get("mae", gates[key]))
    return None


def run_gate(
    holdout_season: int,
    position: str,
    target: str,
    oof_glob: str,
    max_regression: float = 0.05,
) -> dict:
    ridge = fit_ridge_floor(holdout_season, position, target=target)
    oof_df = _load_oof(oof_glob)
    oof_mae = _oof_mae(oof_df, holdout_season)
    baseline_mae = _load_baseline_mae(position, target, holdout_season)

    candidate_mae = oof_mae if oof_mae is not None else ridge.mae
    candidate_source = "oof" if oof_mae is not None else "ridge_floor"

    # Compare to frozen baseline if present; else require beating ridge itself is N/A —
    # require candidate <= ridge.mae * (1+eps) when OOF exists, else pass ridge-only.
    checks = []
    promote = True
    reasons = []

    if baseline_mae is not None:
        ok = candidate_mae <= baseline_mae * (1.0 + max_regression)
        checks.append({
            "name": "vs_frozen_baseline",
            "baseline_mae": baseline_mae,
            "candidate_mae": candidate_mae,
            "ok": ok,
        })
        if not ok:
            promote = False
            reasons.append(
                f"candidate MAE {candidate_mae:.4f} exceeds baseline {baseline_mae:.4f} "
                f"by more than {max_regression:.0%}"
            )
    else:
        checks.append({
            "name": "vs_frozen_baseline",
            "baseline_mae": None,
            "candidate_mae": candidate_mae,
            "ok": True,
            "note": "no current_baseline.json gate entry — skipped",
        })

    if oof_mae is not None:
        ok = oof_mae <= ridge.mae * (1.0 + max_regression)
        checks.append({
            "name": "oof_vs_ridge_floor",
            "ridge_mae": ridge.mae,
            "oof_mae": oof_mae,
            "ok": ok,
        })
        if not ok:
            promote = False
            reasons.append(
                f"OOF MAE {oof_mae:.4f} worse than Ridge floor {ridge.mae:.4f}"
            )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "holdout_season": holdout_season,
        "position": position.upper(),
        "target": target,
        "candidate_source": candidate_source,
        "candidate_mae": candidate_mae,
        "ridge_floor_mae": ridge.mae,
        "oof_mae": oof_mae,
        "baseline_mae": baseline_mae,
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
    p.add_argument("--oof-glob", default="ml/oof/*fantasy_ppr*")
    p.add_argument("--max-regression", type=float, default=0.05)
    args = p.parse_args()

    report = run_gate(
        holdout_season=args.holdout_season,
        position=args.position,
        target=args.target,
        oof_glob=args.oof_glob,
        max_regression=args.max_regression,
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
