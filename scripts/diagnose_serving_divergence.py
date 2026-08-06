#!/usr/bin/env python3
"""
Diagnose serving-path divergence for a target (default: QB passing_yards).

Compares OOF predictions to stored Projection rows on identical keys and
checks whether Ridge coef artifacts exist (vs equal-weight fallback).

Usage:
  python scripts/diagnose_serving_divergence.py \\
      --oof ml/oof/xgb_receiving_yards_20260720.csv \\
      --stat passing_yards --position QB
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _find_ridge(stat: str, position: str) -> list[Path]:
    root = Path("ml/oof")
    patterns = [
        f"ridge_{stat}_{position}_coefs.json",
        f"ridge_{stat}_{position}_*.json",
    ]
    found: list[Path] = []
    for pat in patterns:
        found.extend(root.glob(pat))
    return sorted(set(found))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stat", default="passing_yards")
    parser.add_argument("--position", default="QB")
    parser.add_argument("--oof", type=Path, default=None, help="Optional OOF CSV")
    parser.add_argument("--json-out", type=Path, default=Path("reports/serving_divergence.json"))
    args = parser.parse_args()

    report: dict = {
        "stat": args.stat,
        "position": args.position,
        "ridge_artifacts": [str(p) for p in _find_ridge(args.stat, args.position)],
        "equal_weight_risk": False,
        "oof_vs_db": None,
    }
    if not report["ridge_artifacts"]:
        report["equal_weight_risk"] = True
        report["note"] = (
            "No ridge_{stat}_{pos}_coefs.json found under ml/oof/. "
            "inference_client now fails closed (no equal-weight fallback). "
            "Serving this cell requires regenerating position-specific Ridge coefs."
        )

    if args.oof and args.oof.exists() and os.environ.get("DATABASE_URL"):
        from sqlmodel import Session, create_engine, select
        from backend.app.models.production import Projection

        oof = pd.read_csv(args.oof)
        # Normalize column names across OOF formats
        pred_col = "predicted" if "predicted" in oof.columns else "y_pred"
        actual_col = "actual" if "actual" in oof.columns else "y_true"
        if pred_col not in oof.columns:
            report["oof_vs_db"] = {"error": f"No prediction column in {args.oof}"}
        else:
            if "stat" in oof.columns:
                oof = oof[oof["stat"] == args.stat]
            if "position" in oof.columns:
                oof = oof[oof["position"] == args.position]
            engine = create_engine(os.environ["DATABASE_URL"])
            with Session(engine) as session:
                proj = session.exec(
                    select(Projection).where(
                        Projection.stat == args.stat,
                        Projection.position == args.position,
                    )
                ).all()
            db = pd.DataFrame(
                [
                    {
                        "player_id": p.player_id,
                        "season": p.season,
                        "week": p.week,
                        "db_projection": p.projection,
                    }
                    for p in proj
                ]
            )
            merged = oof.merge(
                db,
                on=["player_id", "season", "week"],
                how="inner",
            )
            if merged.empty:
                report["oof_vs_db"] = {"rows": 0, "note": "No overlapping keys"}
            else:
                delta = merged["db_projection"] - merged[pred_col]
                report["oof_vs_db"] = {
                    "rows": int(len(merged)),
                    "oof_mae": float(np.mean(np.abs(merged[pred_col] - merged[actual_col])))
                    if actual_col in merged.columns
                    else None,
                    "db_mae": float(np.mean(np.abs(merged["db_projection"] - merged[actual_col])))
                    if actual_col in merged.columns
                    else None,
                    "mean_db_minus_oof": float(delta.mean()),
                    "mae_db_vs_oof": float(delta.abs().mean()),
                    "corr": float(np.corrcoef(merged["db_projection"], merged[pred_col])[0, 1]),
                }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
