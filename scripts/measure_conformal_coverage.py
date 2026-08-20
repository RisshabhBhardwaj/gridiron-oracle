#!/usr/bin/env python3
"""Empirical 80% EnbPI coverage + residual-ensemble CRPS on pinned stack OOF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oof-dir",
        type=Path,
        default=ROOT / "releases" / "candidates" / "causal_20260810",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "conformal_coverage.json")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    from ml.backtest import compute_crps_single
    from ml.conformal import empirical_coverage, oof_conformal_bounds, residual_quantiles

    rows = []
    for path in sorted(args.oof_dir.glob("stack_*_*_*.csv")):
        parts = path.stem.split("_")
        # stack_{stat...}_{POS}_{date}
        position = parts[-2]
        stat = "_".join(parts[1:-2])
        frame = pd.read_csv(path)
        if "y_pred" not in frame.columns or "y_true" not in frame.columns:
            continue
        calibrated = oof_conformal_bounds(frame, alpha=0.20)
        usable = calibrated[calibrated["interval_method"] == "mapie_enbpi"]
        if usable.empty:
            rows.append({"stat": stat, "position": position, "n": 0, "coverage_80": None, "crps": None})
            continue
        cover = empirical_coverage(
            usable["y_true"].to_numpy(),
            usable["floor"].to_numpy(),
            usable["ceiling"].to_numpy(),
        )
        prior = frame[pd.to_numeric(frame["season"], errors="coerce") < int(usable["season"].min())]
        resid = (
            pd.to_numeric(prior["y_true"], errors="coerce")
            - pd.to_numeric(prior["y_pred"], errors="coerce")
        ).dropna().to_numpy()
        if resid.size < 20:
            resid = (
                pd.to_numeric(usable["y_true"], errors="coerce")
                - pd.to_numeric(usable["y_pred"], errors="coerce")
            ).dropna().to_numpy()
        rng = np.random.default_rng(0)
        draw = rng.choice(resid, size=min(400, max(len(resid), 1)), replace=True) if resid.size else np.array([0.0])
        crps_vals = []
        for pred, actual in zip(
            pd.to_numeric(usable["y_pred"], errors="coerce"),
            pd.to_numeric(usable["y_true"], errors="coerce"),
        ):
            if not np.isfinite(pred) or not np.isfinite(actual):
                continue
            crps_vals.append(compute_crps_single(pred + draw, float(actual)))
        in_band = cover == cover and 0.75 <= float(cover) <= 0.85
        rows.append({
            "stat": stat,
            "position": position,
            "n": int(len(usable)),
            "coverage_80": None if cover != cover else float(cover),
            "in_75_85": bool(in_band),
            "crps": None if not crps_vals else float(np.mean(crps_vals)),
        })
    payload = {
        "alpha": 0.20,
        "target": [0.75, 0.85],
        "n_cells": len(rows),
        "cells_in_band": int(sum(1 for row in rows if row.get("in_75_85"))),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("n_cells", "cells_in_band")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
