#!/usr/bin/env python3
"""Measure the executable Ridge serving blend against evaluated stack OOF rows.

This deliberately does not compare a projection table to the CSV it was copied
from: that only proves materialization fidelity. Instead it loads the exact
coefficient artifact used by serving, reconstructs predictions from the base
learner columns retained in a stack OOF, and records the resulting divergence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.inference_client import InferenceClient


def diagnose(*, stat: str, position: str, oof: Path, coefficient_dir: Path) -> dict:
    if not oof.exists():
        raise FileNotFoundError(f"Evaluated stack OOF does not exist: {oof}")
    client = InferenceClient(mlflow_tracking_uri="", oof_dir=coefficient_dir)
    loaded = client.load_ridge_coefs(stat, position=position)
    if loaded is None:
        raise ValueError(f"No coefficient artifact for {stat}/{position} in {coefficient_dir}")
    weights, intercept, learners = loaded
    frame = pd.read_csv(oof)
    required = {"y_pred", *[f"{learner}_pred" for learner in learners]}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{oof} cannot support parity check; missing {sorted(missing)}")

    serving = np.full(len(frame), intercept, dtype=float)
    finite = np.isfinite(pd.to_numeric(frame["y_pred"], errors="coerce").to_numpy(dtype=float))
    for learner, weight in zip(learners, weights):
        values = pd.to_numeric(frame[f"{learner}_pred"], errors="coerce").to_numpy(dtype=float)
        finite &= np.isfinite(values)
        serving += weight * values
    if not finite.any():
        raise ValueError(f"{oof} has no finite rows for serving-parity comparison")

    evaluated = pd.to_numeric(frame["y_pred"], errors="coerce").to_numpy(dtype=float)
    delta = serving[finite] - evaluated[finite]
    return {
        "stat": stat,
        "position": position.upper(),
        "oof": str(oof),
        "coefficient_dir": str(coefficient_dir),
        "learner_order": learners,
        "intercept": intercept,
        "n_compared": int(finite.sum()),
        "mean_serving_minus_oof": float(delta.mean()),
        "mae_serving_vs_oof": float(np.abs(delta).mean()),
        "max_abs_serving_vs_oof": float(np.abs(delta).max()),
        "corr_serving_vs_oof": float(np.corrcoef(serving[finite], evaluated[finite])[0, 1]),
        "interpretation": (
            "The served final Ridge fit is trained on all causal base OOF rows, while y_pred is "
            "walk-forward meta-OOF. Non-zero divergence is expected and is release evidence, not a pass-by-copy."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stat", required=True)
    parser.add_argument("--position", required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--coefficient-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(
        stat=args.stat,
        position=args.position,
        oof=args.oof,
        coefficient_dir=args.coefficient_dir,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
