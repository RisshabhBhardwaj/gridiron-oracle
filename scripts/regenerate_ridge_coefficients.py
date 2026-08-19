#!/usr/bin/env python3
"""Regenerate serving Ridge coefficients from a causal base-OOF rebuild.

This intentionally does *not* convert the legacy flat coefficient JSON files.
The input directory must contain the LGBM and CatBoost base OOFs from a named,
causal rebuild.  Each required serving cell is refit through the current stacker
so the result uses the executable coefficient schema consumed by inference.

Example:
  python scripts/regenerate_ridge_coefficients.py \
      --input-dir ml/oof/rebuild_20260809T231500 \
      --output-dir releases/candidates/causal_20260810
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.artifact_manifest import REQUIRED_SERVING_CELLS, sha256_of
from ml.inference_client import InferenceClient
from ml.stacking_ensemble import stack


def _base_oof_path(input_dir: Path, learner: str, stat: str, position: str) -> Path:
    matches = sorted(input_dir.glob(f"{learner}_{stat}_{position}_*.csv"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one causal {learner} base OOF for {stat}/{position} "
            f"in {input_dir}; found {[path.name for path in matches]}"
        )
    return matches[0]


def regenerate(input_dir: Path, output_dir: Path) -> dict[str, object]:
    """Refit and verify a positioned Ridge coefficient artifact for every cell."""
    if not (input_dir / "README.md").exists():
        raise ValueError(f"{input_dir} has no rebuild README; refusing an unproven input directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, dict[str, object]] = {}
    for stat, position in REQUIRED_SERVING_CELLS:
        paths = [
            _base_oof_path(input_dir, "lgbm", stat, position),
            _base_oof_path(input_dir, "catboost", stat, position),
        ]
        result = stack(
            oof_paths=paths,
            target=stat,
            position_filter=position,
            mlflow_tracking_uri="",
            out_dir=output_dir,
        )
        if not result.meta_fold_results:
            raise RuntimeError(f"No walk-forward meta folds for {stat}/{position}")
        coef_path = output_dir / f"ridge_{stat}_{position}_coefs.json"
        if not coef_path.exists():
            raise RuntimeError(f"Stacker did not create {coef_path}")
        generated[f"{stat}:{position}"] = {
            "coefficient_path": coef_path.name,
            "coefficient_sha256": sha256_of(coef_path),
            "inputs": {path.name: sha256_of(path) for path in paths},
            "stacked_mae": result.stacked_mae,
            "stacked_rmse": result.stacked_rmse,
        }

    # Use the production loader, not a duplicate validator, as the release gate.
    client = InferenceClient(mlflow_tracking_uri="", oof_dir=output_dir)
    for stat, position in REQUIRED_SERVING_CELLS:
        loaded = client.load_ridge_coefs(stat, position=position)
        if loaded is None:
            raise RuntimeError(f"Generated coefficient is not discoverable for {stat}/{position}")

    report: dict[str, object] = {
        "schema": "ridge-coefficients-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rebuild": str(input_dir.relative_to(ROOT)),
        "cells": generated,
    }
    (output_dir / "COEFFICIENTS_MANIFEST.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    report = regenerate(input_dir.resolve(), output_dir.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
