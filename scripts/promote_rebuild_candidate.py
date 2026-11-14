#!/usr/bin/env python3
"""
Pin a rebuilt causal OOF directory as the serving release candidate.

Run after ``scripts/retrain_causal_oof.sh`` and
``scripts/select_nonregressing_stacks.py`` (which fixes LightGBM-identity cells
and their attributions in place). For all 32 serving cells this:

  1. checks the rebuild is complete: one stack CSV, one Ridge coefficient JSON
     and one attribution parquet per cell; every attribution row adds up to its
     stack row's served value;
  2. copies them (plus the selection report and LightGBM reference) into
     ``releases/candidates/<release-id>/``, refusing to overwrite;
  3. re-points ``releases/current_baseline.json`` ``artifacts`` at the copies
     (``stacks``, ``ridge_coefficients``, ``bucket_attributions``,
     ``selection_policy``) and recomputes ``artifact_digests`` with repo-relative
     paths. ``serving_parity`` pins describe the superseded stacks and are
     dropped. Run-guard: ids of earlier stack materializations are removed from
     ``projection_policy.approved_pipeline_run_ids`` so the new run may replace
     their rows. ``freeze_baseline.py`` approves the new run afterwards.

It does not touch the database and does not materialize anything. It prints the
``APPROVED_PIPELINE_RUN_IDS`` prefix to carry into ``make freeze-baseline``.

    python scripts/promote_rebuild_candidate.py \\
        --rebuild-dir ml/oof/rebuild_<STAMP> --release-id causal_<YYYYMMDD>
    --dry-run   validate and print the plan; write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.artifact_manifest import REQUIRED_SERVING_CELLS  # noqa: E402
from ml.bucket_attribution import (  # noqa: E402
    ATTR_COLUMNS,
    ATTRIBUTION_TOLERANCE,
    attribution_path_for,
    max_residual,
)

#: Keys of the old ``artifacts`` block that describe the superseded stacks.
SUPERSEDED_KEYS = ("serving_parity",)
#: Stack materialization run ids start with this; they are replaced, not kept.
STACK_RUN_PREFIX = "stack_materialize_"
SELECTION_FILE = "CONSTRAINED_STACK_SELECTION.json"
REFERENCE_DIR = "bootstrap_lgbm_reference"


class PromotionError(RuntimeError):
    """The rebuild is incomplete or inconsistent; nothing was written."""


def _one(directory: Path, pattern: str) -> Path:
    found = sorted(directory.glob(pattern))
    if len(found) != 1:
        raise PromotionError(f"expected exactly one {pattern} in {directory}, found {[p.name for p in found]}")
    return found[0]


def collect_cells(rebuild_dir: Path) -> dict[tuple[str, str], dict[str, Path]]:
    """Return {(stat, position): {stack, attr, coef}} for all 32 cells, validated."""
    cells: dict[tuple[str, str], dict[str, Path]] = {}
    problems: list[str] = []
    for stat, position in sorted(REQUIRED_SERVING_CELLS):
        try:
            stack = _one(rebuild_dir, f"stack_{stat}_{position}_*.csv")
            coef = _one(rebuild_dir, f"ridge_{stat}_{position}_coefs.json")
        except PromotionError as exc:
            problems.append(str(exc))
            continue
        attr = attribution_path_for(stack)
        if not attr.is_file():
            problems.append(f"{stat}/{position}: missing {attr.name}")
            continue
        try:
            _check_attribution(stack, attr, f"{stat}/{position}")
        except PromotionError as exc:
            problems.append(str(exc))
            continue
        cells[(stat, position)] = {"stack": stack, "attr": attr, "coef": coef}
    if problems:
        raise PromotionError("rebuild is not promotable:\n  " + "\n  ".join(problems))
    if not (rebuild_dir / SELECTION_FILE).is_file():
        raise PromotionError(
            f"{SELECTION_FILE} not found in {rebuild_dir}; run scripts/select_nonregressing_stacks.py first"
        )
    return cells


def _check_attribution(stack_path: Path, attr_path: Path, label: str) -> None:
    stack = pd.read_csv(stack_path, usecols=["player_id", "game_id", "y_pred"])
    attr = pd.read_parquet(attr_path)
    missing = {"player_id", "game_id", "y_pred", *ATTR_COLUMNS} - set(attr.columns)
    if missing:
        raise PromotionError(f"{label}: {attr_path.name} lacks columns {sorted(missing)}")
    if len(attr) != len(stack):
        raise PromotionError(f"{label}: {attr_path.name} has {len(attr)} rows, stack has {len(stack)}")
    merged = stack.merge(
        attr[["player_id", "game_id", "y_pred", *ATTR_COLUMNS]],
        on=["player_id", "game_id"], how="left", suffixes=("", "_attr"), validate="1:1",
    )
    if merged["base"].isna().any():
        raise PromotionError(f"{label}: attribution rows do not match the stack's (player_id, game_id)")
    drift = (merged["y_pred"] - merged["y_pred_attr"]).abs().max()
    if not drift <= ATTRIBUTION_TOLERANCE:
        raise PromotionError(
            f"{label}: attribution y_pred differs from the stack's served value (max {drift:.6g}); "
            "re-run selection so the two files describe the same numbers"
        )
    worst = max_residual(merged.assign(y_pred=merged["y_pred"]))
    if not worst <= ATTRIBUTION_TOLERANCE:
        raise PromotionError(f"{label}: attributions do not add up to the stack (max residual {worst:.6g})")


def _walk(value) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote(
    rebuild_dir: Path,
    release_id: str,
    *,
    root: Path = REPO_ROOT,
    dry_run: bool = False,
) -> dict:
    """Copy the rebuild into ``releases/candidates/<release_id>`` and re-point the manifest."""
    if not release_id or "/" in release_id or release_id.startswith("."):
        raise PromotionError(f"bad release id {release_id!r}")
    cells = collect_cells(rebuild_dir)
    dest = root / "releases" / "candidates" / release_id
    if dest.exists():
        raise PromotionError(f"{dest} already exists; choose a new --release-id (never overwrite a candidate)")
    manifest_file = root / "releases" / "current_baseline.json"
    payload = json.loads(manifest_file.read_text())
    old_artifacts = payload.get("artifacts") or {}

    def rel(path: Path) -> str:
        return path.relative_to(root).as_posix()

    stacks = [rel(dest / c["stack"].name) for c in cells.values()]
    coefs = [rel(dest / c["coef"].name) for c in cells.values()]
    attrs = [rel(dest / c["attr"].name) for c in cells.values()]
    selection = [rel(dest / SELECTION_FILE)]
    artifacts = {
        **{k: v for k, v in old_artifacts.items() if k not in SUPERSEDED_KEYS},
        "stacks": sorted(stacks),
        "ridge_coefficients": sorted(coefs),
        "bucket_attributions": sorted(attrs),
        "selection_policy": selection,
    }
    dropped = [k for k in SUPERSEDED_KEYS if k in old_artifacts]

    policy = dict(payload.get("projection_policy") or {})
    approved = list(policy.get("approved_pipeline_run_ids") or [])
    kept = [r for r in approved if not r.startswith(STACK_RUN_PREFIX)]
    removed = [r for r in approved if r.startswith(STACK_RUN_PREFIX)]

    summary = {
        "release_id": release_id,
        "cells": len(cells),
        "candidate_dir": rel(dest),
        "dropped_artifact_groups": dropped,
        "approved_ids_removed": removed,
        "approved_ids_kept": kept,
        "freeze_env": "APPROVED_PIPELINE_RUN_IDS=<new materialize run id>" + "".join(f",{r}" for r in kept),
    }
    if dry_run:
        return summary

    dest.mkdir(parents=True)
    for c in cells.values():
        for key in ("stack", "attr", "coef"):
            shutil.copy2(c[key], dest / c[key].name)
    shutil.copy2(rebuild_dir / SELECTION_FILE, dest / SELECTION_FILE)
    reference = rebuild_dir / REFERENCE_DIR
    if reference.is_dir():
        shutil.copytree(reference, dest / REFERENCE_DIR)

    digests: dict[str, str] = {}
    for relpath in sorted(set(_walk(artifacts))):
        path = root / relpath
        if not path.is_file():
            raise PromotionError(f"pinned artifact missing after copy: {relpath}")
        digests[relpath] = _sha256(path)

    payload["artifacts"] = artifacts
    payload["artifact_digests"] = digests
    payload.pop("artifact_pins_missing", None)
    policy["approved_pipeline_run_ids"] = kept
    payload["projection_policy"] = policy
    payload["promotion"] = {
        "release_id": release_id,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "tool": "scripts/promote_rebuild_candidate.py",
        "superseded_artifact_groups": dropped,
    }
    manifest_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild-dir", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    rebuild = args.rebuild_dir if args.rebuild_dir.is_absolute() else REPO_ROOT / args.rebuild_dir
    try:
        summary = promote(rebuild, args.release_id, dry_run=args.dry_run)
    except PromotionError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        print("dry run: nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
