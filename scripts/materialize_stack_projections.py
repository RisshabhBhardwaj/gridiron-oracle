#!/usr/bin/env python3
"""
Materialize Phase-5 stack OOF predictions into the projections table.

Writes weekly rows for every cell in ml.artifact_manifest.REQUIRED_SERVING_CELLS:
  - fantasy_ppr × QB/RB/WR/TE
  - targets × WR/TE/RB
  - carries × RB
  - pass_attempts × QB, completions × QB
  - passing_yards × QB, passing_tds × QB, interceptions × QB
  - receiving_yards × WR/TE/RB, receptions × WR/TE/RB, receiving_tds × WR/TE/RB
  - rushing_yards × QB/RB, rushing_tds × QB/RB/TE/WR
  - fumbles × QB/RB/TE/WR

The Phase 6 additions (receptions/TDs/interceptions/fumbles/completions)
give ml.scoring.score_fantasy a full component-stat set per position —
without them a "derived" fantasy total built from yardage cells alone
would omit every touchdown and every reception.

Source of truth: the SHA-256-pinned stack artifact for each cell in
releases/current_baseline.json (y_pred). Never "newest by mtime".
Only real posterior quantiles may populate percentile-named fields. Stack OOF
artifacts do not contain per-player posterior samples, so this materializer
leaves those fields NULL and records ``interval_method=unavailable`` in its
report rather than mislabelling a cell-wide residual offset as p10/p90.

Usage:
  PYTHONPATH=. DATABASE_URL=... python scripts/materialize_stack_projections.py
  PYTHONPATH=. python scripts/materialize_stack_projections.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.artifact_manifest import REQUIRED_SERVING_CELLS, get_manifest  # noqa: E402
from ml.bucket_attribution import (  # noqa: E402
    ATTR_COLUMNS,
    ATTRIBUTION_TOLERANCE,
    AttributionError,
    residual,
)
from pipeline.schema import normalize_dsn  # noqa: E402
from pipeline.serving_rows import (  # noqa: E402
    attach_causal_conformal_intervals,
    attach_derived_fantasy_projection,
    upsert_projections,
)

logger = logging.getLogger(__name__)
PIPELINE_RUN_ID = f"stack_materialize_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _git_commit() -> str:
    """Commit of the deployed code. The container ships without .git (.dockerignore),
    so read the platform's commit variable first, then git, else report unknown."""
    sha = os.environ.get("RAILWAY_GIT_COMMIT_SHA") or os.environ.get("GIT_COMMIT_SHA")
    if sha:
        return sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

CELLS: list[tuple[str, str]] = list(REQUIRED_SERVING_CELLS)
# Row preparation and the upsert live in pipeline/serving_rows.py, shared with the
# forward writer. The old private names stay as aliases for existing callers.
_attach_causal_conformal_intervals = attach_causal_conformal_intervals
_attach_derived_fantasy_projection = attach_derived_fantasy_projection
_upsert = upsert_projections


def _pinned_stack(stat: str, position: str) -> Path:
    """
    The release-manifest-pinned stack for a cell, SHA-verified.

    Materialization writes straight into ``projections``, so choosing the wrong
    file here puts wrong numbers in front of users. Selection is manifest-only:
    a missing entry or a digest mismatch raises rather than falling back to
    newest-by-mtime, which is how the four-learner ``_20260807`` TE stack became
    servable.
    """
    return get_manifest().resolve_stack(stat, position)


def _load_cell(stat: str, position: str) -> pd.DataFrame:
    path = _pinned_stack(stat, position)
    if "_20260809" in path.stem:
        raise ValueError(
            f"{path.name} is a pre-02 legacy stack; materialize only rebuilt artifacts"
        )
    df = pd.read_csv(path)
    required = {
        "player_id", "game_id", "season", "week", "y_pred",
        "max_train_season",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns {sorted(missing)}")
    df = df.copy()
    df["stat"] = stat
    df["position"] = position
    df["projection"] = pd.to_numeric(df["y_pred"], errors="coerce")
    if not np.isfinite(df["projection"]).all():
        raise ValueError(f"{path.name} has non-finite y_pred values")
    df["max_train_season"] = pd.to_numeric(df["max_train_season"], errors="coerce")
    if not np.isfinite(df["max_train_season"]).all():
        raise ValueError(f"{path.name} has non-finite max_train_season provenance")
    if not (df["max_train_season"] < df["season"].astype(int)).all():
        raise ValueError(f"{path.name} has non-causal max_train_season provenance")

    # Calibrate interval bounds from earlier OOF seasons only.  They are not
    # posterior quantiles, so percentile-named columns stay NULL.
    df = _attach_causal_conformal_intervals(df)
    df["p25"] = None
    df["p75"] = None
    if stat == "fantasy_ppr":
        df["fantasy_projection"] = df["projection"]
        df["fantasy_floor"] = df["floor"]
        df["fantasy_ceiling"] = df["ceiling"]
    else:
        df["fantasy_projection"] = None
        df["fantasy_floor"] = None
        df["fantasy_ceiling"] = None
    df["pipeline_run_id"] = PIPELINE_RUN_ID
    logger.info("Loaded %s (%d rows) from %s", f"{stat}/{position}", len(df), path.name)
    return df


def _load_cell_attributions(stat: str, position: str, cell: pd.DataFrame) -> pd.DataFrame:
    """The pinned bucket attributions for a cell, checked against the stack rows.

    Every stack row needs exactly one attribution row, and
    ``base + buckets + unweighted`` must equal the row's projection to within
    ATTRIBUTION_TOLERANCE. Anything else raises: attributions that do not add
    up to the served number would make a user's re-weighted number wrong.
    """
    path = get_manifest().resolve_attribution(stat, position)
    attr = pd.read_parquet(path)
    missing = {"player_id", "game_id", *ATTR_COLUMNS} - set(attr.columns)
    if missing:
        raise AttributionError(f"{path.name} missing columns {sorted(missing)}")
    if attr.duplicated(["player_id", "game_id"]).any():
        raise AttributionError(f"{path.name} repeats a (player_id, game_id)")
    keys = cell[["player_id", "game_id", "projection"]]
    merged = keys.merge(
        attr[["player_id", "game_id", *ATTR_COLUMNS]],
        on=["player_id", "game_id"], how="left", validate="1:1",
    )
    if merged["base"].isna().any():
        raise AttributionError(
            f"{path.name}: {int(merged['base'].isna().sum())} of {len(merged)} "
            f"{stat}/{position} stack rows have no attribution row"
        )
    values = merged[list(ATTR_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AttributionError(f"{path.name} has non-finite attributions")
    merged["max_residual"] = np.abs(residual(merged, merged["projection"].to_numpy()))
    worst = float(merged["max_residual"].max()) if len(merged) else 0.0
    if worst > ATTRIBUTION_TOLERANCE:
        raise AttributionError(
            f"{path.name}: attributions do not add up to the pinned stack's y_pred "
            f"(max residual {worst:.6g} > {ATTRIBUTION_TOLERANCE})"
        )
    merged["stat"] = stat
    merged["pipeline_run_id"] = PIPELINE_RUN_ID
    logger.info("Attributions %s/%s: %d rows, max residual %.2e", stat, position, len(merged), worst)
    return merged.drop(columns=["projection"])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL", ""))
    p.add_argument(
        "--pipeline-run-id",
        default=None,
        help="Reuse an approved run id instead of minting a new timestamped id",
    )
    args = p.parse_args()
    global PIPELINE_RUN_ID
    if args.pipeline_run_id:
        PIPELINE_RUN_ID = str(args.pipeline_run_id)
    if not args.database_url and not args.dry_run:
        logger.error("DATABASE_URL required")
        return 2

    frames: list[pd.DataFrame] = []
    attribution_frames: list[pd.DataFrame] = []
    for stat, pos in CELLS:
        # A declared release cell is all-or-nothing.  Selection itself verifies
        # the SHA-256 manifest pin before this reader sees any bytes.
        try:
            cell = _load_cell(stat, pos)
            frames.append(cell)
            # Attributions are part of the cell: a release without them cannot
            # honour the Settings weights, so it is not materialized.
            attribution_frames.append(_load_cell_attributions(stat, pos, cell))
        except Exception as exc:
            logger.error("Refusing materialization: required %s/%s is invalid: %s", stat, pos, exc)
            return 1

    if not frames:
        logger.error("No stack OOFs loaded")
        return 1

    all_df = pd.concat(frames, ignore_index=True)
    attributions = pd.concat(attribution_frames, ignore_index=True)
    # Drop rows without usable projections
    all_df = all_df.dropna(subset=["projection", "player_id", "game_id"])
    all_df = _attach_derived_fantasy_projection(all_df)
    summary = (
        all_df.groupby(["stat", "position"], as_index=False)
        .agg(n=("projection", "size"), seasons=("season", "nunique"))
        .to_dict(orient="records")
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "pipeline_run_id": PIPELINE_RUN_ID,
        "n_rows": int(len(all_df)),
        "cells": summary,
        "intervals": {
            "method": "causal_oof_conformal_90",
            "available_rows": int((all_df["interval_method"] == "causal_oof_conformal_90").sum()),
            "unavailable_rows": int((all_df["interval_method"] == "unavailable").sum()),
            "reason_unavailable": "No strictly prior OOF season exists for calibration.",
            "posterior_percentiles_materialized": False,
        },
        "attributions": {
            "rows": int(len(attributions)),
            "max_residual": float(attributions["max_residual"].max()) if len(attributions) else 0.0,
            "tolerance": ATTRIBUTION_TOLERANCE,
        },
        "dry_run": bool(args.dry_run),
    }
    out = ROOT / "reports" / "materialize_stack_projections.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0

    dsn = normalize_dsn(args.database_url)
    conn = psycopg2.connect(dsn)
    try:
        # No ensure_schema() here: Alembic is the sole schema authority (C-12).
        # Runtime DDL from ~10 call sites meant fresh-database behaviour depended
        # on the current checkout rather than on the migration history. Run
        # `alembic upgrade head` before materializing.
        n = _upsert(conn, all_df, attributions)
        report["n_upserted"] = n
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        logger.info("Upserted %d projection rows", n)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
