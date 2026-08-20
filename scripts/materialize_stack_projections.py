#!/usr/bin/env python3
"""
Materialize Phase-5 stack OOF predictions into the projections table.

Writes weekly rows for:
  - fantasy_ppr × QB/RB/WR/TE
  - targets × WR/TE/RB
  - carries × RB
  - pass_attempts × QB
  - passing_yards × QB (if stack present)

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
from pipeline.schema import normalize_dsn  # noqa: E402

logger = logging.getLogger(__name__)
PIPELINE_RUN_ID = f"stack_materialize_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

CELLS: list[tuple[str, str]] = list(REQUIRED_SERVING_CELLS)
CONFORMAL_LEVEL = 0.90
_MIN_CONFORMAL_HISTORY = 100


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


def _attach_causal_conformal_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Attach 90% empirical intervals using only earlier OOF seasons.

    Stack OOF files contain realised outcomes because they are validation
    evidence.  Those outcomes must never calibrate their own row (or any row
    in the same target season).  For each season we therefore fit residual
    quantiles from strictly earlier seasons.  Prediction-conditioned buckets
    make the width responsive to the forecast level; sparse buckets safely
    fall back to the prior-season pooled residual distribution.

    ``floor`` and ``ceiling`` are lower/upper prediction bounds, not posterior
    percentiles.  ``p25``/``p75`` remain NULL until serving has genuine
    per-player posterior samples.
    """
    out = df.copy()
    out["floor"] = np.nan
    out["ceiling"] = np.nan
    out["interval_method"] = "unavailable"
    if "y_true" not in out.columns:
        return out

    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out["residual"] = out["y_true"] - out["projection"]
    finite = out["residual"].notna() & np.isfinite(out["residual"])
    alpha = (1.0 - CONFORMAL_LEVEL) / 2.0

    for season in sorted(out["season"].unique()):
        target = out["season"] == season
        history = out[(out["season"] < season) & finite].copy()
        if len(history) < _MIN_CONFORMAL_HISTORY:
            continue

        # Use stable, explicit prediction-value cut points rather than qcut's
        # labels, which can collapse when the model emits ties.
        edges = np.unique(history["projection"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy())
        history["_bin"] = np.searchsorted(edges, history["projection"].to_numpy(), side="right")
        target_bins = np.searchsorted(edges, out.loc[target, "projection"].to_numpy(), side="right")
        pooled = history["residual"]
        lower: list[float] = []
        upper: list[float] = []
        for bucket in target_bins:
            bucket_residuals = history.loc[history["_bin"] == bucket, "residual"]
            calibration = bucket_residuals if len(bucket_residuals) >= _MIN_CONFORMAL_HISTORY else pooled
            lower.append(float(calibration.quantile(alpha)))
            upper.append(float(calibration.quantile(1.0 - alpha)))

        predictions = out.loc[target, "projection"].to_numpy(dtype=float)
        # All declared targets are non-negative counting/yards statistics.
        out.loc[target, "floor"] = np.maximum(0.0, predictions + np.asarray(lower))
        out.loc[target, "ceiling"] = np.maximum(0.0, predictions + np.asarray(upper))
        out.loc[target, "interval_method"] = "causal_oof_conformal_90"

    return out.drop(columns=["residual"], errors="ignore")


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


def _upsert(conn, df: pd.DataFrame) -> int:
    sql = """
        INSERT INTO projections
            (player_id, game_id, season, week, stat, position,
             projection, floor, ceiling, p25, p75,
             boom_probability, bust_probability,
             fantasy_projection, fantasy_floor, fantasy_ceiling,
             pipeline_run_id, posterior_samples, max_train_season, interval_method)
        VALUES (
            %(player_id)s, %(game_id)s, %(season)s, %(week)s, %(stat)s, %(position)s,
            %(projection)s, %(floor)s, %(ceiling)s, %(p25)s, %(p75)s,
            NULL, NULL,
            %(fantasy_projection)s, %(fantasy_floor)s, %(fantasy_ceiling)s,
            %(pipeline_run_id)s, NULL, %(max_train_season)s, %(interval_method)s
        )
        ON CONFLICT (player_id, game_id, stat) DO UPDATE SET
            season = EXCLUDED.season,
            week = EXCLUDED.week,
            position = EXCLUDED.position,
            projection = EXCLUDED.projection,
            floor = EXCLUDED.floor,
            ceiling = EXCLUDED.ceiling,
            p25 = EXCLUDED.p25,
            p75 = EXCLUDED.p75,
            fantasy_projection = EXCLUDED.fantasy_projection,
            fantasy_floor = EXCLUDED.fantasy_floor,
            fantasy_ceiling = EXCLUDED.fantasy_ceiling,
            pipeline_run_id = EXCLUDED.pipeline_run_id,
            boom_probability = EXCLUDED.boom_probability,
            bust_probability = EXCLUDED.bust_probability,
            posterior_samples = EXCLUDED.posterior_samples,
            max_train_season = EXCLUDED.max_train_season,
            interval_method = EXCLUDED.interval_method
    """
    rows = []
    for r in df.itertuples(index=False):
        rows.append(
            {
                "player_id": str(r.player_id),
                "game_id": str(r.game_id),
                "season": int(r.season),
                "week": int(r.week),
                "stat": str(r.stat),
                "position": str(r.position),
                "projection": float(r.projection) if pd.notna(r.projection) else None,
                "floor": float(r.floor) if pd.notna(r.floor) else None,
                "ceiling": float(r.ceiling) if pd.notna(r.ceiling) else None,
                "p25": float(r.p25) if pd.notna(r.p25) else None,
                "p75": float(r.p75) if pd.notna(r.p75) else None,
                "fantasy_projection": float(r.fantasy_projection)
                if r.fantasy_projection is not None and pd.notna(r.fantasy_projection)
                else None,
                "fantasy_floor": float(r.fantasy_floor)
                if r.fantasy_floor is not None and pd.notna(r.fantasy_floor)
                else None,
                "fantasy_ceiling": float(r.fantasy_ceiling)
                if r.fantasy_ceiling is not None and pd.notna(r.fantasy_ceiling)
                else None,
                "pipeline_run_id": str(r.pipeline_run_id),
                "max_train_season": int(r.max_train_season),
                "interval_method": str(r.interval_method) if getattr(r, "interval_method", None) else "unavailable",
            }
        )
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
    conn.commit()
    return len(rows)


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
    for stat, pos in CELLS:
        # A declared release cell is all-or-nothing.  Selection itself verifies
        # the SHA-256 manifest pin before this reader sees any bytes.
        try:
            frames.append(_load_cell(stat, pos))
        except Exception as exc:
            logger.error("Refusing materialization: required %s/%s is invalid: %s", stat, pos, exc)
            return 1

    if not frames:
        logger.error("No stack OOFs loaded")
        return 1

    all_df = pd.concat(frames, ignore_index=True)
    # Drop rows without usable projections
    all_df = all_df.dropna(subset=["projection", "player_id", "game_id"])
    summary = (
        all_df.groupby(["stat", "position"], as_index=False)
        .agg(n=("projection", "size"), seasons=("season", "nunique"))
        .to_dict(orient="records")
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
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
        n = _upsert(conn, all_df)
        report["n_upserted"] = n
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        logger.info("Upserted %d projection rows", n)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
