#!/usr/bin/env python3
"""
Materialize Phase-5 stack OOF predictions into the projections table.

Writes weekly rows for:
  - fantasy_ppr × QB/RB/WR/TE
  - targets × WR/TE/RB
  - carries × RB
  - pass_attempts × QB
  - passing_yards × QB (if stack present)

Source of truth: newest ml/oof/stack_{stat}_{pos}_*.csv (y_pred).
Floor/ceiling are simple ± residual MAD proxies so /predict percentiles
are non-null; Bayesian posteriors remain None until full pipeline re-run.

Usage:
  PYTHONPATH=. DATABASE_URL=... python scripts/materialize_stack_projections.py
  PYTHONPATH=. python scripts/materialize_stack_projections.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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

from ml.season_constants import LAST_COMPLETE_SEASON  # noqa: E402
from pipeline.schema import ensure_schema, normalize_dsn  # noqa: E402

logger = logging.getLogger(__name__)
OOF_DIR = ROOT / "ml" / "oof"
PIPELINE_RUN_ID = f"stack_materialize_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

CELLS: list[tuple[str, str]] = [
    ("fantasy_ppr", "QB"),
    ("fantasy_ppr", "RB"),
    ("fantasy_ppr", "WR"),
    ("fantasy_ppr", "TE"),
    ("targets", "WR"),
    ("targets", "TE"),
    ("targets", "RB"),
    ("carries", "RB"),
    ("pass_attempts", "QB"),
    ("passing_yards", "QB"),
]


def _latest_stack(stat: str, position: str) -> Path | None:
    paths = [
        p
        for p in OOF_DIR.glob(f"stack_{stat}_{position}_*.csv")
        if "_archive" not in str(p)
    ]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _load_cell(stat: str, position: str) -> pd.DataFrame:
    path = _latest_stack(stat, position)
    if path is None:
        raise FileNotFoundError(f"No stack OOF for {stat}/{position}")
    df = pd.read_csv(path)
    required = {"player_id", "game_id", "season", "week", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns {sorted(missing)}")
    df = df.copy()
    df["stat"] = stat
    df["position"] = position
    df["projection"] = pd.to_numeric(df["y_pred"], errors="coerce")
    # Residual-based bands when y_true present; else ±20% of |proj|
    if "y_true" in df.columns:
        resid = (df["projection"] - pd.to_numeric(df["y_true"], errors="coerce")).abs()
        mad = float(resid.median()) if resid.notna().any() else 0.0
        band = max(mad, 0.15 * float(df["projection"].abs().median() or 1.0))
    else:
        band = 0.2 * float(df["projection"].abs().median() or 1.0)
    df["floor"] = df["projection"] - band
    df["ceiling"] = df["projection"] + band
    df["p25"] = df["projection"] - 0.5 * band
    df["p75"] = df["projection"] + 0.5 * band
    if stat == "fantasy_ppr":
        df["fantasy_projection"] = df["projection"]
        df["fantasy_floor"] = df["floor"]
        df["fantasy_ceiling"] = df["ceiling"]
    else:
        df["fantasy_projection"] = None
        df["fantasy_floor"] = None
        df["fantasy_ceiling"] = None
    df["pipeline_run_id"] = PIPELINE_RUN_ID
    df["max_train_season"] = np.minimum(
        LAST_COMPLETE_SEASON, df["season"].astype(int) - 1
    )
    logger.info("Loaded %s (%d rows) from %s", f"{stat}/{position}", len(df), path.name)
    return df


def _upsert(conn, df: pd.DataFrame) -> int:
    sql = """
        INSERT INTO projections
            (player_id, game_id, season, week, stat, position,
             projection, floor, ceiling, p25, p75,
             boom_probability, bust_probability,
             fantasy_projection, fantasy_floor, fantasy_ceiling,
             pipeline_run_id, posterior_samples, max_train_season)
        VALUES (
            %(player_id)s, %(game_id)s, %(season)s, %(week)s, %(stat)s, %(position)s,
            %(projection)s, %(floor)s, %(ceiling)s, %(p25)s, %(p75)s,
            NULL, NULL,
            %(fantasy_projection)s, %(fantasy_floor)s, %(fantasy_ceiling)s,
            %(pipeline_run_id)s, NULL, %(max_train_season)s
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
            max_train_season = EXCLUDED.max_train_season
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
    args = p.parse_args()
    if not args.database_url:
        logger.error("DATABASE_URL required")
        return 2

    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for stat, pos in CELLS:
        try:
            frames.append(_load_cell(stat, pos))
        except FileNotFoundError as exc:
            skipped.append(str(exc))
            logger.warning("%s", exc)

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
        "pipeline_run_id": PIPELINE_RUN_ID,
        "n_rows": int(len(all_df)),
        "cells": summary,
        "skipped": skipped,
        "dry_run": bool(args.dry_run),
    }
    out = ROOT / "reports" / "materialize_stack_projections.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0

    dsn = normalize_dsn(args.database_url)
    conn = psycopg2.connect(dsn)
    try:
        ensure_schema(conn)
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
