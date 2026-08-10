#!/usr/bin/env python3
"""Verify the rebuilt feature matrix against the C-01 as-of contract."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import psycopg2

from ml.feature_contract import assert_no_same_game_postgame_equality, assert_model_input_columns
from ml.utils import FEATURE_COLS
from scraper.adapters.nflreadpy_adapter import _psycopg2_dsn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    args = parser.parse_args()
    if not args.database_url:
        raise SystemExit("DATABASE_URL is required")
    assert_model_input_columns(FEATURE_COLS, consumer="rebuild verification")
    with psycopg2.connect(_psycopg2_dsn(args.database_url)) as conn:
        fm = pd.read_sql(
            f"SELECT player_id, game_id, snap_pct_off, {', '.join(FEATURE_COLS)} FROM feature_matrix WHERE season = ANY(%s)",
            conn, params=(args.seasons,),
        )
        logs = pd.read_sql(
            "SELECT gl.player_id, gl.game_id, gl.offense_pct FROM game_logs gl WHERE gl.season = ANY(%s)",
            conn, params=(args.seasons,),
        )
    # The raw source column is intentionally supplied only to make an exact-copy
    # audit possible.  The model allowlist itself is separately validated above.
    if "snap_pct_off" in fm.columns and fm["snap_pct_off"].notna().any():
        raise AssertionError("Rebuilt feature_matrix still contains target-game snap_pct_off values")
    assert_no_same_game_postgame_equality(
        fm, logs, feature_columns=[c for c in FEATURE_COLS if c in logs.columns]
    )
    print(f"feature contract verified: {len(fm)} rows; seasons={args.seasons}; features={len(FEATURE_COLS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
