#!/usr/bin/env python3
"""
Backfill pbp_plays.{yards_gained,interception,fumble_lost,penalty,penalty_yards}
for existing seasons (Phase 5, migration 20260822_0021).

Re-runs only the pbp_plays slice of pipeline/pbp_pipeline.py's per-season
load — _write_pbp_plays is an upsert keyed on (game_id, play_id), so this
is safe to re-run and does not touch feature_matrix, matchups, or ftn
tables.

Usage:
  DATABASE_URL=... python scripts/backfill_pbp_plays_outcomes.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    from pipeline.pbp_pipeline import SEASONS, _build_pbp_plays, _get_conn, _load_pbp_season, _write_pbp_plays

    conn = _get_conn()
    total = 0
    try:
        for season in SEASONS:
            pbp = _load_pbp_season(season)
            if pbp.empty:
                logger.info("Season %d: no PBP data, skipping", season)
                continue
            plays_df = _build_pbp_plays(pbp, season)
            n = _write_pbp_plays(conn, plays_df)
            total += n
            logger.info("Season %d: upserted %d pbp_plays rows", season, n)
    finally:
        conn.close()
    logger.info("Done. %d pbp_plays rows upserted across %d seasons.", total, len(SEASONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
