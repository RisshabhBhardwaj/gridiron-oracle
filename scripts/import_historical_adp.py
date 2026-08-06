#!/usr/bin/env python3
"""Import historical Full-PPR ADP CSVs into fantasy_adp (source='historical').

Expected CSV columns: player_name, position, team, adp
Place files under data/adp/historical/adp_YYYY.csv

  python -m scripts.import_historical_adp --season 2024
  python -m scripts.import_historical_adp --all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import psycopg2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn

logger = logging.getLogger(__name__)
_HIST_DIR = ROOT / "data" / "adp" / "historical"

CREATE_FANTASY_ADP = """
CREATE TABLE IF NOT EXISTS fantasy_adp (
    season INTEGER NOT NULL,
    source TEXT NOT NULL,
    scoring TEXT NOT NULL,
    player_name TEXT NOT NULL,
    position TEXT,
    team TEXT,
    adp FLOAT NOT NULL,
    player_id TEXT,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (season, source, scoring, player_name)
)
"""


def import_season(season: int, database_url: str) -> int:
    path = _HIST_DIR / f"adp_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    required = {"player_name", "adp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    dsn = normalize_dsn(database_url)
    n = 0
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_FANTASY_ADP)
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO fantasy_adp
                        (season, source, scoring, player_name, position, team, adp)
                    VALUES (%s, 'historical', 'ppr', %s, %s, %s, %s)
                    ON CONFLICT (season, source, scoring, player_name) DO UPDATE SET
                        position = EXCLUDED.position,
                        team = EXCLUDED.team,
                        adp = EXCLUDED.adp,
                        imported_at = NOW()
                    """,
                    (
                        season,
                        str(row["player_name"]),
                        row.get("position"),
                        row.get("team"),
                        float(row["adp"]),
                    ),
                )
                n += 1
        conn.commit()
    return n


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    args = p.parse_args()

    seasons: list[int] = []
    if args.all:
        seasons = sorted(
            int(p.stem.split("_")[1])
            for p in _HIST_DIR.glob("adp_*.csv")
        )
    elif args.season:
        seasons = [args.season]
    else:
        p.error("Provide --season or --all")

    total = 0
    for s in seasons:
        n = import_season(s, args.database_url)
        logger.info("Imported %d historical ADP rows for %d", n, s)
        total += n
    logger.info("Done: %d rows", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
