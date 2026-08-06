"""
FantasyPros ADP importer — manual CSV export only.

FantasyPros terms prohibit scraping, and this repository is public.
Download the PPR ADP CSV from FantasyPros and place it under
`data/adp/fantasypros/`, then run:

  python -m scraper.adapters.fantasypros_adp_importer --season 2026 --path data/adp/fantasypros/ppr_2026.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn

logger = logging.getLogger(__name__)

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

# Common FantasyPros export headers → canonical names
_COLUMN_ALIASES = {
    "player": "player_name",
    "player name": "player_name",
    "name": "player_name",
    "pos": "position",
    "team": "team",
    "avg": "adp",
    "average": "adp",
    "adp": "adp",
    "rank": "adp",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in _COLUMN_ALIASES:
            rename[col] = _COLUMN_ALIASES[key]
    out = df.rename(columns=rename)
    required = {"player_name", "adp"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(
            f"FantasyPros CSV missing columns {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )
    return out


def import_csv(
    db_url: str,
    *,
    path: Path,
    season: int,
    scoring: str = "ppr",
    source: str = "fantasypros",
) -> int:
    import psycopg2
    from psycopg2.extras import execute_values

    df = _normalize_columns(pd.read_csv(path))
    conn = psycopg2.connect(normalize_dsn(db_url))
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_FANTASY_ADP)
            rows = [
                (
                    season,
                    source,
                    scoring,
                    str(r.player_name),
                    getattr(r, "position", None),
                    getattr(r, "team", None),
                    float(r.adp),
                    None,
                )
                for r in df.itertuples(index=False)
                if pd.notna(r.adp)
            ]
            execute_values(
                cur,
                """
                INSERT INTO fantasy_adp
                    (season, source, scoring, player_name, position, team, adp, player_id)
                VALUES %s
                ON CONFLICT (season, source, scoring, player_name) DO UPDATE SET
                    position = EXCLUDED.position,
                    team = EXCLUDED.team,
                    adp = EXCLUDED.adp,
                    imported_at = NOW()
                """,
                rows,
            )
        conn.commit()
        logger.info("Imported %d FantasyPros ADP rows from %s", len(rows), path)
        return len(rows)
    finally:
        conn.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()
    import os

    db_url = args.db_url or os.environ.get("DATABASE_URL", DEFAULT_HOST_DATABASE_URL)
    n = import_csv(db_url, path=args.path, season=args.season, scoring=args.scoring)
    print(f"imported {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
