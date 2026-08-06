"""
Coaching / scheme adapter — free, manual CSV seed.

Loads `data/coaching/coaching_{season}.csv` into the `team_coaching` table.
No paid APIs. Refresh the CSV when coaching staff changes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "coaching"

CREATE_TEAM_COACHING = """
CREATE TABLE IF NOT EXISTS team_coaching (
    season INTEGER NOT NULL,
    team TEXT NOT NULL,
    head_coach TEXT,
    offensive_coordinator TEXT,
    defensive_coordinator TEXT,
    scheme_pass_rate_prior FLOAT,
    notes TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (season, team)
)
"""


def load_coaching_csv(season: int) -> pd.DataFrame:
    path = DATA_DIR / f"coaching_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing coaching seed {path}. Add a manual CSV for season {season}."
        )
    df = pd.read_csv(path)
    required = {"season", "team", "head_coach", "offensive_coordinator", "defensive_coordinator"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[df["season"] == season].copy()
    if df.empty:
        raise ValueError(f"{path} has no rows for season={season}")
    return df


def upsert_coaching(db_url: str, season: int) -> int:
    import psycopg2
    from psycopg2.extras import execute_values

    df = load_coaching_csv(season)
    conn = psycopg2.connect(normalize_dsn(db_url))
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TEAM_COACHING)
            rows = [
                (
                    int(r.season),
                    str(r.team),
                    r.head_coach,
                    r.offensive_coordinator,
                    r.defensive_coordinator,
                    float(r.scheme_pass_rate_prior)
                    if pd.notna(getattr(r, "scheme_pass_rate_prior", None))
                    else None,
                    getattr(r, "notes", None),
                )
                for r in df.itertuples(index=False)
            ]
            execute_values(
                cur,
                """
                INSERT INTO team_coaching
                    (season, team, head_coach, offensive_coordinator,
                     defensive_coordinator, scheme_pass_rate_prior, notes)
                VALUES %s
                ON CONFLICT (season, team) DO UPDATE SET
                    head_coach = EXCLUDED.head_coach,
                    offensive_coordinator = EXCLUDED.offensive_coordinator,
                    defensive_coordinator = EXCLUDED.defensive_coordinator,
                    scheme_pass_rate_prior = EXCLUDED.scheme_pass_rate_prior,
                    notes = EXCLUDED.notes,
                    updated_at = NOW()
                """,
                rows,
            )
        conn.commit()
        logger.info("Upserted %d coaching rows for season %d", len(rows), season)
        return len(rows)
    finally:
        conn.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()
    import os

    db_url = args.db_url or os.environ.get("DATABASE_URL", DEFAULT_HOST_DATABASE_URL)
    n = upsert_coaching(db_url, args.season)
    print(f"upserted {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
