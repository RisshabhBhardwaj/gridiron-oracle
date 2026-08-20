"""DynastyProcess / nflverse cross-source player ID map.

Prerequisite for Sleeper weekly consensus capture (sleeper_id → gsis_id).
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import psycopg2

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn

logger = logging.getLogger(__name__)

CREATE_FANTASY_PLAYER_IDS = """
CREATE TABLE IF NOT EXISTS fantasy_player_ids (
    gsis_id TEXT PRIMARY KEY,
    mfl_id TEXT,
    sleeper_id TEXT,
    espn_id TEXT,
    yahoo_id TEXT,
    fantasypros_id TEXT,
    pff_id TEXT,
    sportradar_id TEXT,
    cbs_id TEXT,
    rotowire_id TEXT,
    fleaflicker_id TEXT,
    full_name TEXT,
    position TEXT,
    team TEXT,
    refreshed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

CREATE_SLEEPER_INDEX = """
CREATE INDEX IF NOT EXISTS idx_fantasy_player_ids_sleeper
    ON fantasy_player_ids (sleeper_id)
    WHERE sleeper_id IS NOT NULL
"""


def load_playerids_frame() -> Any:
    import nflreadpy

    frame = nflreadpy.load_ff_playerids()
    if hasattr(frame, "to_pandas"):
        frame = frame.to_pandas()
    return frame


def upsert_playerids(database_url: str) -> int:
    frame = load_playerids_frame()
    dsn = normalize_dsn(database_url)
    n = 0
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_FANTASY_PLAYER_IDS)
            cur.execute(CREATE_SLEEPER_INDEX)
            for raw in frame.to_dict(orient="records"):
                gsis = raw.get("gsis_id")
                if not gsis:
                    continue
                cur.execute(
                    """
                    INSERT INTO fantasy_player_ids (
                        gsis_id, mfl_id, sleeper_id, espn_id, yahoo_id, fantasypros_id,
                        pff_id, sportradar_id, cbs_id, rotowire_id, fleaflicker_id,
                        full_name, position, team, refreshed_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                    )
                    ON CONFLICT (gsis_id) DO UPDATE SET
                        mfl_id = EXCLUDED.mfl_id,
                        sleeper_id = EXCLUDED.sleeper_id,
                        espn_id = EXCLUDED.espn_id,
                        yahoo_id = EXCLUDED.yahoo_id,
                        fantasypros_id = EXCLUDED.fantasypros_id,
                        pff_id = EXCLUDED.pff_id,
                        sportradar_id = EXCLUDED.sportradar_id,
                        cbs_id = EXCLUDED.cbs_id,
                        rotowire_id = EXCLUDED.rotowire_id,
                        fleaflicker_id = EXCLUDED.fleaflicker_id,
                        full_name = EXCLUDED.full_name,
                        position = EXCLUDED.position,
                        team = EXCLUDED.team,
                        refreshed_at = NOW()
                    """,
                    (
                        str(gsis),
                        _opt_str(raw.get("mfl_id")),
                        _opt_str(raw.get("sleeper_id")),
                        _opt_str(raw.get("espn_id")),
                        _opt_str(raw.get("yahoo_id")),
                        _opt_str(raw.get("fantasypros_id")),
                        _opt_str(raw.get("pff_id")),
                        _opt_str(raw.get("sportradar_id")),
                        _opt_str(raw.get("cbs_id")),
                        _opt_str(raw.get("rotowire_id")),
                        _opt_str(raw.get("fleaflicker_id")),
                        _opt_str(raw.get("name") or raw.get("full_name")),
                        _opt_str(raw.get("position")),
                        _opt_str(raw.get("team")),
                    ),
                )
                n += 1
        conn.commit()
    logger.info("Upserted %d fantasy_player_ids rows", n)
    return n


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    args = parser.parse_args()
    upsert_playerids(args.database_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
