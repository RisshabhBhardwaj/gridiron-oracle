"""
scraper/adapters/odds_adapter.py

The Odds API adapter for player props (passing yards, rushing yards, receptions, etc.).
Stores in prop_odds for backtest calibration and line comparison.

Free tier: 500 req/month. Use sparingly (e.g. weekly fetch for upcoming games).

Standalone usage:
  python scraper/adapters/odds_adapter.py --db-url $DATABASE_URL

CLAUDE.md §6: ODDS_API_KEY in .env.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# The Odds API v4 - americanfootball_nfl
ODDS_BASE = "https://api.the-odds-api.com/v4"
PROP_MARKETS = [
    "player_pass_tds",
    "player_pass_yds",
    "player_pass_completions",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_rush_tds",
    "player_reception_tds",
]


def _dsn_for_psycopg2(url: str) -> str:
    """Convert SQLAlchemy-style URL to psycopg2-compatible DSN."""
    s = url.replace("postgresql+asyncpg://", "postgresql://")
    s = s.replace("postgresql+psycopg2://", "postgresql://")
    if "://" not in s and s:
        s = f"postgresql://{s}"
    return s


def _ensure_prop_odds_table(conn) -> None:
    """Create prop_odds table if not exists."""
    from pipeline.schema import ensure_schema

    ensure_schema(conn)


def fetch_and_store_props(db_url: str, api_key: Optional[str] = None) -> int:
    """
    Fetch NFL player props from The Odds API and store in prop_odds.
    Returns count of prop rows stored.
    """
    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        logger.info("ODDS_API_KEY not set; skipping odds fetch")
        return 0

    try:
        import requests
    except ImportError:
        logger.warning("requests not installed; skipping odds fetch")
        return 0

    # Fetch NFL events (upcoming games)
    url = f"{ODDS_BASE}/sports/americanfootball_nfl/events?apiKey={api_key}"
    time.sleep(1.1)
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        events = r.json()
    except Exception as exc:
        logger.warning("Odds API events fetch failed: %s", exc)
        return 0

    if not events:
        logger.info("No NFL events from Odds API")
        return 0

    import psycopg2

    conn = psycopg2.connect(_dsn_for_psycopg2(db_url))
    _ensure_prop_odds_table(conn)

    stored = 0
    for evt in events[:16]:  # limit to 16 games to conserve quota
        evt_id = evt.get("id")
        if not evt_id:
            continue
        # Fetch odds for this event (includes player props if available)
        odds_url = (
            f"{ODDS_BASE}/sports/americanfootball_nfl/events/{evt_id}/odds"
            f"?regions=us&markets={','.join(PROP_MARKETS)}&apiKey={api_key}"
        )
        time.sleep(1.1)
        try:
            ro = requests.get(odds_url, timeout=15)
            if ro.status_code != 200:
                logger.debug("Odds for event %s: %s", evt_id, ro.status_code)
                continue
            odds_data = ro.json()
        except Exception as exc:
            logger.debug("Odds fetch for %s: %s", evt_id, exc)
            continue

        # Parse outcomes and insert
        for book in odds_data.get("bookmakers") or []:
            bm = book.get("key") or "unknown"
            for mkt in book.get("markets") or []:
                mkt_key = mkt.get("key", "")
                if "player_" not in mkt_key:
                    continue
                stat_type = mkt_key.replace("player_", "").replace("_", "_")
                for outcome in mkt.get("outcomes") or []:
                    name = outcome.get("description") or outcome.get("name")
                    point = outcome.get("point")
                    price = outcome.get("price")
                    if not name or point is None:
                        continue
                    try:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                INSERT INTO prop_odds
                                    (odds_event_id, player_name, stat_type, line_value, over_odds, under_odds, bookmaker, fetched_at)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                                ON CONFLICT (odds_event_id, player_name, stat_type, bookmaker)
                                DO UPDATE SET line_value = EXCLUDED.line_value,
                                    over_odds = EXCLUDED.over_odds,
                                    under_odds = EXCLUDED.under_odds,
                                    fetched_at = NOW()
                                """,
                                (
                                    evt_id,
                                    str(name)[:100],
                                    stat_type,
                                    float(point),
                                    float(price) if price is not None else None,
                                    None,  # under_odds in separate outcome
                                    bm,
                                ),
                            )
                        stored += 1
                    except Exception as ins_exc:
                        logger.debug("Insert prop failed: %s", ins_exc)

    conn.commit()
    conn.close()
    if stored:
        logger.info("Stored %d prop_odds rows from Odds API", stored)
    return stored


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NFL player props from The Odds API")
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL"))
    args = parser.parse_args()
    if not args.db_url:
        parser.error("--db-url or DATABASE_URL required")
    fetch_and_store_props(args.db_url)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
