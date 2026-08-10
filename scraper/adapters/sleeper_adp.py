"""
Sleeper draft-aggregate ADP — free public API, no scraping of FantasyPros.

Pulls completed drafts (or a league's draft picks) and aggregates mean pick
number by player. Writes into `fantasy_adp` with source='sleeper'.

Usage:
  # Aggregate from a list of completed draft IDs:
  python -m scraper.adapters.sleeper_adp --season 2025 --scoring ppr \\
      --draft-ids 123 456 789

  # Or discover recent public drafts via a known league:
  python -m scraper.adapters.sleeper_adp --season 2025 --scoring ppr \\
      --league-id <SLEEPER_LEAGUE_ID>
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from typing import Any, Optional

import pandas as pd
import psycopg2
import requests

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn

logger = logging.getLogger(__name__)

_SLEEPER_BASE = "https://api.sleeper.app/v1"

def _get(path: str) -> Any:
    url = f"{_SLEEPER_BASE}{path}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_draft_picks(draft_id: str) -> list[dict]:
    data = _get(f"/draft/{draft_id}/picks")
    return data if isinstance(data, list) else []


def fetch_league_drafts(league_id: str) -> list[str]:
    """Return draft IDs associated with a league."""
    draft_ids: list[str] = []
    league = _get(f"/league/{league_id}")
    if isinstance(league, dict) and league.get("draft_id"):
        draft_ids.append(str(league["draft_id"]))
    try:
        drafts = _get(f"/league/{league_id}/drafts")
        if isinstance(drafts, list):
            for d in drafts:
                if isinstance(d, dict) and d.get("draft_id"):
                    draft_ids.append(str(d["draft_id"]))
    except requests.HTTPError:
        pass
    return list(dict.fromkeys(draft_ids))


def aggregate_adp(draft_ids: list[str]) -> pd.DataFrame:
    """
    Mean pick slot across drafts → ADP.

    Sleeper pick payload fields used:
      metadata.player_id, metadata.first_name, metadata.last_name,
      metadata.position, metadata.team, pick_no
    """
    picks_by_player: dict[str, list[dict]] = defaultdict(list)

    for did in draft_ids:
        try:
            picks = fetch_draft_picks(did)
        except requests.RequestException as exc:
            logger.warning("Failed draft %s: %s", did, exc)
            continue
        logger.info("Draft %s: %d picks", did, len(picks))
        for p in picks:
            meta = p.get("metadata") or {}
            name = (
                f"{meta.get('first_name', '')} {meta.get('last_name', '')}".strip()
                or meta.get("player_id")
                or ""
            )
            if not name:
                continue
            pick_no = p.get("pick_no")
            if pick_no is None:
                continue
            picks_by_player[name].append({
                "player_name": name,
                "position": meta.get("position"),
                "team": meta.get("team"),
                "player_id": meta.get("player_id") or p.get("player_id"),
                "pick_no": float(pick_no),
            })

    rows = []
    for name, entries in picks_by_player.items():
        adp = sum(e["pick_no"] for e in entries) / len(entries)
        sample = entries[0]
        rows.append({
            "player_name": name,
            "position": sample.get("position"),
            "team": sample.get("team"),
            "player_id": sample.get("player_id"),
            "adp": adp,
            "n_drafts": len(entries),
        })
    return pd.DataFrame(rows).sort_values("adp").reset_index(drop=True)


def upsert_adp(
    df: pd.DataFrame,
    season: int,
    scoring: str,
    database_url: str,
) -> int:
    if df.empty:
        return 0
    dsn = normalize_dsn(database_url)
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            n = 0
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO fantasy_adp
                        (season, source, scoring, player_name, position, team, adp, player_id)
                    VALUES (%s, 'sleeper', %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (season, source, scoring, player_name) DO UPDATE SET
                        position = EXCLUDED.position,
                        team = EXCLUDED.team,
                        adp = EXCLUDED.adp,
                        player_id = EXCLUDED.player_id,
                        imported_at = NOW()
                    """,
                    (
                        season,
                        scoring,
                        row["player_name"],
                        row.get("position"),
                        row.get("team"),
                        float(row["adp"]),
                        row.get("player_id"),
                    ),
                )
                n += 1
        conn.commit()
    return n


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Aggregate Sleeper draft ADP into fantasy_adp")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--scoring", default="ppr")
    p.add_argument("--draft-ids", nargs="*", default=[])
    p.add_argument("--league-id", default=None)
    p.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    draft_ids = list(args.draft_ids or [])
    if args.league_id:
        draft_ids.extend(fetch_league_drafts(args.league_id))
    draft_ids = list(dict.fromkeys(draft_ids))
    if not draft_ids:
        logger.error("Provide --draft-ids and/or --league-id")
        return 1

    df = aggregate_adp(draft_ids)
    logger.info("Aggregated ADP for %d players across %d drafts", len(df), len(draft_ids))
    if args.dry_run:
        print(df.head(30).to_string(index=False))
        return 0

    n = upsert_adp(df, args.season, args.scoring, args.database_url)
    logger.info("Upserted %d sleeper ADP rows for season=%d scoring=%s", n, args.season, args.scoring)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
