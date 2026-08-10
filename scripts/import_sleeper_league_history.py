#!/usr/bin/env python3
"""Import completed Sleeper league drafts for separate tendency analysis.

This is intentionally distinct from ``sleeper_adp``: historical league picks
describe manager behaviour and must not be blended into current-year ADP.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn
from scraper.adapters.sleeper_adp import _get, fetch_draft_picks


def _owner_ids(league_id: str | None) -> dict[int, str]:
    if not league_id:
        return {}
    rosters = _get(f"/league/{league_id}/rosters")
    if not isinstance(rosters, list):
        return {}
    return {
        int(roster["roster_id"]): str(roster["owner_id"])
        for roster in rosters
        if isinstance(roster, dict) and roster.get("roster_id") is not None and roster.get("owner_id")
    }


def _members(league_id: str | None) -> list[dict[str, Any]]:
    if not league_id:
        return []
    users = _get(f"/league/{league_id}/users")
    if not isinstance(users, list):
        return []
    return [
        {
            "league_id": league_id, "owner_id": str(user["user_id"]),
            "username": user.get("username"), "display_name": user.get("display_name"),
            "avatar": user.get("avatar"),
        }
        for user in users if isinstance(user, dict) and user.get("user_id")
    ]


def normalized_pick(draft: dict[str, Any], pick: dict[str, Any], owners: dict[int, str]) -> dict[str, Any]:
    metadata = pick.get("metadata") or {}
    roster_id = pick.get("roster_id")
    name = f"{metadata.get('first_name', '')} {metadata.get('last_name', '')}".strip()
    return {
        "draft_id": str(draft["draft_id"]), "pick_no": int(pick["pick_no"]),
        "season": int(draft["season"]), "league_id": draft.get("league_id"),
        "roster_id": roster_id, "owner_id": owners.get(int(roster_id)) if roster_id is not None else None,
        "round": pick.get("round"), "draft_slot": pick.get("draft_slot"),
        "player_id": metadata.get("player_id") or pick.get("player_id"),
        "player_name": name or None, "position": metadata.get("position"), "team": metadata.get("team"),
        "raw_metadata": metadata,
    }


def import_history(draft_ids: list[str], database_url: str) -> dict[str, int]:
    rows: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    completed = 0
    for draft_id in draft_ids:
        draft = _get(f"/draft/{draft_id}")
        if not isinstance(draft, dict) or not draft.get("draft_id"):
            raise ValueError(f"Sleeper draft {draft_id} was not found")
        picks = fetch_draft_picks(draft_id)
        if not picks:
            continue
        owners = _owner_ids(draft.get("league_id"))
        members.extend(_members(draft.get("league_id")))
        rows.extend(normalized_pick(draft, pick, owners) for pick in picks if pick.get("pick_no") is not None)
        completed += 1
    if not rows:
        return {"drafts": 0, "picks": 0}

    sql = """
        INSERT INTO sleeper_league_draft_picks
            (draft_id, pick_no, season, league_id, roster_id, owner_id, round, draft_slot,
             player_id, player_name, position, team, raw_metadata)
        VALUES (%(draft_id)s, %(pick_no)s, %(season)s, %(league_id)s, %(roster_id)s,
                %(owner_id)s, %(round)s, %(draft_slot)s, %(player_id)s, %(player_name)s,
                %(position)s, %(team)s, %(raw_metadata)s::jsonb)
        ON CONFLICT (draft_id, pick_no) DO UPDATE SET
            season = EXCLUDED.season, league_id = EXCLUDED.league_id, roster_id = EXCLUDED.roster_id,
            owner_id = EXCLUDED.owner_id, round = EXCLUDED.round, draft_slot = EXCLUDED.draft_slot,
            player_id = EXCLUDED.player_id, player_name = EXCLUDED.player_name,
            position = EXCLUDED.position, team = EXCLUDED.team, raw_metadata = EXCLUDED.raw_metadata,
            imported_at = NOW()
    """
    with psycopg2.connect(normalize_dsn(database_url)) as conn:
        with conn.cursor() as cur:
            for row in rows:
                row["raw_metadata"] = json.dumps(row["raw_metadata"])
            psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
            if members:
                psycopg2.extras.execute_batch(cur, """
                    INSERT INTO sleeper_league_members (league_id, owner_id, username, display_name, avatar)
                    VALUES (%(league_id)s, %(owner_id)s, %(username)s, %(display_name)s, %(avatar)s)
                    ON CONFLICT (league_id, owner_id) DO UPDATE SET
                        username = EXCLUDED.username, display_name = EXCLUDED.display_name,
                        avatar = EXCLUDED.avatar, imported_at = NOW()
                """, members, page_size=100)
        conn.commit()
    return {"drafts": completed, "picks": len(rows), "members": len(members)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-ids", required=True, nargs="+")
    parser.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    args = parser.parse_args()
    print(json.dumps(import_history(args.draft_ids, args.database_url), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
