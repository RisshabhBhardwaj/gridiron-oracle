#!/usr/bin/env python3
"""Generate transparent per-manager profiles from historical Sleeper picks."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import psycopg2
import psycopg2.extras

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn

EARLY_PICK_CUTOFF = 24  # first three rounds in this eight-team league


def _as_float(values: list[int]) -> float | None:
    return round(float(mean(values)), 2) if values else None


def build_profiles(rows: list[dict[str, Any]], names: dict[str, str]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("owner_id"):
            grouped[str(row["owner_id"])].append(row)
    profiles: list[dict[str, Any]] = []
    for owner_id, picks in grouped.items():
        by_draft: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pick in picks:
            by_draft[str(pick["draft_id"])].append(pick)
        position_counts = Counter(str(pick.get("position") or "Unknown") for pick in picks)
        early = [pick for pick in picks if int(pick["pick_no"]) <= EARLY_PICK_CUTOFF]
        early_positions = Counter(str(pick.get("position") or "Unknown") for pick in early)
        first_qb = [min(int(p["pick_no"]) for p in draft if p.get("position") == "QB") for draft in by_draft.values() if any(p.get("position") == "QB" for p in draft)]
        first_te = [min(int(p["pick_no"]) for p in draft if p.get("position") == "TE") for draft in by_draft.values() if any(p.get("position") == "TE" for p in draft)]
        profiles.append({
            "owner_id": owner_id,
            "display_name": names.get(owner_id, owner_id),
            "drafts": len(by_draft), "picks": len(picks),
            "positions": dict(sorted(position_counts.items())),
            "early_pick_positions": dict(sorted(early_positions.items())),
            "rb_wr_share_first_24": round(sum(1 for pick in early if pick.get("position") in {"RB", "WR"}) / len(early), 3) if early else None,
            "first_qb_pick_average": _as_float(first_qb),
            "first_te_pick_average": _as_float(first_te),
            "first_qb_pick_median": float(median(first_qb)) if first_qb else None,
            "first_te_pick_median": float(median(first_te)) if first_te else None,
        })
    return {
        "method": {
            "early_pick_cutoff": EARLY_PICK_CUTOFF,
            "note": "Profiles use actual overall pick numbers, preserving traded-pick effects. They describe historical tendencies, not player-value recommendations.",
        },
        "profiles": sorted(profiles, key=lambda profile: (profile["display_name"].lower(), profile["owner_id"])),
    }


def load_profiles(database_url: str) -> dict[str, Any]:
    with psycopg2.connect(normalize_dsn(database_url)) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT draft_id, owner_id, pick_no, position FROM sleeper_league_draft_picks ORDER BY draft_id, pick_no")
            rows = [dict(row) for row in cur.fetchall()]
            cur.execute("""
                SELECT DISTINCT ON (owner_id) owner_id, COALESCE(display_name, username, owner_id) AS name
                FROM sleeper_league_members ORDER BY owner_id, imported_at DESC
            """)
            names = {str(row["owner_id"]): str(row["name"]) for row in cur.fetchall()}
    return build_profiles(rows, names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    parser.add_argument("--output", type=Path, default=Path("reports/sleeper_league_profiles.json"))
    args = parser.parse_args()
    payload = load_profiles(args.database_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
