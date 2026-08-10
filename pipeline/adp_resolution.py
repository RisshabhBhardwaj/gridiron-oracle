"""Canonical player-ID resolution and audit records for ADP imports.

ADP vendors identify players inconsistently.  The draft product must never use
those display names as relational keys: imports resolve them once against the
canonical ``players`` table and retain every unresolved input in an auditable
table.
"""

from __future__ import annotations

from collections import defaultdict
import json
from typing import Any, Iterable


CREATE_FANTASY_ADP = """
CREATE TABLE IF NOT EXISTS fantasy_adp (
    season INTEGER NOT NULL,
    source TEXT NOT NULL,
    scoring TEXT NOT NULL,
    player_name TEXT NOT NULL,
    position TEXT,
    team TEXT,
    adp FLOAT NOT NULL,
    player_id TEXT NOT NULL REFERENCES players(id),
    imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (season, source, scoring, player_id)
)
"""

CREATE_ADP_PLAYER_MATCHES = """
CREATE TABLE IF NOT EXISTS adp_player_matches (
    season INTEGER NOT NULL,
    source TEXT NOT NULL,
    scoring TEXT NOT NULL,
    player_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    position TEXT,
    team TEXT,
    matched_player_id TEXT REFERENCES players(id),
    match_method TEXT NOT NULL,
    match_status TEXT NOT NULL,
    candidate_player_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    audited_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (season, source, scoring, player_name)
)
"""


def normalize_name(value: object) -> str:
    name = str(value or "").lower().strip()
    for token in (".", "'", "-", " jr", " sr", " iii", " ii", " iv"):
        name = name.replace(token, "")
    return " ".join(name.split())


def normalize_position(value: object) -> str:
    """Turn vendor values such as ``WR12`` into the canonical ``WR``."""
    return "".join(ch for ch in str(value or "").upper() if ch.isalpha())


def resolve_rows(rows: Iterable[dict[str, Any]], players: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve a batch without database side effects (also useful in tests)."""
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_id: dict[str, dict[str, Any]] = {}
    for player in players:
        pid = str(player["id"])
        by_id[pid] = player
        by_name[normalize_name(player.get("full_name"))].append(player)

    resolved: list[dict[str, Any]] = []
    for raw in rows:
        candidate: dict[str, Any] | None = None
        normalized_position = normalize_position(raw.get("position"))
        supplied_id = raw.get("player_id")
        if supplied_id and str(supplied_id) in by_id:
            candidate = by_id[str(supplied_id)]
            method = "vendor_id"
        else:
            candidates = by_name.get(normalize_name(raw.get("player_name")), [])
            team = str(raw.get("team") or "").upper()
            narrowed = [
                p for p in candidates
                if (not normalized_position or normalize_position(p.get("position")) == normalized_position)
                and (not team or str(p.get("team") or "").upper() == team)
            ]
            if len(narrowed) == 1:
                candidate, method = narrowed[0], "name_position_team"
            elif len(candidates) == 1:
                candidate, method = candidates[0], "unique_name"
            else:
                method = "unmatched" if not candidates else "ambiguous_name"

        row = dict(raw)
        candidates = by_name.get(normalize_name(raw.get("player_name")), [])
        row.update(
            normalized_name=normalize_name(raw.get("player_name")),
            position=normalized_position or None,
            player_id=str(candidate["id"]) if candidate else None,
            match_method=method,
            match_status="matched" if candidate else "unmatched",
            candidate_player_ids=[str(p["id"]) for p in candidates],
        )
        resolved.append(row)
    return resolved


def resolve_and_audit(conn: Any, *, season: int, source: str, scoring: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve rows and upsert audit records.  Unmatched rows are returned too."""
    with conn.cursor() as cur:
        cur.execute(CREATE_FANTASY_ADP)
        cur.execute(CREATE_ADP_PLAYER_MATCHES)
        cur.execute("SELECT id, full_name, position, team FROM players")
        players = [dict(zip((d.name for d in cur.description), values)) for values in cur.fetchall()]
        resolved = resolve_rows(rows, players)
        for row in resolved:
            cur.execute(
                """
                INSERT INTO adp_player_matches
                    (season, source, scoring, player_name, normalized_name, position, team,
                     matched_player_id, match_method, match_status, candidate_player_ids)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (season, source, scoring, player_name) DO UPDATE SET
                    normalized_name = EXCLUDED.normalized_name, position = EXCLUDED.position,
                    team = EXCLUDED.team, matched_player_id = EXCLUDED.matched_player_id,
                    match_method = EXCLUDED.match_method, match_status = EXCLUDED.match_status,
                    candidate_player_ids = EXCLUDED.candidate_player_ids, audited_at = NOW()
                """,
                (season, source, scoring, row.get("player_name"), row["normalized_name"],
                 row.get("position"), row.get("team"), row["player_id"], row["match_method"],
                 row["match_status"], json.dumps(row["candidate_player_ids"])),
            )
    return resolved
