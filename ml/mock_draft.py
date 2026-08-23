"""
ml/mock_draft.py

Mock draft pick engine that simulates opponent selections conditioned on
empirical manager profiles, positional timing gates, roster limits, and reach tendencies.
Appends a VOR-ordered tail from draft_preseason_projections so the draft pool can never exhaust.

Does NOT touch ml/draft_sim.py:snake_draft (the walk-forward evaluator depends on its exact behavior).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence
import numpy as np
import psycopg2
import psycopg2.extras

from ml.vor import FLEX_ELIGIBLE, LEAGUE_TEAMS, SKILL_POSITIONS, STARTING_SLOTS, attach_vor
from pipeline.schema import normalize_dsn

logger = logging.getLogger(__name__)

# Positional roster limits: starting slots + 2
ROSTER_MAX_SLOTS: dict[str, int] = {
    "QB": STARTING_SLOTS.get("QB", 1) + 2,   # 3 max
    "RB": STARTING_SLOTS.get("RB", 2) + 2,   # 4 max (excl flex buffer)
    "WR": STARTING_SLOTS.get("WR", 2) + 2,   # 4 max (excl flex buffer)
    "TE": STARTING_SLOTS.get("TE", 1) + 2,   # 3 max
}


def pick(
    available: list[dict[str, Any]],
    roster: list[dict[str, Any]],
    profile: dict[str, Any],
    rng: np.random.Generator,
    pick_no: int = 1,
    top_k: int = 8,
) -> dict[str, Any]:
    """
    Select the next player for a manager profile given current available board and roster.

    Steps:
    1. Sort available by blended_rank (or model_rank / adp / vor).
    2. Filter by roster legality (hard-cap at STARTING_SLOTS[pos] + 2).
    3. Filter by positional timing gate (drop QB/TE if pick_no < first_qb/te_pick_shrunk).
    4. Top-K candidate subset.
    5. Reach shift & Softmax sampling with temperature scaled by adp_delta_sd_shrunk.
    """
    if not available:
        raise ValueError("Cannot pick from an empty available player pool.")

    # 1. Sort available by best rank
    def _rank_key(row: dict[str, Any]) -> float:
        if row.get("blended_rank") is not None:
            return float(row["blended_rank"])
        if row.get("model_rank") is not None:
            return float(row["model_rank"])
        if row.get("adp_rank") is not None:
            return float(row["adp_rank"])
        if row.get("adp") is not None:
            return float(row["adp"])
        return 9999.0

    sorted_available = sorted(available, key=_rank_key)

    # 2. Roster legality filter
    roster_pos_counts: dict[str, int] = {}
    for p in roster:
        pos = str(p.get("position") or "").upper()
        roster_pos_counts[pos] = roster_pos_counts.get(pos, 0) + 1

    legal_candidates = []
    for player in sorted_available:
        pos = str(player.get("position") or "").upper()
        max_allowed = ROSTER_MAX_SLOTS.get(pos, 5)
        # Add flex allowance buffer if RB/WR/TE
        if pos in FLEX_ELIGIBLE:
            max_allowed += STARTING_SLOTS.get("FLEX", 1)

        if roster_pos_counts.get(pos, 0) < max_allowed:
            legal_candidates.append(player)

    if not legal_candidates:
        legal_candidates = sorted_available

    # 3. Positional timing gate
    first_qb_shrunk = float(profile.get("first_qb_pick_shrunk", 48.0) or 48.0)
    first_te_shrunk = float(profile.get("first_te_pick_shrunk", 64.0) or 64.0)

    gated_candidates = []
    has_non_qb = any(str(p.get("position") or "").upper() != "QB" for p in legal_candidates)
    has_non_te = any(str(p.get("position") or "").upper() != "TE" for p in legal_candidates)

    for p in legal_candidates:
        pos = str(p.get("position") or "").upper()
        if pos == "QB" and pick_no < first_qb_shrunk and has_non_qb:
            continue
        if pos == "TE" and pick_no < first_te_shrunk and has_non_te:
            continue
        gated_candidates.append(p)

    if not gated_candidates:
        gated_candidates = legal_candidates

    # 4. Top-K subset
    candidates = gated_candidates[:top_k]
    if len(candidates) == 1:
        return candidates[0]

    # 5. Reach shift & Softmax sampling
    adp_reach_shift = float(profile.get("adp_delta_mean_shrunk", 0.0) or 0.0)
    adp_sd = float(profile.get("adp_delta_sd_shrunk", 6.0) or 6.0)
    temperature = max(0.5, adp_sd / 3.0)

    # Base utility based on rank position in top candidates
    utilities = np.array([float(top_k - i) for i in range(len(candidates))], dtype=float)
    # Reach shift: reaching managers have higher probability of picking slightly deeper candidates
    if adp_reach_shift < 0:
        # Negative adp delta -> reaches -> flatten top utilities
        reach_factor = min(2.0, abs(adp_reach_shift) / 5.0)
        utilities = utilities + np.linspace(0, reach_factor, len(candidates))

    logits = utilities / temperature
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    chosen_idx = int(rng.choice(len(candidates), p=probs))
    return candidates[chosen_idx]


def build_mock_draft_pool(
    database_url: str,
    season: int = 2026,
    scoring: str = "ppr",
    min_required_picks: int = 136,
) -> list[dict[str, Any]]:
    """
    Build available player pool for mock draft by combining fantasy_adp with
    a VOR-ordered tail from draft_preseason_projections to prevent exhaustion.
    """
    dsn = normalize_dsn(database_url)
    with psycopg2.connect(dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 1. Fetch ADP rows
            cur.execute(
                """
                SELECT player_name, position, team, adp, player_id, source
                FROM fantasy_adp
                WHERE season = %s AND scoring = %s AND player_id IS NOT NULL
                  AND UPPER(position) IN ('QB', 'RB', 'WR', 'TE')
                ORDER BY adp ASC
                """,
                (season, scoring),
            )
            adp_rows = [dict(r) for r in cur.fetchall()]

            # 2. Fetch Preseason Projections
            cur.execute(
                """
                WITH latest AS (
                    SELECT MAX(as_of) AS as_of
                    FROM draft_preseason_projections
                    WHERE season = %s
                )
                SELECT player_id, player_name, position, team, projection AS fantasy_ppr
                FROM draft_preseason_projections
                JOIN latest USING (as_of)
                WHERE season = %s AND UPPER(position) IN ('QB', 'RB', 'WR', 'TE')
                """,
                (season, season),
            )
            proj_rows = [dict(r) for r in cur.fetchall()]

    adp_player_ids = {str(r["player_id"]) for r in adp_rows if r.get("player_id")}
    proj_by_id = {str(r["player_id"]): r for r in proj_rows if r.get("player_id")}

    # Build primary board
    board: list[dict[str, Any]] = []
    for idx, adp_row in enumerate(adp_rows, start=1):
        pid = str(adp_row["player_id"])
        proj = proj_by_id.get(pid, {})
        ppr_val = float(proj.get("fantasy_ppr") or 0.0)
        board.append({
            "player_id": pid,
            "player_name": adp_row["player_name"],
            "position": adp_row.get("position") or proj.get("position"),
            "team": adp_row.get("team") or proj.get("team"),
            "adp": float(adp_row["adp"]),
            "adp_rank": idx,
            "model_fantasy_ppr": ppr_val if ppr_val > 0 else None,
            "model_rank": None,
            "blended_rank": idx,
            "source": adp_row.get("source", "fantasy_adp"),
            "board_tail": False,
        })

    # Build VOR-ordered tail for unlisted players in projections
    unlisted = [p for p in proj_rows if str(p.get("player_id")) not in adp_player_ids]
    if unlisted:
        ranked_unlisted = attach_vor([
            {
                "player_id": str(p["player_id"]),
                "position": p.get("position"),
                "projection": float(p.get("fantasy_ppr") or 0.0),
            }
            for p in unlisted
        ])
        ranked_unlisted.sort(key=lambda item: (-(item["vor"] if item.get("vor") is not None else -1e18), item["player_id"]))

        tail_start_rank = len(board) + 1
        for offset, item in enumerate(ranked_unlisted):
            pid = str(item["player_id"])
            proj = proj_by_id[pid]
            board.append({
                "player_id": pid,
                "player_name": proj["player_name"],
                "position": proj.get("position"),
                "team": proj.get("team"),
                "adp": None,
                "adp_rank": None,
                "model_fantasy_ppr": float(proj.get("fantasy_ppr") or 0.0),
                "model_rank": tail_start_rank + offset,
                "blended_rank": tail_start_rank + offset,
                "source": "draft_preseason_projections",
                "board_tail": True,
            })

    if len(board) <= min_required_picks:
        raise ValueError(
            f"Draft pool size ({len(board)}) is insufficient for {min_required_picks} picks."
        )

    return board
