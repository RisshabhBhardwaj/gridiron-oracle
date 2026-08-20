"""Replacement-level (VOR) ranking for the configured fantasy league.

League (locked): 8 teams, PPR, 1 QB / 2 RB / 2 WR / 1 TE / 1 FLEX (RB/WR/TE).
K and DST are out of scope and must not appear in the ranked universe.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

LEAGUE_TEAMS = 8
STARTING_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
FLEX_ELIGIBLE = frozenset({"RB", "WR", "TE"})
SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})


def league_wide_starters(position: str) -> int:
    if position == "FLEX":
        return LEAGUE_TEAMS * STARTING_SLOTS["FLEX"]
    if position not in STARTING_SLOTS:
        raise KeyError(f"Unknown position {position!r}")
    return LEAGUE_TEAMS * STARTING_SLOTS[position]


def replacement_projections(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Return per-position replacement projection (points, same scale as input)."""
    by_pos: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        pos = str(row.get("position") or "").upper()
        if pos not in SKILL_POSITIONS:
            continue
        value = row.get("projection")
        if value is None:
            continue
        by_pos[pos].append(float(value))
    for pos in by_pos:
        by_pos[pos].sort(reverse=True)

    replacement: dict[str, float] = {}
    qb_index = league_wide_starters("QB")  # 8 starters → 9th is replacement (index 8)
    te_starters = league_wide_starters("TE")
    rb_starters = league_wide_starters("RB")
    wr_starters = league_wide_starters("WR")
    flex_slots = league_wide_starters("FLEX")

    replacement["QB"] = _nth_or_zero(by_pos.get("QB", []), qb_index)

    remainder: list[tuple[str, float]] = []
    remainder.extend(("RB", v) for v in by_pos.get("RB", [])[rb_starters:])
    remainder.extend(("WR", v) for v in by_pos.get("WR", [])[wr_starters:])
    remainder.extend(("TE", v) for v in by_pos.get("TE", [])[te_starters:])
    remainder.sort(key=lambda item: item[1], reverse=True)
    flex_taken = remainder[:flex_slots]
    leftover = remainder[flex_slots:]

    def _first_leftover(position: str, starter_floor: list[float]) -> float:
        for pos, value in leftover:
            if pos == position:
                return value
        if starter_floor:
            return starter_floor[-1]
        return 0.0

    replacement["RB"] = _first_leftover("RB", by_pos.get("RB", [])[:rb_starters])
    replacement["WR"] = _first_leftover("WR", by_pos.get("WR", [])[:wr_starters])
    replacement["TE"] = _first_leftover("TE", by_pos.get("TE", [])[:te_starters])
    if not leftover and flex_taken:
        # Deep leagues can exhaust the bench; last flexed player is replacement.
        last_flex = {pos: value for pos, value in flex_taken}
        for pos in FLEX_ELIGIBLE:
            replacement[pos] = min(replacement[pos], last_flex.get(pos, replacement[pos]))
    return replacement


def attach_vor(rows: Iterable[Mapping[str, object]]) -> list[dict]:
    """Copy rows with ``vor`` = projection − replacement(position)."""
    materialized = [dict(row) for row in rows]
    reps = replacement_projections(materialized)
    for row in materialized:
        pos = str(row.get("position") or "").upper()
        projection = row.get("projection")
        if projection is None or pos not in reps:
            row["vor"] = None
        else:
            row["vor"] = float(projection) - reps[pos]
    return materialized


def _nth_or_zero(values: Sequence[float], index: int) -> float:
    if index < len(values):
        return float(values[index])
    if values:
        return float(values[-1])
    return 0.0
