"""Roster-aware draft evaluation for the configured 8-team PPR league.

Why this module exists
----------------------
``ml.consensus_baseline.points_lost_vs_optimal`` ranks a projection, takes the
top ``slots`` players, and compares their realized points against the top
``slots`` realized scorers. Across positions that oracle is illegal: in this
league the 24 highest-scoring players of a season are 9-11 quarterbacks, and
only one may be started. A board is therefore rewarded for drafting
quarterbacks it could never play, and punished for the replacement-level
correction in :mod:`ml.vor`.

This module scores a board the way the league actually scores it: run a snake
draft, fill a legal starting lineup, count the points that lineup produced.
``points_lost_vs_optimal`` remains correct *within* a position and is still
used that way by :mod:`scripts.draft_walkforward`.

League (locked): 8 teams, PPR, 1 QB / 2 RB / 2 WR / 1 TE / 1 FLEX (RB/WR/TE).
K and DST are out of scope.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ml.vor import FLEX_ELIGIBLE, LEAGUE_TEAMS, STARTING_SLOTS

DEFAULT_ROUNDS = 14


def _board_order(scores: Sequence[float]) -> np.ndarray:
    """Row indices best-first. Non-finite scores sort last, original order kept."""
    values = pd.to_numeric(pd.Series(list(scores)), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    ranked = np.argsort(-values[finite], kind="stable")
    return np.concatenate([np.where(finite)[0][ranked], np.where(~finite)[0]])


def snake_draft(
    subject_order: Sequence[int],
    market_order: Sequence[int],
    *,
    subject_slot: int,
    n_teams: int = LEAGUE_TEAMS,
    rounds: int = DEFAULT_ROUNDS,
) -> list[int]:
    """Return the subject team's roster (row indices) from a snake draft.

    The subject drafts strictly down ``subject_order``; the other seats draft
    strictly down ``market_order``. No positional caps are imposed — a board
    that ranks fourteen quarterbacks first drafts fourteen quarterbacks and
    fields an incomplete lineup. Hiding that behind a cap would suppress the
    exact defect this harness exists to measure.
    """
    if not 0 <= subject_slot < n_teams:
        raise ValueError(f"subject_slot {subject_slot} outside 0..{n_teams - 1}")
    cursors = [0] * n_teams
    orders = [list(market_order)] * n_teams
    orders[subject_slot] = list(subject_order)
    taken: set[int] = set()
    roster: list[int] = []
    for rnd in range(rounds):
        seats = range(n_teams) if rnd % 2 == 0 else reversed(range(n_teams))
        for seat in seats:
            order = orders[seat]
            cursor = cursors[seat]
            while cursor < len(order) and order[cursor] in taken:
                cursor += 1
            cursors[seat] = cursor
            if cursor >= len(order):
                continue
            pick = order[cursor]
            taken.add(pick)
            cursors[seat] = cursor + 1
            if seat == subject_slot:
                roster.append(pick)
    return roster


def starting_lineup_points(
    roster: Sequence[int],
    positions: Sequence[str],
    realized: Sequence[float],
) -> float:
    """Best legal starting lineup from a roster, scored on realized season totals.

    An unfillable slot contributes zero. The same oracle-lineup rule is applied
    to every board, so it cannot favour one.
    """
    pos = np.asarray(positions, dtype=object)
    pts = pd.to_numeric(pd.Series(list(realized)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pool: dict[str, list[float]] = {}
    for idx in roster:
        pool.setdefault(str(pos[idx]).upper(), []).append(float(pts[idx]))
    for key in pool:
        pool[key].sort(reverse=True)

    total = 0.0
    for position, count in STARTING_SLOTS.items():
        if position == "FLEX":
            continue
        available = pool.get(position, [])
        total += sum(available[:count])
        pool[position] = available[count:]
    flex_pool = sorted(
        (value for position in FLEX_ELIGIBLE for value in pool.get(position, [])),
        reverse=True,
    )
    total += sum(flex_pool[: STARTING_SLOTS["FLEX"]])
    return float(total)


def evaluate_board(
    board_scores: Sequence[float],
    market_scores: Sequence[float],
    positions: Sequence[str],
    realized: Sequence[float],
    *,
    n_teams: int = LEAGUE_TEAMS,
    rounds: int = DEFAULT_ROUNDS,
) -> dict[str, float]:
    """Mean starting-lineup points for a board, rotated through every draft slot.

    Rotation removes the seat advantage that a single simulated draft would
    otherwise bake in. ``market_scores`` are ADP values (lower is better) and
    are negated internally so that every board is "higher is better".
    """
    subject_order = _board_order(board_scores)
    market_order = _board_order([-value for value in pd.to_numeric(pd.Series(list(market_scores)), errors="coerce")])
    per_slot = [
        starting_lineup_points(
            snake_draft(subject_order, market_order, subject_slot=slot, n_teams=n_teams, rounds=rounds),
            positions,
            realized,
        )
        for slot in range(n_teams)
    ]
    return {
        "mean_starter_points": float(np.mean(per_slot)),
        "min_starter_points": float(np.min(per_slot)),
        "max_starter_points": float(np.max(per_slot)),
        "per_slot": [round(value, 2) for value in per_slot],
    }


def positional_points_lost(
    frame: pd.DataFrame,
    score_col: str,
    *,
    realized_col: str = "realized",
    position_col: str = "position",
    ascending: bool = False,
) -> dict[str, float]:
    """``points_lost_vs_optimal`` applied within each position, where it is valid.

    Slot counts are league-wide starters (8 QB, 16 RB, 16 WR, 8 TE), so the
    per-position oracle is a roster the league could actually field.
    """
    from ml.consensus_baseline import points_lost_vs_optimal

    slots: Mapping[str, int] = {
        "QB": LEAGUE_TEAMS * STARTING_SLOTS["QB"],
        "RB": LEAGUE_TEAMS * STARTING_SLOTS["RB"],
        "WR": LEAGUE_TEAMS * STARTING_SLOTS["WR"],
        "TE": LEAGUE_TEAMS * STARTING_SLOTS["TE"],
    }
    out: dict[str, float] = {}
    for position, count in slots.items():
        subset = frame[frame[position_col].astype(str).str.upper() == position]
        if len(subset) < count:
            out[position] = float("nan")
            continue
        scores = pd.to_numeric(subset[score_col], errors="coerce")
        if ascending:
            scores = -scores
        out[position] = points_lost_vs_optimal(
            scores.to_numpy(), subset[realized_col].to_numpy(), slots=count
        )
    return out
