"""
ml/scoring.py

L4 (Coherent Prediction Hierarchy, Phase 6) — the single fantasy-scoring
function, replacing three formulas that used to disagree:

  1. The direct `fantasy_ppr` regression (ml/train.py) — retained as a
     calibration anchor per the plan, not replaced. Systematic disagreement
     between it and score_fantasy(component_stats) is a useful alarm, not
     a bug to reconcile away.
  2. ml.monte_carlo.MonteCarloProjector.project()'s FANTASY_PTS_PER_YARD
     scaling — which converts ONE stat's posterior samples into fantasy
     points using that stat's own rate, so a WR's receiving_yards
     projection becomes points with receptions and TDs nowhere in it.
  3. scripts/materialize_stack_projections.py's never-written aggregation —
     `fantasy_projection = None` for every non-fantasy-ppr stat row.

score_fantasy takes a full component-stat dict for one player-game and
returns one number, computed from all of it at once, so a receiving line
is never scored on yards alone.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeagueRules:
    """
    Points per unit of each component stat. Defaults to standard full-PPR
    (DraftKings/FanDuel-style), matching the weights previously hardcoded
    in ml.monte_carlo.FANTASY_PTS_PER_YARD.
    """
    pts_per_reception: float = 1.0
    pts_per_receiving_yard: float = 0.1
    pts_per_rushing_yard: float = 0.1
    pts_per_passing_yard: float = 0.04
    pts_per_receiving_td: float = 6.0
    pts_per_rushing_td: float = 6.0
    pts_per_passing_td: float = 4.0
    pts_per_interception: float = -2.0
    # The "fumbles" stat (ml.stat_resolution._fumbles_combined) sums
    # rushing + receiving + sack fumbles regardless of whether the offense
    # recovered it. Standard scoring only penalizes fumbles LOST; this
    # inherits the existing codebase's fumbles-not-fumbles-lost simplification
    # (ml.monte_carlo.FANTASY_PTS_PER_YARD["fumbles"] = -2.0) rather than
    # introducing a new scoring-rule distinction no upstream model makes.
    pts_per_fumble: float = -2.0
    # Zero-weight in standard PPR; present so passing these keys is a no-op
    # rather than a silently-ignored KeyError risk.
    pts_per_carry: float = 0.0
    pts_per_target: float = 0.0
    pts_per_pass_attempt: float = 0.0
    pts_per_completion: float = 0.0
    pts_per_sack_taken: float = 0.0


STANDARD_PPR = LeagueRules()
HALF_PPR = LeagueRules(pts_per_reception=0.5)
STANDARD_NON_PPR = LeagueRules(pts_per_reception=0.0)

# component stat name -> LeagueRules field name
_STAT_TO_RULE_FIELD: dict[str, str] = {
    "receptions": "pts_per_reception",
    "receiving_yards": "pts_per_receiving_yard",
    "rushing_yards": "pts_per_rushing_yard",
    "passing_yards": "pts_per_passing_yard",
    "receiving_tds": "pts_per_receiving_td",
    "rushing_tds": "pts_per_rushing_td",
    "passing_tds": "pts_per_passing_td",
    "interceptions": "pts_per_interception",
    "fumbles": "pts_per_fumble",
    "carries": "pts_per_carry",
    "targets": "pts_per_target",
    "pass_attempts": "pts_per_pass_attempt",
    "completions": "pts_per_completion",
    "sacks_taken": "pts_per_sack_taken",
}

#: Every component stat score_fantasy will score if present in the input dict.
SCORED_STATS: frozenset[str] = frozenset(_STAT_TO_RULE_FIELD)


def score_fantasy(component_stats: dict[str, float], rules: LeagueRules = STANDARD_PPR) -> float:
    """
    Sum every recognized component stat in `component_stats` against `rules`.

    Keys not in SCORED_STATS (e.g. "fantasy_ppr" itself, or a helper column
    like "target_share") are ignored rather than raising — callers commonly
    pass a full projections row that mixes scored and unscored fields. A
    missing key contributes 0, same as an explicit 0 or None value.
    """
    total = 0.0
    for stat, value in component_stats.items():
        field_name = _STAT_TO_RULE_FIELD.get(stat)
        if field_name is None or value is None:
            continue
        total += float(value) * getattr(rules, field_name)
    return total
