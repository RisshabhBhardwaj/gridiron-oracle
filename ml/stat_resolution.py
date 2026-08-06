"""
Canonical stat-name resolution across GameLog, FeatureMatrix, and model targets.

Four of seven QB stats previously produced zero evaluation rows because
VALID_POSITION_STATS used model names while GameLog used source names
(e.g. pass_attempts vs attempts). This module makes resolution explicit and
fails loudly when a configured (position, stat) cannot be resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Model / evaluation stat name → GameLog attribute (or multi-source sentinel).
STAT_COLUMN_MAP: dict[str, str] = {
    # Passing
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "pass_attempts": "attempts",
    "completions": "completions",
    "interceptions": "passing_interceptions",
    # Rushing
    "rushing_yards": "rushing_yards",
    "rushing_tds": "rushing_tds",
    "carries": "carries",
    # Receiving
    "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_tds",
    "receptions": "receptions",
    "targets": "targets",
    # Fantasy + rare events
    "fantasy_ppr": "fantasy_points_ppr",
    "fumbles": "_fumbles_combined",
}

FUMBLE_COMPONENT_COLS: tuple[str, ...] = (
    "rushing_fumbles",
    "receiving_fumbles",
    "sack_fumbles",
)

# FeatureMatrix Kalman estimate column for each model stat.
KALMAN_EST_MAP: dict[str, str] = {
    stat: f"kalman_est_{stat}" for stat in STAT_COLUMN_MAP
}

# Headline evaluation matrix — fantasy_ppr is required for all skill positions.
VALID_POSITION_STATS: dict[str, list[str]] = {
    "QB": [
        "fantasy_ppr",
        "passing_yards",
        "pass_attempts",
        "completions",
        "passing_tds",
        "interceptions",
        "rushing_yards",
        "fumbles",
    ],
    "RB": [
        "fantasy_ppr",
        "rushing_yards",
        "carries",
        "rushing_tds",
        "receiving_yards",
        "receptions",
        "receiving_tds",
        "targets",
        "fumbles",
    ],
    "WR": [
        "fantasy_ppr",
        "receiving_yards",
        "receptions",
        "receiving_tds",
        "targets",
        "rushing_yards",
        "fumbles",
    ],
    "TE": [
        "fantasy_ppr",
        "receiving_yards",
        "receptions",
        "receiving_tds",
        "targets",
        "fumbles",
    ],
}

YARDAGE_STATS = frozenset({"passing_yards", "rushing_yards", "receiving_yards"})
COUNT_STATS = frozenset(
    {
        "pass_attempts",
        "completions",
        "carries",
        "receptions",
        "targets",
        "passing_tds",
        "rushing_tds",
        "receiving_tds",
        "interceptions",
        "fumbles",
    }
)
FANTASY_STATS = frozenset({"fantasy_ppr"})


@dataclass(frozen=True)
class ResolvedStat:
    model_stat: str
    gamelog_attr: str
    kalman_attr: str
    is_multi_source: bool


def resolve_stat(stat: str) -> ResolvedStat:
    if stat not in STAT_COLUMN_MAP:
        raise KeyError(
            f"Unknown model stat {stat!r}. Known: {sorted(STAT_COLUMN_MAP)}"
        )
    gamelog_attr = STAT_COLUMN_MAP[stat]
    return ResolvedStat(
        model_stat=stat,
        gamelog_attr=gamelog_attr,
        kalman_attr=KALMAN_EST_MAP[stat],
        is_multi_source=gamelog_attr == "_fumbles_combined",
    )


def read_gamelog_stat(gl: Any, stat: str) -> Optional[float]:
    """Read a model stat from a GameLog ORM row / mapping."""
    resolved = resolve_stat(stat)
    if resolved.is_multi_source:
        total = 0.0
        seen = False
        for col in FUMBLE_COMPONENT_COLS:
            val = getattr(gl, col, None) if not isinstance(gl, dict) else gl.get(col)
            if val is not None:
                total += float(val)
                seen = True
        return total if seen else None
    val = (
        getattr(gl, resolved.gamelog_attr, None)
        if not isinstance(gl, dict)
        else gl.get(resolved.gamelog_attr)
    )
    return float(val) if val is not None else None


def assert_position_stats_resolvable(
    position_stats: dict[str, list[str]] | None = None,
) -> None:
    """
    Abort at startup if any configured (position, stat) cannot resolve.

    Also verifies FeatureMatrix Kalman attr names are declared for each stat.
    """
    matrix = position_stats or VALID_POSITION_STATS
    errors: list[str] = []
    for position, stats in matrix.items():
        for stat in stats:
            try:
                resolved = resolve_stat(stat)
            except KeyError as exc:
                errors.append(f"{position}/{stat}: {exc}")
                continue
            if not resolved.kalman_attr.startswith("kalman_est_"):
                errors.append(
                    f"{position}/{stat}: invalid kalman attr {resolved.kalman_attr!r}"
                )
    if errors:
        raise AssertionError(
            "Stat-name resolution failed for configured evaluation matrix:\n  - "
            + "\n  - ".join(errors)
        )


def target_metric_family(stat: str) -> str:
    """Return the metric family used for this target type."""
    if stat in YARDAGE_STATS or stat in FANTASY_STATS:
        return "continuous"  # MAE + CRPS / pinball
    if stat in COUNT_STATS:
        return "count"  # Poisson deviance + reliability
    return "continuous"
