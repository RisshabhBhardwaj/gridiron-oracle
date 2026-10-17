"""
ml/bucket_weights.py

Settings bucket weights, as a re-weighting of stored per-bucket attributions.

A served weekly projection is a Ridge blend over native CatBoost and LightGBM
learners. The blend is SHA-pinned, so a user's weights never change it. What a
weight can change is how much each group of input features moved a stored
number. The materialization step stores, per row, the attribution of the
served value to each bucket (see ml/bucket_attribution.py). Serving then
rescales those attributions:

    r_g      = clamp(w_g / d_g, 0, MAX_BUCKET_RATIO)     (d_g = default weight)
    adjusted = stored + sum_g (r_g - 1) * phi_g

At the defaults every r_g is exactly 1, so the stored value is returned
unchanged. The adjusted number is a counterfactual re-weighting of attributions.
It is not a retrained model and has no backtest, so every response that carries
it says so (weighting = "user").

Keys match the Settings API. "roster_injury" and "weather_venue" are legacy ids
kept for stored files. Their labels are "Rest & Availability" and "Venue"
because the features behind them are rest, availability history, and venue
facts. No injury status is a model input (ml/feature_contract.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

#: Settings keys, in display order. Each is one bucket of FEATURE_COLS.
BUCKET_KEYS: tuple[str, ...] = (
    "kalman_form",
    "seasonal_baseline",
    "matchup",
    "weather_venue",
    "team_context",
    "roster_injury",
    "rule_meta",
)

#: Defaults, in percent. They sum to exactly 100.
DEFAULT_BUCKET_WEIGHTS: dict[str, float] = {
    "kalman_form": 35.0,
    "seasonal_baseline": 20.0,
    "matchup": 20.0,
    "weather_venue": 10.0,
    "team_context": 5.0,
    "roster_injury": 5.0,
    "rule_meta": 5.0,
}

#: No group can be scaled above this multiple of its default. Ratio scaling
#: makes a small group very sensitive: a 5-point group set to 100 would be
#: scaled 20x. Capping keeps every adjustment inside a stated band.
MAX_BUCKET_RATIO: float = 2.0

#: Weights are stored and validated in whole tenths of a percent.
WEIGHT_STEP: float = 0.1

#: Key for the 18 FEATURE_COLS that no Settings group owns. Their attributions
#: are never rescaled. They exist only so the stored attributions can be
#: checked to sum to the served value.
UNWEIGHTED_KEY: str = "unweighted"

#: Member columns of each bucket. Every FEATURE_COLS entry belongs to exactly
#: one group, enforced by validate_bucket_map and by a test.
BUCKET_MEMBERS: dict[str, tuple[str, ...]] = {
    # Bucket 1: Kalman form (ml/utils.py FEATURE_COLS, "Bucket 1")
    "kalman_form": (
        "kalman_est_receiving_yards", "kalman_est_receiving_tds", "kalman_est_targets",
        "kalman_est_receptions", "kalman_est_target_share", "kalman_est_red_zone_target_share",
        "kalman_est_air_yards_share", "kalman_est_fantasy_ppr", "kalman_est_carries",
        "kalman_est_rushing_yards", "kalman_est_rushing_tds", "kalman_est_pass_attempts",
        "kalman_est_completions", "kalman_est_interceptions", "kalman_est_fumbles",
        "kalman_est_passing_yards", "kalman_est_passing_tds",
    ),
    # Bucket 2: season baseline
    "seasonal_baseline": (
        "seas_games_played", "seas_avg_receiving_yards", "seas_avg_targets",
        "seas_avg_receptions", "seas_avg_target_share", "seas_avg_fantasy_ppr",
        "seas_yards_per_target", "seas_yards_per_reception", "seas_avg_carries",
        "seas_avg_rushing_yards", "seas_yards_per_carry", "seas_avg_attempts",
        "seas_avg_passing_yards", "seas_completion_pct", "seas_avg_passing_cpoe",
        "seas_avg_passing_epa", "seas_avg_receiving_epa", "seas_avg_rushing_epa",
        "seas_avg_racr", "seas_avg_wopr", "seas_avg_receiving_yac",
    ),
    # Bucket 3: opponent (matchup)
    "matchup": (
        "opp_avg_receiving_yards_allowed", "opp_avg_targets_allowed",
        "opp_avg_tds_allowed", "opp_avg_fantasy_ppr_allowed",
        "opp_avg_rushing_yards_allowed",
    ),
    # Bucket 4: venue facts only. Temperature and wind are disabled.
    "weather_venue": ("is_dome", "surface_turf"),
    # Bucket 5: team context
    "team_context": ("game_total_line", "spread_line", "is_home"),
    # Bucket 6 (rest) and bucket 8 (availability history). No injury status.
    "roster_injury": (
        "days_rest", "is_short_week", "is_bye_prior",
        "games_missed_streak", "prior_snap_share",
    ),
    # Bucket 7: rule coefficients
    "rule_meta": ("rule_coeff",),
}

#: FEATURE_COLS that no Settings group owns: defensive tendency and scheme
#: (buckets 9), Elo (10), physical profile, trend, and depth.
UNWEIGHTED_MEMBERS: tuple[str, ...] = (
    "opp_zone_pct", "opp_man_pct", "opp_blitz_rate", "opp_pressure_rate",
    "deep_matchup_score", "coverage_matchup_score", "separation_demand_score",
    "team_off_elo", "team_def_elo", "opp_off_elo", "opp_def_elo",
    "elo_matchup_diff", "elo_implied_win_prob",
    "height", "weight", "draft_round",
    "target_share_trend",
    "team_pos_rank",
)

_COLUMN_TO_KEY: dict[str, str] = {
    column: key for key, members in BUCKET_MEMBERS.items() for column in members
}
_COLUMN_TO_KEY.update({column: UNWEIGHTED_KEY for column in UNWEIGHTED_MEMBERS})


def bucket_of(column: str) -> str:
    """Return the bucket key (a Settings key, or UNWEIGHTED_KEY) that owns a feature column.

    Raises KeyError for a column that no group claims, so a new feature cannot
    be added without a decision about which bucket it belongs to.
    """
    return _COLUMN_TO_KEY[column]


def validate_bucket_map(feature_cols: Sequence[str]) -> None:
    """Check that the groups partition feature_cols exactly.

    Every column must appear in exactly one group, and no group may name a
    column that is not in feature_cols. Raises ValueError with the offending
    names. Tests run this against ml.utils.FEATURE_COLS.
    """
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for key, members in list(BUCKET_MEMBERS.items()) + [(UNWEIGHTED_KEY, UNWEIGHTED_MEMBERS)]:
        for column in members:
            if column in seen:
                duplicates.append(f"{column} ({seen[column]}, {key})")
            seen[column] = key
    wanted = set(feature_cols)
    unmapped = sorted(c for c in feature_cols if c not in seen)
    stale = sorted(c for c in seen if c not in wanted)
    problems = []
    if duplicates:
        problems.append(f"in more than one group: {duplicates}")
    if unmapped:
        problems.append(f"feature columns with no group: {unmapped}")
    if stale:
        problems.append(f"group members that are not feature columns: {stale}")
    if problems:
        raise ValueError("bucket map is not a partition of FEATURE_COLS; " + "; ".join(problems))


def weights_are_default(weights: Mapping[str, float]) -> bool:
    """True when every bucket weight equals its default (within 1e-9)."""
    return all(
        math.isclose(float(weights[key]), DEFAULT_BUCKET_WEIGHTS[key], rel_tol=0.0, abs_tol=1e-9)
        for key in BUCKET_KEYS
    )


def bucket_ratio(weight: float, default: float) -> float:
    """The scaling applied to one group, clamped to [0, MAX_BUCKET_RATIO]."""
    return min(max(float(weight) / float(default), 0.0), MAX_BUCKET_RATIO)


@dataclass(frozen=True)
class WeightedValue:
    """One served number after the bucket-weight layer.

    adjusted:  the value to show as adjusted. Equals the stored value whenever
               weighting is "default".
    weighting: "user" only when the weights moved this number. Otherwise "default".
    reason:    why the number is unchanged ("defaults", "no_attribution",
               "unchanged") or None when weighting is "user".
    clipped:   True when a non-negative stat was floored at zero.
    """

    adjusted: float
    weighting: str
    reason: Optional[str]
    clipped: bool = False


def apply_bucket_weights(
    stored: float,
    attributions: Optional[Mapping[str, Optional[float]]],
    weights: Mapping[str, float],
    *,
    non_negative: bool = True,
) -> WeightedValue:
    """Apply user bucket weights to one stored value.

    stored:       the canonical served value. It is never modified.
    attributions: per-bucket attribution of that value, keyed by BUCKET_KEYS. None
                  when the row has no stored attributions.
    weights:      user weights in percent, keyed by BUCKET_KEYS.
    non_negative: floor the result at zero (all served stats are counts or yards).
    """
    if weights_are_default(weights):
        return WeightedValue(adjusted=stored, weighting="default", reason="defaults")
    if attributions is None or any(
        attributions.get(key) is None or not math.isfinite(float(attributions[key]))
        for key in BUCKET_KEYS
    ):
        return WeightedValue(adjusted=stored, weighting="default", reason="no_attribution")

    delta = 0.0
    for key in BUCKET_KEYS:
        ratio = bucket_ratio(weights[key], DEFAULT_BUCKET_WEIGHTS[key])
        delta += (ratio - 1.0) * float(attributions[key])
    adjusted = stored + delta
    clipped = False
    if non_negative and adjusted < 0.0:
        adjusted = 0.0
        clipped = True
    if adjusted == stored:
        return WeightedValue(adjusted=stored, weighting="default", reason="unchanged", clipped=clipped)
    return WeightedValue(adjusted=adjusted, weighting="user", reason=None, clipped=clipped)
