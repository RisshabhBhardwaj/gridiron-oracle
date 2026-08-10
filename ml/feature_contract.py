"""Executable as-of contract for every feature allowed into a model frame.

The contract is deliberately stricter than a column naming convention: callers
must validate their input allowlist before training, evaluating, or serving.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import pandas as pd


class AsOf(str, Enum):
    """Latest legal source time relative to the prediction game's kickoff."""

    PRIOR_GAMES = "prior_completed_player_or_team_games"
    SEASON_PROFILE = "season_scoped_roster_snapshot"
    PREGAME = "pregame_schedule_or_market_snapshot"
    ILLEGAL = "target_game_or_unproven_source"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    as_of: AsOf
    source: str


# These names may exist in storage for historical/rebuild compatibility, but
# are never legal model inputs.  Aliases and descendants are intentionally
# listed here as well; removing one entry from FEATURE_COLS is not enough.
FORBIDDEN_MODEL_FIELDS = frozenset({
    "snap_pct_off", "offense_pct", "routes_run_pct", "blitz_exposure",
    "epa_per_play", "epa_per_target", "epa_per_rush", "qb_epa_per_dropback",
    "adot", "yac_per_reception", "xyac_per_reception", "target_share_pbp",
    "air_yards_share_pbp", "red_zone_targets", "end_zone_targets",
    "red_zone_target_share", "drop_rate", "ol_pressure_rate", "ol_sack_rate",
    "pass_left_rate", "pass_middle_rate", "pass_right_rate", "avg_separation",
    "avg_cushion", "depth_chart_rank", "injury_status_encoded", "temp_f",
    "wind_mph", "temp_bucket", "wind_bucket", "wind_x_qb", "wind_x_wr",
    "precip_x_pass", "years_exp", "exp_bucket",
    "snap_share_trailing", "snap_share_trend", "snap_vs_pos_avg",
    *(f"player_emb_{i}" for i in range(32)),
})


def _spec_for(name: str) -> FeatureSpec:
    """Return the explicit legal source class for a registered feature name."""
    if name in FORBIDDEN_MODEL_FIELDS:
        return FeatureSpec(name, AsOf.ILLEGAL, "removed or disabled pending a causal source")
    if name in {"height", "weight"}:
        return FeatureSpec(name, AsOf.SEASON_PROFILE, "player_season_profiles.effective_season")
    if name in {"game_total_line", "spread_line", "is_home", "days_rest", "is_short_week", "is_bye_prior", "is_dome", "surface_turf", "rule_coeff", "draft_round"}:
        return FeatureSpec(name, AsOf.PREGAME, "schedule, venue, or immutable draft fact")
    return FeatureSpec(name, AsOf.PRIOR_GAMES, "completed games strictly before target kickoff")


def build_feature_registry(feature_names: Iterable[str]) -> dict[str, FeatureSpec]:
    """Create the concrete registry and reject anything not explicitly legal."""
    registry = {name: _spec_for(name) for name in feature_names}
    illegal = [name for name, spec in registry.items() if spec.as_of is AsOf.ILLEGAL]
    if illegal:
        raise AssertionError(f"Illegal features registered for model use: {sorted(illegal)}")
    return registry


def assert_model_input_columns(columns: Iterable[str], *, consumer: str) -> None:
    """Fail closed if a target-game snap path reaches a model adapter."""
    bad = sorted(set(columns).intersection(FORBIDDEN_MODEL_FIELDS))
    if bad:
        raise AssertionError(f"{consumer} received prohibited feature input(s): {bad}")


def assert_model_frame_contract(
    frame: pd.DataFrame, columns: Iterable[str], *, consumer: str
) -> None:
    """Validate both the model allowlist and that selected columns exist."""
    selected = list(columns)
    assert_model_input_columns(selected, consumer=consumer)
    missing = sorted(set(selected).difference(frame.columns))
    if missing:
        raise AssertionError(f"{consumer} model frame is missing registered features: {missing}")


def assert_no_same_game_postgame_equality(
    feature_frame: pd.DataFrame,
    postgame_frame: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
    join_columns: tuple[str, ...] = ("player_id", "game_id"),
) -> None:
    """Catch exact copies of target-game postgame data (the C-01 invariant).

    The postgame frame should be loaded from ``game_logs`` using the same game
    keys.  Equality is tested only on non-null pairs so sparse legal features
    do not create false positives.
    """
    missing_keys = set(join_columns).difference(feature_frame.columns) | set(join_columns).difference(postgame_frame.columns)
    if missing_keys:
        raise KeyError(f"same-game audit requires join columns: {sorted(missing_keys)}")
    candidates = [c for c in feature_columns if c in feature_frame.columns and c in postgame_frame.columns]
    if not candidates:
        return
    merged = feature_frame[list(join_columns) + candidates].merge(
        postgame_frame[list(join_columns) + candidates], on=list(join_columns), suffixes=("__feature", "__postgame"), how="inner"
    )
    equal = []
    for col in candidates:
        left = pd.to_numeric(merged[f"{col}__feature"], errors="coerce")
        right = pd.to_numeric(merged[f"{col}__postgame"], errors="coerce")
        comparable = left.notna() & right.notna()
        if comparable.any() and left[comparable].eq(right[comparable]).all():
            equal.append(col)
    if equal:
        raise AssertionError(f"Feature(s) exactly equal same-game postgame data: {sorted(equal)}")
