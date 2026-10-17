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
    "avg_cushion", "injury_status_encoded", "temp_f",
    "wind_mph", "temp_bucket", "wind_bucket", "wind_x_qb", "wind_x_wr",
    "precip_x_pass", "years_exp", "exp_bucket",
    "snap_share_trailing", "snap_share_trend", "snap_vs_pos_avg",
    # Contemporaneous expected-fantasy (xFP) is same-week realized opportunity.
    # Lagged xFP may enter later as a role-quality feature; same-week *_exp must not.
    "rec_fantasy_points_exp", "rush_fantasy_points_exp", "pass_fantasy_points_exp",
    "fantasy_points_exp", "receptions_exp", "targets_exp", "carries_exp",
    "pass_attempts_exp", "rushing_yards_exp", "receiving_yards_exp",
    "passing_yards_exp", "completions_exp",
    *(f"player_emb_{i}" for i in range(32)),
})


def _spec_for(name: str) -> FeatureSpec:
    """Return the explicit legal source class for a registered feature name.

    Fail-closed: a name with no explicit entry raises instead of silently
    joining PRIOR_GAMES. A new feature must earn its classification here
    before any model frame may carry it.
    """
    if name in FORBIDDEN_MODEL_FIELDS:
        return FeatureSpec(name, AsOf.ILLEGAL, "removed or disabled pending a causal source")
    if name in _SEASON_PROFILE_FEATURES:
        return FeatureSpec(name, AsOf.SEASON_PROFILE, "player_season_profiles.effective_season")
    if name in _PREGAME_FEATURES:
        return FeatureSpec(name, AsOf.PREGAME, "schedule, venue, market, or immutable draft fact")
    if name in _PRIOR_GAMES_FEATURES:
        return FeatureSpec(name, AsOf.PRIOR_GAMES, "completed games strictly before target kickoff")
    raise KeyError(
        f"Feature {name!r} has no as-of classification. Add it to "
        "ml/feature_contract.py (or to FORBIDDEN_MODEL_FIELDS) before use."
    )


#: Physical / career snapshot attributes, fixed for the season.
_SEASON_PROFILE_FEATURES = frozenset({
    "height", "weight",
    # Career snapshot from the season roster context, not a game outcome.
    "age", "career_games",
})

#: Pregame facts: schedule, venue, market lines, and immutable draft facts.
#: Pace/script expectations belong here (not PRIOR_GAMES): they mix prior
#: pace with the target game's spread/total, so the line's pregame
#: provenance governs them.
_PREGAME_FEATURES = frozenset({
    "game_total_line", "spread_line", "is_home", "days_rest", "is_short_week",
    "is_bye_prior", "is_dome", "surface_turf", "rule_coeff", "draft_round",
    "expected_pass_attempts", "expected_pass_rate", "neutral_script_flag",
})

#: Everything else legal today, each derived from completed games strictly
#: before the target kickoff: Kalman estimates, season-baseline averages,
#: opponent-allowed averages, team pace from prior weeks, rest/injury state
#: from prior rows, lagged PBP/NGS replacements, Elo pregame snapshots, and
#: the Phase 4 experiment-group columns (still gated off FEATURE_COLS).
_PRIOR_GAMES_FEATURES = frozenset({
    "depth_chart_rank",  # lagged depth_charts; week N uses chart week < N
    "games_missed_streak", "prior_snap_share",
    "team_pace", "team_pass_rate",
    "target_share_trend", "team_pos_rank",
    "team_off_elo", "team_def_elo", "opp_off_elo", "opp_def_elo",
    "elo_matchup_diff", "elo_implied_win_prob",
    "deep_matchup_score", "coverage_matchup_score", "separation_demand_score",
    "kalman_est_receiving_yards", "kalman_est_receiving_tds", "kalman_est_targets",
    "kalman_est_receptions", "kalman_est_target_share", "kalman_est_red_zone_target_share",
    "kalman_est_air_yards_share",
    "kalman_est_fantasy_ppr", "kalman_est_carries", "kalman_est_rushing_yards",
    "kalman_est_rushing_tds", "kalman_est_pass_attempts", "kalman_est_completions",
    "kalman_est_interceptions", "kalman_est_fumbles", "kalman_est_passing_yards",
    "kalman_est_passing_tds",
    "seas_games_played", "seas_avg_receiving_yards", "seas_avg_targets",
    "seas_avg_receptions", "seas_avg_target_share", "seas_avg_fantasy_ppr",
    "seas_yards_per_target", "seas_yards_per_reception",
    "seas_avg_carries", "seas_avg_rushing_yards", "seas_yards_per_carry",
    "seas_avg_attempts", "seas_avg_passing_yards", "seas_completion_pct",
    "seas_avg_passing_cpoe", "seas_avg_passing_epa",
    "seas_avg_receiving_epa", "seas_avg_rushing_epa",
    "seas_avg_racr", "seas_avg_wopr", "seas_avg_receiving_yac",
    "opp_avg_receiving_yards_allowed", "opp_avg_targets_allowed",
    "opp_avg_tds_allowed", "opp_avg_fantasy_ppr_allowed",
    "opp_avg_rushing_yards_allowed",
    "opp_zone_pct", "opp_man_pct", "opp_blitz_rate", "opp_pressure_rate",
    # Phase 4 experiment groups (enabled only via resolve_feature_cols).
    "carry_share", "ts_vs_league", "rz_ts_vs_league", "carry_share_vs_league",
    "opp_adj_target_share",
    "prior_epa_per_play", "prior_epa_per_target", "prior_epa_per_rush",
    "prior_qb_epa_per_dropback", "prior_adot", "prior_drop_rate",
    "prior_avg_separation", "prior_avg_cushion", "prior_target_share_pbp",
    "prior_air_yards_share_pbp", "prior_red_zone_targets",
    "prior_end_zone_targets", "prior_red_zone_target_share",
    "prior_pass_left_rate", "prior_pass_middle_rate", "prior_pass_right_rate",
    "prior_ol_pressure_rate", "prior_ol_sack_rate", "prior_yac_per_reception",
    "prior_xyac_per_reception", "routes_run_per_game",
})


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
