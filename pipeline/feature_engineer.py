"""
pipeline/feature_engineer.py

Builds the feature matrix that every ML model trains on.
Reads from production tables (game_logs + games) or, in --dry-run mode,
directly from nflreadpy without any database connection.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEATURE BUCKETS (outline §5.1) — DEFAULT WEIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bucket 1  35%  Kalman Player Form       kalman_est_* / kalman_variance_*
                 Scalar Kalman filter per (player, stat): x_k=x_{k-1}+w_k, y_k=x_k+v_k
                 Q=1.0 (ability drift), R=empirical variance, x0=position-average prior

Bucket 2  20%  Season Baseline          seas_*   cumulative season averages
                 Covers: avg yards, targets, efficiency rates, games played

Bucket 3  20%  Matchup Metrics          opp_*    opponent defensive stats
                 Per position (WR/RB/TE/QB), yards/targets/TDs allowed/game

Bucket 4  10%  Weather & Venue          temp/wind/dome/surface
                 Temp bucket: 0=cold(<32F) 1=cool(32-50F) 2=mild(50F+)
                 Wind bucket: 0=calm(<10) 1=breezy(10-20) 2=windy(20+mph)

Bucket 5   5%  Team Context             game_total_line, spread_line, is_home

Bucket 6   5%  Rest                     days_rest, is_short_week, is_bye_prior
                 Short week: days_rest < 6 (Thursday Night Football)
                 Bye prior:  days_rest >= 12 (coming off bye week)

Bucket 7   5%  Rule Coefficients        rule_coeff
                 rule_coeff encodes significant NFL rule changes that shift offensive volume:
                 season < 2023 → 0.0 (baseline), 2023 → 0.5 (new kickoff rules, partial season),
                 season ≥ 2024 → 1.0 (revised kickoff formation rule, full effect)

Target columns (actual_*) are populated for past games only.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Standalone usage:
  # Dry-run acceptance test (prints feature vector, no DB required):
  python pipeline/feature_engineer.py --dry-run \\
      --player "Justin Jefferson" --season 2024 --week 12

  # Full DB run (requires production tables populated by normalize.py):
  python pipeline/feature_engineer.py --seasons 2025
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from dataclasses import fields as dc_fields
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values

from scraper.adapters.nflreadpy_adapter import _coerce_row, _psycopg2_dsn
from ml.kalman_tracker import KalmanFeatureEngineer
from ml.utils import MIN_SNAP_PCT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from pipeline.features.feature_row import FeatureRow
from pipeline.features.buckets import (
    SeasonFeatureContext,
    build_season_feature_context,
    compute_season_baseline,
    compute_matchup_stats,
    compute_pos_rank,
    compute_venue_features,
    compute_rest_features,
    compute_team_context,
    compute_injury_features,
    compute_rule_features,
    compute_scheme_interactions,
    compute_usage_shares,
    compute_opp_adj_usage,
    compute_pace_script,
    compute_progression_priors,
    _safe_div as _safe_div,  # re-exported for legacy tests/importers
    _safe_float,
    _sum_fumbles,
)

# ── Core assembly function ────────────────────────────────────────────────────

def build_feature_row(
    target_row: dict,
    prior_rows: list[dict],
    game: dict,
    all_season_rows: list[dict],
    injury_df: Optional[Any] = None,
    season_ctx: Optional[SeasonFeatureContext] = None,
) -> FeatureRow:
    """
    Pure function — assembles all buckets into a FeatureRow.

    Args:
        target_row:       The player-game we are computing features FOR.
                          Must have: player_id, game_id, season, week, team,
                          opponent_team, position.
        prior_rows:       All completed game rows for this player in this season,
                          with week < target_row["week"], sorted oldest-first.
        game:             The ScheduleRow dict for this game (venue/weather/odds).
        all_season_rows:  All PlayerStatsRow dicts for this season
                          (used to compute opponent defensive stats).

    Returns:
        FeatureRow with all buckets populated (None where data unavailable).
    """
    player_id     = target_row["player_id"]
    game_id       = target_row["game_id"]
    season        = target_row["season"]
    week          = target_row["week"]
    position      = target_row.get("position")
    team          = target_row.get("team")
    opponent_team = target_row.get("opponent_team")

    is_home_flag  = (team == game.get("home_team")) if team else None
    is_home_int   = int(is_home_flag) if is_home_flag is not None else None

    # Filter prior_rows by snap participation for Kalman + season baseline.
    # Bench appearances (offense_pct < MIN_SNAP_PCT) produce high-variance noise
    # that drags Kalman estimates toward zero — excluded for form/baseline only.
    # Rows where offense_pct is None (seasons without snap counts: 2019-2021,2024)
    # are kept so that we do not drop entire seasons of history.
    snap_prior_rows = [
        r for r in prior_rows
        if r.get("offense_pct") is None or (r.get("offense_pct") or 0.0) >= MIN_SNAP_PCT
    ]
    form    = KalmanFeatureEngineer().compute_kalman_form(snap_prior_rows, position=position)
    seas    = compute_season_baseline(snap_prior_rows)
    matchup = compute_matchup_stats(
        opponent_team or "", position, all_season_rows, week
    )
    venue   = compute_venue_features(game)
    rest    = compute_rest_features(game, bool(is_home_flag))
    ctx     = compute_team_context(game, team or "")
    injury  = compute_injury_features(
        player_id, week, season, prior_rows, injury_df=injury_df
    )
    pos_rank = compute_pos_rank(
        player_id, team, position, week, all_season_rows
    )

    # Combine is_home from team context (more reliable)
    is_home_int = ctx.pop("is_home", is_home_int)

    # Snap participation (from target_row.offense_pct, set by snap_counts ETL pass)
    snap_pct = target_row.get("offense_pct")

    # Bucket 9: scheme interactions — extend matchup dict with opp tendency fields
    # (opp_zone_pct, opp_man_pct, opp_blitz_rate, opp_pressure_rate are populated
    # by compute_matchup_stats once play-by-play coverage data is wired in Phase 4;
    # until then they are None and the interaction terms also evaluate to None)
    scheme = compute_scheme_interactions(form, matchup, snap_pct)

    # ── TFT Static Covariates: physical profile (Item 0 fix) ─────────────────
    # nflreadpy game_logs carries player_height (inches), player_weight (lbs),
    # and draft_number / draft_round directly on every player-game row.
    # These were previously hardcoded to 0.0 in TFTDataset._derive_columns().
    def _coerce_float(val: Any) -> Optional[float]:
        """Safe float coercion; returns None on None, NaN, or non-numeric."""
        try:
            v = float(val)
            return v if v == v else None  # NaN check
        except (TypeError, ValueError):
            return None

    height_val      = _coerce_float(
        target_row.get("player_height") or target_row.get("height")
    )
    weight_val      = _coerce_float(
        target_row.get("player_weight") or target_row.get("weight")
    )
    draft_round_val = _coerce_float(
        target_row.get("draft_round") or target_row.get("draft_number")
    )

    # ── Target share trend (velocity over last 3 games) ─────────────────────
    target_share_trend: Optional[float] = None
    if len(prior_rows) >= 3:
        ts_vals = [float(r.get("target_share") or 0.0) for r in prior_rows[-3:]]
        target_share_trend = ts_vals[-1] - ts_vals[0]
    elif len(prior_rows) >= 2:
        ts_vals = [float(r.get("target_share") or 0.0) for r in prior_rows[-2:]]
        target_share_trend = ts_vals[-1] - ts_vals[0]

    # ── Weather × position interaction terms ─────────────────────────────────
    wind_bkt = venue.get("wind_bucket")
    precip_bkt = game.get("precipitation_bucket")
    is_qb_pos = 1 if (position or "").upper() == "QB" else 0
    is_pass_pos = 1 if (position or "").upper() in ("QB", "WR", "TE") else 0
    wind_x_qb: Optional[float] = (float(wind_bkt) * is_qb_pos) if wind_bkt is not None else None
    wind_x_wr: Optional[float] = (float(wind_bkt) * is_pass_pos) if wind_bkt is not None else None
    precip_x_pass: Optional[float] = (float(precip_bkt) * is_pass_pos) if precip_bkt is not None else None

    # ── routes_run_pct from snap count data ──────────────────────────────────
    routes_run_pct = _coerce_float(
        target_row.get("offense_pct") or target_row.get("routes_run_pct")
    )

    # ── Phase 4 feature groups (causal; prior weeks only) ────────────────────
    usage = compute_usage_shares(
        prior_rows, all_season_rows, player_id, team or "", week, position, ctx=season_ctx
    )
    opp_adj = compute_opp_adj_usage(
        seas, matchup, usage, all_season_rows, week, position, ctx=season_ctx
    )
    pace = compute_pace_script(
        team or "", week, game, all_season_rows, is_home_int, ctx=season_ctx
    )
    progression = compute_progression_priors(target_row, prior_rows, season)
    # Prefer joined players.draft_round when present; else game_log fields
    if progression.get("draft_round_progression") is not None and draft_round_val is None:
        draft_round_val = progression["draft_round_progression"]

    return FeatureRow(
        player_id=player_id,
        game_id=game_id,
        season=season,
        week=week,
        position=position,
        team=team,
        opponent_team=opponent_team,
        is_home=is_home_int,
        # Bucket 1
        **form,
        # Bucket 2
        **seas,
        # Bucket 3
        **matchup,
        # Bucket 4
        **venue,
        # Bucket 5
        **ctx,
        # Bucket 6
        **rest,
        # Bucket 7 — Rule Coefficients
        **compute_rule_features(season),
        # Bucket 8 — Injury / Availability
        **injury,
        # Snap participation (populated by snap_counts normalize pass)
        snap_pct_off=snap_pct,
        # Bucket 9 — Defensive Tendency + Scheme Interactions
        **scheme,
        # Bucket 10 — Elo Ratings (populated lazily if elo system is fitted)
        # NOTE: elo fields default to None here; FeatureEngineer.run() calls
        # _enrich_elo_features() after batch build to populate these in bulk.
        team_off_elo=None,
        team_def_elo=None,
        opp_off_elo=None,
        opp_def_elo=None,
        elo_matchup_diff=None,
        elo_implied_win_prob=None,
        # TFT Static Covariates — physical profile (Item 0 fix)
        height=height_val,
        weight=weight_val,
        draft_round=draft_round_val,
        # Bucket 11 — PBP-derived features (populated from pbp_features DB table;
        # all default to None here — _enrich_pbp_features() fills them in bulk
        # in FeatureEngineer.run() via LEFT JOIN on pbp_features).
        # Velocity/trend features (computed above from prior_rows)
        target_share_trend=target_share_trend,
        # Weather × position interactions (computed above)
        wind_x_qb=wind_x_qb,
        wind_x_wr=wind_x_wr,
        precip_x_pass=precip_x_pass,
        # Snap count proxy for routes run
        routes_run_pct=routes_run_pct,
        # Positional depth signal (e.g. 1.0 = WR1, 2.0 = WR2)
        team_pos_rank=pos_rank,
        # Phase 4 groups
        carry_share=opp_adj.get("carry_share"),
        snap_share_trailing=opp_adj.get("snap_share_trailing"),
        snap_share_trend=opp_adj.get("snap_share_trend"),
        ts_vs_league=opp_adj.get("ts_vs_league"),
        rz_ts_vs_league=opp_adj.get("rz_ts_vs_league"),
        snap_vs_pos_avg=opp_adj.get("snap_vs_pos_avg"),
        carry_share_vs_league=opp_adj.get("carry_share_vs_league"),
        opp_adj_target_share=opp_adj.get("opp_adj_target_share"),
        **pace,
        years_exp=progression.get("years_exp"),
        age=progression.get("age"),
        career_games=progression.get("career_games"),
        exp_bucket=progression.get("exp_bucket"),
        # player_emb_*, PBP cols: all None — populated in bulk by FeatureEngineer.run()
        # Targets — populate from target_row if the game is completed
        actual_fantasy_ppr=target_row.get("fantasy_points_ppr"),
        actual_receiving_yards=target_row.get("receiving_yards"),
        actual_rushing_yards=target_row.get("rushing_yards"),
        actual_passing_yards=target_row.get("passing_yards"),
        actual_pass_attempts=target_row.get("pass_attempts") or target_row.get("attempts"),
        actual_completions=target_row.get("completions"),
        actual_passing_tds=target_row.get("passing_tds"),
        actual_interceptions=target_row.get("interceptions") or target_row.get("passing_interceptions"),
        actual_carries=target_row.get("carries"),
        actual_rushing_tds=target_row.get("rushing_tds"),
        actual_receptions=target_row.get("receptions"),
        actual_receiving_tds=target_row.get("receiving_tds"),
        actual_targets=target_row.get("targets"),
        actual_fumbles=_safe_float(
            _sum_fumbles(target_row)
        ),
        actual_sacks_taken=None,
        actual_qb_hits_taken=None,
    )


# ── Dry-run helper (no DB required) ──────────────────────────────────────────

def build_from_nflreadpy(
    player_name: str,
    season: int,
    week: int,
) -> Optional[FeatureRow]:
    """
    Build a FeatureRow using nflreadpy directly — no DB connection required.
    Uses nflreadpy's built-in filesystem cache (24h TTL).

    Matches player by player_display_name (case-insensitive).
    Returns None if the player or game is not found.
    """
    try:
        import nflreadpy as nfl
    except ImportError:
        logger.error("nflreadpy not installed. Run: pip install nflreadpy")
        return None

    from pydantic import ValidationError
    from scraper.adapters.nflreadpy_adapter import PlayerStatsRow, ScheduleRow

    logger.info("Loading player_stats season=%d from nflreadpy…", season)
    stats_df  = nfl.load_player_stats(seasons=season)
    logger.info("Loading schedules season=%d from nflreadpy…", season)
    sched_df  = nfl.load_schedules(seasons=season)

    # Validate all player_stats rows
    all_rows: list[dict] = []
    for raw in stats_df.to_dicts():
        try:
            validated = PlayerStatsRow.model_validate(_coerce_row(raw))
            all_rows.append(validated.model_dump())
        except ValidationError:
            pass

    # Validate all schedule rows → map game_id → dict
    game_map: dict[str, dict] = {}
    for raw in sched_df.to_dicts():
        try:
            validated = ScheduleRow.model_validate(_coerce_row(raw))
            game_map[validated.game_id] = validated.model_dump()
        except ValidationError:
            pass

    # Build a (season, week, team) → game_id fallback lookup from schedules.
    # Needed because nflreadpy player_stats for some seasons (e.g. 2024) omits
    # the game_id field at the weekly summary level.
    team_week_to_game: dict[tuple, str] = {}
    for gid, g in game_map.items():
        key_home = (season, int(g.get("week", 0)) if g.get("week") else 0,
                    g.get("home_team", ""))
        key_away = (season, int(g.get("week", 0)) if g.get("week") else 0,
                    g.get("away_team", ""))
        team_week_to_game[key_home] = gid
        team_week_to_game[key_away] = gid

    # Filter to the target player (case-insensitive display name match)
    name_lower = player_name.lower()
    player_rows = [
        r for r in all_rows
        if (r.get("player_display_name") or "").lower() == name_lower
    ]

    if not player_rows:
        logger.warning("Player '%s' not found in %d stats data.", player_name, season)
        return None

    target_row = next(
        (r for r in player_rows if r["week"] == week and r["season"] == season),
        None,
    )
    if target_row is None:
        logger.warning(
            "No stats found for '%s' week=%d season=%d.", player_name, week, season
        )
        return None

    # Resolve game_id: use from stats if present, else derive from schedule
    game_id = target_row.get("game_id")
    if not game_id:
        team = target_row.get("team", "")
        game_id = team_week_to_game.get((season, week, team))
        if game_id:
            logger.info(
                "game_id not in player_stats; derived from schedules: %s", game_id
            )
            # Patch all rows for this player with the derived game_id so that
            # the target row passed to build_feature_row has a valid game_id.
            target_row = dict(target_row)   # make a copy before mutating
            target_row["game_id"] = game_id

    game = game_map.get(game_id or "")
    if game is None:
        logger.warning("Game '%s' not found in %d schedules.", game_id, season)
        return None

    prior_rows = sorted(
        [r for r in player_rows if int(r["week"]) < week],
        key=lambda r: int(r["week"]),
    )

    return build_feature_row(target_row, prior_rows, game, all_rows)


# ── DDL ───────────────────────────────────────────────────────────────────────

_CREATE_FEATURE_MATRIX = """
CREATE TABLE IF NOT EXISTS feature_matrix (
    id                              SERIAL   PRIMARY KEY,
    player_id                       TEXT     NOT NULL REFERENCES players(id),
    game_id                         TEXT     NOT NULL REFERENCES games(id),
    season                          INTEGER  NOT NULL,
    week                            INTEGER  NOT NULL,
    position                        TEXT,
    team                            TEXT,
    opponent_team                   TEXT,
    is_home                         SMALLINT,
    -- Bucket 1: Kalman Form (posterior mean + variance per stat)
    kalman_est_receiving_yards      FLOAT,
    kalman_variance_receiving_yards FLOAT,
    kalman_est_receiving_tds        FLOAT,
    kalman_variance_receiving_tds   FLOAT,
    kalman_est_targets              FLOAT,
    kalman_variance_targets         FLOAT,
    kalman_est_receptions           FLOAT,
    kalman_variance_receptions      FLOAT,
    kalman_est_target_share         FLOAT,
    kalman_variance_target_share    FLOAT,
    kalman_est_air_yards_share      FLOAT,
    kalman_variance_air_yards_share FLOAT,
    kalman_est_fantasy_ppr          FLOAT,
    kalman_variance_fantasy_ppr     FLOAT,
    kalman_est_carries              FLOAT,
    kalman_variance_carries         FLOAT,
    kalman_est_rushing_yards        FLOAT,
    kalman_variance_rushing_yards   FLOAT,
    kalman_est_rushing_tds          FLOAT,
    kalman_variance_rushing_tds     FLOAT,
    kalman_est_pass_attempts        FLOAT,
    kalman_variance_pass_attempts   FLOAT,
    kalman_est_completions          FLOAT,
    kalman_variance_completions     FLOAT,
    kalman_est_interceptions        FLOAT,
    kalman_variance_interceptions   FLOAT,
    kalman_est_fumbles              FLOAT,
    kalman_variance_fumbles         FLOAT,
    kalman_est_passing_yards        FLOAT,
    kalman_variance_passing_yards   FLOAT,
    kalman_est_passing_tds          FLOAT,
    kalman_variance_passing_tds     FLOAT,
    -- Bucket 2: Season Baseline
    seas_games_played               INTEGER,
    seas_avg_receiving_yards        FLOAT,
    seas_avg_targets                FLOAT,
    seas_avg_receptions             FLOAT,
    seas_avg_target_share           FLOAT,
    seas_avg_fantasy_ppr            FLOAT,
    seas_yards_per_target           FLOAT,
    seas_yards_per_reception        FLOAT,
    seas_avg_carries                FLOAT,
    seas_avg_rushing_yards          FLOAT,
    seas_yards_per_carry            FLOAT,
    seas_avg_attempts               FLOAT,
    seas_avg_passing_yards          FLOAT,
    seas_completion_pct             FLOAT,
    -- Bucket 3: Matchup
    opp_avg_receiving_yards_allowed FLOAT,
    opp_avg_targets_allowed         FLOAT,
    opp_avg_tds_allowed             FLOAT,
    opp_avg_fantasy_ppr_allowed     FLOAT,
    opp_avg_rushing_yards_allowed   FLOAT,
    -- Bucket 4: Venue
    temp_f                          FLOAT,
    wind_mph                        FLOAT,
    is_dome                         SMALLINT,
    surface_turf                    SMALLINT,
    temp_bucket                     SMALLINT,
    wind_bucket                     SMALLINT,
    -- Bucket 5: Team Context
    game_total_line                 FLOAT,
    spread_line                     FLOAT,
    -- Bucket 6: Rest
    days_rest                       INTEGER,
    is_short_week                   SMALLINT,
    is_bye_prior                    SMALLINT,
    -- Bucket 7: Rule Coefficients
    rule_coeff                      FLOAT,
    -- Bucket 8: Injury / Availability (from ESPN adapter; NULL = assumed healthy)
    injury_status_encoded           SMALLINT,
    games_missed_streak             SMALLINT,
    -- Snap participation (from nflreadpy snap_counts, via normalize.py)
    snap_pct_off                    FLOAT,
    -- Bucket 9: Defensive Tendency (from play-by-play; NULL until Phase 4 data wired)
    opp_zone_pct                    FLOAT,
    opp_man_pct                     FLOAT,
    opp_blitz_rate                  FLOAT,
    opp_pressure_rate               FLOAT,
    -- Bucket 9: Scheme Interaction Features (explicit cross-terms)
    deep_matchup_score              FLOAT,  -- air_yards_share × (1 - opp_zone_pct)
    coverage_matchup_score          FLOAT,  -- target_share × opp_man_pct
    blitz_exposure                  FLOAT,  -- snap_pct_off × opp_blitz_rate
    separation_demand_score         FLOAT,  -- target_share × opp_man_pct × (1 - opp_zone_pct)
    -- Bucket 10: Elo Ratings (from ml/team_elo.py; NULL until elo system is fitted)
    team_off_elo                    FLOAT,  -- team's offensive Elo heading into this week
    team_def_elo                    FLOAT,  -- team's defensive Elo heading into this week
    opp_off_elo                     FLOAT,  -- opponent's offensive Elo
    opp_def_elo                     FLOAT,  -- opponent's defensive Elo
    elo_matchup_diff                FLOAT,  -- team_off_elo - opp_def_elo (+ve = team offense advantage)
    elo_implied_win_prob            FLOAT,  -- elo-implied win probability for team
    -- TFT Static Covariates: player physical profile (from game_logs / nflreadpy)
    -- These were previously hardcoded to 0.0 in TFTDataset._derive_columns() (Item 0 fix)
    height                          FLOAT,  -- player height in inches (e.g. 74.0 = 6'2")
    weight                          FLOAT,  -- player weight in pounds
    draft_round                     FLOAT,  -- NFL draft round (1-7; NULL = undrafted)
    -- Targets
    actual_pass_attempts            FLOAT,
    actual_completions              FLOAT,
    actual_passing_tds              FLOAT,
    actual_interceptions            FLOAT,
    actual_carries                  FLOAT,
    actual_rushing_tds              FLOAT,
    actual_receptions               FLOAT,
    actual_receiving_tds            FLOAT,
    actual_targets                  FLOAT,
    actual_fumbles                  FLOAT,
    actual_sacks_taken              FLOAT,
    actual_qb_hits_taken            FLOAT,
    actual_fantasy_ppr              FLOAT,
    actual_receiving_yards          FLOAT,
    actual_rushing_yards            FLOAT,
    actual_passing_yards            FLOAT,
    computed_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_feature_matrix_player_game UNIQUE (player_id, game_id)
);
CREATE INDEX IF NOT EXISTS idx_feature_matrix_lookup
    ON feature_matrix (season, week, position);
CREATE INDEX IF NOT EXISTS idx_feature_matrix_player_season
    ON feature_matrix (player_id, season);
"""

# Derived dynamically from FeatureRow field order so schema drift is impossible.
# Any field added to FeatureRow is automatically included here and in DB writes.
_FM_COLS: list[str] = [f.name for f in dc_fields(FeatureRow)]


# ── Parallel helper (module-level for joblib pickle-ability) ──────────────────

def _compute_player_features(
    pid: str,
    p_rows: list[dict],
    game_map: dict[str, dict],
    all_rows: list[dict],
    season_ctx: Optional[SeasonFeatureContext] = None,
) -> list[FeatureRow]:
    """Compute all FeatureRows for a single player. Called in parallel via joblib."""
    p_sorted = sorted(p_rows, key=lambda r: int(r["week"]))
    results: list[FeatureRow] = []
    for i, target_row in enumerate(p_sorted):
        prior = p_sorted[:i]
        game  = game_map.get(target_row["game_id"], {})
        fr    = build_feature_row(
            target_row, prior, game, all_rows, season_ctx=season_ctx
        )
        results.append(fr)
    return results


# ── FeatureEngineer class ─────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Reads from production tables (game_logs + games) and writes to
    the feature_matrix table.

    Usage:
        with FeatureEngineer(os.environ["DATABASE_URL"]) as fe:
            n = fe.run(seasons=[2025])
            print(f"Wrote {n} feature rows")
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self._conn: Optional[psycopg2.extensions.connection] = None

    def connect(self) -> None:
        dsn = _psycopg2_dsn(self._db_url)
        logger.info("FeatureEngineer: connecting…")
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        self._ensure_table()

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self) -> "FeatureEngineer":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._conn and not self._conn.closed:
            if exc_type:
                self._conn.rollback()
            self.close()

    def _ensure_table(self) -> None:
        assert self._conn
        from pipeline.schema import ensure_schema

        ensure_schema(self._conn)
        with self._conn.cursor() as cur:
            # Auto-migrate all numeric fields defined in FeatureRow until a
            # dedicated Alembic revision owns FeatureRow drift.
            for f in dc_fields(FeatureRow):
                if f.name in ("player_id", "game_id", "position", "team", "opponent_team", "season", "week"):
                    continue
                db_type = "FLOAT"
                type_str = str(f.type) if not isinstance(f.type, str) else f.type
                if "int" in type_str.lower() and "float" not in type_str.lower():
                    db_type = "INTEGER"
                cur.execute(f"ALTER TABLE feature_matrix ADD COLUMN IF NOT EXISTS {f.name} {db_type}")

            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_feature_matrix_player_season "
                "ON feature_matrix (player_id, season)"
            )
        self._conn.commit()

    def _fetch_season_rows(self, season: int) -> list[dict]:
        """Fetch all game_log rows for a season, joined with game context."""
        assert self._conn
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    gl.player_id, gl.game_id, gl.season, gl.week,
                    gl.position, gl.team, gl.opponent_team,
                    gl.receiving_yards, gl.receiving_tds, gl.targets,
                    gl.receptions, gl.target_share, gl.air_yards_share,
                    gl.fantasy_points_ppr, gl.carries, gl.rushing_yards,
                    gl.rushing_tds, gl.attempts, gl.passing_yards,
                    gl.passing_tds, gl.completions,
                    gl.passing_interceptions, gl.rushing_fumbles,
                    gl.receiving_fumbles, gl.sack_fumbles,
                    gl.passing_cpoe, gl.passing_epa,
                    gl.receiving_epa, gl.rushing_epa, gl.racr, gl.wopr,
                    gl.passing_air_yards, gl.receiving_yards_after_catch,
                    gl.offense_pct,
                    g.home_team, g.away_team, g.roof, g.surface,
                    g.temp, g.wind, g.total_line, g.spread_line,
                    g.home_rest, g.away_rest,
                    g.precipitation_bucket,
                    p.height AS player_height,
                    p.weight AS player_weight,
                    p.draft_round,
                    p.years_exp,
                    p.birth_date
                FROM game_logs gl
                JOIN games g ON gl.game_id = g.id
                LEFT JOIN players p ON gl.player_id = p.id
                WHERE gl.season = %s
                ORDER BY gl.player_id, gl.week
                """,
                (season,),
            )
            return [dict(r) for r in cur.fetchall()]

    def _upsert_feature_rows(self, feature_rows: list[FeatureRow]) -> int:
        """Bulk-upsert feature rows into feature_matrix. Returns row count."""
        if not feature_rows:
            return 0

        col_list = ", ".join(_FM_COLS)
        update_set = ", ".join(
            f"{c} = EXCLUDED.{c}"
            for c in _FM_COLS
            if c not in ("player_id", "game_id")
        )

        rows = [
            tuple(getattr(fr, c, None) for c in _FM_COLS)
            for fr in feature_rows
        ]

        assert self._conn
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO feature_matrix ({col_list}, computed_at)
                VALUES %s
                ON CONFLICT (player_id, game_id) DO UPDATE SET
                    {update_set},
                    computed_at = EXCLUDED.computed_at
                """,
                [r + (datetime.utcnow(),) for r in rows],
            )
        self._conn.commit()
        return len(rows)

    def run(self, seasons: list[int]) -> int:
        """Build feature rows for all player-games in the given seasons."""
        total = 0
        for season in seasons:
            logger.info("Building features for season=%d…", season)
            all_rows = self._fetch_season_rows(season)
            if not all_rows:
                logger.warning("No game_log rows found for season=%d", season)
                continue

            # Group by player for rolling/baseline features
            player_rows: dict[str, list[dict]] = defaultdict(list)
            for r in all_rows:
                player_rows[r["player_id"]].append(r)

            # Build game lookup for venue/context features
            {r["game_id"] for r in all_rows}
            game_map: dict[str, dict] = {}
            for r in all_rows:
                if r["game_id"] not in game_map:
                    game_map[r["game_id"]] = {
                        k: r[k] for k in (
                            "home_team", "away_team", "roof", "surface",
                            "temp", "wind", "total_line", "spread_line",
                            "home_rest", "away_rest", "precipitation_bucket",
                        )
                    }

            # Precompute Phase 4 league/team aggregates once per season (causal).
            season_ctx = build_season_feature_context(all_rows)
            logger.info("season=%d: Phase 4 context built.", season)

            # Compute features in parallel across players (threading backend avoids
            # pickling the large all_rows list; all_rows is read-only so thread-safe).
            per_player: list[list[FeatureRow]] = Parallel(
                n_jobs=-1, backend="threading", prefer="threads",
            )(
                delayed(_compute_player_features)(pid, p_rows, game_map, all_rows, season_ctx)
                for pid, p_rows in player_rows.items()
            )

            # Write to DB sequentially — psycopg2 connections are not thread-safe.
            feature_batch: list[FeatureRow] = []
            for player_frs in per_player:
                for fr in player_frs:
                    feature_batch.append(fr)
                    if len(feature_batch) >= 2000:
                        total += self._upsert_feature_rows(feature_batch)
                        feature_batch.clear()

            if feature_batch:
                total += self._upsert_feature_rows(feature_batch)

            # Enrich depth_chart_rank, avg_separation, avg_cushion from depth_charts + nextgen_stats
            self._enrich_depth_nextgen(season)

            logger.info("season=%d: %d feature rows written.", season, total)

        return total

    def _enrich_depth_nextgen(self, season: int) -> None:
        """Bulk UPDATE feature_matrix with depth_chart_rank and nextgen stats from DB."""
        if not self._conn:
            return
        total = 0
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE feature_matrix fm
                SET depth_chart_rank = dc.depth_rank
                FROM depth_charts dc
                WHERE fm.player_id = dc.player_id AND fm.season = dc.season
                    AND fm.week = dc.week AND fm.season = %s AND dc.season = %s
                """,
                (season, season),
            )
            total += cur.rowcount or 0
            cur.execute(
                """
                UPDATE feature_matrix fm
                SET avg_separation = ng.avg_separation, avg_cushion = ng.avg_cushion
                FROM nextgen_stats ng
                WHERE fm.player_id = ng.player_id AND fm.season = ng.season
                    AND fm.week = ng.week AND fm.season = %s AND ng.season = %s
                """,
                (season, season),
            )
            total += cur.rowcount or 0
        self._conn.commit()
        if total > 0:
            logger.info("Enriched %d feature rows with depth/nextgen data", total)


# ── Print helpers ─────────────────────────────────────────────────────────────

def _print_feature_row(fr: FeatureRow, player_name: str) -> None:
    W = 36
    print(f"\n{'═' * 72}")
    print(f"  FEATURE VECTOR: {player_name} | Week {fr.week} | Season {fr.season}")
    print(f"{'═' * 72}")

    sections = [
        ("Identity", [
            ("player_id", fr.player_id), ("game_id", fr.game_id),
            ("position", fr.position),   ("team", fr.team),
            ("opponent_team", fr.opponent_team), ("is_home", fr.is_home),
        ]),
        ("Bucket 1 — Kalman Form (posterior mean)", [
            ("kalman_est_receiving_yards",  fr.kalman_est_receiving_yards),
            ("kalman_est_receiving_tds",    fr.kalman_est_receiving_tds),
            ("kalman_est_targets",          fr.kalman_est_targets),
            ("kalman_est_receptions",       fr.kalman_est_receptions),
            ("kalman_est_target_share",     fr.kalman_est_target_share),
            ("kalman_est_air_yards_share",  fr.kalman_est_air_yards_share),
            ("kalman_est_fantasy_ppr",      fr.kalman_est_fantasy_ppr),
            ("kalman_est_carries",          fr.kalman_est_carries),
            ("kalman_est_rushing_yards",    fr.kalman_est_rushing_yards),
            ("kalman_est_passing_yards",    fr.kalman_est_passing_yards),
        ]),
        ("Bucket 2 — Season Baseline", [
            ("seas_games_played",          fr.seas_games_played),
            ("seas_avg_receiving_yards",   fr.seas_avg_receiving_yards),
            ("seas_avg_targets",           fr.seas_avg_targets),
            ("seas_avg_receptions",        fr.seas_avg_receptions),
            ("seas_avg_target_share",      fr.seas_avg_target_share),
            ("seas_avg_fantasy_ppr",       fr.seas_avg_fantasy_ppr),
            ("seas_yards_per_target",      fr.seas_yards_per_target),
            ("seas_yards_per_reception",   fr.seas_yards_per_reception),
        ]),
        ("Bucket 3 — Matchup (opponent allowed)", [
            ("opp_avg_receiving_yards_allowed", fr.opp_avg_receiving_yards_allowed),
            ("opp_avg_targets_allowed",         fr.opp_avg_targets_allowed),
            ("opp_avg_tds_allowed",             fr.opp_avg_tds_allowed),
            ("opp_avg_fantasy_ppr_allowed",     fr.opp_avg_fantasy_ppr_allowed),
        ]),
        ("Bucket 4 — Weather & Venue", [
            ("temp_f",       fr.temp_f),
            ("wind_mph",     fr.wind_mph),
            ("is_dome",      fr.is_dome),
            ("surface_turf", fr.surface_turf),
            ("temp_bucket",  fr.temp_bucket),
            ("wind_bucket",  fr.wind_bucket),
        ]),
        ("Bucket 5 — Team Context", [
            ("game_total_line", fr.game_total_line),
            ("spread_line",     fr.spread_line),
        ]),
        ("Bucket 6 — Rest", [
            ("days_rest",      fr.days_rest),
            ("is_short_week",  fr.is_short_week),
            ("is_bye_prior",   fr.is_bye_prior),
        ]),
        ("Targets (actual result)", [
            ("actual_fantasy_ppr",      fr.actual_fantasy_ppr),
            ("actual_receiving_yards",  fr.actual_receiving_yards),
            ("actual_rushing_yards",    fr.actual_rushing_yards),
            ("actual_passing_yards",    fr.actual_passing_yards),
        ]),
    ]

    for section_name, pairs in sections:
        print(f"\n  {section_name}")
        print(f"  {'─' * 68}")
        for label, val in pairs:
            fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
            print(f"    {label:<{W}}: {fmt}")

    print(f"\n{'═' * 72}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build feature matrix from production tables."
    )
    parser.add_argument("--seasons", nargs="+", type=int, metavar="YEAR")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print feature vector for one player-game, no DB write.")
    parser.add_argument("--player", default="Justin Jefferson",
                        help="Player display name for --dry-run.")
    parser.add_argument("--season", type=int, default=2024,
                        help="Season year for --dry-run.")
    parser.add_argument("--week", type=int, default=12,
                        help="Week number for --dry-run.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.dry_run:
        fr = build_from_nflreadpy(args.player, args.season, args.week)
        if fr is None:
            raise SystemExit(1)
        _print_feature_row(fr, args.player)
        return

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        logger.error("DATABASE_URL not set. Use --dry-run for no-DB mode.")
        raise SystemExit(1)

    seasons = args.seasons or [2025]
    with FeatureEngineer(db_url) as fe:
        n = fe.run(seasons=seasons)
        print(f"\nFeature matrix: {n} rows written for seasons {seasons}\n")


if __name__ == "__main__":
    main()
