"""
backend/app/models/production.py

SQLModel ORM definitions for the core production tables.

These are populated by pipeline/normalize.py reading from staging tables.
They are the source of truth for the ML pipeline and the FastAPI backend.

Schema design follows outline §4.2.

Tables defined here (in FK dependency order):
  1. Player    — canonical player record (gsis_id PK)
  2. Team      — team reference (abbreviation PK)
  3. Game      — one NFL game per row
  4. GameLog   — per-player, per-game statistics
  5. FeatureMatrix — pre-computed feature vectors for ML training/inference
  6. Projection — final weekly projection results from ml/train.py pipeline
"""

from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import Column, JSON, UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel


class Player(SQLModel, table=True):
    """
    Canonical player record.
    Primary key: gsis_id (NFL's Global Standard Identifier).
    Updated on every full roster ingest.
    """

    __tablename__ = "players"

    id: str = Field(primary_key=True)          # gsis_id
    full_name: Optional[str] = None
    position: Optional[str] = None
    team: Optional[str] = None
    height: Optional[str] = None
    weight: Optional[float] = None
    birth_date: Optional[date] = None
    college: Optional[str] = None
    years_exp: Optional[int] = None
    entry_year: Optional[int] = None
    status: Optional[str] = None               # "ACT", "INA", "RES:IR", etc.
    headshot_url: Optional[str] = None
    espn_id: Optional[str] = None
    pfr_id: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    game_logs: List["GameLog"] = Relationship(back_populates="player")


class Team(SQLModel, table=True):
    """
    NFL team reference table.
    Populated from schedules/rosters data.
    """

    __tablename__ = "teams"

    id: str = Field(primary_key=True)          # team abbreviation e.g. "KC", "PHI"
    name: Optional[str] = None
    city: Optional[str] = None
    stadium: Optional[str] = None
    roof_type: Optional[str] = None            # "dome" | "outdoors" | "retractable"
    surface: Optional[str] = None             # "grass" | "fieldturf" | etc.
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Game(SQLModel, table=True):
    """
    One NFL game.
    game_id format from nflreadpy: "YYYY_WW_AWAY_HOME" e.g. "2025_01_KC_BAL".
    """

    __tablename__ = "games"

    id: str = Field(primary_key=True)          # nflreadpy game_id
    season: int = Field(index=True)
    week: int
    game_type: Optional[str] = None            # "REG" | "WC" | "DIV" | "CON" | "SB"
    home_team: str
    away_team: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    gameday: Optional[date] = None
    gametime: Optional[str] = None
    weekday: Optional[str] = None
    stadium: Optional[str] = None
    roof: Optional[str] = None
    surface: Optional[str] = None
    temp: Optional[float] = None
    wind: Optional[float] = None
    spread_line: Optional[float] = None
    total_line: Optional[float] = None
    away_moneyline: Optional[float] = None
    home_moneyline: Optional[float] = None
    home_rest: Optional[int] = None            # days of rest
    away_rest: Optional[int] = None
    home_qb_name: Optional[str] = None
    away_qb_name: Optional[str] = None

    # Relationships
    game_logs: List["GameLog"] = Relationship(back_populates="game")


class GameLog(SQLModel, table=True):
    """
    Per-player, per-game statistics.
    Populated by pipeline/normalize.py joining player_stats + snap_counts.

    snap_count / offense_pct come from snap_counts source (pfr_player_id join).
    All other stat columns come from player_stats source (gsis_id).
    """

    __tablename__ = "game_logs"
    __table_args__ = (
        UniqueConstraint("player_id", "game_id", name="uq_game_logs_player_game"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    player_id: str = Field(foreign_key="players.id", index=True)
    game_id: str = Field(foreign_key="games.id", index=True)
    season: int
    week: int
    season_type: Optional[str] = None         # "REG" | "POST"
    team: Optional[str] = None
    opponent_team: Optional[str] = None
    position: Optional[str] = None
    # ── Passing ──────────────────────────────────────────────────────────
    completions: Optional[int] = None
    attempts: Optional[int] = None
    passing_yards: Optional[float] = None
    passing_tds: Optional[int] = None
    passing_interceptions: Optional[int] = None
    passing_air_yards: Optional[float] = None
    passing_yards_after_catch: Optional[float] = None
    passing_first_downs: Optional[int] = None
    passing_epa: Optional[float] = None
    passing_cpoe: Optional[float] = None
    # ── Rushing ──────────────────────────────────────────────────────────
    carries: Optional[int] = None
    rushing_yards: Optional[float] = None
    rushing_tds: Optional[int] = None
    rushing_fumbles: Optional[int] = None
    receiving_fumbles: Optional[int] = None
    sack_fumbles: Optional[int] = None
    rushing_epa: Optional[float] = None
    # ── Receiving ────────────────────────────────────────────────────────
    receptions: Optional[int] = None
    targets: Optional[int] = None
    receiving_yards: Optional[float] = None
    receiving_tds: Optional[int] = None
    receiving_air_yards: Optional[float] = None
    receiving_yards_after_catch: Optional[float] = None
    receiving_first_downs: Optional[int] = None
    receiving_epa: Optional[float] = None
    racr: Optional[float] = None
    target_share: Optional[float] = None
    air_yards_share: Optional[float] = None
    wopr: Optional[float] = None
    # ── Snap usage (from snap_counts join) ───────────────────────────────
    offense_snaps: Optional[int] = None
    offense_pct: Optional[float] = None
    # ── Fantasy ──────────────────────────────────────────────────────────
    fantasy_points: Optional[float] = None
    fantasy_points_ppr: Optional[float] = None

    # Relationships
    player: Optional[Player] = Relationship(back_populates="game_logs")
    game: Optional[Game] = Relationship(back_populates="game_logs")


class FeatureMatrix(SQLModel, table=True):
    """
    Pre-computed feature vector for one player-game.
    Populated by pipeline/feature_engineer.py; consumed by all ML models.

    Feature buckets follow outline §5.1:
      Bucket 1 (kalman_*) — Kalman filter posterior mean + variance per stat
      Bucket 2 (seas_*)   — season-to-date baseline stats
      Bucket 3 (opp_*)    — opponent defensive stats allowed to this position
      Bucket 4 (venue)    — weather + venue (temp/wind buckets, dome, surface)
      Bucket 5 (ctx_*)    — team context (implied total, spread, home/away)
      Bucket 6 (rest_*)   — rest days, short week, bye prior
      Bucket 7            — rule coefficients (placeholder, from rule_changes)
      actual_*            — true outcomes (populated for past games only)

    Schema migration: form_* columns removed, kalman_est_*/kalman_variance_*
    added. Drop-and-rebuild required (no production data yet — run make setup).
    See ml/kalman_tracker.py for Kalman filter implementation.
    """

    __tablename__ = "feature_matrix"
    __table_args__ = (
        UniqueConstraint("player_id", "game_id", name="uq_feature_matrix_player_game"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    player_id: str = Field(foreign_key="players.id", index=True)
    game_id: str = Field(foreign_key="games.id", index=True)
    season: int = Field(index=True)
    week: int
    position: Optional[str] = None
    team: Optional[str] = None
    opponent_team: Optional[str] = None
    is_home: Optional[int] = None          # 1 = home, 0 = away

    # ── Bucket 1: Kalman Player Form ──────────────────────────────────────
    # Scalar Kalman filter posterior per stat: x_k = x_{k-1} + w_k, y_k = x_k + v_k
    # Q=1.0 (process noise), R=empirical variance, x0=position-average prior
    # kalman_variance_* feeds the Bayesian uncertainty layer.
    kalman_est_receiving_yards: Optional[float] = None
    kalman_variance_receiving_yards: Optional[float] = None
    kalman_est_receiving_tds: Optional[float] = None
    kalman_variance_receiving_tds: Optional[float] = None
    kalman_est_targets: Optional[float] = None
    kalman_variance_targets: Optional[float] = None
    kalman_est_receptions: Optional[float] = None
    kalman_variance_receptions: Optional[float] = None
    kalman_est_target_share: Optional[float] = None
    kalman_variance_target_share: Optional[float] = None
    kalman_est_air_yards_share: Optional[float] = None
    kalman_variance_air_yards_share: Optional[float] = None
    kalman_est_fantasy_ppr: Optional[float] = None
    kalman_variance_fantasy_ppr: Optional[float] = None
    kalman_est_carries: Optional[float] = None
    kalman_variance_carries: Optional[float] = None
    kalman_est_rushing_yards: Optional[float] = None
    kalman_variance_rushing_yards: Optional[float] = None
    kalman_est_rushing_tds: Optional[float] = None
    kalman_variance_rushing_tds: Optional[float] = None
    kalman_est_pass_attempts: Optional[float] = None
    kalman_variance_pass_attempts: Optional[float] = None
    kalman_est_passing_yards: Optional[float] = None
    kalman_variance_passing_yards: Optional[float] = None
    kalman_est_passing_tds: Optional[float] = None
    kalman_variance_passing_tds: Optional[float] = None

    # ── Bucket 2: Season-to-Date Baseline ─────────────────────────────────
    seas_games_played: Optional[int] = None
    seas_avg_receiving_yards: Optional[float] = None
    seas_avg_targets: Optional[float] = None
    seas_avg_receptions: Optional[float] = None
    seas_avg_target_share: Optional[float] = None
    seas_avg_fantasy_ppr: Optional[float] = None
    seas_yards_per_target: Optional[float] = None
    seas_yards_per_reception: Optional[float] = None
    seas_avg_carries: Optional[float] = None
    seas_avg_rushing_yards: Optional[float] = None
    seas_yards_per_carry: Optional[float] = None
    seas_avg_attempts: Optional[float] = None
    seas_avg_passing_yards: Optional[float] = None
    seas_completion_pct: Optional[float] = None

    # ── Bucket 3: Matchup (opponent defensive stats by position) ──────────
    opp_avg_receiving_yards_allowed: Optional[float] = None
    opp_avg_targets_allowed: Optional[float] = None
    opp_avg_tds_allowed: Optional[float] = None
    opp_avg_fantasy_ppr_allowed: Optional[float] = None
    opp_avg_rushing_yards_allowed: Optional[float] = None

    # ── Bucket 4: Weather & Venue ──────────────────────────────────────────
    temp_f: Optional[float] = None
    wind_mph: Optional[float] = None
    is_dome: Optional[int] = None          # 1 = dome/closed, 0 = outdoor
    surface_turf: Optional[int] = None    # 1 = artificial turf, 0 = natural grass
    temp_bucket: Optional[int] = None     # 0=cold(<32F) 1=cool(32-50F) 2=mild(50F+)
    wind_bucket: Optional[int] = None     # 0=calm(<10) 1=breezy(10-20) 2=windy(20+)

    # ── Bucket 5: Team Context ─────────────────────────────────────────────
    game_total_line: Optional[float] = None
    spread_line: Optional[float] = None    # positive = home favored

    # ── Bucket 6: Rest ────────────────────────────────────────────────────
    days_rest: Optional[int] = None
    is_short_week: Optional[int] = None   # 1 if days_rest < 6 (Thursday game)
    is_bye_prior: Optional[int] = None    # 1 if days_rest >= 12 (came off bye)

    # ── Bucket 7: Rule Coefficients (placeholder) ─────────────────────────
    rule_coeff: Optional[float] = None    # from rule_changes table (TBD)

    # ── Bucket 8: Injury / Availability (from ESPN adapter) ───────────────
    # injury_status_encoded: 0=out/dnp, 1=doubtful, 2=questionable,
    #                        3=limited, 4=full. NULL = not on injury report.
    injury_status_encoded: Optional[int] = None
    # games_missed_streak: consecutive weeks with no game_log before target week.
    games_missed_streak: Optional[int] = None
    # snap_pct_off: offensive snap participation rate (0-1).  NULL before HIGH 2 fix.
    snap_pct_off: Optional[float] = None

    # ── Targets (actual results — NULL for future games) ──────────────────
    actual_fantasy_ppr: Optional[float] = None
    actual_receiving_yards: Optional[float] = None
    actual_rushing_yards: Optional[float] = None
    actual_passing_yards: Optional[float] = None
    actual_carries: Optional[int] = None
    actual_completions: Optional[int] = None
    actual_fumbles: Optional[int] = None
    actual_interceptions: Optional[int] = None
    actual_pass_attempts: Optional[int] = None
    actual_passing_tds: Optional[int] = None
    actual_receiving_tds: Optional[int] = None
    actual_receptions: Optional[int] = None
    actual_rushing_tds: Optional[int] = None
    actual_sacks_taken: Optional[int] = None
    actual_targets: Optional[int] = None

    # Next Gen / Phase 4 PBP stats
    actual_qb_hits_taken: Optional[int] = None
    adot: Optional[float] = None
    air_yards_share_pbp: Optional[float] = None
    avg_cushion: Optional[float] = None
    avg_separation: Optional[float] = None
    blitz_exposure: Optional[float] = None
    coverage_matchup_score: Optional[float] = None
    first_read_share: Optional[float] = None
    light_box_pct: Optional[float] = None
    man_coverage_rate: Optional[float] = None
    pass_blocking_grade: Optional[float] = None
    pass_rush_win_rate: Optional[float] = None
    press_rate: Optional[float] = None
    route_participation_rate: Optional[float] = None
    route_win_rate: Optional[float] = None
    run_blocking_grade: Optional[float] = None
    snap_share_trend: Optional[float] = None
    stacked_box_pct: Optional[float] = None
    yac_expected: Optional[float] = None

    deep_matchup_score: Optional[float] = None
    depth_chart_rank: Optional[float] = None
    draft_round: Optional[float] = None
    drop_rate: Optional[float] = None
    elo_implied_win_prob: Optional[float] = None
    elo_matchup_diff: Optional[float] = None
    end_zone_targets: Optional[float] = None
    epa_per_play: Optional[float] = None
    epa_per_rush: Optional[float] = None
    epa_per_target: Optional[float] = None
    height: Optional[float] = None
    kalman_est_completions: Optional[float] = None
    kalman_est_fumbles: Optional[float] = None
    kalman_est_interceptions: Optional[float] = None
    kalman_est_red_zone_target_share: Optional[float] = None
    kalman_variance_completions: Optional[float] = None
    kalman_variance_fumbles: Optional[float] = None
    kalman_variance_interceptions: Optional[float] = None
    kalman_variance_red_zone_target_share: Optional[float] = None
    ol_pressure_rate: Optional[float] = None
    ol_sack_rate: Optional[float] = None
    opp_avg_carries_allowed: Optional[float] = None
    opp_avg_completions_allowed: Optional[float] = None
    opp_avg_pass_attempts_allowed: Optional[float] = None
    opp_blitz_rate: Optional[float] = None
    opp_def_elo: Optional[float] = None
    opp_man_pct: Optional[float] = None
    opp_off_elo: Optional[float] = None
    opp_pressure_rate: Optional[float] = None
    opp_pressure_rate_pbp: Optional[float] = None
    opp_sack_rate_pbp: Optional[float] = None
    opp_zone_pct: Optional[float] = None
    pass_left_rate: Optional[float] = None
    pass_middle_rate: Optional[float] = None
    pass_right_rate: Optional[float] = None
    player_emb_0: Optional[float] = None
    player_emb_1: Optional[float] = None
    player_emb_2: Optional[float] = None
    player_emb_3: Optional[float] = None
    player_emb_4: Optional[float] = None
    player_emb_5: Optional[float] = None
    player_emb_6: Optional[float] = None
    player_emb_7: Optional[float] = None
    precip_x_pass: Optional[float] = None
    precipitation_bucket: Optional[float] = None
    qb_epa_per_dropback: Optional[float] = None
    red_zone_target_share: Optional[float] = None
    red_zone_targets: Optional[float] = None
    routes_run_pct: Optional[float] = None
    seas_avg_passing_cpoe: Optional[float] = None
    seas_avg_passing_epa: Optional[float] = None
    seas_avg_racr: Optional[float] = None
    seas_avg_receiving_epa: Optional[float] = None
    seas_avg_receiving_yac: Optional[float] = None
    seas_avg_rushing_epa: Optional[float] = None
    seas_avg_wopr: Optional[float] = None
    separation_demand_score: Optional[float] = None
    target_share_pbp: Optional[float] = None
    target_share_trend: Optional[float] = None
    team_def_elo: Optional[float] = None
    team_off_elo: Optional[float] = None
    team_pos_rank: Optional[float] = None
    weight: Optional[float] = None
    wind_x_qb: Optional[float] = None
    wind_x_wr: Optional[float] = None
    xyac_per_reception: Optional[float] = None
    yac_per_reception: Optional[float] = None

    computed_at: datetime = Field(default_factory=datetime.utcnow)


class Projection(SQLModel, table=True):
    """
    Final weekly projection result for one player-stat pair.
    Populated by ml/train.py PipelineRunner.run().
    Consumed by the FastAPI /predict endpoint.

    One row per (player_id, game_id, stat).
    Unique constraint prevents duplicate runs from double-inserting.
    """

    __tablename__ = "projections"
    __table_args__ = (
        UniqueConstraint(
            "player_id", "game_id", "stat",
            name="uq_projections_player_game_stat",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    player_id: str = Field(foreign_key="players.id", index=True)
    game_id: str = Field(foreign_key="games.id", index=True)
    season: int = Field(index=True)
    week: int
    stat: str                              # e.g. "receiving_yards"
    position: Optional[str] = None

    # ── Projection percentiles ─────────────────────────────────────────
    projection: Optional[float] = None    # p50 — primary point estimate
    floor: Optional[float] = None         # p10 — downside scenario
    ceiling: Optional[float] = None       # p90 — upside scenario
    # p25 / p75 are required for accurate 50% calibration interval in backtest.
    # Previously missing, causing run_backtest.py to proxy p25=floor (p10) and
    # p75=ceiling (p90) — making coverage_50 metrics completely wrong.
    p25: Optional[float] = None           # 25th percentile (from MC posterior samples)
    p75: Optional[float] = None           # 75th percentile (from MC posterior samples)

    # ── Boom / bust probabilities ──────────────────────────────────────
    boom_probability: Optional[float] = None
    bust_probability: Optional[float] = None

    # ── Fantasy PPR projection (yardage component) ─────────────────────
    fantasy_projection: Optional[float] = None
    fantasy_floor: Optional[float] = None
    fantasy_ceiling: Optional[float] = None

    # ── Pipeline provenance ────────────────────────────────────────────
    pipeline_run_id: Optional[str] = None  # MLflow run ID; NULL for dry_run
    # Max season used to train the model that produced this row.
    # Required for causal backtest: must be strictly < projection.season.
    max_train_season: Optional[int] = None

    # ── Posterior samples for CRPS computation ─────────────────────────
    # JSON array of up to 500 draws from the Bayesian posterior.
    # NULL for rows written before HIGH 3 fix — backtest falls back to
    # synthetic np.random.normal() when this field is NULL.
    posterior_samples: Optional[List[float]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
