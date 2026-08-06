"""
pipeline/features/feature_row.py

FeatureRow dataclass — the in-memory feature vector for one player-game.
Field names mirror feature_matrix table columns exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FeatureRow:
    """
    In-memory feature vector for one player-game.
    Field names mirror feature_matrix table columns exactly.
    """
    # Identity
    player_id: str
    game_id: str
    season: int
    week: int
    position: Optional[str] = None
    team: Optional[str] = None
    opponent_team: Optional[str] = None
    is_home: Optional[int] = None

    # Bucket 1 — Kalman Form (posterior mean + variance per stat)
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
    kalman_est_red_zone_target_share: Optional[float] = None
    kalman_variance_red_zone_target_share: Optional[float] = None
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
    kalman_est_completions: Optional[float] = None
    kalman_variance_completions: Optional[float] = None
    kalman_est_interceptions: Optional[float] = None
    kalman_variance_interceptions: Optional[float] = None
    kalman_est_fumbles: Optional[float] = None
    kalman_variance_fumbles: Optional[float] = None
    kalman_est_passing_yards: Optional[float] = None
    kalman_variance_passing_yards: Optional[float] = None
    kalman_est_passing_tds: Optional[float] = None
    kalman_variance_passing_tds: Optional[float] = None

    # Bucket 2 — Season Baseline
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
    # Efficiency metrics from game_logs (player_stats); strong QB/WR/RB predictors
    seas_avg_passing_cpoe: Optional[float] = None   # Completion % Over Expectation (QB)
    seas_avg_passing_epa: Optional[float] = None    # QB EPA per game
    seas_avg_receiving_epa: Optional[float] = None  # Per-target efficiency (WR/TE)
    seas_avg_rushing_epa: Optional[float] = None   # Per-carry efficiency (RB)
    seas_avg_racr: Optional[float] = None          # Reception Air Conversion Ratio
    seas_avg_wopr: Optional[float] = None         # Weighted Opportunity (target + air share)
    seas_avg_receiving_yac: Optional[float] = None # Yds after catch per reception

    # Bucket 3 — Matchup (opponent defensive stats per position)
    opp_avg_receiving_yards_allowed:     Optional[float] = None
    opp_avg_targets_allowed:             Optional[float] = None
    opp_avg_tds_allowed:                 Optional[float] = None
    opp_avg_fantasy_ppr_allowed:         Optional[float] = None
    opp_avg_rushing_yards_allowed:       Optional[float] = None
    # Extended matchup: new stats (carries, completions, pass attempts)
    opp_avg_carries_allowed:             Optional[float] = None
    opp_avg_completions_allowed:         Optional[float] = None
    opp_avg_pass_attempts_allowed:       Optional[float] = None

    # Bucket 4 — Weather & Venue
    temp_f:               Optional[float] = None
    wind_mph:             Optional[float] = None
    is_dome:              Optional[int] = None
    surface_turf:         Optional[int] = None
    temp_bucket:          Optional[int] = None
    wind_bucket:          Optional[int] = None
    # precipitation_bucket: 0=none, 1=rain (>0.05"/hr), 2=snow/sleet.
    # Rain/snow reduce passing efficiency. NULL until OpenWeatherMap API (Phase 4).
    precipitation_bucket: Optional[int] = None

    # Bucket 5 — Team Context
    game_total_line: Optional[float] = None
    spread_line: Optional[float] = None

    # Bucket 6 — Rest
    days_rest: Optional[int] = None
    is_short_week: Optional[int] = None
    is_bye_prior: Optional[int] = None

    # Bucket 7 — Rule Coefficients
    # Encodes NFL rule changes that shift offensive volume.
    # See compute_rule_features() for full encoding.
    rule_coeff: Optional[float] = None

    # Bucket 8 — Injury / Availability (from ESPN adapter, optional)
    # injury_status_encoded: 0=out/dnp, 1=doubtful, 2=questionable, 3=limited, 4=full
    # None means the player was not on the ESPN injury report (assumed healthy).
    injury_status_encoded: Optional[int] = None
    # games_missed_streak: consecutive weeks with no game_log entry prior to
    # target_week in the current season. 0 = played every game so far.
    games_missed_streak: Optional[int] = None

    # Snap participation — populated by snap_counts normalize pass.
    # snap_pct_off: offense snap participation rate [0.0, 1.0].
    # None until snap_counts are processed for this player+game.
    snap_pct_off: Optional[float] = None

    # Bucket 9 — Defensive Tendency (from nflreadpy pbp; stubbed 0.0 until Phase 4 data wired)
    # Opponent defensive formation/coverage tendencies accumulated to week-1.
    opp_zone_pct:     Optional[float] = None  # fraction of pass snaps opponent played zone
    opp_man_pct:      Optional[float] = None  # fraction of pass snaps opponent played man
    opp_blitz_rate:   Optional[float] = None  # fraction of pass snaps opponent blitzed
    opp_pressure_rate: Optional[float] = None  # fraction of pass snaps opponent generated pressure

    # Bucket 9 — Scheme Interaction Features
    # Explicit cross-terms between receiver route depth (aDOT proxy) and coverage shell.
    # Prevents the model from learning aDOT and zone_pct independently.
    deep_matchup_score:       Optional[float] = None  # kalman_est_air_yards_share × (1 - opp_zone_pct)
    coverage_matchup_score:   Optional[float] = None  # kalman_est_target_share × opp_man_pct
    blitz_exposure:           Optional[float] = None  # snap_pct_off × opp_blitz_rate
    separation_demand_score:  Optional[float] = None  # kalman_est_target_share × opp_man_pct × (1 - opp_zone_pct)

    # Bucket 10 — Elo Ratings (from ml/team_elo.py; populated when elo system is fitted)
    team_off_elo:          Optional[float] = None  # team's offensive Elo heading into this week
    team_def_elo:          Optional[float] = None  # team's defensive Elo heading into this week
    opp_off_elo:           Optional[float] = None  # opponent's offensive Elo
    opp_def_elo:           Optional[float] = None  # opponent's defensive Elo
    elo_matchup_diff:      Optional[float] = None  # team_off_elo - opp_def_elo (+ve = offense advantage)
    elo_implied_win_prob:  Optional[float] = None  # elo-implied win probability for team

    # TFT Static Covariates — player physical profile (from game_logs via nflreadpy)
    # These are the "Static Real" columns that TFT uses to distinguish player identities.
    # Previously defaulted to 0.0 in _derive_columns() — now populated from source data.
    height:      Optional[float] = None  # player height in inches (e.g. 74.0 = 6'2")
    weight:      Optional[float] = None  # player weight in pounds
    draft_round: Optional[float] = None  # NFL draft round (1-7; None = undrafted)

    # Bucket 11 — PBP-Derived Features (from pbp_features table via pbp_pipeline.py).
    # NULL until pbp_pipeline.py runs after ETL. XGB/LGB treat NULL as missing splits.
    # EPA (Expected Points Added) — strongest single predictor in NFL analytics.
    epa_per_play:          Optional[float] = None  # avg EPA per snap (offense)
    epa_per_target:        Optional[float] = None  # EPA per target (WR/TE quality)
    epa_per_rush:          Optional[float] = None  # EPA per carry (RB efficiency)
    qb_epa_per_dropback:   Optional[float] = None  # QB EPA per dropback
    # Air yards & depth of target
    adot:                  Optional[float] = None  # avg depth of target (yards downfield)
    yac_per_reception:     Optional[float] = None  # avg yards after catch
    xyac_per_reception:    Optional[float] = None  # expected YAC from model
    # Volume & usage from PBP
    target_share_pbp:      Optional[float] = None  # targets / team_pass_attempts
    air_yards_share_pbp:   Optional[float] = None  # player air yards / team air yards
    red_zone_targets:      Optional[int]   = None  # targets inside opp 20-yard-line
    end_zone_targets:      Optional[int]   = None  # targets inside opp 10-yard-line
    red_zone_target_share: Optional[float] = None  # player RZ targets / team RZ targets
    # Pass direction breakdown
    pass_left_rate:        Optional[float] = None  # % of targets to left
    pass_middle_rate:      Optional[float] = None  # % of targets to middle
    pass_right_rate:       Optional[float] = None  # % of targets to right
    # Drop rate — sourced from PFR via load_pfr_advstats(stat_type='rec').
    # NULL when PFR data unavailable for a season (pre-2018).
    drop_rate:             Optional[float] = None  # drops / targets (PFR; ~0.04 avg)
    # OL / protection quality (from qb_hit + sack columns)
    ol_pressure_rate:      Optional[float] = None  # team's qb_hit rate per dropback
    ol_sack_rate:          Optional[float] = None  # team's sack rate per dropback
    # Defensive pressure faced (opponent tendencies vs this team)
    opp_pressure_rate_pbp: Optional[float] = None  # opp qb_hits per dropback
    opp_sack_rate_pbp:     Optional[float] = None  # opp sacks per dropback
    # routes_run_pct: offensive snap participation rate from load_snap_counts().
    # Proxy for routes run per game; offense_pct × team_pass_attempts ≈ routes.
    # Populated by normalize.py snap counts pass alongside snap_pct_off.
    routes_run_pct:        Optional[float] = None  # offense_pct (0.0–1.0)

    # Derived velocity/trend features (computed in build_feature_row)
    # target_share_trend: 3-game delta in target share (positive = trending up).
    # Captures role trajectory that Kalman mean alone misses.
    target_share_trend:    Optional[float] = None

    # Weather × position interaction terms (computed in build_feature_row)
    # Encode the physical reality that wind/rain hurt aerial attacks more than running.
    wind_x_qb:             Optional[float] = None  # wind_bucket × is_QB (0 or wind_bucket)
    wind_x_wr:             Optional[float] = None  # wind_bucket × is_WR_or_TE
    precip_x_pass:         Optional[float] = None  # precipitation_bucket × is_passing_pos

    # Positional depth signal — team rank by receiving volume.
    # 1 = WR1/TE1 (most targets), 2 = WR2, etc. NULL for non-receiving positions.
    team_pos_rank:         Optional[float] = None
    # depth_chart_rank: official depth chart position (1=WR1, 2=WR2; from nflverse depth_charts)
    depth_chart_rank:     Optional[float] = None
    # Next Gen Stats (2016+): separation/cushion for WR/TE; speed for RB
    avg_separation:       Optional[float] = None  # yards from defender at target
    avg_cushion:          Optional[float] = None  # pre-snap distance from defender

    # Player profile embeddings — first 8 PCA dimensions from PlayerProfileEmbedder.
    # Captures physical profile (size, speed, position, experience) for cold-start.
    # Populated by EmbeddingStore.enrich_features() called in FeatureEngineer.run().
    player_emb_0: Optional[float] = None
    player_emb_1: Optional[float] = None
    player_emb_2: Optional[float] = None
    player_emb_3: Optional[float] = None
    player_emb_4: Optional[float] = None
    player_emb_5: Optional[float] = None
    player_emb_6: Optional[float] = None
    player_emb_7: Optional[float] = None

    # Targets — the labels that train.py trains against.
    # These are populated from game_logs (actual observed outcomes).
    # Do NOT use estimated/Kalman values in the actual_* fields.
    actual_fantasy_ppr:      Optional[float] = None
    actual_receiving_yards:  Optional[float] = None
    actual_rushing_yards:    Optional[float] = None
    actual_passing_yards:    Optional[float] = None
    # 11 new target columns added for per-position stat expansion
    actual_pass_attempts:    Optional[float] = None
    actual_completions:      Optional[float] = None
    actual_passing_tds:      Optional[float] = None
    actual_interceptions:    Optional[float] = None
    actual_carries:          Optional[float] = None
    actual_rushing_tds:      Optional[float] = None
    actual_receptions:       Optional[float] = None
    actual_receiving_tds:    Optional[float] = None
    actual_targets:          Optional[float] = None
    actual_fumbles:          Optional[float] = None
    actual_sacks_taken:      Optional[float] = None
    actual_qb_hits_taken:    Optional[float] = None  # QB hits taken (PBP-derived)
