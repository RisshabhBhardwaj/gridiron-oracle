"""Canonical idempotent DDL for Gridiron Oracle.

Owned by Alembic migrations. Pipeline modules may call
`pipeline.schema.ensure_schema()` which applies these statements
only as an empty-database bootstrap path.
"""

from __future__ import annotations

CREATE_PRODUCTION = """
CREATE TABLE IF NOT EXISTS teams (
    id          VARCHAR(10)  PRIMARY KEY,
    name        TEXT,
    city        TEXT,
    stadium     TEXT,
    roof_type   TEXT,
    surface     TEXT,
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS players (
    id           TEXT         PRIMARY KEY,
    full_name    TEXT,
    position     TEXT,
    team         TEXT,
    height       TEXT,
    weight       FLOAT,
    birth_date   DATE,
    college      TEXT,
    years_exp    INTEGER,
    entry_year   INTEGER,
    status       TEXT,
    headshot_url TEXT,
    espn_id      TEXT,
    pfr_id       TEXT,
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS games (
    id               TEXT         PRIMARY KEY,
    season           INTEGER      NOT NULL,
    week             INTEGER      NOT NULL,
    game_type        TEXT,
    home_team        TEXT         NOT NULL,
    away_team        TEXT         NOT NULL,
    home_score       INTEGER,
    away_score       INTEGER,
    gameday          DATE,
    gametime         TEXT,
    weekday          TEXT,
    stadium          TEXT,
    roof             TEXT,
    surface          TEXT,
    temp             FLOAT,
    wind             FLOAT,
    spread_line      FLOAT,
    total_line       FLOAT,
    away_moneyline   FLOAT,
    home_moneyline   FLOAT,
    home_rest        INTEGER,
    away_rest        INTEGER,
    home_qb_name     TEXT,
    away_qb_name     TEXT
);

CREATE TABLE IF NOT EXISTS game_logs (
    id                           SERIAL   PRIMARY KEY,
    player_id                    TEXT     NOT NULL REFERENCES players(id),
    game_id                      TEXT     NOT NULL REFERENCES games(id),
    season                       INTEGER  NOT NULL,
    week                         INTEGER  NOT NULL,
    season_type                  TEXT,
    team                         TEXT,
    opponent_team                TEXT,
    position                     TEXT,
    completions                  INTEGER,
    attempts                     INTEGER,
    passing_yards                FLOAT,
    passing_tds                  INTEGER,
    passing_interceptions        INTEGER,
    passing_air_yards            FLOAT,
    passing_yards_after_catch    FLOAT,
    passing_first_downs          INTEGER,
    passing_epa                  FLOAT,
    passing_cpoe                 FLOAT,
    carries                      INTEGER,
    rushing_yards                FLOAT,
    rushing_tds                  INTEGER,
    rushing_fumbles              INTEGER,
    receiving_fumbles            INTEGER,
    sack_fumbles                 INTEGER,
    rushing_epa                  FLOAT,
    receptions                   INTEGER,
    targets                      INTEGER,
    receiving_yards              FLOAT,
    receiving_tds                INTEGER,
    receiving_air_yards          FLOAT,
    receiving_yards_after_catch  FLOAT,
    receiving_first_downs        INTEGER,
    receiving_epa                FLOAT,
    racr                         FLOAT,
    target_share                 FLOAT,
    air_yards_share              FLOAT,
    wopr                         FLOAT,
    offense_snaps                INTEGER,
    offense_pct                  FLOAT,
    fantasy_points               FLOAT,
    fantasy_points_ppr           FLOAT,
    CONSTRAINT uq_game_logs_player_game UNIQUE (player_id, game_id)
);
CREATE INDEX IF NOT EXISTS idx_game_logs_player_season
    ON game_logs (player_id, season);
CREATE INDEX IF NOT EXISTS idx_game_logs_season_week
    ON game_logs (season, week);

CREATE TABLE IF NOT EXISTS depth_charts (
    player_id   TEXT NOT NULL,
    season      INTEGER NOT NULL,
    week        INTEGER NOT NULL,
    team        TEXT NOT NULL,
    position    TEXT,
    depth_rank  FLOAT NOT NULL,
    PRIMARY KEY (player_id, season, week)
);
CREATE INDEX IF NOT EXISTS idx_depth_charts_lookup
    ON depth_charts (player_id, season, week);

CREATE TABLE IF NOT EXISTS nextgen_stats (
    player_id   TEXT NOT NULL,
    season      INTEGER NOT NULL,
    week        INTEGER NOT NULL,
    avg_separation  FLOAT,
    avg_cushion    FLOAT,
    max_speed      FLOAT,
    avg_completion_above_expectation FLOAT,
    PRIMARY KEY (player_id, season, week)
);
CREATE INDEX IF NOT EXISTS idx_nextgen_stats_lookup
    ON nextgen_stats (player_id, season, week);

CREATE TABLE IF NOT EXISTS team_game_stats (
    team        TEXT NOT NULL,
    season      INTEGER NOT NULL,
    week        INTEGER NOT NULL,
    pass_attempts   INTEGER,
    rush_attempts   INTEGER,
    total_plays     INTEGER,
    total_yards     FLOAT,
    PRIMARY KEY (team, season, week)
);

CREATE TABLE IF NOT EXISTS ftn_play (
    game_id     TEXT NOT NULL,
    play_id     INTEGER NOT NULL,
    season      INTEGER,
    week        INTEGER,
    is_drop     BOOLEAN DEFAULT FALSE,
    is_contested_ball BOOLEAN DEFAULT FALSE,
    is_catchable_ball BOOLEAN,
    is_created_reception BOOLEAN,
    PRIMARY KEY (game_id, play_id)
);
CREATE INDEX IF NOT EXISTS idx_ftn_play_lookup
    ON ftn_play (game_id, season, week);

CREATE TABLE IF NOT EXISTS ftn_player_game (
    player_id   TEXT NOT NULL,
    game_id     TEXT NOT NULL,
    season      INTEGER,
    week        INTEGER,
    drops       INTEGER DEFAULT 0,
    contested_catches INTEGER DEFAULT 0,
    targets     INTEGER DEFAULT 0,
    PRIMARY KEY (player_id, game_id)
);
CREATE INDEX IF NOT EXISTS idx_ftn_player_game_lookup
    ON ftn_player_game (player_id, season, week);

CREATE TABLE IF NOT EXISTS participation_player_game (
    player_id   TEXT NOT NULL,
    game_id     TEXT NOT NULL,
    season      INTEGER,
    week        INTEGER,
    routes_run  INTEGER DEFAULT 0,
    PRIMARY KEY (player_id, game_id)
);
CREATE INDEX IF NOT EXISTS idx_participation_player_game_lookup
    ON participation_player_game (player_id, season, week);

CREATE TABLE IF NOT EXISTS combine (
    player_id   TEXT NOT NULL,
    season      INTEGER NOT NULL,
    forty       FLOAT,
    bench_press INTEGER,
    vertical_jump FLOAT,
    broad_jump  FLOAT,
    PRIMARY KEY (player_id, season)
);
"""

CREATE_FEATURE_MATRIX = """
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

CREATE_INJURY_HISTORY = """
CREATE TABLE IF NOT EXISTS injury_history (
    player_id            VARCHAR NOT NULL,
    full_name            VARCHAR,
    season               INTEGER NOT NULL,
    week                 INTEGER NOT NULL,
    team                 VARCHAR,
    position             VARCHAR,
    report_status        VARCHAR,   -- "Out", "Doubtful", "Questionable", "Limited", "Full", "DNP"
    practice_status      VARCHAR,   -- Mid-week practice designation
    injury_type          VARCHAR,   -- "Knee", "Hamstring", "Ankle", etc.
    is_out               BOOLEAN,   -- TRUE when player missed the game
    games_missed_streak  INTEGER,   -- consecutive games missed heading into this week
    time_to_return       INTEGER,   -- weeks until next active game (NULL if still IR/season end)
    event_type           VARCHAR,   -- 'active', 'injury_start', 'injury_ongoing', 'return'
    PRIMARY KEY (player_id, season, week)
)
"""

CREATE_PBP_FEATURES = """
CREATE TABLE IF NOT EXISTS pbp_features (
    player_id            VARCHAR NOT NULL,
    game_id              VARCHAR NOT NULL,
    season               INTEGER NOT NULL,
    week                 INTEGER NOT NULL,
    team                 VARCHAR,
    -- EPA metrics (Expected Points Added)
    epa_per_play         FLOAT,    -- avg EPA per snap
    epa_per_target       FLOAT,    -- EPA per target (WR/TE quality)
    epa_per_rush         FLOAT,    -- EPA per carry (RB efficiency)
    qb_epa_per_dropback  FLOAT,    -- QB EPA per dropback (strongest QB predictor)
    -- Air yards & depth of target
    adot                 FLOAT,    -- avg depth of target (yards downfield)
    total_air_yards      FLOAT,    -- total air yards on targets
    -- Yards after catch
    yac_per_reception    FLOAT,    -- avg yards after catch
    xyac_per_reception   FLOAT,    -- expected YAC from model
    -- Volume & usage
    routes_run           INTEGER,  -- snaps as eligible receiver (proxy: targets/game / completion %)
    target_share_pbp     FLOAT,    -- targets / team pass attempts
    air_yards_share_pbp  FLOAT,    -- player air yards / team air yards
    red_zone_targets     INTEGER,  -- targets inside opp 20-yard-line
    end_zone_targets     INTEGER,  -- targets inside opp 10-yard-line
    red_zone_target_share FLOAT,   -- player RZ targets / team RZ targets
    -- Pass direction breakdown
    pass_left_rate       FLOAT,    -- % of targets to left
    pass_middle_rate     FLOAT,    -- % of targets to middle
    pass_right_rate      FLOAT,    -- % of targets to right
    -- Drops (incomplete on catchable ball — estimated from PBP)
    drop_rate            FLOAT,    -- drops / targets (0.0 = perfect; avg ~0.04)
    -- OL / protection quality (from qb_hit + sack columns)
    ol_pressure_rate     FLOAT,    -- team's qb_hit rate per dropback
    ol_sack_rate         FLOAT,    -- team's sack rate per dropback
    -- Defensive pressure faced (opponent tendencies vs this team)
    opp_pressure_rate_pbp FLOAT,   -- opp qb_hits per dropback
    opp_sack_rate_pbp    FLOAT,    -- opp sacks per dropback
    -- Player-level sacks / pressure taken (QB)
    sacks_taken          INTEGER,  -- times this QB was sacked
    qb_hits_taken        INTEGER,  -- times this QB was hit (even without sack)
    PRIMARY KEY (player_id, game_id)
)
"""

CREATE_PBP_MATCHUPS = """
CREATE TABLE IF NOT EXISTS pbp_matchups (
    game_id              VARCHAR NOT NULL,
    season               INTEGER NOT NULL,
    week                 INTEGER NOT NULL,
    off_player_id        VARCHAR NOT NULL,  -- offensive skill player
    def_player_id        VARCHAR NOT NULL,  -- primary defender assigned to them
    snap_overlap         FLOAT,             -- fraction of off player snaps where def was on field
    target_overlap       INTEGER,           -- # of targets while this def was on field
    is_primary           BOOLEAN,           -- top-1 coverage assignment this game
    PRIMARY KEY (game_id, off_player_id, def_player_id)
)
"""

CREATE_FTN_PLAYER_GAME = """
CREATE TABLE IF NOT EXISTS ftn_player_game (
    player_id   TEXT NOT NULL,
    game_id     TEXT NOT NULL,
    season      INTEGER,
    week        INTEGER,
    drops       INTEGER DEFAULT 0,
    contested_catches INTEGER DEFAULT 0,
    targets     INTEGER DEFAULT 0,
    PRIMARY KEY (player_id, game_id)
);
CREATE INDEX IF NOT EXISTS idx_ftn_player_game_lookup
    ON ftn_player_game (player_id, season, week);
"""

CREATE_STAGING = """
CREATE TABLE IF NOT EXISTS staging_nflreadpy (
    id           SERIAL       PRIMARY KEY,
    source_type  VARCHAR(50)  NOT NULL,
    season       INTEGER      NOT NULL,
    week         INTEGER,
    raw_data     JSONB        NOT NULL,
    ingested_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    processed    BOOLEAN      NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_staging_nflreadpy_lookup
    ON staging_nflreadpy (source_type, season, week);
CREATE INDEX IF NOT EXISTS idx_staging_nflreadpy_unprocessed
    ON staging_nflreadpy (processed) WHERE NOT processed;
"""

CREATE_DEAD_LETTER = """
CREATE TABLE IF NOT EXISTS dead_letter (
    id             SERIAL       PRIMARY KEY,
    source         VARCHAR(100) NOT NULL,
    error_message  TEXT         NOT NULL,
    raw_payload    JSONB        NOT NULL,
    ingested_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
"""

CREATE_PROP_ODDS = """
CREATE TABLE IF NOT EXISTS prop_odds (
    id                SERIAL PRIMARY KEY,
    odds_event_id     TEXT NOT NULL,
    game_id           TEXT,
    player_name       TEXT,
    player_id         TEXT REFERENCES players(id),
    stat_type         TEXT NOT NULL,
    line_value        FLOAT,
    over_odds         FLOAT,
    under_odds        FLOAT,
    bookmaker         TEXT,
    fetched_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (odds_event_id, player_name, stat_type, bookmaker)
)
"""

CREATE_ALERTS = """
CREATE TABLE IF NOT EXISTS alerts (
    id           TEXT PRIMARY KEY,
    severity     TEXT        NOT NULL,
    title        TEXT        NOT NULL,
    body         TEXT        NOT NULL,
    player_id    TEXT,
    player_name  TEXT,
    stat         TEXT,
    value        DOUBLE PRECISION,
    timestamp    TIMESTAMPTZ NOT NULL
)
"""

CREATE_PROJECTIONS = """
CREATE TABLE IF NOT EXISTS projections (
    id               SERIAL PRIMARY KEY,
    player_id        VARCHAR NOT NULL,
    game_id          VARCHAR NOT NULL,
    season           INTEGER NOT NULL,
    week             INTEGER NOT NULL,
    stat             VARCHAR NOT NULL,
    position         VARCHAR,
    projection       FLOAT,
    floor            FLOAT,
    ceiling          FLOAT,
    p25              FLOAT,
    p75              FLOAT,
    boom_probability FLOAT,
    bust_probability FLOAT,
    fantasy_projection FLOAT,
    fantasy_floor    FLOAT,
    fantasy_ceiling  FLOAT,
    pipeline_run_id  VARCHAR,
    max_train_season INTEGER,
    posterior_samples JSONB,
    created_at       TIMESTAMP DEFAULT NOW(),
    CONSTRAINT uq_projections_player_game_stat
        UNIQUE (player_id, game_id, stat)
)
"""

ALL_DDL: tuple[str, ...] = (
    CREATE_PRODUCTION,
    CREATE_FEATURE_MATRIX,
    CREATE_INJURY_HISTORY,
    CREATE_PBP_FEATURES,
    CREATE_PBP_MATCHUPS,
    CREATE_FTN_PLAYER_GAME,
    CREATE_STAGING,
    CREATE_DEAD_LETTER,
    CREATE_PROP_ODDS,
    CREATE_ALERTS,
    CREATE_PROJECTIONS,
)
