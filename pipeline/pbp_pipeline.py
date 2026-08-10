"""
pipeline/pbp_pipeline.py

Play-by-Play (PBP) ingestion pipeline.

Loads nflreadpy.load_pbp() for seasons 2019-2025, aggregates to game-level
per-player stats, and writes to two tables:
  1. `pbp_features`  — per-player game-level stats (EPA, ADOT, YAC, pressure, etc.)
  2. `pbp_matchups`  — per-game coverage edges for GNN (off_player_id → def_player_id)

Also bulk-UPDATEs relevant Bucket 11 columns in `feature_matrix` so XGB/LGB/TFT
can use EPA, pressure, drop_rate, and direction features starting from this training run.

Run AFTER pipeline.orchestrator (so feature_matrix exists):
    python -m pipeline.pbp_pipeline

Runtime estimate: ~15-30 min for 2019-2025 (nflreadpy caches PBP locally after first load).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pbp_pipeline")

SEASONS = list(range(2019, 2027))  # 2019–2026


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _get_conn():
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        raise RuntimeError("DATABASE_URL not set.")
    return psycopg2.connect(dsn.replace("postgresql+asyncpg://", "postgresql://"))


# ── Schema setup ───────────────────────────────────────────────────────────────

_PBP_FEATURES_DDL = """
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

_PBP_MATCHUPS_DDL = """
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

_FTN_PLAYER_GAME_DDL = """
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


def _ensure_schema(conn) -> None:
    from pipeline.schema import ensure_schema

    ensure_schema(conn)
    logger.info("pbp_features, pbp_matchups, ftn_player_game tables ensured via schema registry.")


# ── PBP aggregation ────────────────────────────────────────────────────────────

def _load_pbp_season(season: int) -> pd.DataFrame:
    """Load one season of PBP data from nflreadpy, return as pandas DataFrame."""
    import nflreadpy
    import warnings
    warnings.filterwarnings("ignore")
    logger.info("  Loading PBP season %d...", season)
    df = nflreadpy.load_pbp([season])
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    logger.info("  Season %d: %d plays loaded.", season, len(df))
    return df

def _load_ftn_charting_season(season: int) -> pd.DataFrame:
    """Load FTN charting (play-level drops, contested) for join with PBP."""
    import nflreadpy
    import warnings
    warnings.filterwarnings("ignore")
    try:
        df = nflreadpy.load_ftn_charting(seasons=season)
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()
        logger.info("  FTN charting season %d: %d plays", season, len(df))
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception as exc:
        logger.warning("  FTN charting season %d failed: %s", season, exc)
        return pd.DataFrame()


def _load_participation_season(season: int) -> pd.DataFrame:
    """Load participation data to get defensive coverage scheme."""
    import nflreadpy
    import warnings
    warnings.filterwarnings("ignore")
    logger.info("  Loading Participation season %d...", season)
    try:
        df = nflreadpy.load_participation([season])
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()
        logger.info("  Season %d: %d participation records loaded.", season, len(df))
        return df
    except Exception as exc:
        logger.warning("  Season %d participation load failed: %s", season, exc)
        return pd.DataFrame()


def _load_pfr_drop_rates(seasons: list[int]) -> pd.DataFrame:
    """
    Load official drop rates from Pro Football Reference via nflreadpy.

    nflreadpy.load_pfr_advstats(stat_type='rec') returns PFR's advanced
    receiving stats including 'drops' and 'drop_pct' columns.

    THIS IS A LIVE PRO-FOOTBALL-REFERENCE DEPENDENCY. `scraper.adapters.
    pro_football_ref` is retired, but that retirement covers direct scraping
    only — this call still ingests PFR-derived data through nflverse's
    redistributed copy, and `drop_rate` on `player_pbp_features` comes from it.
    Any claim that the project no longer uses PFR is false while this call
    exists (audit C-25). See `docs/DATA_SOURCES.md`. Retiring it means dropping
    `drop_rate` or sourcing it elsewhere (FTN charting already supersedes it
    where available — see the FTN merge below), not deleting this docstring.

    Returns a DataFrame with columns:
        player_id       (pfr_player_id or closest match)
        season, week    (for merging by player-season-week)
        drop_rate       (drops / targets, [0.0, 1.0])

    If the load fails for any reason, returns an empty DataFrame so the
    pipeline continues with NULL drop_rate (better than crashing).
    """
    try:
        import nflreadpy as nfl
        dfs = []
        for season in seasons:
            try:
                df = nfl.load_pfr_advstats(seasons=season, stat_type="rec")
                if hasattr(df, "to_pandas"):
                    df = df.to_pandas()
                if df is None or len(df) == 0:
                    continue
                # PFR columns: 'drops' (count) and 'drop_pct' (decimal fraction)
                # Use drop_pct directly if available; else compute drops/targets.
                if "drop_pct" in df.columns:
                    df["drop_rate"] = pd.to_numeric(df["drop_pct"], errors="coerce") / 100.0
                elif "drops" in df.columns and "targets" in df.columns:
                    df["drop_rate"] = np.where(
                        pd.to_numeric(df["targets"], errors="coerce").fillna(0) > 0,
                        pd.to_numeric(df["drops"], errors="coerce").fillna(0) /
                        pd.to_numeric(df["targets"], errors="coerce").fillna(1),
                        np.nan,
                    )
                else:
                    logger.warning("PFR advstats season %d missing drops/drop_pct columns", season)
                    continue
                df["season"] = int(season)
                keep_cols = [c for c in ["pfr_id", "player_id", "player_name", "season", "week", "drop_rate"] if c in df.columns]
                dfs.append(df[keep_cols])
            except Exception as e:
                logger.warning("PFR advstats season %d failed: %s", season, e)
        if not dfs:
            return pd.DataFrame()
        result = pd.concat(dfs, ignore_index=True)
        logger.info("PFR drop rates loaded: %d player-season-week rows", len(result))
        return result
    except Exception as exc:
        logger.warning("PFR advstats load failed entirely (drop_rate will be NULL): %s", exc)
        return pd.DataFrame()


def _aggregate_pbp(pbp: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate raw PBP (one row per play) to:
      - features_df: one row per (player_id, game_id) with all Bucket 11 columns
      - matchups_df: one row per (game_id, off_player_id, def_player_id) for GNN

    Returns: (features_df, matchups_df)
    """
    # ── Filter to active plays only ─────────────────────────────────────────
    pbp = pbp[pbp["play_type"].isin(["pass", "run", "qb_scramble"])].copy()
    pbp["season"] = season

    # ── Receiving stats (per receiver) ──────────────────────────────────────
    rec_plays = pbp[pbp["receiver_player_id"].notna()].copy()
    rec_agg = (
        rec_plays.groupby(["receiver_player_id", "game_id", "week", "posteam"])
        .agg(
            targets=("pass_attempt", "sum"),
            receptions=("complete_pass", "sum"),
            total_air_yards=("air_yards", "sum"),
            total_yac=("yards_after_catch", "sum"),
            total_xyac=("xyac_mean_yardage", "sum"),
            total_epa=("epa", "sum"),
            red_zone_targets=("yardline_100", lambda x: (x <= 20).sum()),
            end_zone_targets=("yardline_100", lambda x: (x <= 10).sum()),
            pass_left=(   "pass_location", lambda x: (x == "left").sum()),
            pass_middle=( "pass_location", lambda x: (x == "middle").sum()),
            pass_right=(  "pass_location", lambda x: (x == "right").sum()),
        )
        .reset_index()
        .rename(columns={"receiver_player_id": "player_id", "posteam": "team"})
    )
    rec_agg["adot"] = np.where(
        rec_agg["targets"] > 0, rec_agg["total_air_yards"] / rec_agg["targets"], np.nan
    )
    rec_agg["yac_per_reception"] = np.where(
        rec_agg["receptions"] > 0, rec_agg["total_yac"] / rec_agg["receptions"], np.nan
    )
    rec_agg["xyac_per_reception"] = np.where(
        rec_agg["receptions"] > 0, rec_agg["total_xyac"] / rec_agg["receptions"], np.nan
    )
    rec_agg["epa_per_target"] = np.where(
        rec_agg["targets"] > 0, rec_agg["total_epa"] / rec_agg["targets"], np.nan
    )
    total_targets = (
        pbp[pbp["pass_attempt"] == 1]
        .groupby(["game_id", "posteam"])["pass_attempt"]
        .sum()
        .reset_index()
        .rename(columns={"pass_attempt": "team_targets", "posteam": "team"})
    )
    rec_agg = rec_agg.merge(total_targets, on=["game_id", "team"], how="left")
    rec_agg["target_share_pbp"] = np.where(
        rec_agg["team_targets"] > 0,
        rec_agg["targets"] / rec_agg["team_targets"],
        np.nan,
    )
    team_air = (
        pbp[pbp["air_yards"].notna()]
        .groupby(["game_id", "posteam"])["air_yards"]
        .sum()
        .reset_index()
        .rename(columns={"air_yards": "team_air_yards", "posteam": "team"})
    )
    rec_agg = rec_agg.merge(team_air, on=["game_id", "team"], how="left")
    rec_agg["air_yards_share_pbp"] = np.where(
        rec_agg["team_air_yards"] > 0,
        rec_agg["total_air_yards"] / rec_agg["team_air_yards"],
        np.nan,
    )
    # Red zone target share
    team_rz = (
        pbp[(pbp["pass_attempt"] == 1) & (pbp["yardline_100"] <= 20)]
        .groupby(["game_id", "posteam"])["pass_attempt"]
        .sum()
        .reset_index()
        .rename(columns={"pass_attempt": "team_rz_targets", "posteam": "team"})
    )
    rec_agg = rec_agg.merge(team_rz, on=["game_id", "team"], how="left")
    rec_agg["red_zone_target_share"] = np.where(
        rec_agg["team_rz_targets"] > 0,
        rec_agg["red_zone_targets"] / rec_agg["team_rz_targets"],
        np.nan,
    )
    # Direction rates
    for col in ["pass_left_rate", "pass_middle_rate", "pass_right_rate"]:
        direction = col.replace("pass_", "").replace("_rate", "")
        rec_agg[col] = np.where(
            rec_agg["targets"] > 0,
            rec_agg[f"pass_{direction}"] / rec_agg["targets"],
            np.nan,
        )

    # ── Rushing stats (per rusher) ───────────────────────────────────────────
    rush_plays = pbp[pbp["rusher_player_id"].notna()].copy()
    rush_agg = (
        rush_plays.groupby(["rusher_player_id", "game_id", "posteam"])
        .agg(rush_epa=("epa", "sum"), rush_attempts=("rush_attempt", "sum"))
        .reset_index()
        .rename(columns={"rusher_player_id": "player_id", "posteam": "team"})
    )
    rush_agg["epa_per_rush"] = np.where(
        rush_agg["rush_attempts"] > 0,
        rush_agg["rush_epa"] / rush_agg["rush_attempts"],
        np.nan,
    )

    # ── QB stats (per passer) ────────────────────────────────────────────────
    pass_plays = pbp[pbp["passer_player_id"].notna()].copy()
    qb_agg = (
        pass_plays.groupby(["passer_player_id", "game_id", "posteam"])
        .agg(
            qb_epa=("qb_epa", "sum"),
            dropbacks=("qb_dropback", "sum"),
            sacks_taken=("sack", "sum"),
            qb_hits_taken=("qb_hit", "sum"),
        )
        .reset_index()
        .rename(columns={"passer_player_id": "player_id", "posteam": "team"})
    )
    qb_agg["qb_epa_per_dropback"] = np.where(
        qb_agg["dropbacks"] > 0, qb_agg["qb_epa"] / qb_agg["dropbacks"], np.nan
    )

    # ── OL + Defensive pressure (team level) ────────────────────────────────
    team_pressure = (
        pbp[pbp["qb_dropback"] == 1]
        .groupby(["game_id", "posteam"])
        .agg(
            team_dropbacks=("qb_dropback", "sum"),
            team_qb_hits=("qb_hit", "sum"),
            team_sacks=("sack", "sum"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    team_pressure["ol_pressure_rate"] = np.where(
        team_pressure["team_dropbacks"] > 0,
        team_pressure["team_qb_hits"] / team_pressure["team_dropbacks"],
        np.nan,
    )
    team_pressure["ol_sack_rate"] = np.where(
        team_pressure["team_dropbacks"] > 0,
        team_pressure["team_sacks"] / team_pressure["team_dropbacks"],
        np.nan,
    )
    # Defensive pressure from opponent's perspective
    if "defense_man_zone_type" not in pbp.columns:
        pbp["defense_man_zone_type"] = pd.Series(dtype=str)
    if "number_of_pass_rushers" not in pbp.columns:
        pbp["number_of_pass_rushers"] = pd.Series(dtype=float)

    def_pressure = (
        pbp[pbp["qb_dropback"] == 1]
        .groupby(["game_id", "week", "defteam"])
        .agg(
            def_dropbacks=("qb_dropback", "sum"), 
            def_hits=("qb_hit", "sum"), 
            def_sacks=("sack", "sum"),
            def_zone_plays=("defense_man_zone_type", lambda x: x.isin(["ZONE_COVERAGE"]).sum()),
            def_man_plays=("defense_man_zone_type", lambda x: x.isin(["MAN_COVERAGE"]).sum()),
            def_blitz_plays=("number_of_pass_rushers", lambda x: (x > 4).sum()),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
        .sort_values(["team", "week"])
    )

    # Compute trailing cumulative sums (shift by 1 to prevent current game leakage)
    shifted_dropbacks = def_pressure.groupby("team")["def_dropbacks"].transform(lambda x: x.shift().cumsum())
    shifted_hits =      def_pressure.groupby("team")["def_hits"].transform(lambda x: x.shift().cumsum())
    shifted_sacks =     def_pressure.groupby("team")["def_sacks"].transform(lambda x: x.shift().cumsum())
    shifted_zone =      def_pressure.groupby("team")["def_zone_plays"].transform(lambda x: x.shift().cumsum())
    shifted_man =       def_pressure.groupby("team")["def_man_plays"].transform(lambda x: x.shift().cumsum())
    shifted_blitz =     def_pressure.groupby("team")["def_blitz_plays"].transform(lambda x: x.shift().cumsum())
    def_pressure["opp_pressure_rate_pbp"] = np.where(
        shifted_dropbacks > 0,
        shifted_hits / shifted_dropbacks,
        np.nan,
    )
    def_pressure["opp_sack_rate_pbp"] = np.where(
        shifted_dropbacks > 0,
        shifted_sacks / shifted_dropbacks,
        np.nan,
    )
    def_pressure["opp_zone_pct"] = np.where(
        shifted_dropbacks > 0,
        shifted_zone / shifted_dropbacks,
        np.nan,
    )
    def_pressure["opp_man_pct"] = np.where(
        shifted_dropbacks > 0,
        shifted_man / shifted_dropbacks,
        np.nan,
    )
    def_pressure["opp_blitz_rate"] = np.where(
        shifted_dropbacks > 0,
        shifted_blitz / shifted_dropbacks,
        np.nan,
    )

    # ── Merge all player-level aggs into features_df ─────────────────────────
    # Start from receiving agg (largest player set), merge rush + QB stats
    features = rec_agg[["player_id", "game_id", "week", "team",
                         "adot", "total_air_yards", "yac_per_reception",
                         "xyac_per_reception", "epa_per_target",
                         "target_share_pbp", "air_yards_share_pbp",
                         "red_zone_targets", "end_zone_targets",
                         "red_zone_target_share",
                         "pass_left_rate", "pass_middle_rate", "pass_right_rate"]].copy()

    features = features.merge(
        rush_agg[["player_id", "game_id", "epa_per_rush"]],
        on=["player_id", "game_id"], how="outer"
    )
    features = features.merge(
        qb_agg[["player_id", "game_id", "qb_epa_per_dropback", "sacks_taken", "qb_hits_taken"]],
        on=["player_id", "game_id"], how="outer"
    )
    features = features.merge(
        team_pressure[["game_id", "team", "ol_pressure_rate", "ol_sack_rate"]],
        on=["game_id", "team"], how="left"
    )
    features = features.merge(
        def_pressure[["game_id", "team", "opp_pressure_rate_pbp", "opp_sack_rate_pbp", "opp_zone_pct", "opp_man_pct", "opp_blitz_rate"]],
        on=["game_id", "team"], how="left"
    )

    # Overall EPA per play (union of all player snaps)
    all_player_epa = pd.concat([
        rec_agg[["player_id", "game_id", "total_epa"]].rename(columns={"total_epa": "epa"}),
        rush_agg[["player_id", "game_id", "rush_epa"]].rename(columns={"rush_epa": "epa"}),
    ]).groupby(["player_id", "game_id"])["epa"].sum().reset_index()
    all_plays_per_player = pd.concat([
        rec_agg[["player_id", "game_id", "targets"]].rename(columns={"targets": "plays"}),
        rush_agg[["player_id", "game_id", "rush_attempts"]].rename(columns={"rush_attempts": "plays"}),
    ]).groupby(["player_id", "game_id"])["plays"].sum().reset_index()
    epa_per_play = all_player_epa.merge(all_plays_per_player, on=["player_id", "game_id"])
    epa_per_play["epa_per_play"] = np.where(
        epa_per_play["plays"] > 0, epa_per_play["epa"] / epa_per_play["plays"], np.nan
    )
    features = features.merge(epa_per_play[["player_id", "game_id", "epa_per_play"]], on=["player_id", "game_id"], how="left")
    
    # Ensure week is populated for all rows, extracting from game_id (e.g. "2019_01_PIT_NE") if missing
    if "week" not in features.columns:
        features["week"] = features["game_id"].str.split("_").str[1].astype(int)
    else:
        features["week"] = features["week"].fillna(features["game_id"].str.split("_").str[1].astype(float)).astype(int)
        
    features["season"] = season

    # ── Vectorized GNN matchup edges ─────────────────────────────────
    # Vectorized via pd.melt rather than iterrows for 50-100x speedup.
    edge_cols_present = [
        c for c in ["pass_defense_1_player_id", "pass_defense_2_player_id"]
        if c in pbp.columns
    ]
    pass_plays_with_rcvr = pbp[
        pbp["receiver_player_id"].notna()
    ][["game_id", "week", "receiver_player_id"] + edge_cols_present].copy()

    if edge_cols_present and not pass_plays_with_rcvr.empty:
        melted = pass_plays_with_rcvr.melt(
            id_vars=["game_id", "week", "receiver_player_id"],
            value_vars=edge_cols_present,
            value_name="def_player_id",
        ).dropna(subset=["def_player_id"])
        melted["season"] = season
        melted = melted.rename(columns={"receiver_player_id": "off_player_id"})
        matchup_agg = (
            melted.groupby(["game_id", "season", "week", "off_player_id", "def_player_id"])
            .size()
            .reset_index(name="target_overlap")
        )
        matchup_agg["is_primary"] = (
            matchup_agg.groupby(["game_id", "off_player_id"])["target_overlap"].transform("max")
            == matchup_agg["target_overlap"]
        )
        group_sum = matchup_agg.groupby(["game_id", "off_player_id"])["target_overlap"].transform("sum").clip(lower=1)
        matchup_agg["snap_overlap"] = matchup_agg["target_overlap"] / group_sum
    else:
        matchup_agg = pd.DataFrame(columns=[
            "game_id", "season", "week", "off_player_id",
            "def_player_id", "target_overlap", "is_primary", "snap_overlap",
        ])

    # Merge PFR drop rates (season-level cumulative; season int → wrapped in list for the function sig)
    drop_rates = _load_pfr_drop_rates([season])
    if not drop_rates.empty and "drop_rate" in drop_rates.columns:
        merge_keys = [k for k in ["player_id", "season", "week"] if k in drop_rates.columns]
        features = features.merge(
            drop_rates[merge_keys + ["drop_rate"]].drop_duplicates(subset=merge_keys),
            on=merge_keys, how="left",
        )

    return features, matchup_agg


# ── FTN + PBP join → ftn_player_game ─────────────────────────────────────────

def _aggregate_ftn_player_game(pbp: pd.DataFrame, ftn: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Join FTN charting (is_drop, is_contested_ball) with PBP (receiver_player_id) on
    (game_id, play_id), then aggregate to (player_id, game_id, season, week) with
    drops, contested_catches, targets.
    """
    if ftn.empty or "receiver_player_id" not in pbp.columns:
        return pd.DataFrame()

    # Align play_id types (PBP may be float, FTN int)
    ftn = ftn.copy()
    ftn["play_id"] = pd.to_numeric(ftn["nflverse_play_id"], errors="coerce")
    pbp_rec = pbp[pbp["receiver_player_id"].notna()][
        ["game_id", "play_id", "receiver_player_id", "week"]
    ].copy()
    pbp_rec["play_id"] = pd.to_numeric(pbp_rec["play_id"], errors="coerce")

    ftn_sub = ftn[["nflverse_game_id", "play_id", "is_drop", "is_contested_ball"]].copy()
    ftn_sub = ftn_sub.rename(columns={"nflverse_game_id": "game_id"})
    merged = pbp_rec.merge(
        ftn_sub,
        on=["game_id", "play_id"],
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame()

    def _sum_bool(s: pd.Series) -> int:
        return int(s.fillna(False).astype(bool).sum())

    agg = (
        merged.groupby(["receiver_player_id", "game_id", "week"])
        .agg(
            drops=("is_drop", _sum_bool),
            contested_catches=("is_contested_ball", _sum_bool),
            targets=("play_id", "count"),
        )
        .reset_index()
    )
    agg = agg.rename(columns={"receiver_player_id": "player_id"})
    agg["season"] = season
    agg["drops"] = agg["drops"].fillna(0).astype(int)
    agg["contested_catches"] = agg["contested_catches"].fillna(0).astype(int)
    agg["targets"] = agg["targets"].fillna(0).astype(int)
    return agg


def _write_ftn_player_game(conn, agg_df: pd.DataFrame) -> int:
    """Upsert ftn_player_game from FTN+PBP join aggregate."""
    if agg_df.empty:
        return 0
    cur = conn.cursor()
    upsert_sql = """
        INSERT INTO ftn_player_game (player_id, game_id, season, week, drops, contested_catches, targets)
        VALUES %s
        ON CONFLICT (player_id, game_id) DO UPDATE SET
            season = EXCLUDED.season,
            week = EXCLUDED.week,
            drops = EXCLUDED.drops,
            contested_catches = EXCLUDED.contested_catches,
            targets = EXCLUDED.targets
    """
    rows = [
        (
            str(r["player_id"]),
            str(r["game_id"]),
            int(r["season"]) if pd.notna(r["season"]) else None,
            int(r["week"]) if pd.notna(r["week"]) else None,
            int(r["drops"]),
            int(r["contested_catches"]),
            int(r["targets"]),
        )
        for _, r in agg_df.iterrows()
    ]
    psycopg2.extras.execute_values(cur, upsert_sql, rows, page_size=2000)
    conn.commit()
    return len(rows)


# ── DB writes ──────────────────────────────────────────────────────────────────

def _write_pbp_features(conn, features_df: pd.DataFrame) -> int:
    cur = conn.cursor()
    col_order = [
        "player_id", "game_id", "season", "week", "team",
        "epa_per_play", "epa_per_target", "epa_per_rush", "qb_epa_per_dropback",
        "adot", "total_air_yards", "yac_per_reception", "xyac_per_reception",
        "target_share_pbp", "air_yards_share_pbp",
        "red_zone_targets", "end_zone_targets", "red_zone_target_share",
        "pass_left_rate", "pass_middle_rate", "pass_right_rate",
        "drop_rate",
        "ol_pressure_rate", "ol_sack_rate",
        "opp_pressure_rate_pbp", "opp_sack_rate_pbp",
        "sacks_taken", "qb_hits_taken",
    ]
    for col in col_order:
        if col not in features_df.columns:
            features_df[col] = None

    upsert_sql = """
        INSERT INTO pbp_features (
            player_id, game_id, season, week, team,
            epa_per_play, epa_per_target, epa_per_rush, qb_epa_per_dropback,
            adot, total_air_yards, yac_per_reception, xyac_per_reception,
            target_share_pbp, air_yards_share_pbp,
            red_zone_targets, end_zone_targets, red_zone_target_share,
            pass_left_rate, pass_middle_rate, pass_right_rate,
            drop_rate,
            ol_pressure_rate, ol_sack_rate,
            opp_pressure_rate_pbp, opp_sack_rate_pbp,
            sacks_taken, qb_hits_taken
        )
        VALUES %s
        ON CONFLICT (player_id, game_id) DO UPDATE SET
            epa_per_play         = EXCLUDED.epa_per_play,
            epa_per_target       = EXCLUDED.epa_per_target,
            epa_per_rush         = EXCLUDED.epa_per_rush,
            qb_epa_per_dropback  = EXCLUDED.qb_epa_per_dropback,
            adot                 = EXCLUDED.adot,
            total_air_yards      = EXCLUDED.total_air_yards,
            yac_per_reception    = EXCLUDED.yac_per_reception,
            xyac_per_reception   = EXCLUDED.xyac_per_reception,
            target_share_pbp     = EXCLUDED.target_share_pbp,
            air_yards_share_pbp  = EXCLUDED.air_yards_share_pbp,
            red_zone_targets     = EXCLUDED.red_zone_targets,
            end_zone_targets     = EXCLUDED.end_zone_targets,
            red_zone_target_share = EXCLUDED.red_zone_target_share,
            pass_left_rate       = EXCLUDED.pass_left_rate,
            pass_middle_rate     = EXCLUDED.pass_middle_rate,
            pass_right_rate      = EXCLUDED.pass_right_rate,
            drop_rate            = EXCLUDED.drop_rate,
            ol_pressure_rate     = EXCLUDED.ol_pressure_rate,
            ol_sack_rate         = EXCLUDED.ol_sack_rate,
            opp_pressure_rate_pbp = EXCLUDED.opp_pressure_rate_pbp,
            opp_sack_rate_pbp    = EXCLUDED.opp_sack_rate_pbp,
            sacks_taken          = EXCLUDED.sacks_taken,
            qb_hits_taken        = EXCLUDED.qb_hits_taken
    """
    rows = [
        tuple(
            _safe(features_df.iloc[i][c])
            for c in col_order
        )
        for i in range(len(features_df))
    ]
    psycopg2.extras.execute_values(cur, upsert_sql, rows, page_size=2000)
    conn.commit()
    return len(rows)


def _write_matchups(conn, matchups_df: pd.DataFrame) -> int:
    if matchups_df.empty:
        return 0
    cur = conn.cursor()
    upsert_sql = """
        INSERT INTO pbp_matchups
            (game_id, season, week, off_player_id, def_player_id,
             snap_overlap, target_overlap, is_primary)
        VALUES %s
        ON CONFLICT (game_id, off_player_id, def_player_id) DO UPDATE SET
            snap_overlap   = EXCLUDED.snap_overlap,
            target_overlap = EXCLUDED.target_overlap,
            is_primary     = EXCLUDED.is_primary
    """
    rows = [
        (str(r["game_id"]), int(r["season"]), int(r["week"]),
         str(r["off_player_id"]), str(r["def_player_id"]),
         _safe(r.get("snap_overlap")), int(r["target_overlap"]),
         bool(r.get("is_primary", False)))
        for _, r in matchups_df.iterrows()
    ]
    psycopg2.extras.execute_values(cur, upsert_sql, rows, page_size=2000)
    conn.commit()
    return len(rows)


def _update_feature_matrix(conn, features_df: pd.DataFrame) -> None:
    """
    Bulk-UPDATE Bucket 11 PBP columns in feature_matrix for every matching
    (player_id, game_id) row. Adds columns if they don't exist yet.
    """
    cur = conn.cursor()
    pbp_cols = [
        "epa_per_play", "epa_per_target", "epa_per_rush", "qb_epa_per_dropback",
        "adot", "yac_per_reception", "xyac_per_reception",
        "target_share_pbp", "air_yards_share_pbp",
        "red_zone_targets", "end_zone_targets", "red_zone_target_share",
        "pass_left_rate", "pass_middle_rate", "pass_right_rate",
        "drop_rate", "ol_pressure_rate", "ol_sack_rate",
        "opp_pressure_rate_pbp", "opp_sack_rate_pbp",
        "opp_zone_pct", "opp_man_pct", "opp_blitz_rate",
        "routes_run_per_game", "slot_rate",
    ]
    for col in pbp_cols:
        cur.execute(f"ALTER TABLE feature_matrix ADD COLUMN IF NOT EXISTS {col} FLOAT")
    conn.commit()

    set_clause = ", ".join([f"{c} = %({c})s" for c in pbp_cols if c in features_df.columns])
    update_sql = f"""
        UPDATE feature_matrix SET {set_clause}
        WHERE player_id = %(player_id)s AND game_id = %(game_id)s
    """

    batch, n_updated = [], 0
    for _, row in features_df.iterrows():
        r = {"player_id": str(row["player_id"]), "game_id": str(row["game_id"])}
        for c in pbp_cols:
            r[c] = _safe(row.get(c)) if c in row.index else None
        batch.append(r)
        if len(batch) >= 5000:
            psycopg2.extras.execute_batch(cur, update_sql, batch)
            conn.commit()
            n_updated += len(batch)
            logger.info("  feature_matrix PBP update: %d rows...", n_updated)
            batch = []
    if batch:
        psycopg2.extras.execute_batch(cur, update_sql, batch)
        conn.commit()
        n_updated += len(batch)
    logger.info("✓ feature_matrix PBP columns updated: %d rows.", n_updated)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe(v):
    if v is None:
        return None
    try:
        import math
        import numpy as np
        if isinstance(v, (np.integer, np.int64, np.int32)):
            return int(v)
        if isinstance(v, (np.floating, np.float64, np.float32)):
            if np.isnan(v):
                return None
            return float(v)
        if isinstance(v, float) and math.isnan(v):
            return None
        return v
    except Exception:
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("  PBP Pipeline — Seasons 2019–2025")
    logger.info("=" * 60)

    conn = _get_conn()
    try:
        _ensure_schema(conn)

        total_features, total_matchups, total_ftn = 0, 0, 0
        all_features = []

        for season in SEASONS:
            try:
                pbp = _load_pbp_season(season)
                part = _load_participation_season(season)
                if part is not None and not part.empty:
                    part_cols = ["nflverse_game_id", "play_id", "defense_man_zone_type", "number_of_pass_rushers", "defenders_in_box"]
                    part_cols = [c for c in part_cols if c in part.columns]
                    part_subset = part[part_cols].rename(columns={"nflverse_game_id": "game_id"}).drop_duplicates(subset=["game_id", "play_id"])
                    pbp = pbp.merge(part_subset, on=["game_id", "play_id"], how="left")

                ftn = _load_ftn_charting_season(season)
                ftn_agg = pd.DataFrame()
                if not ftn.empty:
                    ftn_agg = _aggregate_ftn_player_game(pbp, ftn, season)
                    n_ftn = _write_ftn_player_game(conn, ftn_agg)
                    total_ftn += n_ftn
                    logger.info("Season %d: %d ftn_player_game rows", season, n_ftn)

                features_df, matchups_df = _aggregate_pbp(pbp, season)
                # Prefer FTN drop_rate over PFR when available (game-level, charted)
                if not ftn_agg.empty and "drops" in ftn_agg.columns and "targets" in ftn_agg.columns:
                    ftn_dr = ftn_agg[["player_id", "game_id", "drops", "targets"]].copy()
                    ftn_dr["drop_rate_ftn"] = np.where(
                        ftn_dr["targets"] > 0,
                        ftn_dr["drops"].astype(float) / ftn_dr["targets"],
                        np.nan,
                    )
                    ftn_dr = ftn_dr[["player_id", "game_id", "drop_rate_ftn"]].drop_duplicates(
                        subset=["player_id", "game_id"]
                    )
                    features_df = features_df.merge(ftn_dr, on=["player_id", "game_id"], how="left")
                    features_df["drop_rate"] = features_df["drop_rate_ftn"].combine_first(
                        features_df["drop_rate"]
                    )
                    features_df = features_df.drop(columns=["drop_rate_ftn"], errors="ignore")

                n_feat = _write_pbp_features(conn, features_df)
                n_match = _write_matchups(conn, matchups_df)
                total_features += n_feat
                total_matchups += n_match
                all_features.append(features_df)
                logger.info(
                    "Season %d: %d pbp_feature rows, %d matchup edges",
                    season, n_feat, n_match
                )
            except Exception as e:
                logger.warning("Season %d PBP failed: %s — skipping.", season, e)

        # Update feature_matrix with all PBP-derived Bucket 11 columns
        if all_features:
            combined = pd.concat(all_features, ignore_index=True)
            _update_feature_matrix(conn, combined)

    finally:
        conn.close()

    logger.info("=" * 60)
    logger.info(
        "  PBP DONE — %d feature rows, %d matchup edges, %d ftn_player_game",
        total_features, total_matchups, total_ftn
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
