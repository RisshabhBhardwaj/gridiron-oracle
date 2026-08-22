"""
ml/utils.py

Shared utilities for ML stacking ensemble models (XGBoost, LightGBM, TFT).
Centralizes data loading, metric computation, fold generation, and OOF saving.

THREAD SAFETY (Apple Silicon / macOS ARM)
-----------------------------------------
XGBoost, LightGBM, and PyTorch each ship with their own OpenMP runtime
(libomp). When multiple runtimes are loaded in the same process they collide
in the OpenMP thread barrier → SIGSEGV (exit code 139). configure_thread_env()
sets the necessary env vars BEFORE any of those libraries are imported.

Rules:
  1. Call configure_thread_env() at the top of every ml/ module, before any
     import of xgboost / lightgbm / torch / pytorch_forecasting.
  2. The env vars must be set before the first dlopen() of libomp — setting
     them after the import has no effect.
  3. OMP_NUM_THREADS=1 forces all OMP runtimes to use a single thread,
     eliminating cross-runtime contention entirely.
"""

from __future__ import annotations

import hashlib
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ml.feature_contract import (
    assert_model_frame_contract,
    build_feature_registry,
)

logger = logging.getLogger(__name__)


def configure_thread_env() -> None:
    """
    Set OpenMP / threading env vars to prevent SIGSEGV on Apple Silicon.

    MUST be called at the top of every ml/ module, before importing
    xgboost, lightgbm, torch, or pytorch_forecasting. Setting these vars
    after the library import has no effect because libomp is already loaded.

    Variables set:
      KMP_DUPLICATE_LIB_OK=True  — allows coexistence of multiple OMP runtimes
                                    (Intel MKL-OMP + LLVM-OMP + Apple-OMP).
      OMP_NUM_THREADS=1          — caps every OMP runtime to 1 thread,
                                    eliminating cross-runtime contention.

    If torch is already importable, also calls torch.set_num_threads(1)
    to prevent PyTorch's thread pool from spawning workers that conflict
    with XGBoost's or LightGBM's thread pools.
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Cap PyTorch thread pool only if torch is already imported into sys.modules.
    # Using a try/import here would trigger torch's dlopen and defeat the
    # purpose of setting env vars before the library loads.
    import sys
    if "torch" in sys.modules:
        try:
            import torch  # noqa: PLC0415 — only called if already loaded
            torch.set_num_threads(1)
        except Exception:  # noqa: BLE001
            pass


# Apply immediately at module load — this file is imported before any ML lib.
configure_thread_env()


def suppress_training_warnings() -> None:
    """
    Suppress noisy warnings during training, stacking, and inference.

    Applied automatically when ml.utils is imported. Covers:
      - pandas PerformanceWarning (DataFrame fragmentation from column assignment)
      - pandas FutureWarning (fillna downcasting, infer_objects)
      - sklearn UserWarning (LGBM/XGB/CatBoost feature names when passing numpy)
      - ArviZ FutureWarning (library refactor notice)
      - PyTorch/sklearn deprecation and convergence noise
    """
    # pandas: DataFrame fragmentation (column assignment in loops)
    warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented.*")
    # pandas: fillna downcasting
    warnings.filterwarnings(
        "ignore",
        message=".*Downcasting object dtype arrays on .fillna.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*future.no_silent_downcasting.*",
        category=FutureWarning,
    )
    # sklearn: LGBM/XGB/CatBoost fitted with feature names, predict gets numpy
    warnings.filterwarnings(
        "ignore",
        message=".*does not have valid feature names.*",
        category=UserWarning,
    )
    # ArviZ: refactor notice
    warnings.filterwarnings(
        "ignore",
        message=".*ArviZ.*",
        category=FutureWarning,
    )
    # sklearn: RidgeCV/ElasticNetCV convergence (often benign)
    warnings.filterwarnings(
        "ignore",
        message=".*Ill-conditioned matrix.*",
        category=UserWarning,
    )


suppress_training_warnings()

# ── Shared Definitions ────────────────────────────────────────────────────────

# Pregame eligibility: at least one completed player game in the current
# season.  This deliberately does not condition on participation in the game
# being predicted; `seas_games_played` is assembled before target kickoff.
MIN_PRIOR_GAMES: int = 1

# LightGBM and CatBoost training (ml/lgbm_model.py, ml/catboost_model.py)
# fillna() missing features with this sentinel before .fit() — both libraries'
# sklearn APIs need a concrete float, not real NaN, and the trees learn split
# thresholds around it as a de facto "missing" indicator. Inference must feed
# the exact same sentinel for the exact same features, or the model silently
# treats "missing" as a real, in-range observed value. XGBoost is exempt: its
# training path (ml/xgb_model.py) never fillna's, relying on native NaN
# routing instead, so XGBoost inference must receive real NaN, not this value.
MISSING_VALUE_SENTINEL: float = -9999.0

FEATURE_COLS: list[str] = [
    # Bucket 1 — Kalman Form
    "kalman_est_receiving_yards", "kalman_est_receiving_tds", "kalman_est_targets",
    "kalman_est_receptions", "kalman_est_target_share", "kalman_est_red_zone_target_share",
    "kalman_est_air_yards_share",
    "kalman_est_fantasy_ppr", "kalman_est_carries", "kalman_est_rushing_yards",
    "kalman_est_rushing_tds", "kalman_est_pass_attempts", "kalman_est_completions",
    "kalman_est_interceptions", "kalman_est_fumbles", "kalman_est_passing_yards",
    "kalman_est_passing_tds",
    # Bucket 2 — Season Baseline
    "seas_games_played", "seas_avg_receiving_yards", "seas_avg_targets",
    "seas_avg_receptions", "seas_avg_target_share", "seas_avg_fantasy_ppr",
    "seas_yards_per_target", "seas_yards_per_reception",
    "seas_avg_carries", "seas_avg_rushing_yards", "seas_yards_per_carry",
    "seas_avg_attempts", "seas_avg_passing_yards", "seas_completion_pct",
    "seas_avg_passing_cpoe", "seas_avg_passing_epa",
    "seas_avg_receiving_epa", "seas_avg_rushing_epa",
    "seas_avg_racr", "seas_avg_wopr", "seas_avg_receiving_yac",
    # Bucket 3 — Matchup (opponent defensive stats)
    "opp_avg_receiving_yards_allowed", "opp_avg_targets_allowed",
    "opp_avg_tds_allowed", "opp_avg_fantasy_ppr_allowed",
    "opp_avg_rushing_yards_allowed",
    # Bucket 4 — immutable venue facts only. Historical weather needs a
    # timestamped forecast source and is disabled until recaptured.
    "is_dome", "surface_turf",
    # Bucket 5 — Team Context
    "game_total_line", "spread_line", "is_home",
    # Bucket 6 — Rest
    "days_rest", "is_short_week", "is_bye_prior",
    # Bucket 7 — Rule Coefficients
    "rule_coeff",
    # Bucket 8 — injury reports require an as-of publication timestamp.
    "games_missed_streak", "prior_snap_share",
    # Bucket 9 — Defensive Tendency (NULL until Phase 4 play-by-play data wired)
    # XGB/LGB treat NaN as missing (skipped splits); TFT fills with median.
    # When populated: season-to-date opponent coverage shell tendencies per game-week.
    "opp_zone_pct", "opp_man_pct", "opp_blitz_rate", "opp_pressure_rate",
    # Bucket 9 — Scheme Interaction Features (cross-terms; NULL when tendency data absent)
    # deep_matchup_score:      air_yards_share × (1 - opp_zone_pct)   → field-stretchers vs man
    # coverage_matchup_score:  target_share × opp_man_pct             → separators vs man coverage
    # blitz_exposure:          snap_pct_off × opp_blitz_rate           → slot receivers vs blitz
    # separation_demand_score: target_share × opp_man_pct × (1-zone)  → three-way matchup signal
    "deep_matchup_score", "coverage_matchup_score",
    "separation_demand_score",
    # Bucket 10 — Elo Ratings (NULL until ml/team_elo.py fitted from DB game results)
    # XGB/LGB skip NaN splits; TFT median-fills. Enrich via team_elo.enrich_features().
    # team_off_elo / opp_def_elo: captures season-to-date team quality trajectory.
    # elo_matchup_diff: single summary of offensive vs defensive advantage.
    # elo_implied_win_prob: calibrated game-level Vegas-style probability.
    "team_off_elo", "team_def_elo", "opp_off_elo", "opp_def_elo",
    "elo_matchup_diff", "elo_implied_win_prob",
    # TFT Static Covariates — physical profile (Item 0 fix)
    "height", "weight", "draft_round",
    # Velocity / trend features (computed in feature_engineer.build_feature_row)
    # target_share_trend: 3-game delta in kalman target share (+ve = trending up in role)
    "target_share_trend",
    # Positional depth signal — 1=WR1/RB1/TE1, 2=WR2, etc. (team rank by receiving volume)
    "team_pos_rank",
]

# Concrete, executable registry; a new default feature cannot bypass an as-of
# declaration simply by being appended to FEATURE_COLS.
FEATURE_REGISTRY = build_feature_registry(FEATURE_COLS)

# Phase 4 A/B feature groups — NOT in FEATURE_COLS until held-out delta is positive.
# Enable via ml.feature_groups.resolve_feature_cols(groups=[...]).
FEATURE_GROUP_OPP_ADJ_USAGE: list[str] = [
    "carry_share",
    "ts_vs_league",
    "rz_ts_vs_league",
    "carry_share_vs_league",
    "opp_adj_target_share",
]
FEATURE_GROUP_PACE_SCRIPT: list[str] = [
    "team_pace",
    "team_pass_rate",
    "expected_pass_attempts",
    "expected_pass_rate",
    "neutral_script_flag",
]
FEATURE_GROUP_PROGRESSION: list[str] = [
    "years_exp",
    "age",
    "career_games",
    "exp_bucket",
]
# Phase 3 — legal lagged (strictly-prior-games) counterparts of the
# permanently-forbidden contemporaneous PBP/NGS fields, plus routes_run_per_game
# which was already computed this way but never wired to any group. Covers
# every PBP-family field with a real source in pbp_features/nextgen_stats
# (pipeline/pbp_pipeline.py) — the only ones NOT here are weather (observed,
# not forecast) and the xFP family (fantasy_points_exp etc.), which has no
# source anywhere in the pipeline to lag from. See feature_engineer.
# _fill_prior_pbp_ngs_features / _fill_prior_routes_run.
FEATURE_GROUP_TRAILING_PBP_NGS: list[str] = [
    "prior_epa_per_play",
    "prior_epa_per_target",
    "prior_epa_per_rush",
    "prior_qb_epa_per_dropback",
    "prior_adot",
    "prior_drop_rate",
    "prior_avg_separation",
    "prior_avg_cushion",
    "prior_target_share_pbp",
    "prior_air_yards_share_pbp",
    "prior_red_zone_targets",
    "prior_end_zone_targets",
    "prior_red_zone_target_share",
    "prior_pass_left_rate",
    "prior_pass_middle_rate",
    "prior_pass_right_rate",
    "prior_ol_pressure_rate",
    "prior_ol_sack_rate",
    "prior_yac_per_reception",
    "prior_xyac_per_reception",
    "routes_run_per_game",
]
FEATURE_GROUPS: dict[str, list[str]] = {
    "opp_adj_usage": FEATURE_GROUP_OPP_ADJ_USAGE,
    "pace_script": FEATURE_GROUP_PACE_SCRIPT,
    "progression": FEATURE_GROUP_PROGRESSION,
    "trailing_pbp_ngs": FEATURE_GROUP_TRAILING_PBP_NGS,
}

# Naming convention: TARGET_COL_MAP maps a MODEL STAT NAME (what we train
# the model to predict, also the column key in --target CLI flags) to its
# ACTUAL OUTCOME COLUMN in feature_matrix (where the real game result is stored).
#
# Why two names? feature_matrix stores BOTH:
#   - Kalman estimates (inputs, e.g. kalman_est_passing_yards)
#   - Actual outcomes (targets, e.g. actual_passing_yards)
# The model sees estimates as features and actual_* as the training label.
# This separation makes the pipeline fully causal (no future leakage).
#
# All 7 NFL seasons are supported: 2019, 2020, 2021, 2022, 2023, 2024, 2025.
TARGET_COL_MAP: dict[str, str] = {
    # ── Shared (all positions produce these via rushing/receiving/passing) ──────
    "receiving_yards":   "actual_receiving_yards",
    "rushing_yards":     "actual_rushing_yards",
    "passing_yards":     "actual_passing_yards",
    "fantasy_ppr":       "actual_fantasy_ppr",
    # ── Receiving stats (WR / TE / RB catching) ────────────────────────────────
    "receptions":        "actual_receptions",
    "receiving_tds":     "actual_receiving_tds",
    "targets":           "actual_targets",
    # ── Rushing stats (RB / QB scramble / WR jet sweep) ───────────────────────
    "carries":           "actual_carries",
    "rushing_tds":       "actual_rushing_tds",
    # ── QB-specific stats ──────────────────────────────────────────────────────
    "pass_attempts":     "actual_pass_attempts",
    "completions":       "actual_completions",
    "passing_tds":       "actual_passing_tds",
    "interceptions":     "actual_interceptions",
    # ── Fumbles (RB / WR / TE / QB) ───────────────────────────────────────────
    # Zero-inflated: most rows = 0. XGB handles this naturally via weighted splits.
    # Mean fumble rate 2019-2025: ~0.04/game for RBs, ~0.01/game for WRs.
    "fumbles":           "actual_fumbles",
    # ── QB pressure / protection stats (PBP-derived, Phase 4+ populated) ───────
    # sacks_taken: times QB was sacked (negative yardage play type='sack')
    # qb_hits_taken: defenders who hit QB without sacking (qb_hit col from PBP)
    "sacks_taken":       "actual_sacks_taken",
    "qb_hits_taken":     "actual_qb_hits_taken",
}


@dataclass
class FoldResult:
    """Metrics for one walk-forward fold."""
    fold_idx: int
    train_seasons: list[int]
    val_season: int
    mae: float
    rmse: float
    n_train: int
    n_val: int


# ── Shared Tooling ────────────────────────────────────────────────────────────

def load_feature_matrix(
    db_url: str,
    seasons: list[int],
    position_filter: Optional[str] = None,
    feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load feature_matrix rows from PostgreSQL.
    """
    import psycopg2
    import psycopg2.extras
    from scraper.adapters.nflreadpy_adapter import _psycopg2_dsn

    id_cols  = ["player_id", "game_id", "season", "week", "position", "team"]
    requested_features = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    all_cols = id_cols + requested_features + list(TARGET_COL_MAP.values())

    col_list = ", ".join(all_cols)
    position_clause = ""
    params: list = [seasons]

    if position_filter:
        position_clause = "AND position = %s"
        params.append(position_filter)

    query = f"""
        SELECT {col_list}
        FROM feature_matrix
        WHERE season = ANY(%s)
        {position_clause}
        ORDER BY season, week, player_id
    """

    dsn  = _psycopg2_dsn(db_url)
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=all_cols)

    df = pd.DataFrame([dict(r) for r in rows])
    original_len = len(df)
    df = df[df["seas_games_played"].fillna(0) >= MIN_PRIOR_GAMES].copy()
    logger.info(
        "  Applied pregame eligibility: removed %d rows (<%d completed prior games). Remaining: %d",
        original_len - len(df), MIN_PRIOR_GAMES, len(df),
    )
    assert_model_frame_contract(df, requested_features, consumer="load_feature_matrix")
    return df


def load_feature_matrix_prior(
    db_url: str,
    target_season: int,
    target_week: int,
    player_ids: list[str],
    position_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load feature_matrix rows for players BEFORE (target_season, target_week).

    Used by TFT inference to build per-player historical sequences.
    Rows are ordered by (season, week) ascending per player.

    Args:
        db_url: PostgreSQL connection URL.
        target_season: Target NFL season (exclusive upper bound).
        target_week: Target NFL week (exclusive upper bound for target_season).
        player_ids: List of player_id strings to load.
        position_filter: Optional position filter (e.g. "WR").

    Returns:
        DataFrame with same columns as load_feature_matrix, or empty if no rows.
    """
    if not player_ids:
        return pd.DataFrame()

    import psycopg2
    import psycopg2.extras
    from scraper.adapters.nflreadpy_adapter import _psycopg2_dsn

    id_cols = ["player_id", "game_id", "season", "week", "position", "team"]
    all_cols = id_cols + FEATURE_COLS + list(TARGET_COL_MAP.values())
    col_list = ", ".join(all_cols)
    position_clause = ""
    params: list = [tuple(player_ids), target_season, target_week]

    if position_filter:
        position_clause = "AND position = %s"
        params.append(position_filter)

    # (season < target_season) OR (season = target_season AND week < target_week)
    query = f"""
        SELECT {col_list}
        FROM feature_matrix
        WHERE player_id = ANY(%s)
          AND ((season < %s) OR (season = %s AND week < %s))
        {position_clause}
        ORDER BY player_id, season, week
    """
    params = [list(player_ids), target_season, target_season, target_week]
    if position_filter:
        params.append(position_filter)

    dsn = _psycopg2_dsn(db_url)
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame([dict(r) for r in rows])
    df = df[df["seas_games_played"].fillna(0) >= MIN_PRIOR_GAMES].copy()
    assert_model_frame_contract(df, FEATURE_COLS, consumer="load_feature_matrix_prior")
    return df


def _make_walk_forward_folds(seasons: list[int]) -> list[tuple[list[int], int]]:
    """Generate expanding-window walk-forward folds."""
    seasons = sorted(set(seasons))
    if len(seasons) < 2:
        raise ValueError(f"Walk-forward CV requires at least 2 seasons, got: {seasons}")
    return [(seasons[:i], seasons[i]) for i in range(1, len(seasons))]


def _data_hash(df: pd.DataFrame, features: list[str], target_col: str) -> str:
    """SHA-256 hash of the training data (features + target columns only)."""
    subset = df[features + [target_col]].to_csv(index=False)
    return hashlib.sha256(subset.encode()).hexdigest()[:16]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Returns (mae, rmse), securely dropping any paired NaN values."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return float('nan'), float('nan')
        
    mae  = float(mean_absolute_error(y_true_clean, y_pred_clean))
    rmse = float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)))
    return mae, rmse


def save_oof(
    oof_df: pd.DataFrame,
    target: str,
    run_id: str,
    out_dir: Path,
    prefix: str = "xgb",
    position: Optional[str] = None,
) -> Path:
    """Save OOF predictions to CSV. Returns path written.

    Filename includes position when provided so multi-position same-day runs
    cannot overwrite each other (stacking discovery depends on this).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_part = f"_{position}" if position else ""
    filename = f"{prefix}_{target}{pos_part}_{run_id[:8]}.csv"
    path = out_dir / filename
    oof_df.to_csv(path, index=False)
    logger.info("OOF predictions saved → %s  (%d rows)", path, len(oof_df))
    return path


_ONNX_DIR = Path(__file__).parent / "onnx"


def export_onnx(
    model: object,
    features: list[str],
    learner: str,
    target: str,
    position: str,
    onnx_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Export a trained tree model to ONNX for fast CPU inference.

    Supports XGBoost (via onnxmltools), LightGBM (via onnxmltools), and
    CatBoost (native ONNX export). Silently skips if the required packages
    are not installed so training never fails due to missing ONNX tooling.

    Returns:
        Path to saved .onnx file, or None on failure / missing deps.
    """
    if model is None:
        return None
    onnx_dir = onnx_dir or _ONNX_DIR
    onnx_dir.mkdir(parents=True, exist_ok=True)
    pos_tag = (position or "all").replace("/", "_")
    onnx_path = onnx_dir / f"{learner}_{target}_{pos_tag}.onnx"
    n_features = len(features)

    try:
        if learner == "catboost":
            # CatBoost has first-class ONNX support — no extra packages needed.
            model.save_model(  # type: ignore[union-attr]
                str(onnx_path),
                format="onnx",
                export_parameters={"prediction_type": "RawFormulaVal"},
            )
        else:
            import onnx  # noqa: F401 — presence check
            from onnxmltools.convert.common.data_types import FloatTensorType

            if learner == "xgb":
                from onnxmltools.convert import convert_xgboost
                onnx_model = convert_xgboost(
                    model,
                    initial_types=[("input", FloatTensorType([None, n_features]))],
                )
            elif learner == "lgbm":
                from onnxmltools.convert import convert_lightgbm
                onnx_model = convert_lightgbm(
                    model.booster_,  # type: ignore[union-attr]
                    initial_types=[("input", FloatTensorType([None, n_features]))],
                    target_opset=15,
                )
            else:
                logger.debug("export_onnx: unsupported learner '%s', skipping.", learner)
                return None

            import onnx as onnx_lib
            onnx_lib.save_model(onnx_model, str(onnx_path))

        logger.info("ONNX model saved → %s", onnx_path)
        return onnx_path

    except ImportError:
        logger.debug(
            "ONNX export skipped for %s/%s/%s — install onnx + onnxmltools to enable.",
            learner, target, position,
        )
        return None
    except Exception as exc:
        logger.warning("ONNX export failed for %s/%s/%s: %s", learner, target, position, exc)
        return None


def run_onnx_inference(onnx_path: Path, X: np.ndarray) -> Optional[np.ndarray]:
    """
    Run inference using a saved ONNX model.

    Args:
        onnx_path: Path to .onnx file.
        X:         Input feature array of shape (n_samples, n_features), float32.

    Returns:
        Predictions as np.ndarray of shape (n_samples,), or None on failure.
    """
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: X.astype(np.float32)})
        preds = np.asarray(result[0], dtype=float).ravel()
        return preds
    except ImportError:
        logger.debug("run_onnx_inference: onnxruntime not installed, falling back.")
        return None
    except Exception as exc:
        logger.warning("ONNX inference failed (%s): %s", onnx_path.name, exc)
        return None


def _parse_seasons(season_str: str, *, complete_only: bool = True) -> list[int]:
    """
    Parse '2018-2024' (range) or '2018 2019 2020' (list) into a list of ints.

    Seasons beyond the cap RAISE `ValueError`. They used to be dropped with a
    warning (audit C-28), so a trainer invoked with `--seasons 2019-2026`
    quietly trained on 2019–2025 and reported success — the pre-Week-1
    guarantee rested on a log line nobody read.

    The lower bound is not capped at all: `TRAIN_SEASON_START` is the default
    start of the walk-forward range, not a hard floor, and clamping to it
    silently rewrote `2018-2024` as `2019-2024`.

    `complete_only=False` still caps at `CURRENT_SEASON` — a season that does
    not exist yet is never a valid request.
    """
    from ml.season_constants import assert_seasons_within_cap

    season_str = season_str.strip()
    if "-" in season_str and not season_str.startswith("-"):
        parts = season_str.split("-")
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            start, end = int(parts[0]), int(parts[1])
            seasons = list(range(start, end + 1))
        else:
            seasons = [int(s.strip()) for s in season_str.replace(",", " ").split()]
    else:
        seasons = [int(s.strip()) for s in season_str.replace(",", " ").split()]

    if not seasons:
        raise ValueError(f"No seasons parsed from {season_str!r}")

    assert_seasons_within_cap(seasons, complete_only=complete_only)
    return seasons
