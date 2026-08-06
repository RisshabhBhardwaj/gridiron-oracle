"""
ml/train.py

Full-stack projection pipeline orchestrator.

ARCHITECTURE
------------
Wires the 4 projection layers in sequence for every (player, stat) pair
for a target NFL week:

  Layer 1 — Kalman form features  (ml/kalman_tracker.py)
    KalmanFeatureEngineer.compute_kalman_form(prior_rows, position)
    → kalman_est_{stat}, kalman_variance_{stat} per player

  Layer 2 — Stacking ensemble estimate  (ml/stacking_ensemble.py)
    XGB + LGB + TFT base predictions → Ridge meta-learner
    → stacked point estimate per player per stat

  Layer 3 — Bayesian posterior samples  (ml/bayesian_layer.py)
    BayesianProjection.posterior_samples(stacked_est, kalman_variance)
    → 2000 NUTS draws per player per stat

  Layer 4 — Monte Carlo projection  (ml/monte_carlo.py)
    MonteCarloProjector.batch_project(samples_dict, stat, position)
    → ProjectionResult: projection, floor, ceiling, boom/bust, fantasy

WEEKLY PRODUCTION RUN
---------------------
  1. Load FeatureMatrix rows for target (season, week) from DB — these
     contain all pre-computed Kalman, seas_*, opp_*, venue, ctx_*, rest_*
     features but no actual_* targets (those are NULL for future games).

  2. _run_kalman_step: re-run KalmanFeatureEngineer on each player's
     prior game rows to refresh kalman_est_*/kalman_variance_* for
     the exact target week.

  3. _run_stacking_step: apply saved base learner models to the feature
     matrix → [xgb_pred, lgbm_pred, tft_pred] per player, then apply
     the saved Ridge meta-learner to combine them into stacked_estimate.

     NOTE: Base-model inference requires serialized MLflow artifacts.
     When those artifacts are missing, the pipeline uses the available
     base learners and a Kalman proxy for TFT instead of pretending full
     model serving is active.

  4. _run_bayesian_step: for each player, call BayesianProjection
     .posterior_samples(stacked_estimate, kalman_variance). Uses a
     per-position BayesianProjection fitted on historical stacking
     residuals. In dry_run mode, skips MCMC and returns fast synthetic
     samples (normal distribution around stacked_estimate).

  5. _run_mc_step: MonteCarloProjector.batch_project(samples_dict) →
     DataFrame of ProjectionResults for all players.

  6. _write_db: Upsert Projection rows into the DB.
     Skipped in dry_run mode.

  7. _log_mlflow: Log pipeline run metadata + projection summary stats
     to MLflow (projection count, mean projection per position/stat,
     timestamp). Skipped in dry_run mode.

TESTING
-------
All 4 step methods are designed to be patchable for unit testing:
  - patch.object(runner, '_run_kalman_step', ...)
  - patch.object(runner, '_run_stacking_step', ...)
  - patch.object(runner, '_run_bayesian_step', ...)
  - patch.object(runner, '_run_mc_step', ...)

dry_run() bypasses MCMC (returns fast synthetic samples) and skips
DB writes and MLflow — safe to call in tests and manual validation.
"""

from __future__ import annotations

import json
import logging
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backend.app.core.tracing import start_span
from ml.utils import FEATURE_COLS, suppress_training_warnings
from ml.reliability import YARDAGE_STATS, split_conformal_interval

# Ensure noisy warnings are suppressed before any pipeline code runs
suppress_training_warnings()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_POSITIONS: list[str] = ["QB", "RB", "WR", "TE"]

# All stats the pipeline will predict. Some are position-specific (e.g. pass_attempts
# only meaningful for QB) but the pipeline trains a separate model per (stat, position)
# so position-irrelevant combos produce near-zero predictions and are ignored downstream.
#
# sacks_taken: excluded — actual_sacks_taken is NULL until PBP pipeline (Phase 4).
# TFT would get 0 rows and crash; XGB/LGB would train on all-NaN target.
_DEFAULT_STATS: list[str] = [
    # Passing (QB)
    "pass_attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "interceptions",
    # Rushing (QB + RB + WR jet sweeps)
    "rushing_yards",
    "rushing_tds",
    "carries",
    # Receiving (WR + TE + RB out of backfield)
    "receiving_yards",
    "receiving_tds",
    "receptions",
    "targets",
    # Cross-position
    "fantasy_ppr",
    "fumbles",
]

# Per-position stat sets used by train_all_models.sh to skip obviously inapplicable
# stat/position combos (QB receiving_yards, WR pass_attempts, etc.).
# XGB tolerates irrelevant combos gracefully but this saves compute.
_QB_STATS: list[str] = [
    "pass_attempts", "completions", "passing_yards", "passing_tds",
    "interceptions", "rushing_yards", "rushing_tds", "carries",
    "fantasy_ppr", "fumbles",
]
_RB_STATS: list[str] = [
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds", "targets",
    "fantasy_ppr", "fumbles",
]
_WR_STATS: list[str] = [
    "receptions", "receiving_yards", "receiving_tds", "targets",
    "carries", "rushing_yards", "rushing_tds",
    "fantasy_ppr", "fumbles",
]
_TE_STATS: list[str] = [
    "receptions", "receiving_yards", "receiving_tds", "targets",
    "carries", "rushing_yards", "rushing_tds",
    "fantasy_ppr", "fumbles",
]

# Maps position → stat list for position-aware training loops
POSITION_STAT_MAP: dict[str, list[str]] = {
    "QB": _QB_STATS,
    "RB": _RB_STATS,
    "WR": _WR_STATS,
    "TE": _TE_STATS,
}

# Synthetic player set for dry_run (2 per position = 8 players total).
# game_id uses a placeholder format safe for structural tests.
_DRY_RUN_PLAYERS: list[tuple[str, str]] = [
    ("dry_WR_001", "WR"),
    ("dry_WR_002", "WR"),
    ("dry_RB_001", "RB"),
    ("dry_RB_002", "RB"),
    ("dry_TE_001", "TE"),
    ("dry_TE_002", "TE"),
    ("dry_QB_001", "QB"),
    ("dry_QB_002", "QB"),
]

# Number of posterior draws in dry_run mode — small for test speed.
_DRY_RUN_N_SAMPLES: int = 50
_DRY_RUN_SAMPLE_STD: float = 0.5


@dataclass
class PipelineMetadata:
    """Metadata returned alongside the projection DataFrame."""
    season: int
    week: int
    positions: list[str]
    stats: list[str]
    n_projections: int
    run_id: Optional[str]        # MLflow run ID; None for dry_run
    dry_run: bool


# ---------------------------------------------------------------------------
# PipelineRunner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """
    Weekly projection pipeline orchestrator.

    Usage:
        runner = PipelineRunner()

        # Quick structural test — no DB, no MCMC, no MLflow:
        df = runner.dry_run(season=2025, week=1)

        # Full production run (requires DB session + trained models):
        df = runner.run(season=2025, week=18, db_session=session)

    The 4 internal step methods are designed to be individually patchable
    for unit testing. Each has a clear input/output contract documented
    below.

    Args:
        db_session:           SQLModel/SQLAlchemy session for DB reads/writes.
                              Required for run(); ignored in dry_run().
        mlflow_tracking_uri:  MLflow URI. Pass "" to disable (default).
        oof_dir:              Directory containing base learner OOF CSVs.
        n_bayesian_samples:   MCMC draws per player per stat (production).
    """

    def __init__(
        self,
        db_session=None,
        mlflow_tracking_uri: str = "",
        oof_dir: Optional[Path] = None,
        n_bayesian_samples: int = 2000,
        fast: bool = False,
    ) -> None:
        self.db_session = db_session
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.oof_dir = oof_dir or Path(__file__).parent / "oof"
        self.n_bayesian_samples = n_bayesian_samples
        # fast=True: skip MCMC; use Laplace/Gaussian approximation (~100x faster).
        self.fast = fast
        # MLflow run ID for current run() call; None outside run().
        self._run_id: Optional[str] = None
        # One-time MLflow reachability flag — checked once on first stacking call,
        # then cached. Avoids 7 retries × N stat/pos/model lookups when server is down.
        self._mlflow_reachable: Optional[bool] = None

        # BayesianProjection instances cached per (position, stat) key.
        # Populated lazily on first use so each instance compiles the PyMC
        # graph only once, then pm.set_data() re-uses the compiled graph.
        self._bayesian_cache: dict[str, object] = {}

        from ml.monte_carlo import MonteCarloProjector
        self._mc = MonteCarloProjector()

        from ml.inference_client import InferenceClient
        self._inference_client = InferenceClient(
            oof_dir=self.oof_dir,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(
        self,
        season: int,
        week: int,
        positions: List[str] = _DEFAULT_POSITIONS,
        stats: List[str] = _DEFAULT_STATS,
    ) -> pd.DataFrame:
        """
        Full production pipeline — loads from DB, writes results, logs MLflow.

        Args:
            season:    NFL season year.
            week:      NFL week number (1-18 regular, 19-22 postseason).
            positions: Player positions to project. Default all 4.
            stats:     Stats to project. Default 4 core stats.

        Returns:
            DataFrame with columns: player_id, stat, position,
            projection, floor, ceiling, boom_probability, bust_probability,
            fantasy_projection, fantasy_floor, fantasy_ceiling.
            Indexed 0..n_rows-1.
        """
        return self._execute(
            season=season,
            week=week,
            positions=list(positions),
            stats=list(stats),
            dry_run_mode=False,
        )

    def dry_run(
        self,
        season: int,
        week: int,
        positions: List[str] = _DEFAULT_POSITIONS,
        stats: List[str] = _DEFAULT_STATS,
    ) -> pd.DataFrame:
        """
        Dry-run: uses synthetic players, bypasses MCMC, skips DB writes
        and MLflow logging. Safe for tests and manual validation.

        Returns same column schema as run().
        """
        return self._execute(
            season=season,
            week=week,
            positions=list(positions),
            stats=list(stats),
            dry_run_mode=True,
        )

    # ------------------------------------------------------------------
    # Core execution (shared by run() and dry_run())
    # ------------------------------------------------------------------

    def _execute(
        self,
        season: int,
        week: int,
        positions: list[str],
        stats: list[str],
        dry_run_mode: bool,
    ) -> pd.DataFrame:
        """
        Shared execution engine for both run() and dry_run().

        Layer execution order (strictly sequential, guaranteed by code):
          1. _run_kalman_step    — once, for all players
          1.5 _run_volume_redistribution_step
                                — once per team; injects injury-aware
                                  projected_targets / projected_carries
                                  into kalman_df so stacking/Bayesian steps
                                  automatically use injury-adjusted volumes.
          2. _run_stacking_step — once per (stat, position)
          3. _run_bayesian_step — once per (stat, position)
          4. _run_mc_step       — once per (stat, position)
        """
        with start_span(
            "pipeline.execute",
            attributes={
                "pipeline.season": season,
                "pipeline.week": week,
                "pipeline.positions": positions,
                "pipeline.stats": stats,
                "pipeline.dry_run": dry_run_mode,
                "pipeline.fast_mode": self.fast,
            },
            tracer_name="ml.train",
        ):
            # ── Step 0: get players ───────────────────────────────────────
            if not positions or not stats:
                return pd.DataFrame(columns=_PROJECTION_COLUMNS)

            if dry_run_mode:
                players_df = self._build_synthetic_players(season, week, positions)
                prior_rows_by_player: Dict[str, List[dict]] = {
                    pid: self._build_synthetic_prior_rows(pos, n=5)
                    for pid, pos in zip(players_df["player_id"], players_df["position"])
                }
            else:
                players_df, prior_rows_by_player = self._load_from_db(season, week, positions)

            if players_df.empty:
                logger.warning("No players found for season=%d week=%d positions=%s", season, week, positions)
                return pd.DataFrame(columns=_PROJECTION_COLUMNS)

            # ── Layer 1: Kalman form features ─────────────────────────────
            # Contract: receives players_df (player_id, game_id, position) +
            # prior_rows_by_player dict → returns augmented df with
            # kalman_est_{stat} and kalman_variance_{stat} for all KALMAN_STATS.
            with start_span(
                "pipeline.kalman",
                attributes={
                    "pipeline.season": season,
                    "pipeline.week": week,
                    "pipeline.player_count": len(players_df),
                },
                tracer_name="ml.train",
            ):
                kalman_df = self._run_kalman_step(players_df, prior_rows_by_player)

            # ── Layer 1.5: Volume Redistribution (injury-aware) ───────────
            # Runs once across ALL players (not per stat) so that target share
            # redistribution is team-level and zero-sum constrained.
            # Skipped in dry_run (no real injury report available).
            if not dry_run_mode:
                with start_span(
                    "pipeline.volume_redistribution",
                    attributes={
                        "pipeline.season": season,
                        "pipeline.week": week,
                        "pipeline.player_count": len(kalman_df),
                    },
                    tracer_name="ml.train",
                ):
                    kalman_df = self._run_volume_redistribution_step(
                        kalman_df=kalman_df,
                        season=season,
                        week=week,
                    )

            all_rows: list[dict] = []

            n_samples = _DRY_RUN_N_SAMPLES if dry_run_mode else self.n_bayesian_samples

            for stat in stats:
                for position in positions:
                    mc_df = self._run_single_stat_position(
                        stat=stat,
                        position=position,
                        kalman_df=kalman_df,
                        n_samples=n_samples,
                        season=season,
                        week=week,
                        dry_run_mode=dry_run_mode,
                    )
                    if mc_df is not None:
                        all_rows.append(mc_df)

            if not all_rows:
                return pd.DataFrame(columns=_PROJECTION_COLUMNS)

            result_df = pd.concat(all_rows, ignore_index=True)

            # ── Step 6: log to MLflow (skipped in dry_run) ────────────────
            # MLflow first so _run_id is available for DB provenance tracking.
            if not dry_run_mode and self.mlflow_tracking_uri:
                with start_span(
                    "pipeline.mlflow_log",
                    attributes={
                        "pipeline.output_rows": len(result_df),
                        "pipeline.stats": stats,
                        "pipeline.positions": positions,
                    },
                    tracer_name="ml.train",
                ):
                    self._run_id = self._log_mlflow(result_df, stats=stats, positions=positions)

            # ── Step 7: write to DB (skipped in dry_run) ─────────────────
            # _write_db reads DATABASE_URL from env and handles missing URL
            # gracefully, so db_session is not required for CLI runs.
            if not dry_run_mode:
                with start_span(
                    "pipeline.write_db",
                    attributes={"pipeline.output_rows": len(result_df)},
                    tracer_name="ml.train",
                ):
                    self._write_db(result_df)

            return result_df[_PROJECTION_COLUMNS].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Inner loop — single (stat, position) projection unit
    # ------------------------------------------------------------------

    def _run_single_stat_position(
        self,
        stat: str,
        position: str,
        kalman_df: pd.DataFrame,
        n_samples: int,
        season: int,
        week: int,
        dry_run_mode: bool,
    ) -> Optional[pd.DataFrame]:
        """
        Run all 3 projection layers for one (stat, position) pair.

        Returns a DataFrame of ProjectionResults with metadata columns attached,
        or None if no players exist for this position.

        Extracted from _execute() to reduce nesting depth and allow independent
        testing of the per-stat/position projection logic.
        """
        pos_mask = kalman_df["position"] == position
        pos_df = kalman_df[pos_mask].copy()
        if pos_df.empty:
            return None

        player_ids = pos_df["player_id"].tolist()

        # ── Layer 2: stacked estimate ─────────────────────────────────────
        with start_span(
            "pipeline.stacking",
            attributes={
                "projection.stat": stat,
                "player.position": position,
                "pipeline.player_count": len(pos_df),
            },
            tracer_name="ml.train",
        ):
            stacked_estimates = self._run_stacking_step(
                pos_df, stat, position,
                season=season, week=week, dry_run_mode=dry_run_mode,
            )

        # ── Layer 3: Bayesian posterior samples ───────────────────────────
        kv_col = f"kalman_variance_{stat}"
        kalman_variances = (
            pos_df[kv_col].fillna(100.0).values
            if kv_col in pos_df.columns
            else np.full(len(pos_df), 100.0)
        )
        with start_span(
            "pipeline.bayesian",
            attributes={
                "projection.stat": stat,
                "player.position": position,
                "pipeline.player_count": len(player_ids),
                "bayesian.sample_count": n_samples,
                "bayesian.fast_mode": self.fast or dry_run_mode,
            },
            tracer_name="ml.train",
        ):
            samples_dict = self._run_bayesian_step(
                stacked_estimates=stacked_estimates,
                kalman_variances=kalman_variances,
                stat=stat,
                position=position,
                player_ids=player_ids,
                n_samples=n_samples,
                dry_run_mode=dry_run_mode,
                fast=self.fast,
            )

        # ── Layer 4: Monte Carlo projection ───────────────────────────────
        with start_span(
            "pipeline.monte_carlo",
            attributes={
                "projection.stat": stat,
                "player.position": position,
                "pipeline.player_count": len(samples_dict),
            },
            tracer_name="ml.train",
        ):
            mc_df = self._run_mc_step(samples_dict, stat, position)

        # Yardage intervals are calibrated from stacked OOF residuals.  This
        # widens/narrows posterior intervals using held-out empirical error,
        # rather than trusting a parametric posterior alone.
        if stat in YARDAGE_STATS:
            mc_df = self._apply_conformal_yardage_intervals(
                mc_df, stacked_estimates, stat=stat, position=position,
            )

        # Attach metadata columns
        mc_df = mc_df.reset_index()  # player_id moves from index to column
        mc_df["stat"]     = stat
        mc_df["position"] = position
        mc_df["season"]   = season
        mc_df["week"]     = week

        id_map = pos_df.set_index("player_id")["game_id"].to_dict()
        mc_df["game_id"] = mc_df["player_id"].map(id_map)

        # Attach first 500 posterior draws for CRPS storage (accurate yet compact).
        mc_df["posterior_samples"] = mc_df["player_id"].map(
            {pid: s[:500].tolist() for pid, s in samples_dict.items()}
        )

        return mc_df

    # ------------------------------------------------------------------
    # Layer 1 — Kalman step (patchable for ordering tests)
    # ------------------------------------------------------------------

    def _run_kalman_step(
        self,
        players_df: pd.DataFrame,
        prior_rows_by_player: Dict[str, List[dict]],
    ) -> pd.DataFrame:
        """
        Layer 1: Compute Kalman form features for each player.

        Iterates over players_df rows, calls compute_kalman_form() with
        that player's prior game rows, and adds kalman_est_{stat} and
        kalman_variance_{stat} columns to each row.

        Returns augmented DataFrame (all original columns + kalman_* cols).
        """
        # If kalman_est_* columns are already present (loaded from feature_matrix),
        # pass through unchanged — avoids redundant Kalman re-computation.
        if any(c.startswith("kalman_est_") for c in players_df.columns):
            return players_df

        from ml.kalman_tracker import KalmanFeatureEngineer
        kfe = KalmanFeatureEngineer()

        augmented_rows = []
        for _, player_row in players_df.iterrows():
            pid = str(player_row["player_id"])
            pos = str(player_row.get("position", ""))
            prior = prior_rows_by_player.get(pid, [])
            kalman_feats = kfe.compute_kalman_form(prior, position=pos or None)
            augmented_rows.append({**player_row.to_dict(), **kalman_feats})

        return pd.DataFrame(augmented_rows)

    # ------------------------------------------------------------------
    # Layer 1.5 — Volume Redistribution (injury-aware, patchable)
    # ------------------------------------------------------------------

    def _run_volume_redistribution_step(
        self,
        kalman_df: pd.DataFrame,
        season: int,
        week: int,
    ) -> pd.DataFrame:
        """
        Layer 1.5: Redistribute target/carry share among teammates when
        a player is ruled OUT or DNP.

        Fetches the current injury report, then calls VolumeRedistributor
        to produce Dirichlet-sampled, zero-sum-constrained projected_targets
        and projected_carries for each team's full player group.

        The redistributed values are written into kalman_df as:
            - projected_targets:  injury-adjusted expected targets per game
            - projected_carries:  injury-adjusted expected carries per game
            - target_share_mean:  posterior mean target share (post-injury)
            - carry_share_mean:   posterior mean carry share (post-injury)

        If the ESPN injury adapter is unavailable (ImportError, network error,
        or any exception), this step logs a WARNING and returns kalman_df
        unchanged. The pipeline never blocks on this step.

        Args:
            kalman_df: DataFrame output of Layer 1 (Kalman features present).
            season:    Current NFL season.
            week:      Current NFL week.

        Returns:
            kalman_df with redistributed volume columns appended/updated.
        """
        try:
            # Import lazily to avoid circular imports and allow graceful skip
            from ml.volume_redistribution import get_redistributor

            # Build injury report dict: {player_id: injury_status_str}
            injury_report: dict[str, str] = {}
            try:
                from scraper.adapters.espn_adapter import EspnAdapter
                db_url = os.environ.get("DATABASE_URL")
                espn = EspnAdapter(db_url=db_url)
                raw_df = espn.fetch_injury_report(week=week, season=season)
                if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                    valid = raw_df["player_id"].notna() & (raw_df["player_id"] != "")
                    status_col = "practice_status" if "practice_status" in raw_df.columns else "player_name"
                    injury_report = raw_df.loc[valid].set_index("player_id")[status_col].astype(str).to_dict()
            except Exception as espn_exc:
                logger.warning(
                    "_run_volume_redistribution_step: could not fetch ESPN "
                    "injury report (%s). Proceeding with empty report "
                    "(all players assumed healthy).",
                    espn_exc,
                )

            if not injury_report:
                logger.debug(
                    "_run_volume_redistribution_step: empty injury report for "
                    "S%dW%d — skipping redistribution.", season, week,
                )
                return kalman_df

            vr = get_redistributor(df=None)  # fit lazily (defaults if not pre-fitted)

            # Build game context per team from kalman_df if spread/total are available
            game_context_by_team: dict[str, dict] = {}
            if all(c in kalman_df.columns for c in ("team", "spread_line", "total_line")):
                for team, grp in kalman_df.groupby("team"):
                    row = grp.iloc[0]
                    game_context_by_team[str(team)] = {
                        "spread_line": float(row.get("spread_line") or 0.0),
                        "total_line":  float(row.get("total_line") or 45.0),
                        "is_home":     int(row.get("is_home") or 0),
                    }

            redistributed_df = vr.redistribute_team_projections(
                projections_df=kalman_df,
                injury_report=injury_report,
                game_context_by_team=game_context_by_team or None,
                n_samples=500,   # lighter sampling for speed; full 2000 in backtest
            )

            n_injured = sum(
                1 for pid, status in injury_report.items()
                if str(status).lower().strip() in
                {"out", "dnp", "ir", "injured reserve", "o", "0"}
            )
            logger.info(
                "_run_volume_redistribution_step: S%dW%d — %d injured players "
                "redistributed across their teammates.",
                season, week, n_injured,
            )
            return redistributed_df

        except Exception as exc:
            logger.warning(
                "_run_volume_redistribution_step: unexpected error (%s). "
                "Returning kalman_df unchanged.", exc,
            )
            return kalman_df

    # ------------------------------------------------------------------
    # Layer 2 — Stacking step (patchable for ordering tests)
    # ------------------------------------------------------------------

    def _run_stacking_step(
        self,
        kalman_df: pd.DataFrame,
        stat: str,
        position: str,
        *,
        season: Optional[int] = None,
        week: Optional[int] = None,
        dry_run_mode: bool = False,
    ) -> np.ndarray:
        """
        Layer 2: Produce stacked point estimate per player for (stat, position).

        Loads the most recent XGB, LGB, CatBoost, and TFT base models from MLflow
        (when mlflow_tracking_uri is configured), runs inference, and combines
        predictions via Ridge meta-learner coefficients saved to
        ml/oof/ridge_{stat}_coefs.json — or equal weights when absent.

        TFT inference uses the trained TFT model with full TimeSeriesDataSet
        sequences (prior feature_matrix rows + current week). Falls back to
        kalman_est_{stat} proxy when: dry_run, no TFT model, or no prior rows.

        Falls back to kalman_est_{stat} proxy (logged as WARNING) when:
          - mlflow_tracking_uri is empty (default — disables MLflow entirely)
          - No XGB or LGB model artifacts are found for this stat
          - MLflow server is unreachable or raises any exception

        Returns:
            np.ndarray of shape (n_players,) — stacked estimates.
        """
        # Fast path: no MLflow configured (covers dry_run and default CLI runs).
        if not self.mlflow_tracking_uri:
            col = f"kalman_est_{stat}"
            if col in kalman_df.columns:
                return kalman_df[col].fillna(0.0).values.astype(float)
            return np.zeros(len(kalman_df), dtype=float)

        # One-time connectivity check — avoids 7 retries × N lookups when MLflow is down.
        if self._mlflow_reachable is None:
            try:
                import urllib.request
                urllib.request.urlopen(self.mlflow_tracking_uri + "/health", timeout=3)
                self._mlflow_reachable = True
                logger.info("MLflow reachable at %s — using stacking inference.", self.mlflow_tracking_uri)
            except Exception:
                self._mlflow_reachable = False
                logger.warning(
                    "MLflow at %s is unreachable — stacking inference disabled for this run. "
                    "Projections will use kalman_est_* proxy. Start MLflow and re-run to use "
                    "full stacking ensemble.",
                    self.mlflow_tracking_uri,
                )

        if not self._mlflow_reachable:
            col = f"kalman_est_{stat}"
            if col in kalman_df.columns:
                return kalman_df[col].fillna(0.0).values.astype(float)
            return np.zeros(len(kalman_df), dtype=float)

        try:
            with start_span(
                "pipeline.stacking.inference",
                attributes={
                    "projection.stat": stat,
                    "player.position": position,
                    "stacking.learners": "xgb,lgbm,catboost,tft",
                    "stacking.dry_run": dry_run_mode,
                },
                tracer_name="ml.train",
            ):
                return self._load_and_run_stacking(
                    kalman_df, stat, position,
                    season=season, week=week, dry_run_mode=dry_run_mode,
                )
        except Exception as exc:
            logger.warning(
                "Stacking inference failed for stat=%s (%s) — "
                "falling back to kalman_est_%s proxy.",
                stat, exc, stat,
            )
            col = f"kalman_est_{stat}"
            if col in kalman_df.columns:
                return kalman_df[col].fillna(0.0).values.astype(float)
            return np.zeros(len(kalman_df), dtype=float)

    def _load_and_run_stacking(
        self,
        kalman_df: pd.DataFrame,
        stat: str,
        position: str,
        *,
        season: Optional[int] = None,
        week: Optional[int] = None,
        dry_run_mode: bool = False,
    ) -> np.ndarray:
        # Delegates to InferenceClient. Passes self._load_latest_mlflow_model so
        # patch.object(runner, "_load_latest_mlflow_model") still intercepts in tests.
        return self._inference_client.load_and_run_stacking(
            kalman_df, stat, position,
            season=season, week=week, dry_run_mode=dry_run_mode,
            model_loader=self._load_latest_mlflow_model,
        )

    def _run_tft_inference(
        self,
        kalman_df: pd.DataFrame,
        stat: str,
        position: str,
        tft_model: Optional[object],
        *,
        season: Optional[int] = None,
        week: Optional[int] = None,
        dry_run_mode: bool = False,
    ) -> np.ndarray:
        return self._inference_client.run_tft_inference(
            kalman_df, stat, position, tft_model,
            season=season, week=week, dry_run_mode=dry_run_mode,
        )

    def _align_features_to_model(
        self, model: object, X_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, bool]:
        return self._inference_client.align_features_to_model(model, X_df)

    def _build_inference_features(self, kalman_df: pd.DataFrame) -> pd.DataFrame:
        return self._inference_client.build_inference_features(kalman_df)

    def _load_latest_mlflow_model(
        self,
        learner: str,
        stat: str,
        position: Optional[str] = None,
    ) -> Optional[object]:
        return self._inference_client.load_latest_mlflow_model(learner, stat, position)

    def _try_onnx_pred(
        self,
        learner: str,
        stat: str,
        position: Optional[str],
        X_arr: np.ndarray,
        expected_rows: int,
    ) -> Optional[np.ndarray]:
        return self._inference_client.try_onnx_pred(learner, stat, position, X_arr, expected_rows)

    def _load_ridge_coefs(
        self, stat: str, position: Optional[str] = None
    ) -> Optional[tuple[list[float], float]]:
        return self._inference_client.load_ridge_coefs(stat, position)

    # ------------------------------------------------------------------
    # Layer 3 — Bayesian step (patchable for ordering tests)
    # ------------------------------------------------------------------

    def _run_bayesian_step(
        self,
        stacked_estimates: np.ndarray,
        kalman_variances: np.ndarray,
        stat: str,
        position: str,
        player_ids: List[str],
        n_samples: int = 2000,
        dry_run_mode: bool = False,
        fast: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Layer 3: Produce Bayesian posterior samples per player.

        dry_run_mode: synthetic normal samples — no MCMC (for tests).
        fast=True:    Laplace/Gaussian approximation — no MCMC (~100x faster).
                      Samples N(kalman_est, sqrt(kalman_variance)); suitable
                      for historical backtest population where throughput matters.
        production:   Full NUTS sampling via BayesianProjection.

        Returns:
            Dict[player_id, np.ndarray] — posterior draws per player.
        """
        result: Dict[str, np.ndarray] = {}

        if dry_run_mode or fast:
            # Gaussian approximation: N(stacked_est, sqrt(kalman_variance)).
            # dry_run uses a fixed seed + small std floor for test reproducibility.
            # fast mode uses kalman_variance directly (real data path).
            seed = 0 if dry_run_mode else 42
            rng = np.random.default_rng(seed)
            min_std = _DRY_RUN_SAMPLE_STD if dry_run_mode else 1.0
            for pid, est, kv in zip(player_ids, stacked_estimates, kalman_variances):
                std = max(float(kv) ** 0.5, min_std)
                if fast and not dry_run_mode:
                    # Calibration correction: kalman_variance tracks uncertainty
                    # about the player's true mean, not per-game outcome variance.
                    # Steady-state kalman_variance ≈ sqrt(Q×R) ≈ sqrt(R) for Q=1,
                    # so sqrt(kv) ≈ R^0.25 — far too narrow for interval forecasting.
                    # ×3 widens [p10, p90] to approximately match observed per-game
                    # variance, targeting ~80% empirical coverage.
                    std *= 3.0
                result[pid] = rng.normal(float(est), std, n_samples)
            return result

        # Production path: BayesianProjection with MCMC sampling.
        from ml.bayesian_layer import BayesianProjection

        cache_key = f"{position}_{stat}"
        if cache_key not in self._bayesian_cache:
            proj = BayesianProjection()
            residuals = self._load_oof_residuals(position, stat)
            proj.fit(position, residuals)
            self._bayesian_cache[cache_key] = proj

        projector = self._bayesian_cache[cache_key]

        for pid, est, kv in zip(player_ids, stacked_estimates, kalman_variances):
            samples = projector.posterior_samples(
                stacked_estimate=float(est),
                kalman_variance=float(kv),
                n_samples=n_samples,
            )
            result[pid] = samples

        return result

    # ------------------------------------------------------------------
    # OOF residual loader — feeds real calibration data into BayesianProjection
    # ------------------------------------------------------------------

    # Position-calibrated sigma fallbacks used when no OOF files exist yet
    # (first training run before any model is saved).
    # Values are approximate per-game standard deviations from historical data:
    #   WR/TE receiving_yards ~35yds std, RB rushing_yards ~30yds, QB passing ~55yds.
    # Per-position, per-stat sigma fallbacks for Bayesian calibration.
    # Values are approximate per-game standard deviations from 2019-2025 data.
    # Used when no OOF files exist yet (first training run).
    # Key format: "{stat}_{position}" — falls back to position-only, then default.
    _POSITION_SIGMA_FALLBACK: dict[str, float] = {
        # Passing yards — highest variance stat
        "passing_yards_QB":   55.0,
        # Pass attempts — count stat, lower variance
        "pass_attempts_QB":    8.0,
        "completions_QB":      6.0,
        "passing_tds_QB":      1.2,
        "interceptions_QB":    0.7,   # near-binary; most games 0 or 1
        # Rushing yards
        "rushing_yards_QB":   15.0,
        "rushing_yards_RB":   30.0,
        "rushing_yards_WR":    5.0,
        "rushing_yards_TE":    2.0,
        # Rushing touchdowns
        "rushing_tds_QB":      0.3,
        "rushing_tds_RB":      0.5,
        "rushing_tds_WR":      0.1,
        # Carries
        "carries_QB":          3.0,
        "carries_RB":          5.0,
        "carries_WR":          1.0,
        # Receiving yards
        "receiving_yards_WR":  35.0,
        "receiving_yards_TE":  28.0,
        "receiving_yards_RB":  18.0,
        # Receptions
        "receptions_WR":        2.0,
        "receptions_TE":        2.0,
        "receptions_RB":        2.0,
        # Targets
        "targets_WR":           2.5,
        "targets_TE":           2.0,
        "targets_RB":           2.0,
        # Receiving TDs
        "receiving_tds_WR":    0.45,
        "receiving_tds_TE":    0.35,
        "receiving_tds_RB":    0.25,
        # Rare events — zero-inflated, use small sigma
        "fumbles_QB":          0.25,
        "fumbles_RB":          0.18,
        "fumbles_WR":          0.10,
        "fumbles_TE":          0.10,
        "sacks_taken_QB":      1.8,
        # Fantasy PPR — composite, moderate variance
        "fantasy_ppr_QB":     10.0,
        "fantasy_ppr_RB":      9.0,
        "fantasy_ppr_WR":      9.0,
        "fantasy_ppr_TE":      7.0,
        # Position-only fallbacks (stat not in above dict)
        "QB": 55.0, "RB": 30.0, "WR": 35.0, "TE": 28.0,
    }
    _SIGMA_FALLBACK_DEFAULT: float = 35.0

    def _load_oof_residuals(self, position: str, stat: str) -> np.ndarray:
        """
        Load real stacking residuals from XGB+LGBM OOF CSVs for Bayesian calibration.

        Strategy:
          1. Scan self.oof_dir for xgb_{stat}_*.csv and lgbm_{stat}_*.csv files.
          2. Load them, filter rows to `position`, compute residual = y_true - y_pred.
          3. Return combined residuals from both learners as a 1-D float array.

        Fallback (when no OOF files found):
          Returns an array whose std matches the position-calibrated fallback sigma
          rather than synthetic Gaussian noise. This ensures BayesianProjection
          learns a physically meaningful sigma even on the first training run.

        Args:
            position: e.g. "WR", "RB", "QB", "TE"
            stat:     e.g. "receiving_yards", "rushing_yards" — matches OOF filename prefix

        Returns:
            np.ndarray of residuals (1-D, dtype float). Length >= 2 guaranteed.
        """
        import glob

        oof_dir = self.oof_dir
        all_residuals: list[np.ndarray] = []

        for prefix in ("xgb", "lgbm"):
            pattern = str(oof_dir / f"{prefix}_{stat}_*.csv")
            for filepath in glob.glob(pattern):
                try:
                    df = pd.read_csv(filepath)
                    # OOF files: player_id, game_id, season, week, y_true, y_pred, fold_idx
                    # Filter to position if the column is present.
                    if "position" in df.columns:
                        df = df[df["position"] == position]
                    if "y_true" in df.columns and "y_pred" in df.columns:
                        residuals = (df["y_true"] - df["y_pred"]).dropna().values
                        if len(residuals) > 0:
                            all_residuals.append(residuals.astype(float))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("_load_oof_residuals: could not read %s: %s", filepath, exc)

        if all_residuals:
            combined = np.concatenate(all_residuals)
            logger.info(
                "Bayesian fit: loaded %d OOF residuals for position=%s stat=%s",
                len(combined), position, stat,
            )
            return combined

        # No OOF files yet — use stat+position-calibrated fallback sigma.
        sigma = self._POSITION_SIGMA_FALLBACK.get(
            f"{stat}_{position}",
            self._POSITION_SIGMA_FALLBACK.get(position, self._SIGMA_FALLBACK_DEFAULT),
        )
        logger.warning(
            "No OOF files found for stat=%s. Using position-calibrated fallback "
            "sigma=%.1f for position=%s. Run train_all_models.sh to generate OOF files.",
            stat, sigma, position,
        )
        rng = np.random.default_rng(42)
        # Generate enough samples so std(ddof=1) is stable; use exact sigma as scale.
        fallback = rng.normal(0.0, sigma, 500)
        return fallback

    def _apply_conformal_yardage_intervals(
        self,
        mc_df: pd.DataFrame,
        stacked_estimates: np.ndarray,
        *,
        stat: str,
        position: str,
    ) -> pd.DataFrame:
        """Replace p10/p90 with 80% split-conformal intervals when available."""
        import glob

        actual_parts: list[np.ndarray] = []
        prediction_parts: list[np.ndarray] = []
        pattern = str(self.oof_dir / f"stack_{stat}_{position}_*.csv")
        for path in glob.glob(pattern):
            try:
                oof = pd.read_csv(path)
                if {"y_true", "y_pred"}.issubset(oof.columns):
                    actual_parts.append(oof["y_true"].to_numpy(dtype=float))
                    prediction_parts.append(oof["y_pred"].to_numpy(dtype=float))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load conformal OOF %s: %s", path, exc)

        if not actual_parts:
            message = f"No stacked OOF residuals for conformal {stat}/{position}"
            if os.environ.get("PRODUCT_MODE", "graceful_fallback") == "artifact_backed":
                raise RuntimeError(message)
            logger.warning("%s; retaining Bayesian intervals in fallback mode", message)
            return mc_df

        interval = split_conformal_interval(
            np.concatenate(actual_parts),
            np.concatenate(prediction_parts),
            np.asarray(stacked_estimates, dtype=float),
            coverage=0.80,
        )
        if len(interval.lower) != len(mc_df):
            raise RuntimeError("Conformal prediction count does not match Monte Carlo output")
        calibrated = mc_df.copy()
        calibrated["floor"] = interval.lower
        calibrated["ceiling"] = interval.upper
        calibrated["conformal_radius"] = interval.radius
        logger.info(
            "Applied 80%% split-conformal interval for %s/%s (radius=%.3f)",
            stat, position, interval.radius,
        )
        return calibrated

    # ------------------------------------------------------------------
    # Layer 4 — Monte Carlo step (patchable for ordering tests)
    # ------------------------------------------------------------------

    def _run_mc_step(
        self,
        samples_dict: Dict[str, np.ndarray],
        stat: str,
        position: str,
    ) -> pd.DataFrame:
        """
        Layer 4: Convert posterior samples to projection results.

        Returns pd.DataFrame indexed by player_id with columns:
            projection, floor, ceiling, boom_probability, bust_probability,
            fantasy_projection, fantasy_floor, fantasy_ceiling, n_samples.
        """
        return self._mc.batch_project(samples_dict, stat=stat, position=position)

    # ------------------------------------------------------------------
    # DB write (skipped in dry_run)
    # ------------------------------------------------------------------

    def _write_db(self, results_df: pd.DataFrame) -> None:
        """
        Upsert Projection rows into the projections table via psycopg2.

        Uses ON CONFLICT (player_id, game_id, stat) DO UPDATE to allow
        re-running the pipeline without duplicate errors.

        Creates the projections table if it does not yet exist.
        """
        import os
        import psycopg2

        dsn = os.environ.get("DATABASE_URL", "")
        if not dsn:
            logger.warning("DATABASE_URL not set — DB write skipped.")
            return
        dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")

        conn = psycopg2.connect(dsn)
        try:
            cur = conn.cursor()

            # Ensure table exists (idempotent).
            cur.execute("""
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
                    posterior_samples JSONB,
                    created_at       TIMESTAMP DEFAULT NOW(),
                    CONSTRAINT uq_projections_player_game_stat
                        UNIQUE (player_id, game_id, stat)
                )
            """)
            # Migrate existing tables: add new columns if absent.
            for _col, _type in [
                ("posterior_samples", "JSONB"),
                ("p25", "FLOAT"),
                ("p75", "FLOAT"),
            ]:
                cur.execute(f"ALTER TABLE projections ADD COLUMN IF NOT EXISTS {_col} {_type}")
            conn.commit()

            upsert_sql = """
                INSERT INTO projections
                    (player_id, game_id, season, week, stat, position,
                     projection, floor, ceiling, p25, p75,
                     boom_probability, bust_probability,
                     fantasy_projection, fantasy_floor, fantasy_ceiling,
                     pipeline_run_id, posterior_samples)
                VALUES (%s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s, %s,%s, %s,%s,%s, %s, %s)
                ON CONFLICT (player_id, game_id, stat) DO UPDATE SET
                    season             = EXCLUDED.season,
                    week               = EXCLUDED.week,
                    position           = EXCLUDED.position,
                    projection         = EXCLUDED.projection,
                    floor              = EXCLUDED.floor,
                    ceiling            = EXCLUDED.ceiling,
                    p25                = EXCLUDED.p25,
                    p75                = EXCLUDED.p75,
                    boom_probability   = EXCLUDED.boom_probability,
                    bust_probability   = EXCLUDED.bust_probability,
                    fantasy_projection = EXCLUDED.fantasy_projection,
                    fantasy_floor      = EXCLUDED.fantasy_floor,
                    fantasy_ceiling    = EXCLUDED.fantasy_ceiling,
                    pipeline_run_id    = EXCLUDED.pipeline_run_id,
                    posterior_samples  = EXCLUDED.posterior_samples
            """

            for _, row in results_df.iterrows():
                # Serialize posterior samples to JSON for storage.
                # Stored as first 500 draws; None if samples absent (legacy rows).
                samples = row.get("posterior_samples")
                samples_json = json.dumps(samples) if samples is not None and len(samples) > 0 else None
                cur.execute(upsert_sql, (
                    str(row["player_id"]),
                    str(row.get("game_id", "")),
                    int(row.get("season", 0)),
                    int(row.get("week", 0)),
                    str(row["stat"]),
                    row.get("position"),
                    _float_or_none(row.get("projection")),
                    _float_or_none(row.get("floor")),
                    _float_or_none(row.get("ceiling")),
                    # p25/p75 — generated by MonteCarloProjector as percentile_25/75
                    _float_or_none(row.get("p25") if row.get("p25") is not None else row.get("percentile_25")),
                    _float_or_none(row.get("p75") if row.get("p75") is not None else row.get("percentile_75")),
                    _float_or_none(row.get("boom_probability")),
                    _float_or_none(row.get("bust_probability")),
                    _float_or_none(row.get("fantasy_projection")),
                    _float_or_none(row.get("fantasy_floor")),
                    _float_or_none(row.get("fantasy_ceiling")),
                    self._run_id,
                    samples_json,
                ))
            conn.commit()
            logger.info("Wrote %d projection rows to DB.", len(results_df))
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # MLflow logging (skipped in dry_run)
    # ------------------------------------------------------------------

    def _log_mlflow(
        self,
        results_df: pd.DataFrame,
        stats: list[str],
        positions: list[str],
    ) -> Optional[str]:
        """
        Log pipeline run metadata to MLflow.

        Required fields per CLAUDE.md §3 (adapted for pipeline runs):
            n_projections, mean_projection per (position, stat), timestamp.

        Returns:
            MLflow run ID string, or None if MLflow is unavailable.
        """
        if not self.mlflow_tracking_uri:
            return None

        try:
            import mlflow
            from datetime import datetime, timezone

            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment("pipeline_weekly")

            with mlflow.start_run() as run:
                mlflow.log_params({
                    "stats":     str(stats),
                    "positions": str(positions),
                })
                mlflow.log_metrics({"n_projections": len(results_df)})

                for stat in stats:
                    for pos in positions:
                        mask = (
                            (results_df["stat"] == stat) &
                            (results_df["position"] == pos)
                        )
                        subset = results_df.loc[mask, "projection"].dropna()
                        if not subset.empty:
                            mlflow.log_metric(
                                f"mean_proj_{pos}_{stat}",
                                float(subset.mean()),
                            )

                mlflow.log_param("timestamp", datetime.now(timezone.utc).isoformat())
                return run.info.run_id

        except Exception as exc:
            logger.warning("MLflow logging failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Synthetic data helpers (dry_run only)
    # ------------------------------------------------------------------

    def _build_synthetic_players(
        self,
        season: int,
        week: int,
        positions: list[str],
    ) -> pd.DataFrame:
        """Build a synthetic players DataFrame for dry_run testing."""
        rows = [
            {
                "player_id": pid,
                "position":  pos,
                "game_id":   f"{season}_{week:02d}_DRY_DRY",
                "season":    season,
                "week":      week,
            }
            for pid, pos in _DRY_RUN_PLAYERS
            if pos in positions
        ]
        return pd.DataFrame(rows)

    def _build_synthetic_prior_rows(
        self,
        position: str,
        n: int = 5,
    ) -> list[dict]:
        """
        Generate n synthetic prior game rows for a player.

        Values are sampled around POSITION_PRIORS to give the Kalman filter
        realistic-ish input without requiring a live DB.
        """
        from ml.kalman_tracker import POSITION_PRIORS, _DEFAULT_PRIOR, KALMAN_STATS, STAT_SOURCE_COL

        priors = POSITION_PRIORS.get(position, _DEFAULT_PRIOR)
        rng = np.random.default_rng(abs(hash(position)) % 2**31)

        rows = []
        for _ in range(n):
            row: dict = {}
            for stat in KALMAN_STATS:
                src_col = STAT_SOURCE_COL[stat]
                mu = priors.get(stat, 0.0)
                # ±30% noise around the prior
                row[src_col] = float(rng.normal(mu, max(mu * 0.3, 0.5)))
            rows.append(row)

        return rows

    # ------------------------------------------------------------------
    # DB load (production only)
    # ------------------------------------------------------------------

    def _load_from_db(
        self,
        season: int,
        week: int,
        positions: list[str],
    ) -> tuple[pd.DataFrame, Dict[str, List[dict]]]:
        """
        Load FeatureMatrix rows + prior game rows from the DB.

        Returns:
            players_df:           Rows for target (season, week, positions).
            prior_rows_by_player: {player_id: [game_log dicts sorted oldest-first]}
        """
        import os
        import psycopg2

        dsn = os.environ.get("DATABASE_URL", "")
        if not dsn:
            raise RuntimeError(
                "DATABASE_URL environment variable not set. "
                "Export it before calling run(), e.g.:\n"
                "  export DATABASE_URL=postgresql://oracle:oracle@localhost:5432/oracle"
            )
        # Convert asyncpg DSN (used by FastAPI) to psycopg2 synchronous DSN.
        dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")

        pos_placeholders = ",".join(["%s"] * len(positions))

        conn = psycopg2.connect(dsn)
        try:
            cur = conn.cursor()

            # ── Load FeatureMatrix rows for (season, week, positions) ──────
            cols_str = ", ".join(FEATURE_COLS)
            cur.execute(
                f"""
                SELECT
                    player_id, game_id, season, week, position,
                    {cols_str}
                FROM feature_matrix
                WHERE season = %s
                  AND week   = %s
                  AND position IN ({pos_placeholders})
                """,
                [season, week] + list(positions),
            )
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            logger.info(
                "No feature_matrix rows for season=%d week=%d positions=%s",
                season, week, positions,
            )
            return pd.DataFrame(), {}

        players_df = pd.DataFrame(rows, columns=cols)

        # prior_rows_by_player is empty — Kalman features come pre-computed
        # from feature_matrix; _run_kalman_step detects this and passes through.
        prior_rows_by_player: Dict[str, List[dict]] = {
            pid: [] for pid in players_df["player_id"].unique()
        }
        return players_df, prior_rows_by_player


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float_or_none(v) -> Optional[float]:
    """Convert a value to float, returning None if it is None/NaN."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else f   # NaN check: NaN != NaN
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Output column spec
# ---------------------------------------------------------------------------

_PROJECTION_COLUMNS: list[str] = [
    "player_id",
    "game_id",
    "season",
    "week",
    "stat",
    "position",
    "projection",
    "floor",
    "ceiling",
    "p5",
    "p25",
    "p75",
    "p95",
    "boom_probability",
    "bust_probability",
    "fantasy_projection",
    "fantasy_floor",
    "fantasy_ceiling",
    "n_samples",
]


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------

def _week_already_projections(season: int, week: int) -> bool:
    """True if projections table has rows for this (season, week)."""
    import os
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        return False
    dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")
    try:
        import psycopg2
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM projections WHERE season = %s AND week = %s LIMIT 1",
            (season, week),
        )
        has = cur.fetchone() is not None
        conn.close()
        return has
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from ml.utils import suppress_training_warnings
    suppress_training_warnings()

    parser = argparse.ArgumentParser(description="Gridiron Oracle — weekly projection pipeline")

    # Single-week shorthand (kept for backwards compatibility)
    parser.add_argument("--season", type=int, default=None,
                        help="Single season (use --seasons for multi-season)")
    parser.add_argument("--week",   type=int, default=None,
                        help="Single week (use --all-weeks for full season sweep)")

    # Multi-season / all-weeks batch mode
    parser.add_argument("--seasons", type=int, nargs="+", default=None,
                        help="One or more seasons, e.g. --seasons 2019 2020 2021")
    parser.add_argument("--all-weeks", action="store_true", default=False,
                        help="Run all 18 regular-season weeks for each season")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Skip (season,week) pairs already in projections table")

    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--fast",    action="store_true", default=False,
                        help="Laplace/Gaussian approximation instead of NUTS (~100x faster)")
    parser.add_argument(
        "--positions", nargs="+", default=_DEFAULT_POSITIONS,
        choices=["QB", "RB", "WR", "TE"],
    )
    parser.add_argument("--stats", nargs="+", default=_DEFAULT_STATS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Build season/week pairs to run
    seasons = args.seasons or ([args.season] if args.season else None)
    if not seasons:
        parser.error("Specify either --season YEAR or --seasons YEAR [YEAR ...]")

    runner = PipelineRunner(
        fast=args.fast,
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", ""),
    )

    total_projections: int = 0
    for season in seasons:
        if args.all_weeks:
            weeks = list(range(1, 19))
        elif args.week:
            weeks = [args.week]
        else:
            parser.error("Specify either --week N or --all-weeks")

        for week in weeks:
            if args.resume and not args.dry_run:
                if _week_already_projections(season, week):
                    logger.info("Season %d Week %2d → SKIP (already in DB)", season, week)
                    continue
            if args.dry_run:
                df = runner.dry_run(season, week, args.positions, args.stats)
            else:
                df = runner.run(season, week, args.positions, args.stats)
            n = len(df)
            total_projections += n
            logger.info("Season %d Week %2d → %d projections", season, week, n)

    logger.info("Total projections written: %d", total_projections)
