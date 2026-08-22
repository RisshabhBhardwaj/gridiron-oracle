"""
ml/markov_simulator.py

Drive-Level Markov Chain Simulator.

OVERVIEW
--------
Extracts empirical transition matrices from Play-By-Play (PBP) data
and prepares them for the C++ DriveMCMC engine.

Football plays are modeled as state transitions:
    State:  (field_position_bucket, down, yards_to_go_bucket,
              score_diff_bucket, quarter)
    Action: p_pass (probability the play call is a pass, learned per state —
             this is what lets leading/trailing game state change the mix)
    Transition: (mean_gain, gain_std, p_turnover, p_penalty_gain, p_penalty_loss)

The score_diff/quarter dimensions exist so that game script — a team
leaning run when leading late — emerges from the fitted state rather than
being assumed. Cells are sparse once the state space is 5D (11*4*6*5*4 =
5,280 cells against ~320K plays, ~60/cell on average, worse in the
long-ydstogo / trailing-early tails), so each cell is shrunk toward the
3D (fp, down, ytg) marginal in proportion to its own sample count — see
_shrink_to_marginal.

USAGE
-----
    from ml.markov_simulator import DriveMarkovModel

    model = DriveMarkovModel()
    model.fit(pbp_df)
    model.export_transitions("ml/oof/transitions.csv")
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Buckets defined by engine/src/drive_mcmc.cpp
N_FP_BUCKETS = 11
N_DOWN = 4
N_YTG_BUCKETS = 6

# Score-differential buckets (posteam perspective): trailing_big, trailing_small,
# tied, leading_small, leading_big. Thresholds match the empirical check in
# scripts/verify_drive_transitions.py (>= 9 / 1-8 / 0 / -8..-1 / <= -9).
N_SCORE_BUCKETS = 5
N_QUARTER = 4  # quarter 1-4; OT (5) folds into 4

# Below this many plays, a (fp, down, ytg, score, quarter) cell is shrunk
# heavily toward the coarser (fp, down, ytg) marginal — see _shrink_to_marginal.
MIN_CELL_COUNT_FOR_CONFIDENCE = 50


def _score_diff_bucket(score_differential: pd.Series) -> pd.Series:
    """Map posteam score differential to one of N_SCORE_BUCKETS (0=trailing_big..4=leading_big)."""
    sd = score_differential.to_numpy()
    conditions = [sd <= -9, (sd >= -8) & (sd <= -1), sd == 0, (sd >= 1) & (sd <= 8), sd >= 9]
    choices = [0, 1, 2, 3, 4]
    bucket = np.select(conditions, choices, default=2)
    return pd.Series(bucket, index=score_differential.index, dtype=int)

class DriveMarkovModel:
    """
    Builds the state-transition probability matrices for drove-level Markov Monte Carlo.
    """
    def __init__(self):
        self.transitions: Optional[pd.DataFrame] = None
        self.transitions_by_game_state: Optional[pd.DataFrame] = None

    def fit(self, pbp_df: pd.DataFrame) -> "DriveMarkovModel":
        """
        Extract Markov states from play-by-play DataFrame.
        Expected columns:
            - yardline_100 (0 to 100)
            - down (1 to 4)
            - ydstogo (1 to 99)
            - yards_gained
            - interception
            - fumble_lost
            - penalty
            - penalty_yards
            - play_type, score_differential, quarter/qtr (only needed for
              the game-state-aware table; transitions_by_game_state stays
              None without them)
        """
        logger.info(f"Fitting DriveMarkovModel on {len(pbp_df)} PBP plays...")
        df = pbp_df.copy()

        # Standardize missing columns safely
        for col in ["yardline_100", "down", "ydstogo", "yards_gained"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        if "interception" not in df.columns:
            df["interception"] = 0
        if "fumble_lost" not in df.columns:
            df["fumble_lost"] = 0
        if "penalty" not in df.columns:
            df["penalty"] = 0
        if "penalty_yards" not in df.columns:
            df["penalty_yards"] = 0

        # Drop plays without down/distance
        df = df.dropna(subset=["yardline_100", "down", "ydstogo"]).copy()
        df = df[df["down"] <= 4]

        # Map to discrete buckets matching C++ (table_index)
        # uint32_t fp_bucket = std::min(fp / 10, 10)
        df["fp_bucket"] = np.clip(df["yardline_100"] // 10, 0, N_FP_BUCKETS - 1).astype(int)

        # uint32_t down_idx = std::min(down - 1, 3)
        df["down_idx"] = np.clip(df["down"] - 1, 0, N_DOWN - 1).astype(int)

        # uint32_t ytg_bucket = std::min((ytg - 1) / 5, 5)
        df["ytg_bucket"] = np.clip((df["ydstogo"] - 1) // 5, 0, N_YTG_BUCKETS - 1).astype(int)

        # Define outcome booleans
        df["is_turnover"] = ((df["interception"] == 1) | (df["fumble_lost"] == 1)).astype(int)
        df["is_pen_gain"] = ((df["penalty"] == 1) & (df["penalty_yards"] > 0)).astype(int)
        df["is_pen_loss"] = ((df["penalty"] == 1) & (df["penalty_yards"] < 0)).astype(int)

        # Group by the 3D state
        group = df.groupby(["fp_bucket", "down_idx", "ytg_bucket"])

        agg_df = group.agg(
            mean_gain=("yards_gained", "mean"),
            gain_std=("yards_gained", "std"),
            p_turnover=("is_turnover", "mean"),
            p_penalty_gain=("is_pen_gain", "mean"),
            p_penalty_loss=("is_pen_loss", "mean"),
            count=("yards_gained", "count")
        ).reset_index()

        # Fill missing variance with global average if 1 sample
        global_std = df["yards_gained"].std()
        agg_df["gain_std"] = agg_df["gain_std"].fillna(global_std)

        # Save internally
        self.transitions = agg_df
        logger.info(f"Extracted {len(self.transitions)} unique state transitions.")

        self.transitions_by_game_state = self._fit_game_state(df)
        return self

    def _fit_game_state(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Extend the (fp, down, ytg) state with (score_diff_bucket, quarter) and
        add p_pass — the play-calling mix, not just the yardage outcome. This
        is what lets "leading teams run more in Q4" emerge from the fitted
        table rather than being hardcoded. Requires play_type,
        score_differential, and quarter (or qtr); returns None if any are
        missing rather than fitting on a silently-degraded state space.
        """
        needed = {"play_type", "score_differential"}
        quarter_col = "quarter" if "quarter" in df.columns else ("qtr" if "qtr" in df.columns else None)
        if not needed.issubset(df.columns) or quarter_col is None:
            logger.warning(
                "Skipping game-state-aware fit: missing one of play_type/"
                "score_differential/quarter(qtr)."
            )
            return None

        gs = df.dropna(subset=["score_differential", quarter_col, "play_type"]).copy()
        gs = gs[gs["play_type"].isin(["run", "pass"])]
        if gs.empty:
            logger.warning("Skipping game-state-aware fit: no run/pass plays with full state.")
            return None

        gs["score_diff_bucket"] = _score_diff_bucket(gs["score_differential"])
        gs["quarter_idx"] = np.clip(gs[quarter_col].astype(int), 1, N_QUARTER) - 1
        gs["is_pass"] = (gs["play_type"] == "pass").astype(int)

        state_cols = ["fp_bucket", "down_idx", "ytg_bucket", "score_diff_bucket", "quarter_idx"]
        cell = gs.groupby(state_cols).agg(
            mean_gain=("yards_gained", "mean"),
            gain_std=("yards_gained", "std"),
            p_turnover=("is_turnover", "mean"),
            p_penalty_gain=("is_pen_gain", "mean"),
            p_penalty_loss=("is_pen_loss", "mean"),
            p_pass=("is_pass", "mean"),
            count=("yards_gained", "count"),
        ).reset_index()

        marginal = gs.groupby(["fp_bucket", "down_idx", "ytg_bucket"]).agg(
            mean_gain_m=("yards_gained", "mean"),
            gain_std_m=("yards_gained", "std"),
            p_turnover_m=("is_turnover", "mean"),
            p_penalty_gain_m=("is_pen_gain", "mean"),
            p_penalty_loss_m=("is_pen_loss", "mean"),
            p_pass_m=("is_pass", "mean"),
        ).reset_index()

        global_std = gs["yards_gained"].std()
        marginal["gain_std_m"] = marginal["gain_std_m"].fillna(global_std)

        cell = cell.merge(marginal, on=["fp_bucket", "down_idx", "ytg_bucket"], how="left")
        cell["gain_std"] = cell["gain_std"].fillna(cell["gain_std_m"]).fillna(global_std)
        cell = self._shrink_to_marginal(cell)

        logger.info(
            "Extracted %d game-state transitions (%d marginal cells) from %d plays.",
            len(cell), len(marginal), len(gs),
        )
        return cell

    @staticmethod
    def _shrink_to_marginal(cell: pd.DataFrame, k: float = MIN_CELL_COUNT_FOR_CONFIDENCE) -> pd.DataFrame:
        """
        Empirical-Bayes-style shrinkage: a cell with `count` observations is
        weighted count/(count+k) toward its own mean and k/(count+k) toward
        the coarser (fp, down, ytg) marginal. A cell with 0 samples (didn't
        occur, e.g. 4th-and-25 while leading big in Q1) falls back entirely
        to the marginal; a cell with hundreds of samples barely moves.
        """
        n = cell["count"].astype(float)
        w = n / (n + k)
        for col in ["mean_gain", "p_turnover", "p_penalty_gain", "p_penalty_loss", "p_pass"]:
            cell[col] = w * cell[col] + (1 - w) * cell[f"{col}_m"]
        cell = cell.drop(columns=[c for c in cell.columns if c.endswith("_m")])
        return cell

    def export_transitions(self, out_path: str) -> None:
        """Export to CSV for C++ `drive_mcmc.cpp` ingestion."""
        if self.transitions is None:
            raise RuntimeError("Must call fit() before export_transitions().")
            
        out_df = self.transitions.copy()
        
        # Global fill for any completely sparse states requested by C++ engine
        # But C++ defaults to base rates if a cell isn't loaded, so we just
        # export what we have.
        
        cols = [
            "fp_bucket", "down_idx", "ytg_bucket", 
            "mean_gain", "gain_std", "p_turnover", 
            "p_penalty_gain", "p_penalty_loss"
        ]
        
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        out_df[cols].to_csv(out_file, index=False)
        logger.info(f"Exported transitions to {out_file}")

    def export_transitions_by_game_state(self, out_path: str) -> None:
        """Export the (fp, down, ytg, score_diff_bucket, quarter) table with p_pass."""
        if self.transitions_by_game_state is None:
            raise RuntimeError(
                "No game-state transitions to export — fit() either wasn't called "
                "or the input frame was missing play_type/score_differential/quarter."
            )

        cols = [
            "fp_bucket", "down_idx", "ytg_bucket", "score_diff_bucket", "quarter_idx",
            "mean_gain", "gain_std", "p_turnover",
            "p_penalty_gain", "p_penalty_loss", "p_pass", "count",
        ]

        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        self.transitions_by_game_state[cols].to_csv(out_file, index=False)
        logger.info(f"Exported game-state transitions to {out_file}")
