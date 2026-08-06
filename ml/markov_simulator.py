"""
ml/markov_simulator.py

Drive-Level Markov Chain Simulator.

OVERVIEW
--------
Extracts empirical transition matrices from Play-By-Play (PBP) data
and prepares them for the C++ DriveMCMC engine.

Football plays are modeled as state transitions:
    State:  (field_position_bucket, down, yards_to_go_bucket)
    Action: (Implicitly determined by NFL average coaching or specific team)
    Transition: (mean_gain, gain_std, p_turnover, p_penalty_gain, p_penalty_loss)

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

class DriveMarkovModel:
    """
    Builds the state-transition probability matrices for drove-level Markov Monte Carlo.
    """
    def __init__(self):
        self.transitions: Optional[pd.DataFrame] = None
        
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
        return self
        
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
