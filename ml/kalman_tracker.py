"""
ml/kalman_tracker.py

Scalar Kalman filter for tracking latent player ability per stat.
Replaces compute_rolling_form() in pipeline/feature_engineer.py.

STATE MODEL
-----------
  x_k = x_{k-1} + w_k        (w_k ~ N(0, Q))  -- latent ability random walk
  y_k = x_k + v_k             (v_k ~ N(0, R))  -- noisy observation

Kalman equations for F=1, H=1 (scalar, identity transitions):

  Predict:
    x_pred = x_{k-1}
    P_pred = P_{k-1} + Q

  Update:
    K   = P_pred / (P_pred + R)
    x_k = x_pred + K * (y_k - x_pred)
    P_k = (1 - K) * P_pred

Parameters:
  Q  -- process noise variance (ability drift), constructor hyperparameter,
        default=1.0, tune via Optuna in Phase 4.
  R  -- measurement noise variance, estimated from player historical variance.
        Falls back to 100.0 if fewer than 2 observations are available.
  x0 -- initial state mean: position-average prior for cold-start.
  P0 -- initial state variance: INITIAL_VARIANCE (large = high uncertainty).

CAUSALITY GUARANTEE
-------------------
predict_sequence(observations) returns the POSTERIOR after incorporating each
observation. The caller (build_feature_row) passes only prior_rows -- game
rows from weeks BEFORE the target game. This ensures no future leakage.

For feature engineering at week k+1 (after k prior games):
  estimates = tracker.predict_sequence([y_1, ..., y_k])
  est, var  = estimates[-1]   # posterior after y_1..y_k
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Initial state covariance -- large value signals high initial uncertainty.
INITIAL_VARIANCE: float = 1_000.0

# Default process noise Q. Same value used for all stats.
# Q=1.0 may be too large for rate/share stats (target_share, red_zone_target_share)
# — known design limitation; Phase 4 will add stat-specific Q tuning via Optuna.
DEFAULT_Q: float = 1.0

# Minimum allowed R to avoid division-by-zero for perfectly consistent players.
_MIN_R: float = 1e-6

# Stats tracked per player -- one KalmanTracker instance per (player, stat).
# These names appear in output keys: kalman_est_{stat}, kalman_variance_{stat}.
# All 7 seasons supported: 2019-2025 (complete season data available).
KALMAN_STATS: list[str] = [
    # Receiving
    "receiving_yards",
    "receiving_tds",
    "targets",
    "receptions",
    "target_share",
    "red_zone_target_share",
    "air_yards_share",
    # General
    "fantasy_ppr",       # source col in prior_rows: "fantasy_points_ppr"
    # Rushing
    "carries",
    "rushing_yards",
    "rushing_tds",
    # Passing (QB)
    "pass_attempts",     # source col: "attempts" in nflreadpy game_logs
    "completions",       # source col: "completions" in nflreadpy game_logs
    "passing_yards",
    "passing_tds",
    "interceptions",
    # Zero-inflated rare events (fumbles apply to all positions)
    "fumbles",           # source col: "sack_fumbles" + "rushing_fumbles" + "receiving_fumbles"
]

# Maps KALMAN_STATS name → source column name in prior_rows dicts (game_logs schema).
STAT_SOURCE_COL: dict[str, str] = {stat: stat for stat in KALMAN_STATS}
STAT_SOURCE_COL["fantasy_ppr"]    = "fantasy_points_ppr"
STAT_SOURCE_COL["pass_attempts"]  = "attempts"               # nflreadpy column name
STAT_SOURCE_COL["interceptions"]  = "passing_interceptions"   # nflreadpy / game_logs column name
# "_fumbles_combined" is a sentinel: compute_kalman_form() sums the component columns.
STAT_SOURCE_COL["fumbles"]        = "_fumbles_combined"

# Multi-source columns: stat keys that map to a list of source columns to be summed.
# All fumble types: rushing, receiving (WR/RB/TE), and sack (QB).
# game_logs must have receiving_fumbles and sack_fumbles (add via migration if absent).
STAT_MULTI_SOURCE_COLS: dict[str, list[str]] = {
    "_fumbles_combined": [
        "rushing_fumbles",
        "receiving_fumbles",
        "sack_fumbles",
    ],
}

# Position-average priors for cold-start (zero prior games).
# Values are 2019-2025 season averages per game for each position,
# derived from nflreadpy player_stats via scripts/compute_position_priors.py.
POSITION_PRIORS: dict[str, dict[str, float]] = {
    "WR": {
        "receiving_yards":    55.0,
        "receiving_tds":       0.40,
        "targets":             5.5,
        "receptions":          3.5,
        "target_share":        0.15,
        "red_zone_target_share": 0.15,
        "air_yards_share":     0.12,
        "fantasy_ppr":        10.0,
        "carries":             0.3,
        "rushing_yards":       2.0,
        "rushing_tds":         0.02,
        "pass_attempts":       0.0,
        "completions":         0.0,
        "passing_yards":       0.0,
        "passing_tds":         0.0,
        "interceptions":       0.0,
        "fumbles":             0.01,   # WRs average ~0.01 fumbles/game
    },
    "RB": {
        "receiving_yards":    20.0,
        "receiving_tds":       0.20,
        "targets":             3.0,
        "receptions":          2.5,
        "target_share":        0.07,
        "red_zone_target_share": 0.05,
        "air_yards_share":     0.03,
        "fantasy_ppr":        12.0,
        "carries":            14.0,
        "rushing_yards":      60.0,
        "rushing_tds":         0.50,
        "pass_attempts":       0.0,
        "completions":         0.0,
        "passing_yards":       0.0,
        "passing_tds":         0.0,
        "interceptions":       0.0,
        "fumbles":             0.04,   # RBs average ~0.04 fumbles/game (highest)
    },
    "TE": {
        "receiving_yards":    35.0,
        "receiving_tds":       0.30,
        "targets":             4.0,
        "receptions":          3.0,
        "target_share":        0.10,
        "red_zone_target_share": 0.12,
        "air_yards_share":     0.07,
        "fantasy_ppr":          8.0,
        "carries":             0.05,
        "rushing_yards":       0.3,
        "rushing_tds":         0.01,
        "pass_attempts":       0.0,
        "completions":         0.0,
        "passing_yards":       0.0,
        "passing_tds":         0.0,
        "interceptions":       0.0,
        "fumbles":             0.01,   # TEs average ~0.01 fumbles/game
    },
    "QB": {
        "receiving_yards":     0.0,
        "receiving_tds":       0.0,
        "targets":             0.0,
        "receptions":          0.0,
        "target_share":        0.0,
        "red_zone_target_share": 0.0,
        "air_yards_share":     0.0,
        "fantasy_ppr":        20.0,
        "carries":             5.0,
        "rushing_yards":      25.0,
        "rushing_tds":         0.25,
        "pass_attempts":      35.0,   # NFL avg starter ~35 att/game 2019-2025
        "completions":        22.0,   # NFL avg ~22 comp/game (comp% ~63%)
        "passing_yards":     240.0,
        "passing_tds":         1.60,
        "interceptions":       0.90,  # NFL avg ~0.9 INTs/game per QB
        "fumbles":             0.30,  # QBs fumble most (scrambles + sack fumbles)
    },
}

# Fallback when position is unknown -- safe zero prior.
_DEFAULT_PRIOR: dict[str, float] = {stat: 0.0 for stat in KALMAN_STATS}


# ---------------------------------------------------------------------------
# KalmanTracker
# ---------------------------------------------------------------------------

class KalmanTracker:
    """
    Scalar Kalman filter for one (player, stat) pair.

    Usage:
        tracker = KalmanTracker(Q=1.0, x0=55.0)
        tracker.fit(player_history_df, stat_col="receiving_yards")
        estimates = tracker.predict_sequence([70.0, 85.0, 60.0, 120.0])
        est, var = estimates[-1]   # posterior after all 4 observations
    """

    def __init__(
        self,
        Q:  float = DEFAULT_Q,
        x0: float = 0.0,
        P0: float = INITIAL_VARIANCE,
    ) -> None:
        self.Q  = Q
        self.x0 = x0
        self.P0 = P0
        self.R: Optional[float] = None   # set by fit()

    def fit(self, player_history: pd.DataFrame, stat_col: str) -> "KalmanTracker":
        """
        Estimate measurement noise R from player historical data variance.

        R = sample variance (ddof=1) of stat_col over all non-null rows.
        Falls back to 100.0 when fewer than 2 non-null observations exist.

        Returns self for chaining.
        """
        if stat_col not in player_history.columns:
            self.R = 100.0
            return self

        vals = player_history[stat_col].dropna().values
        if len(vals) < 2:
            self.R = 100.0
        else:
            r_est = float(np.var(vals, ddof=1))
            self.R = max(r_est, _MIN_R)

        return self

    def predict_sequence(
        self,
        observations: List[float],
    ) -> List[Tuple[float, float]]:
        """
        Run Kalman filter over a sequence of observations.

        Returns the POSTERIOR (mean, variance) after incorporating each
        observation in turn. Entry at position k incorporates observations[0..k].

        STRICTLY CAUSAL for feature engineering: caller passes only prior game
        observations. The last element is the correct Kalman estimate for the
        upcoming game.

        Args:
            observations: [y_1, y_2, ..., y_n] in chronological order.

        Returns:
            [(mean_1, var_1), ..., (mean_n, var_n)] -- posteriors after each y_k.
            Empty list when observations is empty.

        Raises:
            RuntimeError: if fit() has not been called.
        """
        if self.R is None:
            raise RuntimeError(
                "KalmanTracker.R is None -- call fit() before predict_sequence()."
            )

        results: List[Tuple[float, float]] = []
        x: float = self.x0
        P: float = self.P0

        for y in observations:
            # Predict
            x_pred = x
            P_pred = P + self.Q

            # Update
            K = P_pred / (P_pred + self.R)
            x = x_pred + K * (y - x_pred)
            P = (1.0 - K) * P_pred

            results.append((x, P))

        return results


# ---------------------------------------------------------------------------
# KalmanFeatureEngineer
# ---------------------------------------------------------------------------

class KalmanFeatureEngineer:
    """
    Applies KalmanTracker per stat to produce Kalman form features.
    Replaces compute_rolling_form() in pipeline/feature_engineer.py.

    Args:
        Q: Process noise hyperparameter (default 1.0).
    """

    def __init__(self, Q: float = DEFAULT_Q) -> None:
        self.Q = Q

    def compute_kalman_form(
        self,
        prior_rows: list[dict],
        position: Optional[str] = None,
    ) -> dict[str, Optional[float]]:
        """
        Compute Kalman posterior estimates from a player's prior game rows.

        For each stat in KALMAN_STATS:
          - Seeds x_0 from position-average prior (cold-start safety).
          - Estimates R from empirical variance in prior_rows
            (falls back to 100.0 when fewer than 2 non-null values exist).
          - Runs Kalman filter over all prior observations.
          - Returns kalman_est_{stat} and kalman_variance_{stat}.

        Cold-start (0 prior rows):
          kalman_est_{stat}      = position-average prior (x_0)
          kalman_variance_{stat} = INITIAL_VARIANCE (1000.0)

        Args:
            prior_rows: game rows from weeks BEFORE the target game,
                        sorted oldest-first.
            position:   player position ("WR"/"RB"/"QB"/"TE").

        Returns:
            dict with kalman_est_{stat} and kalman_variance_{stat} for
            every stat in KALMAN_STATS.
        """
        pos_priors = POSITION_PRIORS.get(position or "", None)
        if pos_priors is None:
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "KalmanFeatureEngineer: unknown position %r — using zero priors.",
                position,
            )
            pos_priors = _DEFAULT_PRIOR
        result: dict[str, Optional[float]] = {}

        for stat in KALMAN_STATS:
            x0 = pos_priors.get(stat, 0.0)

            if not prior_rows:
                # Cold-start: no history.
                result[f"kalman_est_{stat}"]      = x0
                result[f"kalman_variance_{stat}"] = INITIAL_VARIANCE
                continue

            # Use source column name (e.g. "fantasy_points_ppr" for "fantasy_ppr").
            src_col = STAT_SOURCE_COL[stat]

            # Multi-source sentinel: sum component columns instead of looking up a single col.
            is_multi = src_col.startswith("_") and src_col in STAT_MULTI_SOURCE_COLS

            def _extract_val(row: dict) -> Optional[float]:
                if is_multi:
                    parts = [row.get(c) for c in STAT_MULTI_SOURCE_COLS[src_col]]
                    non_none = [v for v in parts if v is not None]
                    return sum(float(v) for v in non_none) if non_none else None
                v = row.get(src_col)
                return float(v) if v is not None else None

            # Estimate R from player's historical variance.
            raw_vals = [_extract_val(r) for r in prior_rows]
            raw_vals_clean = [v for v in raw_vals if v is not None]
            if len(raw_vals_clean) >= 2:
                r_est = float(np.var(raw_vals_clean, ddof=1))
                R = max(r_est, _MIN_R)
            else:
                R = 100.0

            tracker = KalmanTracker(Q=self.Q, x0=x0, P0=INITIAL_VARIANCE)
            tracker.R = R

            # Replace None with position prior (best guess for missing games).
            observations = [
                v if v is not None else x0
                for v in (_extract_val(r) for r in prior_rows)
            ]

            estimates = tracker.predict_sequence(observations)
            est, var = estimates[-1]
            result[f"kalman_est_{stat}"]      = est
            result[f"kalman_variance_{stat}"] = var

        return result
