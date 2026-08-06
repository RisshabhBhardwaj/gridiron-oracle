"""
ml/survival_model.py

Player Availability Survival Analysis — Cox Proportional Hazards Model.

PROBLEM
-------
When a player goes on IR or misses a game, downstream Monte Carlo simulations
need to answer: "What is the probability this player is available for week W?"

A naive approach — assume the player is always healthy — overestimates volume
for injured rosters and inflates SGP hit rates.

SOLUTION — SURVIVAL ANALYSIS
-----------------------------
Model each player's "time to return from injury" as a survival function:
    S(t) = P(player returns by week t | missed streak k, status, snap rate)

We use the Cox Proportional Hazards (CoxPH) model from the `lifelines` library:
    h(t | x) = h₀(t) × exp(β·x)

where:
    - h₀(t) is the baseline hazard (fitted from historical IR/missed games data)
    - x is a covariate vector derived from our feature_matrix
    - β are the learned coefficients

Covariates:
    - games_missed_streak:   consecutive weeks missed (duration feature)
    - injury_status_encoded: 0=Out, 1=Doubtful, 2=Questionable, 3=Limited, 4=Full
    - snap_pct_off:          snap share pre-injury (proxy for player importance)
    - age:                   older players take longer to recover
    - position_encoded:      0=QB, 1=RB, 2=WR, 3=TE (positional recovery differences)

USAGE
-----
    from ml.survival_model import PlayerSurvivalModel

    model = PlayerSurvivalModel()
    model.fit(survival_df)   # DataFrame from build_survival_dataset()

    # P(player available in each of next 3 weeks)
    probs = model.predict_return_probability(
        games_missed_streak=2,
        injury_status_encoded=2,   # Questionable
        snap_pct_off=0.65,
        age=28.0,
        position="WR",
        n_weeks=3,
    )
    # → [0.45, 0.72, 0.88]   P(available wk+1, wk+2, wk+3)

Integration:
    SeasonSimulator calls model.predict_return_probability() to apply
    week-specific injury discount multipliers instead of binary out/in flags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

POSITION_ENCODING = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
_MIN_TRAINING_ROWS = 50    # minimum rows to fit CoxPH (fall back to heuristic otherwise)
_RETURN_HEURISTIC_BY_STATUS = {
    0: 0.05,   # Out → 5% per week
    1: 0.20,   # Doubtful → 20% per week
    2: 0.50,   # Questionable → 50% per week
    3: 0.75,   # Limited → 75% per week
    4: 0.95,   # Full → 95% per week
}


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class SurvivalRecord:
    """
    One observation in the survival dataset.
    Used to build the training DataFrame for CoxPH fitting.
    """
    player_id:             str
    season:                int
    week_missed_start:     int
    duration:              int    # weeks until return (or censoring)
    event_observed:        bool   # True if player actually returned
    games_missed_streak:   int    # consecutive games missed at time of entry
    injury_status_encoded: int    # ESPN injury status
    snap_pct_off:          float  # pre-injury snap rate
    age:                   float
    position_encoded:      int


# ── Heuristic Fallback ─────────────────────────────────────────────────────────

def _heuristic_return_probabilities(
    injury_status_encoded: int,
    n_weeks: int,
) -> list[float]:
    """
    Fallback to per-week return probability heuristic when CoxPH not fitted.

    Uses simplified geometric decay model based on ESPN status encoding.
    """
    p_week = _RETURN_HEURISTIC_BY_STATUS.get(injury_status_encoded, 0.50)
    probs = []
    survival = 1.0 - p_week  # probability still injured after each week passes
    for _ in range(n_weeks):
        probs.append(1.0 - survival)
        survival *= (1.0 - p_week)
    return probs


# ── PlayerSurvivalModel ────────────────────────────────────────────────────────

class PlayerSurvivalModel:
    """
    Cox Proportional Hazards survival model for player availability.

    Fits from historical missed-game sequences and predicts week-by-week
    return probabilities for currently injured players.

    State:
        _model:   lifelines.CoxPHFitter (or None before fit())
        _fitted:  bool
    """

    def __init__(self) -> None:
        self._model = None
        self._fitted = False
        self._covariates = [
            "games_missed_streak",
            "injury_status_encoded",
            "snap_pct_off",
            "age",
            "position_encoded",
        ]

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, survival_df: pd.DataFrame) -> "PlayerSurvivalModel":
        """
        Fit the Cox Proportional Hazards model.

        Args:
            survival_df: DataFrame with columns:
                [duration, event_observed, games_missed_streak,
                 injury_status_encoded, snap_pct_off, age, position_encoded]
                Produced by build_survival_dataset() or manually constructed.

        Returns:
            self (for chaining)
        """
        try:
            from lifelines import CoxPHFitter
        except ImportError:
            logger.warning(
                "lifelines not installed — survival model will use heuristic fallback. "
                "Install: pip install lifelines"
            )
            self._fitted = False
            return self

        required_cols = ["duration", "event_observed"] + self._covariates
        missing = [c for c in required_cols if c not in survival_df.columns]
        if missing:
            logger.warning(
                "survival_df missing columns %s — using heuristic fallback.", missing
            )
            self._fitted = False
            return self

        df = survival_df[required_cols].dropna()
        if len(df) < _MIN_TRAINING_ROWS:
            logger.warning(
                "survival_df has only %d rows (minimum %d). Using heuristic fallback.",
                len(df), _MIN_TRAINING_ROWS,
            )
            self._fitted = False
            return self

        try:
            cph = CoxPHFitter(penalizer=0.1)  # L2 regularization for stability
            cph.fit(
                df,
                duration_col="duration",
                event_col="event_observed",
                show_progress=False,
            )
            self._model = cph
            self._fitted = True
            logger.info(
                "PlayerSurvivalModel fitted: %d observations, concordance=%.3f",
                len(df),
                cph.concordance_index_,
            )
        except Exception as exc:
            logger.warning("CoxPHFitter.fit() failed (%s) — using heuristic fallback.", exc)
            self._fitted = False

        return self

    def predict_return_probability(
        self,
        games_missed_streak: int,
        injury_status_encoded: int,
        snap_pct_off: float = 0.6,
        age: float = 26.0,
        position: str = "WR",
        n_weeks: int = 4,
    ) -> list[float]:
        """
        Predict P(player available by week t) for t = 1 … n_weeks.

        Args:
            games_missed_streak:   consecutive games already missed (>=0)
            injury_status_encoded: ESPN status code (0=Out … 4=Full)
            snap_pct_off:          pre-injury snap share [0, 1]
            age:                   player age in years
            position:              position string (QB/RB/WR/TE)
            n_weeks:               number of future weeks to project

        Returns:
            list[float] of length n_weeks. Each value is P(available) for
            that week, ranging from (low) to (high) over time.
        """
        if not self._fitted or self._model is None:
            return _heuristic_return_probabilities(injury_status_encoded, n_weeks)

        pos_enc = POSITION_ENCODING.get(position.upper(), 2)  # default WR

        # Build covariate DataFrame for lifelines prediction
        covariate_df = pd.DataFrame([{
            "games_missed_streak":   games_missed_streak,
            "injury_status_encoded": injury_status_encoded,
            "snap_pct_off":          snap_pct_off,
            "age":                   age,
            "position_encoded":      pos_enc,
        }])

        try:
            # predict_survival_function returns a DataFrame indexed by time
            sf = self._model.predict_survival_function(covariate_df)
            # sf.columns = [0] (one player)
            # sf.index = sorted unique durations seen during training
            # We interpolate at t = 1, 2, ..., n_weeks
            t_values = np.arange(1, n_weeks + 1)
            sf_values = sf.iloc[:, 0]

            probs = []
            for t in t_values:
                # S(t) = P(not yet returned by week t)
                # Return probability = 1 - S(t)
                # Interpolate since sf is defined at training time points
                idx = sf_values.index.searchsorted(t, side="right") - 1
                idx = max(0, min(idx, len(sf_values) - 1))
                s_t = float(sf_values.iloc[idx])
                probs.append(float(np.clip(1.0 - s_t, 0.0, 1.0)))

            return probs
        except Exception as exc:
            logger.warning(
                "survival prediction failed (%s), falling back to heuristic.", exc
            )
            return _heuristic_return_probabilities(injury_status_encoded, n_weeks)

    def predict_availability_for_week(
        self,
        games_missed_streak: int,
        injury_status_encoded: int,
        weeks_until_target: int,
        **kwargs,
    ) -> float:
        """
        Convenience method: return P(available) for a single target week.

        Args:
            games_missed_streak:   weeks already missed
            injury_status_encoded: ESPN status
            weeks_until_target:    how many weeks from now is the target week (>=1)
            **kwargs:              forwarded to predict_return_probability

        Returns:
            float P(available for target week)
        """
        probs = self.predict_return_probability(
            games_missed_streak=games_missed_streak,
            injury_status_encoded=injury_status_encoded,
            n_weeks=weeks_until_target,
            **kwargs,
        )
        return probs[-1] if probs else 0.0

    def summary(self) -> Optional[pd.DataFrame]:
        """Return coefficient summary DataFrame if model is fitted."""
        if self._fitted and self._model is not None:
            return self._model.summary
        return None

    def plot_survival_function(self, out_path: str = "ml/survival_function.png") -> None:
        """Plot and save the baseline survival function."""
        if not self._fitted or self._model is None:
            logger.warning("Model not fitted — cannot plot survival function.")
            return
        try:
            import matplotlib.pyplot as plt
            ax = self._model.baseline_survival_.plot(
                title="Baseline Survival Function — Player Return from Injury",
                xlabel="Weeks",
                ylabel="P(still injured)",
            )
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            ax.get_figure().savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close()
            logger.info("Survival function plot saved: %s", out_path)
        except ImportError:
            logger.warning("matplotlib not installed — skipping plot.")

    def save(self, path: str) -> None:
        """Pickle the fitted model to disk."""
        if not self._fitted:
            logger.warning("Cannot save unfitted model.")
            return
        import pickle
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f, protocol=4)
        logger.info("PlayerSurvivalModel saved: %s", path)

    def load(self, path: str) -> "PlayerSurvivalModel":
        """Load a pickled model."""
        import pickle
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        self._fitted = True
        logger.info("PlayerSurvivalModel loaded: %s", path)
        return self


# ── Dataset Builder ────────────────────────────────────────────────────────────

def build_survival_dataset(
    feature_matrix_df: pd.DataFrame,
    game_logs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a survival analysis training dataset from feature_matrix + game_logs.

    A "missed game event" starts when a player has no game_log entry for a
    given week. Duration is the number of consecutive missed weeks. Event is
    observed if the player returned within the season.

    Args:
        feature_matrix_df: DataFrame from FeatureMatrix with:
            [player_id, season, week, games_missed_streak,
             injury_status_encoded, snap_pct_off]
        game_logs_df: DataFrame from GameLog with [player_id, season, week]

    Returns:
        DataFrame with [duration, event_observed, games_missed_streak,
        injury_status_encoded, snap_pct_off, age, position_encoded]
    """
    logger.info("Building survival dataset from %d feature rows…", len(feature_matrix_df))

    played_set = set(
        (r["player_id"], r["season"], r["week"])
        for _, r in game_logs_df.iterrows()
    )

    records = []
    for player_id, grp in feature_matrix_df.groupby("player_id"):
        grp = grp.sort_values("week")
        for _, row in grp.iterrows():
            streak = int(row.get("games_missed_streak") or 0)
            if streak < 1:
                continue  # player was active — not an injury entry

            # Determine if player returned in subsequent weeks
            season = int(row["season"])
            week   = int(row["week"])
            returned = any((player_id, season, w) in played_set for w in range(week + 1, 24))

            records.append({
                "duration":              max(1, streak),
                "event_observed":        int(returned),
                "games_missed_streak":   streak,
                "injury_status_encoded": int(row.get("injury_status_encoded") or 2),
                "snap_pct_off":          float(row.get("snap_pct_off") or 0.6),
                "age":                   float(row.get("age") or 26.0),
                "position_encoded":      POSITION_ENCODING.get(
                    str(row.get("position") or "WR").upper(), 2
                ),
            })

    df = pd.DataFrame(records)
    logger.info("Survival dataset: %d injury events.", len(df))
    return df


# ── Singleton ─────────────────────────────────────────────────────────────────

_GLOBAL_SURVIVAL_MODEL: Optional[PlayerSurvivalModel] = None


def get_survival_model() -> PlayerSurvivalModel:
    global _GLOBAL_SURVIVAL_MODEL
    if _GLOBAL_SURVIVAL_MODEL is None:
        _GLOBAL_SURVIVAL_MODEL = PlayerSurvivalModel()
    return _GLOBAL_SURVIVAL_MODEL


def reset_survival_model() -> None:
    global _GLOBAL_SURVIVAL_MODEL
    _GLOBAL_SURVIVAL_MODEL = None
