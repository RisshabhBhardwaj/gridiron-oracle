"""Evaluate the Gaussian copula against an independence assumption (SP5).

Run this before committing drive-level MCMC as the correlation engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pairwise_residual_correlation(
    frame: pd.DataFrame,
    *,
    player_a: str,
    player_b: str,
    pred_col: str = "y_pred",
    actual_col: str = "y_true",
) -> float:
    """Same-game residual correlation; NaN if fewer than 8 overlapping games."""
    a = frame[frame["player_id"] == player_a][["game_id", pred_col, actual_col]].rename(
        columns={pred_col: "pred_a", actual_col: "act_a"}
    )
    b = frame[frame["player_id"] == player_b][["game_id", pred_col, actual_col]].rename(
        columns={pred_col: "pred_b", actual_col: "act_b"}
    )
    merged = a.merge(b, on="game_id")
    if len(merged) < 8:
        return float("nan")
    resid_a = pd.to_numeric(merged["act_a"], errors="coerce") - pd.to_numeric(merged["pred_a"], errors="coerce")
    resid_b = pd.to_numeric(merged["act_b"], errors="coerce") - pd.to_numeric(merged["pred_b"], errors="coerce")
    return float(resid_a.corr(resid_b))


def independence_overstates_joint(
    p_a: float,
    p_b: float,
    empirical_joint: float,
) -> bool:
    """True when P(A)P(B) is further from the empirical joint than a copula-adjusted value would need to be."""
    independent = float(p_a) * float(p_b)
    return abs(independent - empirical_joint) > 0.02
