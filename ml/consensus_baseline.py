"""Consensus / ADP baselines for the SP1 scoreboard.

Contemporaneous xFP is not a baseline. Rank metrics weight the top of the
draft more than the tail.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def adp_ranks(frame: pd.DataFrame, *, adp_col: str = "adp") -> pd.Series:
    return pd.to_numeric(frame[adp_col], errors="coerce").rank(method="average", ascending=True)


def top_n_hit_rate(
    model_rank: Sequence[float],
    market_rank: Sequence[float],
    *,
    n: int = 24,
) -> float:
    model = np.asarray(model_rank, dtype=float)
    market = np.asarray(market_rank, dtype=float)
    mask = np.isfinite(model) & np.isfinite(market)
    if mask.sum() < n:
        return float("nan")
    model_top = set(np.where(mask)[0][np.argsort(model[mask])[:n]])
    market_top = set(np.where(mask)[0][np.argsort(market[mask])[:n]])
    return float(len(model_top & market_top) / n)


def points_lost_vs_optimal(
    projection: Sequence[float],
    realized: Sequence[float],
    *,
    slots: int = 24,
) -> float:
    """How many realized points a projection ranking leaves on the table vs oracle."""
    proj = np.asarray(projection, dtype=float)
    actual = np.asarray(realized, dtype=float)
    mask = np.isfinite(proj) & np.isfinite(actual)
    if mask.sum() < slots:
        return float("nan")
    picked = actual[mask][np.argsort(-proj[mask])[:slots]].sum()
    oracle = np.sort(actual[mask])[-slots:].sum()
    return float(oracle - picked)
