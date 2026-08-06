"""Market-aware evaluation for historical player-prop backtests.

Rows must be timestamped snapshots selected by the caller's published closing
policy.  This module never selects a favourable bookmaker or line after seeing
the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def american_to_implied_probability(odds: np.ndarray) -> np.ndarray:
    odds = np.asarray(odds, dtype=float)
    if np.any(odds == 0):
        raise ValueError("American odds cannot be zero")
    return np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0))


def no_vig_probability(over_odds: np.ndarray, under_odds: np.ndarray) -> np.ndarray:
    over = american_to_implied_probability(over_odds)
    under = american_to_implied_probability(under_odds)
    return over / (over + under)


@dataclass(frozen=True)
class MarketMetrics:
    brier_score: float
    log_loss: float
    pnl_units: float
    roi: float
    sharpe: float
    max_drawdown: float
    n_bets: int


def evaluate_over_strategy(
    actual: np.ndarray,
    line: np.ndarray,
    model_probability: np.ndarray,
    over_odds: np.ndarray,
    *,
    edge_threshold: float = 0.02,
    stake_fraction: float = 0.01,
) -> MarketMetrics:
    """Score a fixed, predeclared over-only strategy in bankroll units."""
    actual, line, probability, odds = [np.asarray(x, dtype=float) for x in (actual, line, model_probability, over_odds)]
    if not (len(actual) == len(line) == len(probability) == len(odds)):
        raise ValueError("market evaluation arrays must have equal length")
    implied = american_to_implied_probability(odds)
    outcome = (actual > line).astype(float)
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    brier = float(np.mean((probability - outcome) ** 2))
    log_loss = float(-np.mean(outcome * np.log(probability) + (1 - outcome) * np.log(1 - probability)))
    bet = probability - implied >= edge_threshold
    returns = np.zeros(len(actual), dtype=float)
    decimal_profit = np.where(odds < 0, 100.0 / -odds, odds / 100.0)
    returns[bet] = np.where(outcome[bet] == 1.0, stake_fraction * decimal_profit[bet], -stake_fraction)
    equity = np.cumsum(returns)
    drawdown = equity - np.maximum.accumulate(np.r_[0.0, equity])[1:]
    selected = returns[bet]
    sharpe = float(selected.mean() / selected.std(ddof=1) * np.sqrt(len(selected))) if len(selected) > 1 and selected.std(ddof=1) > 0 else 0.0
    return MarketMetrics(
        brier_score=brier, log_loss=log_loss, pnl_units=float(returns.sum()),
        roi=float(returns.sum() / (stake_fraction * len(selected))) if len(selected) else 0.0,
        sharpe=sharpe, max_drawdown=float(drawdown.min()) if len(drawdown) else 0.0,
        n_bets=int(bet.sum()),
    )
