"""
backend/app/api/backtest.py

GET /backtest endpoint.

CLAUDE.md §3 (API Rules):
  "/backtest endpoint must include: MAE, RMSE, Brier score, simulated P&L,
   Sharpe ratio, max drawdown, calibration data points."

This is the page that proves the model has edge (Backtest Explorer).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from backend.app.core.rate_limit import limiter
from backend.app.core.runtime_mode import ArtifactRequiredError
from backend.app.services.backtest import BacktestService

router = APIRouter(prefix="", tags=["backtest"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class CalibrationPoint(BaseModel):
    predicted_prob: float
    observed_freq:  float
    n_samples:      int


class SeasonMetrics(BaseModel):
    season:                   int
    position:                 str
    stat:                     str
    n_games:                  int
    stack_mae:                float
    stack_rmse:               float
    stack_crps:               float
    naive_mae:                float
    coverage_80:              float
    coverage_50:              float
    baseline_improvement_pct: float


class BacktestResponse(BaseModel):
    """
    Full backtest summary returned by GET /backtest.

    Required fields per CLAUDE.md §3:
      mae, rmse, brier_score, simulated_pnl, sharpe_ratio,
      max_drawdown, calibration data points.
    """
    model_version:            str
    seasons:                  list[int]
    positions:                list[str]
    stat:                     str
    # ── Core accuracy metrics ────────────────────────────────────────
    overall_mae:              float = Field(description="Mean Absolute Error (primary metric)")
    overall_rmse:             float = Field(description="Root Mean Squared Error")
    overall_crps:             float = Field(description="Continuous Ranked Probability Score")
    # ── Calibration ──────────────────────────────────────────────────
    brier_score:              Optional[float] = Field(default=None, description="Binary-event calibration metric when available")
    calibration:              list[CalibrationPoint] = Field(description="Reliability diagram data")
    # ── Simulated P&L (flat-bet strategy over backtest period) ───────
    simulated_pnl:            Optional[float] = Field(default=None, description="Backtest P&L from a real wagering policy when available")
    sharpe_ratio:             Optional[float] = Field(default=None, description="Risk-adjusted return when simulated P&L is available")
    max_drawdown:             Optional[float] = Field(default=None, description="Maximum drawdown when simulated P&L is available")
    # ── Breakdown ────────────────────────────────────────────────────
    by_season:                list[SeasonMetrics]
    data_source:              str
    metric_notes:             dict[str, str] = Field(default_factory=dict)
    data_freshness:           datetime


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.get("/backtest", response_model=BacktestResponse)
@limiter.limit("10/minute")
def get_backtest(
    request:       Request,
    stat:          str           = Query("receiving_yards", max_length=64, description="Stat to analyse"),
    positions:     list[str]     = Query(default=["WR", "RB", "TE", "QB"]),
    model_version: Optional[str] = Query(None, max_length=128, description="Filter by model version tag"),
) -> BacktestResponse:
    """
    Historical walk-forward backtest results.

    Used by the Backtest Explorer page to display:
      - Season-by-season MAE / RMSE / CRPS time series
      - Simulated P&L equity curve with Sharpe and max drawdown
      - Calibration reliability diagram
      - Baseline improvement percentage

    Results are loaded from ml/backtest_results/ CSV or the backtest_results DB table.
    """
    svc = BacktestService(
        db_url=settings.database_url,
        model_version=settings.model_version,
    )

    try:
        summary = svc.get_summary(
            stat=stat,
            positions=list(positions),
            model_version=model_version,
        )
    except ArtifactRequiredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return BacktestResponse(
        model_version=summary.model_version,
        seasons=summary.seasons,
        positions=summary.positions,
        stat=summary.stat,
        overall_mae=summary.overall_mae,
        overall_rmse=summary.overall_rmse,
        overall_crps=summary.overall_crps,
        brier_score=summary.brier_score,
        calibration=[
            CalibrationPoint(
                predicted_prob=c.predicted_prob,
                observed_freq=c.observed_freq,
                n_samples=c.n_samples,
            )
            for c in summary.calibration
        ],
        simulated_pnl=summary.simulated_pnl,
        sharpe_ratio=summary.sharpe_ratio,
        max_drawdown=summary.max_drawdown,
        by_season=[
            SeasonMetrics(
                season=s.season,
                position=s.position,
                stat=s.stat,
                n_games=s.n_games,
                stack_mae=s.stack_mae,
                stack_rmse=s.stack_rmse,
                stack_crps=s.stack_crps,
                naive_mae=s.naive_mae,
                coverage_80=s.coverage_80,
                coverage_50=s.coverage_50,
                baseline_improvement_pct=s.baseline_improvement_pct,
            )
            for s in summary.by_season
        ],
        data_source=summary.data_source,
        metric_notes=summary.metric_notes,
        data_freshness=summary.data_freshness,
    )
