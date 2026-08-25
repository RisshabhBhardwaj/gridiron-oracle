"""
backend/app/services/backtest.py

Backtest service — wraps ml/backtest.py for the /backtest API endpoint.

Loads pre-computed backtest CSV results if available, otherwise returns
a summary from the DB backtest_results table.

CLAUDE.md §3 (API Rules) requires /backtest to include:
  MAE, RMSE, Brier score, simulated P&L, Sharpe ratio,
  max drawdown, calibration data points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from backend.app.core.runtime_mode import ArtifactRequiredError, fallbacks_allowed

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path(__file__).parents[3] / "ml" / "backtest_results"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CalibrationPoint:
    predicted_prob: float
    observed_freq:  float
    n_samples:      int


@dataclass
class SeasonMetrics:
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


@dataclass
class BacktestSummary:
    """Aggregated backtest metrics required by CLAUDE.md §3."""
    model_version:            str
    seasons:                  list[int]
    positions:                list[str]
    stat:                     str
    overall_mae:              float
    overall_rmse:             float
    overall_crps:             float
    brier_score:              Optional[float]
    simulated_pnl:            Optional[float]
    sharpe_ratio:             Optional[float]
    max_drawdown:             Optional[float]
    calibration:              list[CalibrationPoint]
    by_season:                list[SeasonMetrics]
    data_source:              str
    metric_notes:             dict[str, str]
    data_freshness:           datetime


# ---------------------------------------------------------------------------
# BacktestService
# ---------------------------------------------------------------------------

class BacktestService:
    """
    Loads and aggregates backtest results for the API.

    Priority:
      1. Load from CSV file in ml/backtest_results/ (fast, no DB).
      2. Query backtest_results table in PostgreSQL.
      3. Return synthetic placeholder if neither available.
    """

    def __init__(self, db_url: str, model_version: str = "latest") -> None:
        self._db_url = db_url
        self._model_version = model_version

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_summary(
        self,
        stat: str = "receiving_yards",
        positions: list[str] | None = None,
        model_version: str | None = None,
    ) -> BacktestSummary:
        """
        Return aggregated backtest metrics for the /backtest endpoint.

        Loads from CSV if available, falls back to DB, then synthetic.
        """
        pos = positions or ["WR", "RB", "TE", "QB"]
        mv = model_version or self._model_version

        rows = self._load_csv(stat)
        data_source = "csv"
        if not rows:
            rows = self._load_db(stat, pos, mv)
            data_source = "db"

        if not rows:
            if not fallbacks_allowed():
                raise ArtifactRequiredError(
                    "Backtest artifacts unavailable while PRODUCT_MODE=artifact_backed"
                )
            logger.warning("No backtest results found — returning placeholder")
            return self._placeholder_summary(stat, pos, mv)

        return self._aggregate(rows, stat, pos, mv, data_source)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_csv(self, stat: str) -> list[dict] | None:
        """Scan ml/backtest_results/ for a CSV matching the stat."""
        import pandas as pd

        if not _RESULTS_DIR.exists():
            return None

        # Pick the latest run by *filename*, which carries the run stamp.
        # Sorting by st_mtime made the choice depend on filesystem timestamps: a
        # `touch` on an old result, or a fresh clone where every mtime is the
        # checkout time, silently changes which backtest gets reported. The key
        # must stay a scalar — sorting by the bare `p.stat()` struct raises
        # TypeError as soon as more than one CSV exists.
        candidates = sorted(
            _RESULTS_DIR.glob("backtest_*.csv"),
            key=lambda p: p.name,
            reverse=True,
        )
        if not candidates:
            return None

        path = candidates[0]
        try:
            df = pd.read_csv(path)
            if "stat" in df.columns:
                df = df[df["stat"] == stat]
            return df.to_dict("records") if not df.empty else None
        except Exception as exc:
            logger.warning("CSV load error: %s", exc)
            return None

    def _load_db(self, stat: str, positions: list[str], model_version: str) -> list[dict] | None:
        """Load from backtest_results DB table if it exists."""
        import psycopg2

        placeholders = ",".join(["%s"] * len(positions))
        try:
            conn = psycopg2.connect(self._db_url)
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT eval_season, position, stat, n_games,
                       stack_mae, stack_rmse, stack_crps,
                       naive_mae, coverage_80, coverage_50,
                       baseline_improvement_pct
                FROM   backtest_results
                WHERE  stat     = %s
                  AND  position IN ({placeholders})
                ORDER BY eval_season, position
                """,
                [stat] + positions,
            )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            conn.close()
            return [dict(zip(cols, r)) for r in rows] if rows else None
        except Exception as exc:
            logger.debug("backtest DB load error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        rows: list[dict],
        stat: str,
        positions: list[str],
        model_version: str,
        data_source: str,
    ) -> BacktestSummary:
        """Aggregate per-(season, position) rows into a BacktestSummary."""
        import pandas as pd

        df = pd.DataFrame(rows)

        # Rename column aliases between CSV and DB schemas
        col_map = {
            "eval_season": "season",
            "stack_mae":   "stack_mae",
            "stack_rmse":  "stack_rmse",
            "stack_crps":  "stack_crps",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        for col in ("stack_mae", "stack_rmse", "stack_crps", "naive_mae",
                    "coverage_80", "coverage_50", "baseline_improvement_pct"):
            if col not in df.columns:
                df[col] = float("nan")

        # ── Position filter ─────────────────────────────────────────────
        # `positions` used to be echoed back in the response and applied to
        # nothing. _load_csv filters on `stat` only, and there is no
        # backtest_results table in production, so the CSV path always ran and
        # the Season Review position chips were inert: selecting WR returned
        # WR + RB + TE rows, and the "Evaluated Sample Count" card summed all
        # of them. That is the reported inconsistency — WR alone and WR+TE
        # could not be reconciled because neither was actually a WR figure.
        # (The DB path did filter, so the two loaders disagreed as well.)
        if "position" in df.columns and positions:
            wanted = {str(p).upper() for p in positions}
            df = df[df["position"].astype(str).str.upper().isin(wanted)]
            rows = df.to_dict("records")
            if df.empty:
                logger.warning(
                    "No backtest rows for stat=%s positions=%s", stat, sorted(wanted)
                )

        # ── Overall metrics ─────────────────────────────────────────────
        # Weighted by n_games, not a flat mean over (season, position) rows.
        # An unweighted mean gave a TE season with 1,301 evaluated player-games
        # the same say as a WR season with 2,500, so the headline MAE was not
        # the error over the selected sample.
        def _weighted(col: str) -> float:
            if df.empty or col not in df.columns:
                return float("nan")
            values = df[col].astype(float)
            weights = df["n_games"].astype(float) if "n_games" in df.columns else None
            valid = values.notna() & (weights.notna() & (weights > 0) if weights is not None else True)
            if not valid.any():
                return float("nan")
            if weights is None:
                return float(values[valid].mean())
            return float(np.average(values[valid], weights=weights[valid]))

        overall_mae  = _weighted("stack_mae")
        overall_rmse = _weighted("stack_rmse")
        overall_crps = _weighted("stack_crps")
        if not np.isfinite(overall_crps):
            overall_crps = overall_mae

        # ── Calibration reliability diagram (derived from coverage stats) ──
        calibration = _real_calibration(df)

        # ── Per-season breakdown ─────────────────────────────────────────
        by_season: list[SeasonMetrics] = []
        for _, row in df.iterrows():
            by_season.append(SeasonMetrics(
                season=int(row.get("season", row.get("eval_season", 0))),
                position=str(row.get("position", "")),
                stat=stat,
                n_games=int(row.get("n_games", 0)),
                stack_mae=float(row["stack_mae"]),
                stack_rmse=float(row["stack_rmse"]),
                stack_crps=float(row.get("stack_crps", row["stack_mae"])),
                naive_mae=float(row.get("naive_mae", 0.0)),
                coverage_80=float(row.get("coverage_80", 0.0)),
                coverage_50=float(row.get("coverage_50", 0.0)),
                baseline_improvement_pct=float(row.get("baseline_improvement_pct", 0.0)),
            ))

        seasons = sorted({int(r.get("season", r.get("eval_season", 0))) for r in rows})

        return BacktestSummary(
            model_version=model_version,
            seasons=seasons,
            positions=positions,
            stat=stat,
            overall_mae=round(overall_mae, 3),
            overall_rmse=round(overall_rmse, 3),
            overall_crps=round(overall_crps, 3),
            brier_score=None,
            simulated_pnl=None,
            sharpe_ratio=None,
            max_drawdown=None,
            calibration=calibration,
            by_season=by_season,
            data_source=data_source,
            metric_notes={
                "brier_score": "Unavailable for regression forecasts until the backtest pipeline emits binary-event probabilities.",
                "simulated_pnl": "Unavailable until the project defines a real wagering policy and links forecasts to priced market data.",
                "sharpe_ratio": "Unavailable because simulated P&L is intentionally disabled.",
                "max_drawdown": "Unavailable because simulated P&L is intentionally disabled.",
            },
            data_freshness=datetime.now(timezone.utc),
        )

    def _placeholder_summary(
        self, stat: str, positions: list[str], model_version: str
    ) -> BacktestSummary:
        """Return a clearly-marked placeholder when no real data exists."""
        return BacktestSummary(
            model_version=model_version,
            seasons=[],
            positions=positions,
            stat=stat,
            overall_mae=float("nan"),
            overall_rmse=float("nan"),
            overall_crps=float("nan"),
            brier_score=None,
            simulated_pnl=None,
            sharpe_ratio=None,
            max_drawdown=None,
            calibration=[],
            by_season=[],
            data_source="placeholder",
            metric_notes={
                "overall": "No backtest CSV or database rows are available in this environment.",
            },
            data_freshness=datetime.now(timezone.utc),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _real_calibration(df) -> list[CalibrationPoint]:
    """
    Derive a calibration reliability diagram from backtest coverage statistics.

    Anchors the diagram at four empirical points:
        (0.0 → 0.0), (0.5 → coverage_50_mean), (0.8 → coverage_80_mean), (1.0 → 1.0)

    Linear interpolation fills the remaining decile points (0.1 … 1.0).
    This avoids the previous sine-curve placeholder and grounds each point in
    real walk-forward coverage measurements.
    """
    n_games = int(df["n_games"].sum()) if "n_games" in df.columns else 100
    n_per_bucket = max(n_games // 10, 1)

    cov80 = float(np.clip(
        df["coverage_80"].mean() if "coverage_80" in df.columns else 0.80, 0.0, 1.0
    ))
    cov50 = float(np.clip(
        df["coverage_50"].mean() if "coverage_50" in df.columns else 0.50, 0.0, 1.0
    ))

    anchors_x = [0.0, 0.5, 0.8, 1.0]
    anchors_y = [0.0, cov50, cov80, 1.0]

    points = []
    for i in range(1, 11):
        pred = i / 10.0
        obs = float(np.clip(np.interp(pred, anchors_x, anchors_y), 0.0, 1.0))
        points.append(CalibrationPoint(
            predicted_prob=round(pred, 1),
            observed_freq=round(obs, 3),
            n_samples=n_per_bucket,
        ))
    return points
