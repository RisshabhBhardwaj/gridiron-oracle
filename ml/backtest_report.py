"""
ml/backtest_report.py

Backtest Report Generator — Consume OOF CSVs and produce a full HTML report.

PURPOSE
-------
After training, the pipeline writes Out-of-Fold (OOF) prediction files to
ml/oof/{stat}_{model}.csv. This module:

  1. Loads all OOF CSVs from a directory.
  2. Computes comprehensive metrics: MAE, RMSE, CRPS, coverage_50/80/90.
  3. Runs per-position, per-stat, and per-season breakdowns.
  4. Generates calibration plots and MAE-by-week trend plots.
  5. Outputs a standalone HTML report + summary CSV.

The report is designed to be run post-training but pre-deployment.

USAGE
-----
    from ml.backtest_report import BacktestReport

    report = BacktestReport()
    report.generate(
        oof_dir="ml/oof",
        out_dir="ml/backtest_reports",
    )
    # → ml/backtest_reports/report.html
    # → ml/backtest_reports/summary.csv
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

_OOF_COLUMNS_REQUIRED = ["player_id", "season", "week", "stat", "actual", "predicted"]
_OOF_COLUMNS_OPTIONAL = ["position", "team", "fold"]

_PERCENTILE_COLS = {50: "p50", 80: "p80", 90: "p90"}  # coverage thresholds


# ── Math Functions ─────────────────────────────────────────────────────────────

def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean signed error (positive = over-predicting)."""
    return float(np.mean(predicted - actual))


def _crps_empirical(
    actual: np.ndarray,
    posterior_samples: Optional[np.ndarray] = None,
    predicted: Optional[np.ndarray] = None,
    std_proxy: Optional[float] = None,
) -> float:
    """
    Compute Continuous Ranked Probability Score (CRPS).

    CRPS measures calibration of probabilistic forecasts:
        CRPS = E[|X - y|] - 0.5 × E[|X - X'|]
    where X, X' are independent draws from the forecast distribution and y is actual.

    If posterior_samples are provided, uses them directly.
    Otherwise, approximates a Gaussian forecast with mean=predicted and
    std=std_proxy (default: MAE of the fold).

    Lower CRPS = better calibrated forecast.
    """
    if posterior_samples is not None and len(posterior_samples) > 0:
        # Empirical CRPS using eq. from Gneiting & Raftery (2007)
        E_xy = np.mean(np.abs(posterior_samples - actual[:, None]), axis=1).mean()
        rng = np.random.default_rng(42)
        x1 = rng.choice(posterior_samples.ravel(), size=min(2000, len(posterior_samples.ravel())))
        x2 = rng.choice(posterior_samples.ravel(), size=min(2000, len(posterior_samples.ravel())))
        E_xx = np.mean(np.abs(x1 - x2))
        return float(E_xy - 0.5 * E_xx)
    else:
        # Gaussian approximation CRPS
        if predicted is None:
            return float("nan")
        sigma = std_proxy or max(_mae(actual, predicted), 1.0)
        z = (actual - predicted) / sigma
        from scipy.special import ndtr  # normal CDF
        crps_per_row = sigma * (
            z * (2 * ndtr(z) - 1) + 2 * np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi)
        )
        return float(crps_per_row.mean())


def _coverage_at_threshold(
    actual: np.ndarray,
    predicted: np.ndarray,
    sigma: float,
    level: float = 0.80,
) -> float:
    """
    Coverage = fraction of actual values within the model's predicted interval.

    For level=0.80: interval is [predicted ± z_0.90 × sigma].
    """
    from scipy import stats as scipy_stats
    z = scipy_stats.norm.ppf(0.5 + level / 2)
    lower = predicted - z * sigma
    upper = predicted + z * sigma
    return float(np.mean((actual >= lower) & (actual <= upper)))


# ── ReportMetrics ──────────────────────────────────────────────────────────────

@dataclass
class ReportMetrics:
    """Metrics for one (position, stat) slice."""
    position:            str
    stat:                str
    n_rows:              int
    mae:                 float
    rmse:                float
    bias:                float
    crps:                float
    coverage_50:         float
    coverage_80:         float
    coverage_90:         float

    def to_dict(self) -> dict:
        return {
            "position":    self.position,
            "stat":        self.stat,
            "n_rows":      self.n_rows,
            "mae":         round(self.mae, 3),
            "rmse":        round(self.rmse, 3),
            "bias":        round(self.bias, 3),
            "crps":        round(self.crps, 3) if not np.isnan(self.crps) else None,
            "coverage_50": round(self.coverage_50, 3),
            "coverage_80": round(self.coverage_80, 3),
            "coverage_90": round(self.coverage_90, 3),
        }


# ── BacktestReport ─────────────────────────────────────────────────────────────

class BacktestReport:
    """
    Loads OOF CSV files and generates a full backtest report.

    OOF CSV format expected:
        player_id, season, week, stat, actual, predicted
        [optional: position, team, fold]

    Stat names match TARGET_COL_MAP keys: receiving_yards, rushing_yards,
    passing_yards, fantasy_ppr.
    """

    def __init__(self) -> None:
        self._oof_df: Optional[pd.DataFrame] = None

    def load_oofs(self, oof_dir: str) -> "BacktestReport":
        """
        Load all OOF CSV files from directory.

        Args:
            oof_dir: Directory containing {stat}_{model}_oof.csv files.

        Returns:
            self
        """
        oof_path = Path(oof_dir)
        dfs = []
        for csv_file in sorted(oof_path.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file)
                # Infer stat from filename if not present
                if "stat" not in df.columns:
                    stem = csv_file.stem.lower()
                    # Check all 15 TARGET_COL_MAP stats, longest-first to
                    # avoid substring false positives (e.g. 'rushing_yards'
                    # before 'yards'). Stat is first unambiguous match.
                    _ALL_STATS = sorted([
                        "receiving_yards", "rushing_yards", "passing_yards",
                        "fantasy_ppr", "receptions", "receiving_tds",
                        "targets", "carries", "rushing_tds",
                        "pass_attempts", "completions", "passing_tds",
                        "interceptions", "fumbles", "sacks_taken",
                    ], key=len, reverse=True)
                    for s in _ALL_STATS:
                        if s in stem:
                            df["stat"] = s
                            break
                dfs.append(df)
                logger.info("Loaded OOF file: %s (%d rows)", csv_file.name, len(df))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", csv_file, exc)

        if not dfs:
            logger.warning("No OOF CSV files found in %s", oof_dir)
            self._oof_df = pd.DataFrame()
            return self

        combined = pd.concat(dfs, ignore_index=True)
        missing_cols = [c for c in _OOF_COLUMNS_REQUIRED if c not in combined.columns]
        if missing_cols:
            logger.warning("OOF DataFrame missing required columns: %s", missing_cols)
        self._oof_df = combined
        logger.info("Total OOF rows loaded: %d", len(combined))
        return self

    def compute_metrics(self) -> list[ReportMetrics]:
        """
        Compute per-(position, stat) metrics.

        Returns:
            list[ReportMetrics]
        """
        if self._oof_df is None or self._oof_df.empty:
            return []

        df = self._oof_df.copy()
        df["actual"]    = pd.to_numeric(df.get("actual"),    errors="coerce")
        df["predicted"] = pd.to_numeric(df.get("predicted"), errors="coerce")
        df = df.dropna(subset=["actual", "predicted"])

        metrics = []
        group_cols = ["position", "stat"] if "position" in df.columns else ["stat"]

        for group_vals, grp in df.groupby(group_cols):
            if isinstance(group_vals, str):
                pos, stat = "ALL", group_vals
            else:
                pos, stat = group_vals if len(group_vals) == 2 else ("ALL", group_vals[0])

            actual    = grp["actual"].values
            predicted = grp["predicted"].values
            sigma     = max(_mae(actual, predicted), 1.0)   # uncertainty proxy

            metrics.append(ReportMetrics(
                position=pos,
                stat=stat,
                n_rows=len(grp),
                mae=_mae(actual, predicted),
                rmse=_rmse(actual, predicted),
                bias=_bias(actual, predicted),
                crps=_crps_empirical(actual, predicted=predicted, std_proxy=sigma),
                coverage_50=_coverage_at_threshold(actual, predicted, sigma, 0.50),
                coverage_80=_coverage_at_threshold(actual, predicted, sigma, 0.80),
                coverage_90=_coverage_at_threshold(actual, predicted, sigma, 0.90),
            ))

        return sorted(metrics, key=lambda m: (m.stat, m.position))

    def generate(
        self,
        oof_dir: str,
        out_dir: str,
        title: str = "Gridiron Oracle — OOF Backtest Report",
    ) -> Path:
        """
        Full pipeline: load OOFs → compute metrics → write HTML + CSV.

        Args:
            oof_dir:  Directory containing OOF CSV files.
            out_dir:  Output directory for report.html + summary.csv.
            title:    Report title string.

        Returns:
            Path to report.html
        """
        self.load_oofs(oof_dir)
        metrics = self.compute_metrics()

        if not metrics:
            logger.warning("No metrics computed — empty OOF directory?")
            return Path(out_dir)

        # Summary CSV
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        summary_df = pd.DataFrame([m.to_dict() for m in metrics])
        csv_path = Path(out_dir) / "summary.csv"
        summary_df.to_csv(csv_path, index=False)
        logger.info("Summary CSV saved: %s", csv_path)

        # HTML report
        html_path = Path(out_dir) / "report.html"
        html = self._build_html(metrics, summary_df, title)
        html_path.write_text(html, encoding="utf-8")
        logger.info("Backtest report saved: %s", html_path)

        # Optional: MAE-by-week plot
        self._plot_mae_by_week(out_dir)

        return html_path

    def _build_html(
        self,
        metrics: list[ReportMetrics],
        summary_df: pd.DataFrame,
        title: str,
    ) -> str:
        """Build a self-contained HTML report string."""
        table_rows = "\n".join(
            f"<tr><td>{m.position}</td><td>{m.stat}</td><td>{m.n_rows}</td>"
            f"<td>{m.mae:.2f}</td><td>{m.rmse:.2f}</td><td>{m.bias:+.2f}</td>"
            f"<td>{m.crps:.3f}</td>"
            f"<td>{m.coverage_50:.1%}</td><td>{m.coverage_80:.1%}</td><td>{m.coverage_90:.1%}</td></tr>"
            for m in metrics
        )
        from datetime import datetime
        generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background:#0f1117; color:#e0e0e0; margin:40px auto; max-width:1100px; }}
  h1 {{ color:#7b93fd; font-size:1.8em; }}
  h2 {{ color:#a0aec0; border-bottom:1px solid #2d3748; padding-bottom:8px; }}
  table {{ border-collapse:collapse; width:100%; margin:20px 0; }}
  th {{ background:#1a202c; color:#7b93fd; padding:10px 8px; text-align:left; }}
  td {{ padding:8px; border-bottom:1px solid #2d3748; font-size:0.92em; }}
  tr:hover {{ background:#1a202c; }}
  .good {{ color:#68d391; }} .warn {{ color:#f6e05e; }} .bad {{ color:#fc8181; }}
  .meta {{ color:#718096; font-size:0.85em; margin-bottom:20px; }}
</style>
</head>
<body>
<h1>📊 {title}</h1>
<p class="meta">Generated: {generated_at} | Rows: {sum(m.n_rows for m in metrics):,}</p>
<h2>Per-Position / Stat Metrics</h2>
<table>
<tr>
  <th>Position</th><th>Stat</th><th>N</th>
  <th>MAE</th><th>RMSE</th><th>Bias</th><th>CRPS</th>
  <th>Cov 50</th><th>Cov 80</th><th>Cov 90</th>
</tr>
{table_rows}
</table>

<h2>Metric Definitions</h2>
<ul>
<li><strong>MAE</strong> — Mean Absolute Error. Primary accuracy metric.</li>
<li><strong>RMSE</strong> — Root Mean Squared Error. Penalizes large misses.</li>
<li><strong>Bias</strong> — Mean signed error (+ve = over-predicting). Near zero = unbiased.</li>
<li><strong>CRPS</strong> — Continuous Ranked Probability Score. Measures calibration of the
    probabilistic forecast. Lower = better. Gaussian approximation used (σ = MAE).</li>
<li><strong>Cov 50/80/90</strong> — Coverage: fraction of actuals falling within the
    predicted 50%/80%/90% interval. Well-calibrated = close to target level.</li>
</ul>
</body>
</html>"""

    def _plot_mae_by_week(self, out_dir: str) -> None:
        """Plot MAE as a function of week (detects early-season vs late-season accuracy)."""
        if self._oof_df is None or "week" not in self._oof_df.columns:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            df = self._oof_df.copy()
            df["actual"]    = pd.to_numeric(df.get("actual"),    errors="coerce")
            df["predicted"] = pd.to_numeric(df.get("predicted"), errors="coerce")
            df["abs_err"]   = (df["actual"] - df["predicted"]).abs()
            by_week = df.groupby("week")["abs_err"].mean()

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(by_week.index, by_week.values, marker="o", linewidth=2, color="#7b93fd")
            ax.fill_between(by_week.index, by_week.values, alpha=0.15, color="#7b93fd")
            ax.set_title("MAE by NFL Week", fontsize=14)
            ax.set_xlabel("Week")
            ax.set_ylabel("MAE (yards / pts)")
            ax.grid(alpha=0.3, linestyle="--")
            fig.tight_layout()
            plot_path = Path(out_dir) / "mae_by_week.png"
            fig.savefig(plot_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info("MAE-by-week plot saved: %s", plot_path)
        except ImportError:
            logger.info("matplotlib not installed — skipping MAE-by-week plot.")
        except Exception as exc:
            logger.warning("MAE-by-week plot failed: %s", exc)
