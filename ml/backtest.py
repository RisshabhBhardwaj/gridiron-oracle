"""
ml/backtest.py

Walk-forward backtesting framework for the full 4-layer projection stack.

PURPOSE
-------
Prove the Kalman → Stacking → Bayesian → Monte Carlo pipeline produces
projections that beat a naive baseline on held-out historical seasons.
This is the module that gives the project its claimed "edge" — without a
validated backtest the model is a hypothesis, not a result.

WALK-FORWARD DESIGN
-------------------
For each eval_season in [2019, 2020, 2021, 2022, 2023, 2024]:
  1. Train all layers on all seasons < eval_season (no future leakage).
  2. Run the full pipeline on eval_season games.
  3. Compare projections to actuals. Compute metrics.

METRICS
-------
  MAE          — Mean Absolute Error (primary ranking metric)
  RMSE         — Root Mean Squared Error (penalises large misses)
  CRPS         — Continuous Ranked Probability Score; single metric
                 combining accuracy AND calibration. Lower is better.
                 Uses the energy-form estimator (O(N log N) per player).
  Coverage_80  — % of actuals within [p10, p90]. Target: ~80%.
  Coverage_50  — % of actuals within [p25, p75]. Target: ~50%.
  Baseline MAE — MAE of the naive (prev-season avg) baseline.
  Rolling MAE  — MAE of the 4-week rolling mean baseline.
  baseline_improvement_pct — (naive_MAE - stack_MAE) / naive_MAE * 100.
                             Positive = stack beats naive.

CRPS FORMULA (energy form, sample approximation)
-------------------------------------------------
  CRPS(F, y) = E_F[|X - y|] - (1/2) * E_F[|X - X'|]

  Efficient O(N log N) implementation via sorted samples:
    Let s = sorted(samples), n = len(s), i = 0..n-1 (0-indexed)
    spread = (2/n²) * Σ_i  s_i * (2i - n + 1)
    CRPS   = mean(|s - y|) - spread / 2

CAUSALITY GUARANTEE
-------------------
BacktestRunner.run() uses data_provider(eval_season) to get actuals
and projection_provider(eval_season, ...) for projections. Callers are
responsible for ensuring projection_provider was trained on seasons
< eval_season only. The framework asserts this at runtime when a
season manifest is provided.

INJECTABLE PROVIDERS FOR TESTING
---------------------------------
Both data_provider and projection_provider are callables injected at
__init__ time. Tests supply synthetic implementations — no DB or trained
models required.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestTrustPolicy:
    """
    Policy describing which stored projection rows are eligible for trusted replay.

    Backtests should fail fast when rows come from unknown or invalidated runs
    rather than silently synthesizing missing data or grading legacy projections.
    """

    baseline_manifest_path: str
    invalidation_manifest_path: str
    approved_pipeline_run_ids: frozenset[str]
    invalidated_pipeline_run_ids: frozenset[str]
    invalidated_position_stats: frozenset[tuple[str, str]]
    require_posterior_samples: bool = True
    require_interval_columns: bool = True


def load_backtest_trust_policy(
    baseline_manifest_path: str,
    invalidation_manifest_path: str,
) -> BacktestTrustPolicy:
    """
    Load trusted replay policy from the promoted baseline + invalidation manifests.
    """
    baseline_path = Path(baseline_manifest_path)
    invalidation_path = Path(invalidation_manifest_path)

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline manifest not found: {baseline_manifest_path}"
        )
    if not invalidation_path.exists():
        raise FileNotFoundError(
            f"Artifact invalidation manifest not found: {invalidation_manifest_path}"
        )

    baseline = json.loads(baseline_path.read_text())
    invalidations = json.loads(invalidation_path.read_text())

    projection_policy = baseline.get("projection_policy") or {}
    approved_run_ids = frozenset(
        str(run_id).strip()
        for run_id in projection_policy.get("approved_pipeline_run_ids", [])
        if str(run_id).strip()
    )
    invalidated_run_ids = frozenset(
        str(run_id).strip()
        for run_id in invalidations.get("invalid_pipeline_run_ids", [])
        if str(run_id).strip()
    )
    invalidated_position_stats = frozenset(
        (
            str(item.get("position", "")).strip(),
            str(item.get("stat", "")).strip(),
        )
        for item in invalidations.get("invalid_projection_targets", [])
        if str(item.get("position", "")).strip() and str(item.get("stat", "")).strip()
    )

    return BacktestTrustPolicy(
        baseline_manifest_path=str(baseline_path),
        invalidation_manifest_path=str(invalidation_path),
        approved_pipeline_run_ids=approved_run_ids,
        invalidated_pipeline_run_ids=invalidated_run_ids,
        invalidated_position_stats=invalidated_position_stats,
        require_posterior_samples=bool(
            projection_policy.get("require_posterior_samples", True)
        ),
        require_interval_columns=bool(
            projection_policy.get("require_interval_columns", True)
        ),
    )


def validate_projection_trust(
    projections_df: pd.DataFrame,
    policy: BacktestTrustPolicy,
    *,
    eval_season: int,
) -> pd.DataFrame:
    """
    Reject legacy or invalidated projection rows before scoring them in backtest.
    """
    if projections_df.empty:
        return projections_df

    if not policy.approved_pipeline_run_ids:
        raise ValueError(
            "Backtest trust policy has no approved pipeline_run_id values. "
            "Populate projection_policy.approved_pipeline_run_ids in the baseline manifest."
        )

    required_cols = {"pipeline_run_id", "position", "stat"}
    missing = required_cols - set(projections_df.columns)
    if missing:
        raise ValueError(
            f"Projection trust validation requires columns {sorted(required_cols)}; "
            f"missing {sorted(missing)}."
        )

    violations: list[str] = []
    valid_indices: list[int] = []

    for idx, row in projections_df.iterrows():
        position = str(row.get("position") or "").strip()
        stat = str(row.get("stat") or "").strip()
        run_id = str(row.get("pipeline_run_id") or "").strip()

        if (position, stat) in policy.invalidated_position_stats:
            violations.append(
                f"season={eval_season} idx={idx}: {position}/{stat} is explicitly invalidated."
            )
            continue
        if not run_id:
            violations.append(
                f"season={eval_season} idx={idx}: missing pipeline_run_id for {position}/{stat}."
            )
            continue
        if run_id in policy.invalidated_pipeline_run_ids:
            violations.append(
                f"season={eval_season} idx={idx}: pipeline_run_id={run_id} is invalidated."
            )
            continue
        if run_id not in policy.approved_pipeline_run_ids:
            violations.append(
                f"season={eval_season} idx={idx}: pipeline_run_id={run_id} is not approved."
            )
            continue
        if policy.require_interval_columns and (
            pd.isna(row.get("p25")) or pd.isna(row.get("p75"))
        ):
            violations.append(
                f"season={eval_season} idx={idx}: missing p25/p75 for approved run {run_id}."
            )
            continue
        if policy.require_posterior_samples:
            samples = row.get("posterior_samples")
            if not isinstance(samples, (list, tuple, np.ndarray)) or len(samples) == 0:
                violations.append(
                    f"season={eval_season} idx={idx}: missing posterior_samples for approved run {run_id}."
                )
                continue

        valid_indices.append(idx)

    if violations:
        raise ValueError(
            "Backtest trust validation failed:\n- " + "\n- ".join(violations[:20])
        )

    return projections_df.loc[valid_indices].copy()


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

_RESULTS_DIR = Path(__file__).parent / "backtest_results"


# ---------------------------------------------------------------------------
# Pure metric functions (standalone, fully testable)
# ---------------------------------------------------------------------------

def compute_mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(predictions, dtype=float)
                                - np.asarray(actuals, dtype=float))))


def compute_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Root Mean Squared Error."""
    diff = np.asarray(predictions, dtype=float) - np.asarray(actuals, dtype=float)
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_crps_single(samples: np.ndarray, actual: float) -> float:
    """
    CRPS for one probabilistic forecast (posterior samples) vs one observation.

    Uses the O(N log N) energy-form estimator via sorted samples:
        CRPS = mean(|s - y|) - (1/n²) * Σ_i s_i * (2i - n + 1)

    where i is the 0-based rank in the sorted sample array s.

    Reference value (hand-verified):
        samples = [0, 1, 2, 3, 4], actual = 2.0  →  CRPS = 0.4

    Args:
        samples: 1-D posterior draw array.
        actual:  Observed outcome.

    Returns:
        CRPS scalar ≥ 0. Lower is better; 0 = perfect deterministic forecast.
    """
    s = np.sort(np.asarray(samples, dtype=float))
    n = len(s)
    if n == 0:
        raise ValueError("samples must not be empty.")

    mae_term = float(np.mean(np.abs(s - float(actual))))

    # Spread term: (1/n²) * Σ_i s_i * (2i - n + 1)
    # = half of E[|X - X'|] (energy form)
    idx = np.arange(n, dtype=float)
    spread = float(np.sum(s * (2.0 * idx - n + 1.0)) / (n * n))

    return mae_term - spread


def compute_crps_batch(
    samples_list: List[np.ndarray],
    actuals: np.ndarray,
) -> float:
    """
    Mean CRPS over a batch of (forecast, observation) pairs.

    Args:
        samples_list: List of posterior sample arrays, one per game.
        actuals:      1-D array of observed outcomes, same length.

    Returns:
        Mean CRPS across all pairs.
    """
    actuals = np.asarray(actuals, dtype=float)
    if len(samples_list) != len(actuals):
        raise ValueError(
            f"samples_list length ({len(samples_list)}) must match "
            f"actuals length ({len(actuals)})."
        )
    scores = [
        compute_crps_single(np.asarray(s, dtype=float), float(y))
        for s, y in zip(samples_list, actuals)
    ]
    return float(np.mean(scores))


def compute_coverage(
    floors: np.ndarray,
    ceilings: np.ndarray,
    actuals: np.ndarray,
) -> float:
    """
    Fraction of actuals falling within [floor, ceiling].

    For 80% coverage: pass floors=p10, ceilings=p90.
    For 50% coverage: pass floors=p25, ceilings=p75.

    Returns:
        Float in [0, 1].
    """
    floors   = np.asarray(floors,   dtype=float)
    ceilings = np.asarray(ceilings, dtype=float)
    actuals  = np.asarray(actuals,  dtype=float)
    return float(np.mean((actuals >= floors) & (actuals <= ceilings)))


# ---------------------------------------------------------------------------
# BacktestResult dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestResult:
    """
    Metrics for one (eval_season, position, stat) cohort.

    Frozen → hashable → usable as dict key / set element.
    All numeric fields are Python floats/ints (not numpy scalars).

    Fields:
        eval_season:              Season used as held-out test data.
        position:                 Player position ("WR"/"RB"/"TE"/"QB").
        stat:                     Projected stat (e.g. "receiving_yards").
        n_games:                  Number of player-game observations.
        stack_mae:                MAE of the full 4-layer stack (p50).
        stack_rmse:               RMSE of the stack.
        stack_crps:               Mean CRPS of the stack (accuracy + calibration).
        naive_mae:                MAE of the naive baseline (prev-season average).
        rolling_mae:              MAE of the rolling 4-week mean baseline.
        coverage_80:              Fraction of actuals in [p10, p90]. Target ~0.80.
        coverage_50:              Fraction of actuals in [p25, p75]. Target ~0.50.
        baseline_improvement_pct: (naive_mae - stack_mae) / naive_mae × 100.
                                  Positive = stack beats naive.
    """

    eval_season:              int
    position:                 str
    stat:                     str
    n_games:                  int
    stack_mae:                float
    stack_rmse:               float
    stack_crps:               float
    naive_mae:                float
    rolling_mae:              float
    coverage_80:              float
    coverage_50:              float
    baseline_improvement_pct: float


# ---------------------------------------------------------------------------
# BacktestRunner
# ---------------------------------------------------------------------------

class BacktestRunner:
    """
    Walk-forward backtesting framework for the full projection stack.

    Designed with injectable providers so the framework can be tested
    without a live DB or trained models.

    Args:
        data_provider:        Callable(eval_season, positions, stats)
                              → pd.DataFrame with columns:
                                player_id, season, week, position, stat,
                                actual_value, naive_baseline, rolling_baseline.
                              Default: raises NotImplementedError (requires DB).
        projection_provider:  Callable(eval_season, positions, stats)
                              → pd.DataFrame with columns:
                                player_id, season, week, stat,
                                projection, floor, ceiling, p25, p75,
                                posterior_samples (list[np.ndarray] per row).
                              Default: raises NotImplementedError.
        mlflow_tracking_uri:  MLflow URI for save_results. "" = disabled.
    """

    def __init__(
        self,
        data_provider: Optional[Callable] = None,
        projection_provider: Optional[Callable] = None,
        mlflow_tracking_uri: str = "",
    ) -> None:
        self._data_provider       = data_provider       or _default_data_provider
        self._projection_provider = projection_provider or _default_projection_provider
        self.mlflow_tracking_uri  = mlflow_tracking_uri

    # ------------------------------------------------------------------
    # run — walk-forward backtest
    # ------------------------------------------------------------------

    def run(
        self,
        seasons: List[int],
        positions: List[str],
        stats: List[str],
    ) -> pd.DataFrame:
        """
        Walk-forward backtest over the provided eval_seasons.

        For each eval_season:
          1. Load actual outcomes via data_provider(eval_season, ...).
          2. Load projections via projection_provider(eval_season, ...).
             Caller ensures the projection model was trained on seasons
             strictly less than eval_season.
          3. Inner-join actuals + projections on (player_id, season, week, stat).
          4. Compute BacktestResult per (position, stat).

        Returns:
            pd.DataFrame with one BacktestResult row per
            (eval_season, position, stat) combination that has ≥ 2 games.
        """
        all_results: list[BacktestResult] = []

        for eval_season in sorted(seasons):
            logger.info("Backtesting eval_season=%d ...", eval_season)

            actuals_df     = self._data_provider(eval_season, positions, stats)
            projections_df = self._projection_provider(eval_season, positions, stats)

            if actuals_df.empty or projections_df.empty:
                logger.warning("Empty data for eval_season=%d — skipping.", eval_season)
                continue

            merged = _merge_actuals_projections(actuals_df, projections_df)

            for position in positions:
                for stat in stats:
                    mask = (
                        (merged["position"] == position) &
                        (merged["stat"]     == stat)
                    )
                    cohort = merged[mask]
                    if len(cohort) < 2:
                        continue

                    result = self.evaluate(
                        actuals=cohort["actual_value"].values,
                        projections=cohort["projection"].values,
                        floors=cohort["floor"].values,
                        ceilings=cohort["ceiling"].values,
                        p25s=cohort["p25"].values,
                        p75s=cohort["p75"].values,
                        posterior_samples_list=cohort["posterior_samples"].tolist(),
                        naive_baselines=cohort["naive_baseline"].values,
                        rolling_baselines=cohort["rolling_baseline"].values,
                        eval_season=eval_season,
                        position=position,
                        stat=stat,
                    )
                    all_results.append(result)

        if not all_results:
            logger.warning("No backtest results produced.")
            return pd.DataFrame(columns=list(BacktestResult.__dataclass_fields__))

        return pd.DataFrame([vars(r) for r in all_results])

    # ------------------------------------------------------------------
    # evaluate — pure metric computation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        actuals: np.ndarray,
        projections: np.ndarray,
        floors: np.ndarray,
        ceilings: np.ndarray,
        p25s: np.ndarray,
        p75s: np.ndarray,
        posterior_samples_list: List[np.ndarray],
        naive_baselines: np.ndarray,
        rolling_baselines: np.ndarray,
        eval_season: int,
        position: str,
        stat: str,
    ) -> BacktestResult:
        """
        Compute all backtest metrics for one (season, position, stat) cohort.

        All inputs are plain numpy arrays — no DB or trained models needed.
        Designed for direct use in unit tests.

        Args:
            actuals:                Ground-truth outcomes, shape (n,).
            projections:            Model p50 estimates, shape (n,).
            floors:                 p10 values, shape (n,).
            ceilings:               p90 values, shape (n,).
            p25s:                   25th percentile, shape (n,).
            p75s:                   75th percentile, shape (n,).
            posterior_samples_list: List of n arrays of posterior draws.
            naive_baselines:        Prev-season avg per player, shape (n,).
            rolling_baselines:      4-week rolling mean per player, shape (n,).
            eval_season:            Season integer label.
            position:               Position string.
            stat:                   Stat name string.

        Returns:
            BacktestResult (frozen dataclass).
        """
        actuals           = np.asarray(actuals,           dtype=float)
        projections       = np.asarray(projections,       dtype=float)
        floors            = np.asarray(floors,            dtype=float)
        ceilings          = np.asarray(ceilings,          dtype=float)
        p25s              = np.asarray(p25s,              dtype=float)
        p75s              = np.asarray(p75s,              dtype=float)
        naive_baselines   = np.asarray(naive_baselines,   dtype=float)
        rolling_baselines = np.asarray(rolling_baselines, dtype=float)

        stack_mae  = compute_mae(projections, actuals)
        stack_rmse = compute_rmse(projections, actuals)
        stack_crps = compute_crps_batch(posterior_samples_list, actuals)

        naive_mae   = compute_mae(naive_baselines,   actuals)
        rolling_mae = compute_mae(rolling_baselines, actuals)

        cov_80 = compute_coverage(floors,   ceilings, actuals)
        cov_50 = compute_coverage(p25s,     p75s,     actuals)

        if naive_mae > 0.0:
            improvement = (naive_mae - stack_mae) / naive_mae * 100.0
        else:
            improvement = 0.0

        return BacktestResult(
            eval_season=int(eval_season),
            position=str(position),
            stat=str(stat),
            n_games=int(len(actuals)),
            stack_mae=float(stack_mae),
            stack_rmse=float(stack_rmse),
            stack_crps=float(stack_crps),
            naive_mae=float(naive_mae),
            rolling_mae=float(rolling_mae),
            coverage_80=float(cov_80),
            coverage_50=float(cov_50),
            baseline_improvement_pct=float(improvement),
        )

    # ------------------------------------------------------------------
    # save_results
    # ------------------------------------------------------------------

    def save_results(self, df: pd.DataFrame, path: str) -> None:
        """
        Save backtest results to CSV and optionally log to MLflow.

        Args:
            df:   DataFrame returned by run().
            path: Output file path (e.g. "ml/backtest_results/run_2025.csv").
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Saved backtest results → %s (%d rows)", out_path, len(df))

        if not self.mlflow_tracking_uri:
            return

        try:
            import mlflow
            from datetime import datetime, timezone

            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment("backtest")

            with mlflow.start_run():
                if "stack_mae" in df.columns:
                    mlflow.log_metric("mean_stack_mae",  float(df["stack_mae"].mean()))
                    mlflow.log_metric("mean_stack_crps", float(df["stack_crps"].mean()))
                    mlflow.log_metric("mean_coverage_80", float(df["coverage_80"].mean()))
                    mlflow.log_metric(
                        "mean_baseline_improvement_pct",
                        float(df["baseline_improvement_pct"].mean()),
                    )
                mlflow.log_param("timestamp", datetime.now(timezone.utc).isoformat())
                mlflow.log_artifact(str(out_path))

            logger.info("Logged backtest run to MLflow.")
        except Exception as exc:
            logger.warning("MLflow logging failed: %s", exc)

    # ------------------------------------------------------------------
    # plot_improvement
    # ------------------------------------------------------------------

    def plot_improvement(self, df: pd.DataFrame) -> None:
        """
        Grouped bar chart: baseline_improvement_pct by position and stat.

        Saves to ml/backtest_results/improvement_chart.png.
        Each bar group = one position; bars within group = one stat each.
        Positive bars = stack beats naive baseline.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — skipping plot.")
            return

        if df.empty or "baseline_improvement_pct" not in df.columns:
            logger.warning("No data to plot.")
            return

        pivot = (
            df.groupby(["position", "stat"])["baseline_improvement_pct"]
            .mean()
            .unstack("stat")
            .fillna(0.0)
        )

        positions  = pivot.index.tolist()
        stats_cols = pivot.columns.tolist()
        n_stats = len(stats_cols)
        n_pos   = len(positions)

        x     = np.arange(n_pos)
        width = 0.8 / max(n_stats, 1)

        fig, ax = plt.subplots(figsize=(max(8, n_pos * 2), 5))
        for i, stat in enumerate(stats_cols):
            offsets = x + i * width - (n_stats - 1) * width / 2
            ax.bar(offsets, pivot[stat].values, width=width * 0.9, label=stat)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(positions)
        ax.set_ylabel("Baseline improvement (%)")
        ax.set_title("Stack vs Naive Baseline: Improvement by Position & Stat")
        ax.legend(title="stat", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()

        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = _RESULTS_DIR / "improvement_chart.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved improvement chart → %s", out)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _merge_actuals_projections(
    actuals_df: pd.DataFrame,
    projections_df: pd.DataFrame,
) -> pd.DataFrame:
    join_keys = ["player_id", "season", "week", "stat"]
    before = len(actuals_df)
    merged = actuals_df.merge(projections_df, on=join_keys, how="inner")
    after  = len(merged)
    if after < before:
        logger.info(
            "Actuals-projections join: %d → %d rows (%d unmatched)",
            before, after, before - after,
        )
    return merged


def _default_data_provider(
    eval_season: int,
    positions: List[str],
    stats: List[str],
) -> pd.DataFrame:
    raise NotImplementedError(
        "Inject a data_provider at BacktestRunner.__init__(). "
        "Required columns: player_id, season, week, position, stat, "
        "actual_value, naive_baseline, rolling_baseline."
    )


def _default_projection_provider(
    eval_season: int,
    positions: List[str],
    stats: List[str],
) -> pd.DataFrame:
    raise NotImplementedError(
        "Inject a projection_provider at BacktestRunner.__init__(). "
        "Required columns: player_id, season, week, stat, "
        "projection, floor, ceiling, p25, p75, posterior_samples."
    )
