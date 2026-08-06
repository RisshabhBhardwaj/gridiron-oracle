"""
ml/run_backtest.py

Manual backtest runner — connects BacktestRunner to the real DB and
trained models. Run from project root:

    python3 ml/run_backtest.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

import ml.utils  # noqa: F401 — ensures suppress_training_warnings runs for backtest
from backend.app.core.config import settings
from ml.backtest import BacktestRunner, load_backtest_trust_policy, validate_projection_trust

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_TRUST_POLICY = None


def _get_trust_policy():
    global _TRUST_POLICY
    if _TRUST_POLICY is None:
        _TRUST_POLICY = load_backtest_trust_policy(
            settings.baseline_manifest_path,
            settings.artifact_invalidation_path,
        )
    return _TRUST_POLICY


# ---------------------------------------------------------------------------
# Providers — connect to real data
# ---------------------------------------------------------------------------

def real_data_provider(
    eval_season: int,
    positions: list[str],
    stats: list[str],
) -> pd.DataFrame:
    """
    Load actual outcomes + Kalman baselines from the DB.

    Actuals come from GameLog (authoritative source; has all stats including
    receptions). Kalman estimates come from FeatureMatrix (joined in memory
    on player_id + game_id) and serve as the naive/rolling baseline proxy.

    Bug fixed: FeatureMatrix.actual_{stat} only exists for 3 stats and was
    not the right column anyway — GameLog has the true observed values.
    """
    from backend.app.models.production import GameLog, FeatureMatrix
    from sqlmodel import Session, create_engine, select
    import os

    engine = create_engine(os.environ["DATABASE_URL"])

    with Session(engine) as session:
        gl_rows = session.exec(
            select(GameLog).where(GameLog.season == eval_season)
        ).all()

        # Index FeatureMatrix by (player_id, game_id) for O(1) lookup.
        fm_index = {
            (fm.player_id, fm.game_id): fm
            for fm in session.exec(
                select(FeatureMatrix).where(FeatureMatrix.season == eval_season)
            ).all()
        }

    rows = []
    for gl in gl_rows:
        if gl.position not in positions:
            continue
        fm = fm_index.get((gl.player_id, gl.game_id))
        if fm is None:
            continue
        for stat in stats:
            actual    = getattr(gl, stat, None)
            kalman_est = getattr(fm, f"kalman_est_{stat}", None)
            if actual is None or kalman_est is None:
                continue
            rows.append({
                "player_id":        gl.player_id,
                "season":           gl.season,
                "week":             gl.week,
                "position":         gl.position,
                "stat":             stat,
                "actual_value":     float(actual),
                "naive_baseline":   float(kalman_est),   # Kalman prior as proxy
                "rolling_baseline": float(kalman_est),   # same proxy
            })

    return pd.DataFrame(rows)


def real_projection_provider(
    eval_season: int,
    positions: list[str],
    stats: list[str],
) -> pd.DataFrame:
    """Load only approved projections written by PipelineRunner for eval_season."""
    from backend.app.models.production import Projection
    from sqlmodel import Session, create_engine, select
    import os

    engine = create_engine(os.environ["DATABASE_URL"])
    rows = []

    with Session(engine) as session:
        results = session.exec(
            select(Projection).where(
                Projection.season == eval_season
            )
        ).all()

        for row in results:
            if row.position not in positions or row.stat not in stats:
                continue
            if row.posterior_samples:
                try:
                    samples = np.array(row.posterior_samples, dtype=float)
                except (TypeError, ValueError):
                    samples = None
            else:
                samples = None

            rows.append({
                "player_id":         row.player_id,
                "season":            row.season,
                "week":              row.week,
                "position":          row.position,
                "stat":              row.stat,
                "projection":        row.projection,
                "floor":             row.floor,
                "ceiling":           row.ceiling,
                "p25": row.p25,
                "p75": row.p75,
                "pipeline_run_id":   row.pipeline_run_id,
                "posterior_samples": samples,
            })

    projections_df = pd.DataFrame(rows)
    if projections_df.empty:
        return projections_df

    return validate_projection_trust(
        projections_df,
        _get_trust_policy(),
        eval_season=eval_season,
    )


# ---------------------------------------------------------------------------
# Valid position/stat pairs
# ---------------------------------------------------------------------------

# Only meaningful stat/position combinations. Running a QB on receiving_yards
# or an RB on passing_yards produces near-zero actuals AND near-zero projections
# — trivially low MAE with nonsense improvement metrics and artifically high
# coverage (everything fits in a [0, ~0] interval).
VALID_POSITION_STATS: dict[str, list[str]] = {
    "QB": [
        "passing_yards", "pass_attempts", "completions",
        "passing_tds", "interceptions", "rushing_yards", "fumbles",
    ],
    "RB": [
        "rushing_yards", "carries", "rushing_tds",
        "receiving_yards", "receptions", "receiving_tds", "targets", "fumbles",
    ],
    "WR": [
        "receiving_yards", "receptions", "receiving_tds", "targets",
        "rushing_yards", "fumbles",
    ],
    "TE": [
        "receiving_yards", "receptions", "receiving_tds", "targets", "fumbles",
    ],
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

    runner = BacktestRunner(
        data_provider=real_data_provider,
        projection_provider=real_projection_provider,
    )

    logger.info("Starting walk-forward backtest over seasons %s", SEASONS)

    # Run once per position with only that position's valid stats.
    # This avoids cross-position contamination (QB on receiving_yards, etc.).
    all_dfs = []
    for pos, stats in VALID_POSITION_STATS.items():
        logger.info("Position=%s  stats=%s", pos, stats)
        pos_df = runner.run(seasons=SEASONS, positions=[pos], stats=stats)
        if not pos_df.empty:
            all_dfs.append(pos_df)

    if not all_dfs:
        logger.error("Backtest produced no results — check that projections "
                     "exist in the DB for these seasons.")
        sys.exit(1)

    df = pd.concat(all_dfs, ignore_index=True)

    out_path = "ml/backtest_results/backtest_2019_2025.csv"
    runner.save_results(df, out_path)
    runner.plot_improvement(df)

    print("\n=== BACKTEST SUMMARY ===")
    print(df.groupby(["position", "stat"])[
        ["stack_mae", "naive_mae", "baseline_improvement_pct", "coverage_80"]
    ].mean().round(2).to_string())
