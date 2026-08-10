"""
ml/run_backtest.py

Manual backtest runner. Prefer OOF-based evaluation (`ml/eval_causal.py`) for
causal model scores. This module still supports DB replay, but now:
  - baselines are true prev-season / trailing-3 (not Kalman proxies)
  - cohort membership is explicit and pregame-knowable (prior games)
  - stat names resolve through STAT_COLUMN_MAP with startup asserts
  - projection rows must carry max_train_season < eval_season

Run from project root:

    DATABASE_URL=... python ml/run_backtest.py
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

import ml.utils  # noqa: F401 — OpenMP / warning setup
from backend.app.core.config import settings
from ml.backtest import BacktestRunner, load_backtest_trust_policy, validate_projection_trust
from ml.baselines import attach_baselines
from ml.eval_cohort import DEFAULT_COHORT, filter_cohort_frame, normalize_offense_pct
from ml.stat_resolution import (
    VALID_POSITION_STATS,
    assert_position_stats_resolvable,
    read_gamelog_stat,
    resolve_stat,
)

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


def _load_gamelog_history(seasons: list[int]) -> pd.DataFrame:
    from backend.app.models.production import GameLog
    from sqlmodel import Session, create_engine, select

    engine = create_engine(os.environ["DATABASE_URL"])
    rows: list[dict] = []
    with Session(engine) as session:
        gl_rows = session.exec(
            select(GameLog).where(GameLog.season.in_(seasons))
        ).all()
        for gl in gl_rows:
            rows.append(
                {
                    "player_id": gl.player_id,
                    "game_id": gl.game_id,
                    "season": gl.season,
                    "week": gl.week,
                    "position": gl.position,
                    "offense_pct": normalize_offense_pct(gl.offense_pct),
                    "attempts": gl.attempts,
                    "completions": gl.completions,
                    "passing_yards": gl.passing_yards,
                    "passing_tds": gl.passing_tds,
                    "passing_interceptions": gl.passing_interceptions,
                    "carries": gl.carries,
                    "rushing_yards": gl.rushing_yards,
                    "rushing_tds": gl.rushing_tds,
                    "receptions": gl.receptions,
                    "targets": gl.targets,
                    "receiving_yards": gl.receiving_yards,
                    "receiving_tds": gl.receiving_tds,
                    "fantasy_points_ppr": gl.fantasy_points_ppr,
                    "rushing_fumbles": gl.rushing_fumbles,
                    "receiving_fumbles": getattr(gl, "receiving_fumbles", None),
                    "sack_fumbles": getattr(gl, "sack_fumbles", None),
                }
            )
    return pd.DataFrame(rows)


def real_data_provider(
    eval_season: int,
    positions: list[str],
    stats: list[str],
) -> pd.DataFrame:
    """
    Load actual outcomes + causal baselines from the DB.

    Cohort: GameLog rows for eval_season in `positions` with at least one
    completed prior game. Baselines: prev-season mean and trailing-3; Kalman is attached as
    a third incumbent column when FeatureMatrix has the estimate.
    """
    from backend.app.models.production import FeatureMatrix
    from sqlmodel import Session, create_engine, select

    history_seasons = list(range(min(2019, eval_season - 1), eval_season + 1))
    history = _load_gamelog_history(history_seasons)
    if history.empty:
        return pd.DataFrame()

    cohort = filter_cohort_frame(
        history[history["season"] == eval_season],
        spec=DEFAULT_COHORT,
    )
    cohort = cohort[cohort["position"].isin(positions)].copy()
    if cohort.empty:
        return pd.DataFrame()

    engine = create_engine(os.environ["DATABASE_URL"])
    with Session(engine) as session:
        fm_index = {
            (fm.player_id, fm.game_id): fm
            for fm in session.exec(
                select(FeatureMatrix).where(FeatureMatrix.season == eval_season)
            ).all()
        }

    rows: list[dict] = []
    for stat in stats:
        resolved = resolve_stat(stat)
        if resolved.is_multi_source:
            history_stat = history.copy()
            history_stat[stat] = history_stat.apply(
                lambda r: read_gamelog_stat(r.to_dict(), stat), axis=1
            )
        else:
            history_stat = history.rename(columns={resolved.gamelog_attr: stat})
            if stat not in history_stat.columns:
                history_stat[stat] = history.apply(
                    lambda r: read_gamelog_stat(r.to_dict(), stat), axis=1
                )

        eval_part = cohort.copy()
        eval_part["stat"] = stat
        eval_part["actual_value"] = eval_part.apply(
            lambda r: read_gamelog_stat(r.to_dict(), stat), axis=1
        )
        eval_part = eval_part[eval_part["actual_value"].notna()].copy()
        if eval_part.empty:
            continue

        kalman_vals = []
        for row in eval_part.itertuples(index=False):
            fm = fm_index.get((row.player_id, row.game_id))
            if fm is None:
                kalman_vals.append(None)
                continue
            val = getattr(fm, resolved.kalman_attr, None)
            kalman_vals.append(float(val) if val is not None else None)
        eval_part["kalman_est"] = kalman_vals

        with_base = attach_baselines(
            eval_part,
            history_stat,
            stat_col=stat,
            trailing_n=3,
            kalman_col="kalman_est",
        )
        with_base = with_base[
            with_base["naive_baseline"].notna() | with_base["rolling_baseline"].notna()
        ]
        for _, row in with_base.iterrows():
            naive = row["naive_baseline"]
            rolling = row["rolling_baseline"]
            if pd.isna(naive) and pd.isna(rolling):
                continue
            rows.append(
                {
                    "player_id": row["player_id"],
                    "season": int(row["season"]),
                    "week": int(row["week"]),
                    "position": row["position"],
                    "stat": stat,
                    "actual_value": float(row["actual_value"]),
                    "naive_baseline": float(naive if pd.notna(naive) else rolling),
                    "rolling_baseline": float(rolling if pd.notna(rolling) else naive),
                    "kalman_baseline": (
                        float(row["kalman_baseline"])
                        if pd.notna(row["kalman_baseline"])
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def real_projection_provider(
    eval_season: int,
    positions: list[str],
    stats: list[str],
) -> pd.DataFrame:
    """
    Load stored projections for eval_season.

    WARNING: DB-replay projections are only causal when each row carries
    max_train_season < eval_season. Prefer OOF evaluation until that contract
    is enforced end-to-end by the training writer.
    """
    from backend.app.models.production import Projection
    from sqlmodel import Session, create_engine, select

    engine = create_engine(os.environ["DATABASE_URL"])
    rows = []

    with Session(engine) as session:
        results = session.exec(
            select(Projection).where(Projection.season == eval_season)
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

            max_train_season = getattr(row, "max_train_season", None)
            rows.append(
                {
                    "player_id": row.player_id,
                    "season": row.season,
                    "week": row.week,
                    "position": row.position,
                    "stat": row.stat,
                    "projection": row.projection,
                    "floor": row.floor,
                    "ceiling": row.ceiling,
                    "p25": row.p25,
                    "p75": row.p75,
                    "pipeline_run_id": row.pipeline_run_id,
                    "posterior_samples": samples,
                    "max_train_season": max_train_season,
                }
            )

    projections_df = pd.DataFrame(rows)
    if projections_df.empty:
        return projections_df

    return validate_projection_trust(
        projections_df,
        _get_trust_policy(),
        eval_season=eval_season,
    )


if __name__ == "__main__":
    assert_position_stats_resolvable(VALID_POSITION_STATS)

    from ml.season_constants import train_seasons

    SEASONS = train_seasons()

    runner = BacktestRunner(
        data_provider=real_data_provider,
        projection_provider=real_projection_provider,
        require_causal_projections=True,
    )

    logger.info("Starting walk-forward backtest over seasons %s", SEASONS)
    logger.info("Cohort: %s", DEFAULT_COHORT.describe())

    all_dfs = []
    for pos, stats in VALID_POSITION_STATS.items():
        logger.info("Position=%s  stats=%s", pos, stats)
        pos_df = runner.run(seasons=SEASONS, positions=[pos], stats=stats)
        if not pos_df.empty:
            all_dfs.append(pos_df)

    if not all_dfs:
        logger.error(
            "Backtest produced no results — check projections / OOF availability."
        )
        sys.exit(1)

    df = pd.concat(all_dfs, ignore_index=True)
    out_path = "ml/backtest_results/backtest_2019_2025.csv"
    runner.save_results(df, out_path)
    runner.plot_improvement(df)

    print("\n=== BACKTEST SUMMARY ===")
    print(
        df.groupby(["position", "stat"])[
            [
                "stack_mae",
                "naive_mae",
                "rolling_mae",
                "baseline_improvement_pct",
                "coverage_80",
            ]
        ]
        .mean()
        .round(2)
        .to_string()
    )
