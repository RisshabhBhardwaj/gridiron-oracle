#!/usr/bin/env python3
"""
Materialize team-game outcome predictions (Phase 4) for one forward week.

Fits Ridge models for points/yards/pass_rate on every season strictly before
--season (plays is excluded — see ml.team_game_model.TARGET_COLS), derives
win_probability from the points model (see
ml.team_game_model.derive_win_probability), predicts for every team playing
in --season/--week, and writes to team_game_predictions.

Usage:
  DATABASE_URL=... python scripts/materialize_team_game_predictions.py --season 2026 --week 1
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from scipy.stats import norm
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.team_game_model import FEATURE_COLS, TARGET_COLS, _prepare_x, compute_oof_residual_std
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.team_game_features import build_team_game_forward_frame, build_team_game_frame

logger = logging.getLogger(__name__)


def _fit(train_df: pd.DataFrame, target_col: str, alpha: float) -> tuple[Ridge, pd.Series]:
    train_df = train_df[train_df[target_col].notna()]
    fill_values = train_df[FEATURE_COLS].median(numeric_only=True)
    X = _prepare_x(train_df, fill_values)
    y = train_df[target_col].astype(float).values
    model = Ridge(alpha=alpha).fit(X, y)
    return model, fill_values


def materialize(season: int, week: int, database_url: str, alpha: float = 10.0) -> int:
    forward = build_team_game_forward_frame(database_url, season, week)
    if forward.empty:
        raise ValueError(f"No scheduled games found for season={season} week={week}")

    train_df = build_team_game_frame(database_url, list(range(2019, season)))
    if train_df.empty:
        raise ValueError(f"No training data available strictly before season={season}")

    predictions = forward[["game_id", "team", "opponent", "season", "week", "is_home"]].copy()

    points_model, points_fill = _fit(train_df, "points", alpha)
    X_forward_points = _prepare_x(forward, points_fill)
    predictions["points"] = points_model.predict(X_forward_points)

    for target, target_col in TARGET_COLS.items():
        if target == "points":
            continue
        model, fill_values = _fit(train_df, target_col, alpha)
        X_forward = _prepare_x(forward, fill_values)
        predictions[target] = model.predict(X_forward)

    # Win probability: same points model, paired against the opponent's own
    # predicted points for the same game — see ml.team_game_model.derive_win_probability
    # for the full rationale (kept in sync with predictions["points"] here).
    # residual_std is pooled OUT-OF-FOLD (walk-forward), not in-sample —
    # an in-sample std understates uncertainty and overstates confidence.
    points_train_df = train_df[train_df["points"].notna()]
    residual_std = compute_oof_residual_std(points_train_df, alpha, target_col="points")
    margin_std = residual_std * np.sqrt(2)
    opp_points = predictions[["game_id", "team", "points"]].rename(
        columns={"team": "opponent", "points": "opp_points"}
    )
    paired = predictions.merge(opp_points, on=["game_id", "opponent"], how="left")
    margin = paired["points"].values - paired["opp_points"].values
    predictions["win_probability"] = norm.cdf(margin / margin_std)

    model_run_id = f"team_game_{season}_{week}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    rows = [
        (
            r.game_id, r.team, r.opponent, int(r.season), int(r.week), int(r.is_home),
            float(r.points), float(r.yards), float(r.pass_rate),
            float(r.win_probability), model_run_id,
        )
        for r in predictions.itertuples()
    ]

    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO team_game_predictions
                    (game_id, team, opponent, season, week, is_home,
                     points, yards, pass_rate, win_probability, model_run_id)
                VALUES %s
                ON CONFLICT (game_id, team) DO UPDATE SET
                    opponent = EXCLUDED.opponent, is_home = EXCLUDED.is_home,
                    points = EXCLUDED.points,
                    yards = EXCLUDED.yards, pass_rate = EXCLUDED.pass_rate,
                    win_probability = EXCLUDED.win_probability,
                    model_run_id = EXCLUDED.model_run_id, created_at = now()
                """,
                rows,
            )
        conn.commit()
    finally:
        conn.close()

    logger.info("Wrote %d team_game_predictions rows (model_run_id=%s)", len(rows), model_run_id)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    n = materialize(args.season, args.week, args.database_url, alpha=args.alpha)
    print(f"Wrote {n} team_game_predictions rows for season={args.season} week={args.week}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
