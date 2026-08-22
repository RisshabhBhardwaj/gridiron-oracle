#!/usr/bin/env python3
"""
Verify the Phase 5 game-state-aware Markov fit reproduces the empirical
"leading teams run more in Q4" effect (plan's stated Phase 5 acceptance
criterion).

Computes two independent numbers per (quarter, score_diff_bucket):
  1. Raw empirical run rate straight from pbp_plays (no bucketing on
     fp/down/ytg — this is the ground truth the plan's criterion refers to).
  2. The fitted DriveMarkovModel's count-weighted p_pass for the same
     (quarter, score_diff_bucket) cells, averaged over the finer
     (fp, down, ytg) state it's actually conditioned on.

Passes if leading-big teams have a materially lower Q4 pass rate (run more)
than trailing-big teams in both the raw and fitted numbers, in the same
direction as each other.

Usage:
  DATABASE_URL=... python scripts/verify_drive_transitions.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_BUCKET_LABELS = {0: "trailing_big", 1: "trailing_small", 2: "tied", 3: "leading_small", 4: "leading_big"}


def main() -> int:
    import numpy as np
    import pandas as pd
    import psycopg2

    from ml.markov_simulator import DriveMarkovModel, _score_diff_bucket
    from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL

    conn = psycopg2.connect(DEFAULT_HOST_DATABASE_URL)
    try:
        df = pd.read_sql(
            """
            SELECT play_type, down, ydstogo, yardline_100, quarter, score_differential,
                   yards_gained, interception, fumble_lost, penalty, penalty_yards
            FROM pbp_plays
            WHERE play_type IN ('run','pass') AND quarter BETWEEN 1 AND 4
              AND score_differential IS NOT NULL
            """,
            conn,
        )
    finally:
        conn.close()
    logger.info("Loaded %d run/pass plays", len(df))

    # 1. Raw empirical (no fp/down/ytg conditioning)
    df["score_diff_bucket"] = _score_diff_bucket(df["score_differential"])
    df["is_pass"] = (df["play_type"] == "pass").astype(int)
    raw = df.groupby(["quarter", "score_diff_bucket"]).agg(
        p_pass=("is_pass", "mean"), n=("is_pass", "count")
    ).reset_index()

    # 2. Fitted model
    model = DriveMarkovModel()
    model.fit(df)
    gs = model.transitions_by_game_state
    if gs is None:
        logger.error("Model failed to produce a game-state-aware fit.")
        return 1

    def weighted_p_pass(quarter: int, score_bucket: int) -> float:
        sub = gs[(gs["quarter_idx"] == quarter - 1) & (gs["score_diff_bucket"] == score_bucket)]
        if sub.empty or sub["count"].sum() == 0:
            return float("nan")
        return float(np.average(sub["p_pass"], weights=sub["count"]))

    print(f"{'quarter':>8} {'score_bucket':>15} {'raw_p_pass':>11} {'fitted_p_pass':>14} {'n':>8}")
    for _, row in raw.sort_values(["quarter", "score_diff_bucket"]).iterrows():
        q, sb = int(row["quarter"]), int(row["score_diff_bucket"])
        fitted = weighted_p_pass(q, sb)
        print(f"{q:>8} {_BUCKET_LABELS[sb]:>15} {row['p_pass']:>11.3f} {fitted:>14.3f} {int(row['n']):>8}")

    q4 = raw[raw["quarter"] == 4]
    raw_leading = q4[q4["score_diff_bucket"] == 4]["p_pass"].iloc[0]
    raw_trailing = q4[q4["score_diff_bucket"] == 0]["p_pass"].iloc[0]
    fitted_leading = weighted_p_pass(4, 4)
    fitted_trailing = weighted_p_pass(4, 0)

    print()
    print(f"Q4 leading_big:  raw p_pass={raw_leading:.3f}  fitted p_pass={fitted_leading:.3f}")
    print(f"Q4 trailing_big: raw p_pass={raw_trailing:.3f}  fitted p_pass={fitted_trailing:.3f}")

    ok = (
        raw_leading < raw_trailing - 0.15
        and fitted_leading < fitted_trailing - 0.15
    )
    if ok:
        print("\nPASS: leading teams pass materially less (run more) in Q4, in both raw and fitted numbers.")
        return 0
    print("\nFAIL: expected leading teams to run materially more than trailing teams in Q4.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
