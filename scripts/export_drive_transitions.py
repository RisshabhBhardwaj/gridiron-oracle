#!/usr/bin/env python3
"""Fit DriveMarkovModel and write ml/oof/transitions.csv for C++ DriveMCMC."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbp", type=Path, default=None, help="Optional parquet/csv of PBP plays")
    parser.add_argument("--from-nflverse", action="store_true", help="Load PBP via nflreadpy")
    parser.add_argument("--start-season", type=int, default=2019)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--out", type=Path, default=ROOT / "ml" / "oof" / "transitions.csv")
    parser.add_argument(
        "--game-state-out",
        type=Path,
        default=ROOT / "ml" / "oof" / "transitions_by_game_state.csv",
        help="5D (fp, down, ytg, score_diff_bucket, quarter) table consumed by "
             "the C++ DriveMCMC engine — see ml.drive_engine.TRANSITIONS_PATH.",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Load play_type/score_differential/quarter/outcome columns from "
             "pbp_plays (DATABASE_URL) instead of --pbp/--from-nflverse. "
             "Needed for the game-state export; --pbp/--from-nflverse frames "
             "may lack quarter/score_differential.",
    )
    args = parser.parse_args()

    import pandas as pd
    from ml.markov_simulator import DriveMarkovModel

    if args.db:
        import psycopg2

        from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL

        conn = psycopg2.connect(DEFAULT_HOST_DATABASE_URL)
        try:
            pbp = pd.read_sql(
                """
                SELECT play_type, down, ydstogo, yardline_100, quarter, score_differential,
                       yards_gained, interception, fumble_lost, penalty, penalty_yards
                FROM pbp_plays
                WHERE play_type IN ('run','pass')
                """,
                conn,
            )
        finally:
            conn.close()
    elif args.pbp is not None:
        if args.pbp.suffix == ".parquet":
            pbp = pd.read_parquet(args.pbp)
        else:
            pbp = pd.read_csv(args.pbp)
    elif args.from_nflverse:
        import nflreadpy

        seasons = list(range(args.start_season, args.end_season + 1))
        pbp = nflreadpy.load_pbp(seasons=seasons)
        if hasattr(pbp, "to_pandas"):
            pbp = pbp.to_pandas()
    else:
        logger.error("Pass --db, --pbp, or --from-nflverse; refusing to invent transitions")
        return 2
    model = DriveMarkovModel()
    model.fit(pbp)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.export_transitions(str(args.out))
    logger.info("Wrote %s", args.out)
    if model.transitions_by_game_state is not None:
        args.game_state_out.parent.mkdir(parents=True, exist_ok=True)
        model.export_transitions_by_game_state(str(args.game_state_out))
        logger.info("Wrote %s", args.game_state_out)
    else:
        logger.warning(
            "No game-state transitions produced (missing play_type/"
            "score_differential/quarter) — pass --db to fit from pbp_plays."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
