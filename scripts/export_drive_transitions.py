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
    args = parser.parse_args()

    import pandas as pd
    from ml.markov_simulator import DriveMarkovModel

    if args.pbp is not None:
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
        logger.error("Pass --pbp or --from-nflverse; refusing to invent transitions")
        return 2
    model = DriveMarkovModel()
    model.fit(pbp)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.export_transitions(str(args.out))
    logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
