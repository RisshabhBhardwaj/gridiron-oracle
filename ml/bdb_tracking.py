"""Big Data Bowl tracking loader — unblocks gnn_matchup after SP5 wiring.

FTN charting does not provide coverage shell. Tracking fills that gap when
the CSVs are present; otherwise this module fails closed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_BDB_DIR = Path(__file__).resolve().parents[1] / "data" / "bdb"


def load_tracking_week(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"BDB tracking file missing: {path}")
    frame = pd.read_csv(path)
    required = {"gameId", "playId", "nflId", "x", "y", "event"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} missing tracking columns {sorted(missing)}")
    return frame


def tracking_available(directory: Path | None = None) -> bool:
    root = directory or DEFAULT_BDB_DIR
    if not root.is_dir():
        return False
    return any(root.glob("tracking_week_*.csv")) or any(root.glob("*.parquet"))
