"""Deterministic probe of the history-writer path, used to prove that moving the
code into pipeline/serving_rows.py changed nothing (golden file next to it)."""

from __future__ import annotations

import hashlib
import json
from typing import Callable

import numpy as np
import pandas as pd


def synthetic_history_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two cells over three seasons: enough history for conformal in the later seasons."""
    rng = np.random.default_rng(20261114)
    frames = []
    for stat, position, scale in (("receiving_yards", "WR", 55.0), ("receiving_tds", "WR", 0.4), ("fantasy_ppr", "WR", 12.0)):
        for season in (2021, 2022, 2023):
            n = 140
            proj = np.abs(rng.normal(scale, scale / 3, n))
            frames.append(pd.DataFrame({
                "player_id": [f"p{i:03d}" for i in range(n)],
                "game_id": [f"{season}_{i:03d}" for i in range(n)],
                "season": season, "week": (np.arange(n) % 17) + 1,
                "stat": stat, "position": position,
                "projection": proj,
                "y_true": np.abs(proj + rng.normal(0, scale / 4, n)),
                "max_train_season": season - 1,
            }))
    df = pd.concat(frames, ignore_index=True)
    attr = df[["player_id", "game_id", "stat"]].copy()
    attr["pipeline_run_id"] = "probe_run"
    for i, col in enumerate(("base", "kalman_form", "seasonal_baseline", "matchup", "weather_venue",
                             "team_context", "roster_injury", "rule_meta", "unweighted")):
        attr[col] = float(i) / 7.0
    attr["max_residual"] = 1e-9
    return df, attr


def digest(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.sort_index(axis=1).to_csv(index=False).encode()).hexdigest()


def run_probe(
    attach_intervals: Callable[[pd.DataFrame], pd.DataFrame],
    attach_derived: Callable[[pd.DataFrame], pd.DataFrame],
    upsert: Callable,
    capture_batches: Callable[[], list],
) -> dict:
    df, attr = synthetic_history_frame()
    parts = []
    for (_, _), cell in df.groupby(["stat", "position"], sort=True):
        out = attach_intervals(cell.copy())
        out["p25"] = None
        out["p75"] = None
        if (out["stat"] == "fantasy_ppr").all():
            out["fantasy_projection"] = out["projection"]
            out["fantasy_floor"] = out["floor"]
            out["fantasy_ceiling"] = out["ceiling"]
        else:
            out["fantasy_projection"] = None
            out["fantasy_floor"] = None
            out["fantasy_ceiling"] = None
        out["pipeline_run_id"] = "probe_run"
        parts.append(out)
    full = attach_derived(pd.concat(parts, ignore_index=True))

    class _Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Conn:
        committed = 0
        def cursor(self): return _Cur()
        def commit(self): self.committed += 1
        def rollback(self): raise AssertionError("rollback")

    conn = _Conn()
    n = upsert(conn, full, attr)
    batches = capture_batches()
    return {
        "frame_sha256": digest(full),
        "n_rows": int(n),
        "committed": conn.committed,
        "batches": [
            {"sql": " ".join(sql.split()), "n_rows": len(rows), "rows_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()}
            for sql, rows in batches
        ],
    }
