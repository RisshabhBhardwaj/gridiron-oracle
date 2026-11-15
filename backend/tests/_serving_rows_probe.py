"""Deterministic probe of the history-writer path, used to prove that moving the
code into pipeline/serving_rows.py changed nothing (golden file next to it).

Two things make the digests the same on every machine (found 2026-11-15, when the
golden matched on the Mac and failed on GitHub's x86-64 Linux runner):

1. The synthetic input is built with plain numpy ufuncs only. ``Generator.normal(loc,
   scale)`` computes ``loc + scale * z`` inside one compiled expression, which arm64
   fuses into a single rounding (FMA) and x86-64 does in two. Same random stream, but
   about 8 % of the values differed in the last bit. ``loc + scale * standard_normal()``
   is two separate ufunc calls, which never fuse.
2. Floats are written to the digests at 9 significant digits (relative 1e-9), so a
   last-bit difference from another libm, BLAS or SIMD width cannot flip a hash. The
   worst platform difference measured with the old generator was 6.4e-14 relative; a
   real row change is many orders of magnitude larger. SQL text and every non-float
   value are still hashed exactly, and row order and row count are hashed too.

Regenerating the golden (only if the probe's format changes): run ``run_probe`` against
the OLD in-script writer, ``scripts/materialize_stack_projections.py`` at ``de79477^``
(``_attach_causal_conformal_intervals``, ``_attach_derived_fantasy_projection``,
``_upsert``), never against ``pipeline/serving_rows.py``, then compare it with the
shared module. See backend/tests/golden/README.md.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable

import numpy as np
import pandas as pd


#: Significant digits kept for floats in a digest (``%.8e`` writes 9).
DIGEST_FLOAT_DECIMALS = 8


def _normal(rng: np.random.Generator, loc: float, scale: float, n: int) -> np.ndarray:
    """``rng.normal(loc, scale, n)`` without the platform-dependent fused multiply-add."""
    return loc + scale * rng.standard_normal(n)


def canonical(value):
    """Floats become fixed-precision strings; everything else is kept exactly.

    ``+ 0.0`` turns -0.0 into 0.0. NaN and the infinities get one spelling each.
    """
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if x != x:
            return "nan"
        if x in (float("inf"), float("-inf")):
            return "inf" if x > 0 else "-inf"
        return f"{x + 0.0:.{DIGEST_FLOAT_DECIMALS}e}"
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


def synthetic_history_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two cells over three seasons: enough history for conformal in the later seasons."""
    rng = np.random.default_rng(20261114)
    frames = []
    for stat, position, scale in (("receiving_yards", "WR", 55.0), ("receiving_tds", "WR", 0.4), ("fantasy_ppr", "WR", 12.0)):
        for season in (2021, 2022, 2023):
            n = 140
            proj = np.abs(_normal(rng, scale, scale / 3, n))
            frames.append(pd.DataFrame({
                "player_id": [f"p{i:03d}" for i in range(n)],
                "game_id": [f"{season}_{i:03d}" for i in range(n)],
                "season": season, "week": (np.arange(n) % 17) + 1,
                "stat": stat, "position": position,
                "projection": proj,
                "y_true": np.abs(proj + _normal(rng, 0.0, scale / 4, n)),
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
    """SHA-256 of the frame as CSV: float columns at fixed precision, all else exact."""
    out = df.sort_index(axis=1).copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = [canonical(v) for v in out[col]]
    return hashlib.sha256(out.to_csv(index=False).encode()).hexdigest()


def digest_rows(rows: list) -> str:
    """SHA-256 of the rows handed to the database, in order."""
    return hashlib.sha256(json.dumps(canonical(rows), sort_keys=True, default=str).encode()).hexdigest()


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
            {"sql": " ".join(sql.split()), "n_rows": len(rows), "rows_sha256": digest_rows(rows)}
            for sql, rows in batches
        ],
    }
