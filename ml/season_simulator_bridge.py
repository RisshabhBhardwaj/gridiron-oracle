"""
ml/season_simulator_bridge.py

High-level Python bridge to the C++ Season Simulator.

RESPONSIBILITY
--------------
Provides a single `sim_season()` function that:
  1. Tries the C++ fast path (engine/python_bindings.CppSeasonSimulator).
  2. Falls back automatically to the Python SeasonSimulator on any failure
     (library not built, mismatched struct layout, unsupported platform).

The caller never needs to know which path was taken.

CTYPES STRUCT
-------------
`_PlayerProjectionC` mirrors `gridiron::PlayerProjection` in C++ exactly.
It is used for static type checking and documentation; the actual C++ call
uses the numpy PLAYER_DTYPE structured array (same binary layout).

Struct layout (56 bytes, matches engine/include/season_simulator.hpp):
    char     player_id[32]        — null-terminated ASCII player GSIS ID
    char     position[4]          — "QB\x00\x00", "WR\x00\x00", etc.
    char     team[4]              — "KC\x00\x00", "SF\x00\x00", etc.
    float32  kalman_mean          — Kalman estimate (e.g. 92.3 receiving yards)
    float32  kalman_variance      — Kalman variance (not std dev)
    float32  injury_multiplier    — 1.0 = healthy, 0.0 = OUT
    float32  snap_share           — fraction [0,1] of offensive snaps

USAGE
-----
    from ml.season_simulator_bridge import sim_season

    results = sim_season(
        players_df=active_roster_df,     # DataFrame with player_id, position, team,
                                         # kalman_est_{stat}, kalman_variance_{stat}
        stat="receiving_yards",
        n_simulations=500,
        start_week=9,
        end_week=18,
        corr_mat=None,                   # optional (n_players × n_players) float32
        use_cpp=True,                    # set False to force Python path
    )
    # → {"mean": ndarray, "p10": ndarray, "p50": ndarray, "p90": ndarray}
    # Each array has shape (n_players,), aligned to players_df row order.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Ctypes Structure (mirrors gridiron::PlayerProjection) ─────────────────────

class _PlayerProjectionC(ctypes.Structure):
    """
    Ctypes mirror of gridiron::PlayerProjection (engine/include/season_simulator.hpp).

    Binary layout: 56 bytes, packed, matches the C++ struct exactly.
    Used for type documentation and optional direct ctypes calls.
    For bulk operations, prefer the numpy PLAYER_DTYPE structured array
    (same binary layout, faster batch construction).
    """
    _fields_ = [
        ("player_id",          ctypes.c_char * 32),   # null-terminated ASCII GSIS ID
        ("position",           ctypes.c_char * 4),    # "QB\x00\x00" etc.
        ("team",               ctypes.c_char * 4),    # "KC\x00\x00" etc.
        ("kalman_mean",        ctypes.c_float),        # point estimate
        ("kalman_variance",    ctypes.c_float),        # uncertainty (not std)
        ("injury_multiplier",  ctypes.c_float),        # 1.0 = healthy, 0.0 = OUT
        ("snap_share",         ctypes.c_float),        # offensive snap fraction [0,1]
    ]

    @classmethod
    def from_row(cls, row: dict, stat: str) -> "_PlayerProjectionC":
        """Build a struct from a feature_matrix row dict for a given stat."""
        est = float(row.get(f"kalman_est_{stat}") or 0.0)
        var = float(row.get(f"kalman_variance_{stat}") or max(est * 0.5, 5.0) ** 2)
        c = cls()
        c.player_id         = str(row.get("player_id", ""))[:31].encode()
        c.position          = str(row.get("position", ""))[:3].encode()
        c.team              = str(row.get("team", ""))[:3].encode()
        c.kalman_mean       = est
        c.kalman_variance   = var
        c.injury_multiplier = float(row.get("injury_multiplier", 1.0))
        c.snap_share        = float(row.get("snap_share", 1.0))
        return c


# ── Internal: build numpy PLAYER_DTYPE array from DataFrame ───────────────────

def _build_player_array(players_df: pd.DataFrame, stat: str) -> np.ndarray:
    """
    Convert a players DataFrame into the numpy PLAYER_DTYPE structured array
    expected by engine.python_bindings.CppSeasonSimulator.

    Handles missing kalman columns gracefully (substitutes 0 mean / 25 variance).
    """
    from engine.python_bindings import PLAYER_DTYPE

    n = len(players_df)
    arr = np.zeros(n, dtype=PLAYER_DTYPE)

    est_col = f"kalman_est_{stat}"
    var_col = f"kalman_variance_{stat}"

    for i, (_, row) in enumerate(players_df.iterrows()):
        est = float(row.get(est_col) or 0.0)
        var = float(row.get(var_col) or max(est * 0.5, 5.0) ** 2)

        arr[i]["player_id"]          = str(row.get("player_id", ""))[:31].encode()
        arr[i]["position"]           = str(row.get("position", ""))[:3].encode()
        arr[i]["team"]               = str(row.get("team", ""))[:3].encode()
        arr[i]["kalman_mean"]        = est
        arr[i]["kalman_variance"]    = var
        arr[i]["injury_multiplier"]  = float(row.get("injury_multiplier", 1.0))
        arr[i]["snap_share"]         = float(row.get("snap_share", 1.0))

    return arr


# ── Python fallback path ──────────────────────────────────────────────────────

def _python_sim_season(
    players_df: pd.DataFrame,
    stat: str,
    n_simulations: int,
    start_week: int,
    end_week: int,
    rng_seed: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """
    Pure-Python season simulation fallback.

    Runs SeasonSimulator (ml/season_simulator.py) and extracts the requested
    stat into the standard {mean, p10, p50, p90} ndarray format.
    """
    from ml.season_simulator import SeasonSimulator

    sim = SeasonSimulator(
        season=0,             # season not needed for within-season simulation
        start_week=start_week,
        end_week=end_week,
        n_simulations=n_simulations,
        stats=[stat],
        use_copula=False,     # faster; copula requires per-stat correlation data
        use_cpp=False,
    )

    # Build prior_game_rows as empty (Kalman estimates already baked in players_df)
    player_ids = players_df["player_id"].astype(str).tolist()
    prior_game_rows: dict[str, list[dict]] = {pid: [] for pid in player_ids}

    result = sim.run(
        players_df=players_df,
        prior_game_rows=prior_game_rows,
        rng_seed=rng_seed,
    )

    n = len(player_ids)
    out_mean = np.zeros(n, dtype=np.float32)
    out_p10  = np.zeros(n, dtype=np.float32)
    out_p50  = np.zeros(n, dtype=np.float32)
    out_p90  = np.zeros(n, dtype=np.float32)

    for i, pid in enumerate(player_ids):
        totals = result.player_season_totals.get(pid, {}).get(stat, {})
        out_mean[i] = float(totals.get("mean", 0.0))
        out_p10[i]  = float(totals.get("p10",  0.0))
        out_p50[i]  = float(totals.get("p50",  0.0))
        out_p90[i]  = float(totals.get("p90",  0.0))

    return {"mean": out_mean, "p10": out_p10, "p50": out_p50, "p90": out_p90}


# ── Public API ────────────────────────────────────────────────────────────────

def sim_season(
    players_df: pd.DataFrame,
    stat: str,
    n_simulations: int = 500,
    start_week: int = 1,
    end_week: int = 18,
    corr_mat: Optional[np.ndarray] = None,
    use_cpp: bool = True,
    rng_seed: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """
    Simulate rest-of-season projections for all players in `players_df`.

    Tries the C++ fast path first (engine/python_bindings.CppSeasonSimulator).
    Falls back silently to the Python SeasonSimulator if:
      - The C++ shared library is not built.
      - Any ctypes call fails.
      - `use_cpp=False` is explicitly requested.

    Args:
        players_df:    DataFrame with [player_id, position, team,
                       kalman_est_{stat}, kalman_variance_{stat}] columns.
        stat:          Stat to simulate (e.g. "receiving_yards").
        n_simulations: Number of Monte Carlo season paths (default: 500).
        start_week:    First week to simulate (inclusive, default: 1).
        end_week:      Last week to simulate (inclusive, default: 18).
        corr_mat:      Optional (n_players × n_players) correlation matrix as
                       float32 ndarray. When None, independent draws are used.
        use_cpp:       If False, skips C++ and uses Python simulator directly.
        rng_seed:      Optional seed for reproducibility (Python path only).

    Returns:
        Dict with keys "mean", "p10", "p50", "p90" — each an ndarray of shape
        (n_players,) aligned to players_df row order.
    """
    n_players = len(players_df)
    if n_players == 0:
        empty = np.zeros(0, dtype=np.float32)
        return {"mean": empty, "p10": empty, "p50": empty, "p90": empty}

    if use_cpp:
        try:
            from engine.python_bindings import CppSeasonSimulator, LIB
            if LIB is None:
                raise ImportError("C++ shared library not loaded.")

            cpp_sim = CppSeasonSimulator(n_simulations=n_simulations)
            p_arr = _build_player_array(players_df, stat)

            if corr_mat is not None and corr_mat.size > 0:
                import ctypes as _ct
                c_data = np.ascontiguousarray(corr_mat, dtype=np.float32)
                c_data.ctypes.data_as(_ct.c_void_p)

            result = cpp_sim.sim_full_season(p_arr, corr_mat, start_week, end_week)
            logger.debug(
                "sim_season: C++ path (%d players, %d sims, weeks %d-%d, stat=%s)",
                n_players, n_simulations, start_week, end_week, stat,
            )
            return result

        except Exception as exc:
            logger.warning(
                "sim_season: C++ path failed (%s) — falling back to Python simulator.", exc
            )

    # Python fallback
    logger.debug(
        "sim_season: Python path (%d players, %d sims, weeks %d-%d, stat=%s)",
        n_players, n_simulations, start_week, end_week, stat,
    )
    return _python_sim_season(
        players_df, stat, n_simulations, start_week, end_week, rng_seed=rng_seed,
    )
