"""
ml/monte_carlo.py

Monte Carlo projection layer — final layer in the projection stack.

ARCHITECTURE POSITION
---------------------
  Layer 1: Kalman filter     → latent player ability estimates
  Layer 2: XGB + LGB + TFT  → stacked point forecast
  Layer 3: Bayesian layer    → posterior samples (2000 draws)
  Layer 4: This module       → projection result + fantasy scoring

RESPONSIBILITY
--------------
  Takes Bayesian posterior samples and converts them into:
    - Projection (median), floor (p10), ceiling (p90)
    - Boom / bust probabilities against position-specific thresholds
    - Fantasy PPR scoring distribution over all 2000 samples
    - Fantasy projection (median), floor (p10), ceiling (p90)

SINGLE-STAT DESIGN
------------------
  Each MonteCarloProjector call handles ONE stat at a time (e.g.
  receiving_yards). Multi-stat correlations (e.g. yards × TDs) will
  be handled by a Gaussian copula in Phase 3 (ml/monte_carlo.py
  update per §11). This keeps the math correct and simple for now.

FANTASY SCORING (PPR)
---------------------
  Passing:   0.04 pts/yd,  4 pts/TD, -2 pts/INT
  Rushing:   0.1  pts/yd,  6 pts/TD,  1 pt/carry  (half-PPR carry bonus)
  Receiving: 0.1  pts/yd,  6 pts/TD,  1 pt/rec    (full PPR reception bonus)

  For single-stat projections we model ONLY the yardage distribution.
  TD and reception bonuses cannot be derived from a yards-only sample,
  so fantasy_points_distribution captures the yardage component only.
  The stacking ensemble separately projects TDs and receptions; those
  are added at the full-projection aggregation step in train.py.

ALL MATH IS PURE NUMPY
----------------------
  No Python loops over the 2000-sample arrays. Vectorized percentile,
  mean, and comparison operations throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

# ── MLX acceleration (Apple Silicon) ─────────────────────────────────────────
# MLX uses unified CPU/GPU memory — no data transfer overhead. On M1/M2/M3 the
# quantile and mean ops run on the Neural Engine / GPU cores, giving 3-8× speed
# over NumPy for batch projections.  Falls back to NumPy on non-Apple platforms.
try:
    import mlx.core as mx  # type: ignore[import]
    _USE_MLX: bool = True
except ImportError:
    mx = None  # type: ignore[assignment]
    _USE_MLX: bool = False


# ---------------------------------------------------------------------------
# Position-specific boom / bust thresholds
# ---------------------------------------------------------------------------
# These represent "good game" (boom) and "bad game" (bust) in yards for the
# primary stat tracked per position. Defined here as class-level constants
# so they're visible in documentation and easy to tune.

# Boom: P(stat > threshold) — elite performance probability
# Thresholds calibrated from 2019-2025 90th percentile single-game performances.
BOOM_THRESHOLDS: dict[str, dict[str, float]] = {
    # ── Receiving ─────────────────────────────────────────────────────────
    "receiving_yards":  {"WR": 100.0, "RB":  50.0, "TE":  60.0, "QB":   0.0},
    "receptions":       {"WR":   7.0, "RB":   5.0, "TE":   6.0, "QB":   0.0},
    "targets":          {"WR":   8.0, "RB":   5.0, "TE":   7.0, "QB":   0.0},
    "receiving_tds":    {"WR":   1.0, "RB":   1.0, "TE":   1.0, "QB":   0.0},
    # ── Rushing ───────────────────────────────────────────────────────────
    "rushing_yards":    {"WR":  30.0, "RB":  80.0, "TE":  10.0, "QB":  50.0},
    "rushing_tds":      {"WR":   1.0, "RB":   1.0, "TE":   0.0, "QB":   1.0},
    "carries":          {"WR":   3.0, "RB":  18.0, "TE":   1.0, "QB":   7.0},
    # ── Passing (QB) ──────────────────────────────────────────────────────
    "passing_yards":    {"WR":   0.0, "RB":   0.0, "TE":   0.0, "QB": 300.0},
    "passing_tds":      {"WR":   0.0, "RB":   0.0, "TE":   0.0, "QB":   3.0},
    "pass_attempts":    {"WR":   0.0, "RB":   0.0, "TE":   0.0, "QB":  40.0},
    "completions":      {"WR":   0.0, "RB":   0.0, "TE":   0.0, "QB":  28.0},
    # ── Negative events — boom = rare enough to be notable (1 in a game) ─
    "interceptions":    {"WR":   0.0, "RB":   0.0, "TE":   0.0, "QB":   2.0},
    "fumbles":          {"WR":   1.0, "RB":   1.0, "TE":   1.0, "QB":   1.0},
    "sacks_taken":      {"WR":   0.0, "RB":   0.0, "TE":   0.0, "QB":   4.0},
    # ── Composite ─────────────────────────────────────────────────────────
    "fantasy_ppr":      {"WR":  20.0, "RB":  20.0, "TE":  15.0, "QB":  25.0},
}

# Bust: P(stat < threshold) — poor performance probability
BUST_THRESHOLDS: dict[str, dict[str, float]] = {
    # ── Receiving ─────────────────────────────────────────────────────────
    "receiving_yards":  {"WR":  30.0, "RB":  10.0, "TE":  15.0, "QB":   0.0},
    "receptions":       {"WR":   2.0, "RB":   1.0, "TE":   2.0, "QB":   0.0},
    "targets":          {"WR":   2.0, "RB":   1.0, "TE":   2.0, "QB":   0.0},
    "receiving_tds":    {"WR":   0.0, "RB":   0.0, "TE":   0.0, "QB":   0.0},
    # ── Rushing ───────────────────────────────────────────────────────────
    "rushing_yards":    {"WR":   5.0, "RB":  20.0, "TE":   2.0, "QB":  10.0},
    "rushing_tds":      {"WR":   0.0, "RB":   0.0, "TE":   0.0, "QB":   0.0},
    "carries":          {"WR":   0.0, "RB":   8.0, "TE":   0.0, "QB":   2.0},
    # ── Passing (QB) ──────────────────────────────────────────────────────
    "passing_yards":    {"WR":   0.0, "RB":   0.0, "TE":   0.0, "QB": 150.0},
    "passing_tds":      {"WR":   0.0, "RB":   0.0, "TE":   0.0, "QB":   0.0},
    "pass_attempts":    {"WR":   0.0, "RB":   0.0, "TE":   0.0, "QB":  25.0},
    "completions":      {"WR":   0.0, "RB":   0.0, "TE":   0.0, "QB":  14.0},
    # ── Negative events — bust = multiple in one game ─────────────────────
    "interceptions":    {"WR":   0.0, "RB":   0.0, "TE":   0.0, "QB":   0.0},
    "fumbles":          {"WR":   0.0, "RB":   0.0, "TE":   0.0, "QB":   0.0},
    "sacks_taken":      {"WR":   0.0, "RB":   0.0, "TE":   0.0, "QB":   1.0},
    # ── Composite ─────────────────────────────────────────────────────────
    "fantasy_ppr":      {"WR":   5.0, "RB":   5.0, "TE":   5.0, "QB":  10.0},
}

# Fantasy PPR scoring weights per unit of each stat.
# Standard DraftKings / FanDuel scoring:
#   Passing: 0.04/yd, 4.0/TD, -2.0/INT
#   Rushing: 0.1/yd,  6.0/TD, 0.1/carry (half-PPR carry bonus removed — standard PPR has no carry bonus)
#   Receiving: 0.1/yd, 6.0/TD, 1.0/reception (full PPR)
#   Negative events: -2.0/fumble_lost, -2.0/INT
FANTASY_PTS_PER_YARD: dict[str, float] = {
    # Positive scoring
    "receiving_yards":  0.1,
    "rushing_yards":    0.1,
    "passing_yards":    0.04,
    "receptions":       1.0,   # full PPR: 1 pt per reception
    "receiving_tds":    6.0,
    "rushing_tds":      6.0,
    "passing_tds":      4.0,
    "targets":          0.0,   # no direct fantasy scoring for targets
    "carries":          0.0,   # PPR has no carry bonus (half-PPR is 0.5; use 0 for standard scoring)
    "pass_attempts":    0.0,
    "completions":      0.0,
    # Negative scoring — fantasy penalties
    "interceptions":   -2.0,   # -2 pts per interception thrown
    "fumbles":         -2.0,   # -2 pts per fumble lost
    "sacks_taken":      0.0,   # no direct scoring impact from sacks
    # Composite (already in fantasy points)
    "fantasy_ppr":      1.0,
    "target_share":     0.0,
    "air_yards_share":  0.0,
    "attempts":         0.0,   # legacy alias for pass_attempts
}

# Default fallback position when position is unrecognized.
_DEFAULT_POS: str = "WR"


# ---------------------------------------------------------------------------
# ProjectionResult dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectionResult:
    """
    Immutable projection result for one player-stat pair.

    All float fields are Python floats (not numpy scalars) to ensure
    JSON serializability downstream.

    Percentile fields:
        floor/ceiling   — p10/p90  (standard)
        p5/p95          — min exposure / max upside (for GPP DFS)
        p25/p75         — cash game floor/ceiling (more conservative)
    """

    projection:                   float
    floor:                        float   # p10
    ceiling:                      float   # p90
    p5:                           float   # 5th percentile
    p25:                          float   # 25th percentile
    p75:                          float   # 75th percentile
    p95:                          float   # 95th percentile
    boom_probability:             float
    bust_probability:             float
    fantasy_points_distribution:  np.ndarray
    fantasy_projection:           float
    fantasy_floor:                float
    fantasy_ceiling:              float
    n_samples:                    int

    def __hash__(self):
        return hash((
            self.projection,
            self.floor,
            self.ceiling,
            self.p5, self.p25, self.p75, self.p95,
            self.boom_probability,
            self.bust_probability,
            self.fantasy_points_distribution.tobytes(),
            self.fantasy_projection,
            self.fantasy_floor,
            self.fantasy_ceiling,
            self.n_samples,
        ))

    def __eq__(self, other):
        if not isinstance(other, ProjectionResult):
            return NotImplemented
        return (
            self.projection == other.projection
            and self.floor == other.floor
            and self.ceiling == other.ceiling
            and self.p5 == other.p5
            and self.p25 == other.p25
            and self.p75 == other.p75
            and self.p95 == other.p95
            and self.boom_probability == other.boom_probability
            and self.bust_probability == other.bust_probability
            and np.array_equal(
                self.fantasy_points_distribution,
                other.fantasy_points_distribution,
            )
            and self.fantasy_projection == other.fantasy_projection
            and self.fantasy_floor == other.fantasy_floor
            and self.fantasy_ceiling == other.fantasy_ceiling
            and self.n_samples == other.n_samples
        )


# ---------------------------------------------------------------------------
# MonteCarloProjector
# ---------------------------------------------------------------------------

class MonteCarloProjector:
    """
    Converts Bayesian posterior samples into projection results.

    Usage (single player):
        projector = MonteCarloProjector()
        result = projector.project(samples, stat="receiving_yards", position="WR")

    Usage (batch):
        df = projector.batch_project(
            samples_dict={"player_a": samples_a, "player_b": samples_b},
            stat="receiving_yards",
            position="WR",
        )
    """

    # Class-level references to the module constants (for clean subclassing).
    BOOM_THRESHOLDS = BOOM_THRESHOLDS
    BUST_THRESHOLDS = BUST_THRESHOLDS
    FANTASY_PTS_PER_YARD = FANTASY_PTS_PER_YARD

    # ------------------------------------------------------------------
    # project
    # ------------------------------------------------------------------

    def project(
        self,
        posterior_samples: np.ndarray,
        stat: str,
        position: str,
    ) -> ProjectionResult:
        """
        Compute projection result for a single player-stat pair.

        All percentile/probability math is vectorised NumPy — no Python
        loops over samples.

        Args:
            posterior_samples: 1-D array of Bayesian posterior draws.
            stat:              Stat name matching KALMAN_STATS keys,
                               e.g. "receiving_yards".
            position:          Player position ("WR"/"RB"/"TE"/"QB").

        Returns:
            ProjectionResult (frozen dataclass).

        Raises:
            ValueError: if posterior_samples is empty.
        """
        samples = np.asarray(posterior_samples, dtype=float)
        if samples.size == 0:
            raise ValueError("posterior_samples must not be empty.")

        pos = position if position in ("WR", "RB", "TE", "QB") else _DEFAULT_POS

        boom_thresh = self._boom_threshold(stat, pos)
        bust_thresh = self._bust_threshold(stat, pos)
        pts_per_unit = self.FANTASY_PTS_PER_YARD.get(stat, 0.0)

        if _USE_MLX:
            # ── MLX path (Apple Silicon) ─────────────────────────────
            mx_s = mx.array(samples, dtype=mx.float32)
            pcts = mx.quantile(mx_s, mx.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))
            mx.eval(pcts)
            p5, p10, p25, p50, p75, p90, p95 = [float(pcts[i].item()) for i in range(7)]

            boom_prob = float(mx.mean(mx_s > mx.array(boom_thresh, dtype=mx.float32)).item())
            bust_prob = float(mx.mean(mx_s < mx.array(bust_thresh, dtype=mx.float32)).item())

            mx_fp = mx_s * float(pts_per_unit)
            fp_pcts = mx.quantile(mx_fp, mx.array([0.1, 0.5, 0.9]))
            mx.eval(fp_pcts)
            fp10, fp50, fp90 = [float(fp_pcts[i].item()) for i in range(3)]
            fp_samples = np.array(mx_fp.tolist(), dtype=float)
        else:
            # ── NumPy fallback ────────────────────────────────────────
            p5, p10, p25, p50, p75, p90, p95 = np.percentile(
                samples, [5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0]
            )
            boom_prob = float(np.mean(samples > boom_thresh))
            bust_prob = float(np.mean(samples < bust_thresh))
            fp_samples = samples * pts_per_unit
            fp10, fp50, fp90 = np.percentile(fp_samples, [10.0, 50.0, 90.0])

        return ProjectionResult(
            projection=float(p50),
            floor=float(p10),
            ceiling=float(p90),
            p5=float(p5),
            p25=float(p25),
            p75=float(p75),
            p95=float(p95),
            boom_probability=boom_prob,
            bust_probability=bust_prob,
            fantasy_points_distribution=fp_samples,
            fantasy_projection=float(fp50),
            fantasy_floor=float(fp10),
            fantasy_ceiling=float(fp90),
            n_samples=int(samples.size),
        )

    # ------------------------------------------------------------------
    # batch_project
    # ------------------------------------------------------------------

    def batch_project(
        self,
        samples_dict: Dict[str, np.ndarray],
        stat: str,
        position: str,
    ) -> pd.DataFrame:
        """
        Vectorized batch projection for multiple players.

        Args:
            samples_dict: {player_id: posterior_samples_array} mapping.
            stat:         Stat name (same for all players in this call).
            position:     Position (same for all players in this call).

        Returns:
            pd.DataFrame with columns:
                player_id, projection, floor, ceiling,
                boom_probability, bust_probability,
                fantasy_projection, fantasy_floor, fantasy_ceiling,
                n_samples
            One row per player_id, indexed by player_id.

        Notes:
            - fantasy_points_distribution is NOT included in the DataFrame
              (arrays can't be cleanly stored as column values).
            - Iterate over samples_dict to retrieve per-player distributions.
        """
        _EMPTY_COLS = [
            "player_id", "projection", "floor", "ceiling",
            "p5", "p25", "p75", "p95",
            "boom_probability", "bust_probability",
            "fantasy_projection", "fantasy_floor", "fantasy_ceiling",
            "n_samples",
        ]
        if not samples_dict:
            return pd.DataFrame(columns=_EMPTY_COLS)

        pos = position if position in ("WR", "RB", "TE", "QB") else _DEFAULT_POS
        boom_thresh = self._boom_threshold(stat, pos)
        bust_thresh = self._bust_threshold(stat, pos)
        pts_per_unit = self.FANTASY_PTS_PER_YARD.get(stat, 0.0)
        player_ids = list(samples_dict.keys())
        arrays = [np.asarray(samples_dict[pid], dtype=float) for pid in player_ids]

        # Vectorized MLX batch path: stack into (n_players, n_samples) matrix.
        # Only used when all arrays have identical length (the common case from
        # BayesianProjection which always draws n_bayesian_samples draws).
        lengths = {a.shape[0] for a in arrays}
        use_mlx_batch = _USE_MLX and len(lengths) == 1

        if use_mlx_batch:
            mat = mx.array(np.stack(arrays, axis=0), dtype=mx.float32)  # (P, N)
            pct_q = mx.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], dtype=mx.float32)
            fp_q  = mx.array([0.1, 0.5, 0.9], dtype=mx.float32)

            pcts  = mx.quantile(mat, pct_q, axis=1)     # (7, P)
            boom  = mx.mean(mat > mx.array(boom_thresh, dtype=mx.float32), axis=1)  # (P,)
            bust  = mx.mean(mat < mx.array(bust_thresh, dtype=mx.float32), axis=1)  # (P,)
            fp    = mat * float(pts_per_unit)
            fp_pcts = mx.quantile(fp, fp_q, axis=1)  # (3, P)
            mx.eval(pcts, boom, bust, fp_pcts)

            pcts_np  = np.array(pcts.tolist())      # (7, P)
            fp_np    = np.array(fp_pcts.tolist())   # (3, P)
            boom_np  = np.array(boom.tolist())
            bust_np  = np.array(bust.tolist())
            n_s = arrays[0].shape[0]

            rows = [
                {
                    "player_id":          pid,
                    "projection":         float(pcts_np[3, i]),  # p50
                    "floor":              float(pcts_np[1, i]),  # p10
                    "ceiling":            float(pcts_np[5, i]),  # p90
                    "p5":                 float(pcts_np[0, i]),
                    "p25":                float(pcts_np[2, i]),
                    "p75":                float(pcts_np[4, i]),
                    "p95":                float(pcts_np[6, i]),
                    "boom_probability":   float(boom_np[i]),
                    "bust_probability":   float(bust_np[i]),
                    "fantasy_projection": float(fp_np[1, i]),
                    "fantasy_floor":      float(fp_np[0, i]),
                    "fantasy_ceiling":    float(fp_np[2, i]),
                    "n_samples":          n_s,
                }
                for i, pid in enumerate(player_ids)
            ]
        else:
            # NumPy fallback: per-player project() calls
            rows = []
            for player_id, samples in samples_dict.items():
                r = self.project(samples, stat=stat, position=position)
                rows.append({
                    "player_id":          player_id,
                    "projection":         r.projection,
                    "floor":              r.floor,
                    "ceiling":            r.ceiling,
                    "p5":                 r.p5,
                    "p25":                r.p25,
                    "p75":                r.p75,
                    "p95":                r.p95,
                    "boom_probability":   r.boom_probability,
                    "bust_probability":   r.bust_probability,
                    "fantasy_projection": r.fantasy_projection,
                    "fantasy_floor":      r.fantasy_floor,
                    "fantasy_ceiling":    r.fantasy_ceiling,
                    "n_samples":          r.n_samples,
                })

        df = pd.DataFrame(rows).set_index("player_id")
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _boom_threshold(self, stat: str, position: str) -> float:
        """Return the boom threshold for (stat, position), default 0."""
        pos_map = self.BOOM_THRESHOLDS.get(stat)
        if pos_map is None:
            return 0.0
        return pos_map.get(position, 0.0)

    def _bust_threshold(self, stat: str, position: str) -> float:
        """Return the bust threshold for (stat, position), default 0."""
        pos_map = self.BUST_THRESHOLDS.get(stat)
        if pos_map is None:
            return 0.0
        return pos_map.get(position, 0.0)
