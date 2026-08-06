"""
backend/app/services/kalman.py

Kalman service — exposes player form estimates to the API layer.

Wraps ml/kalman_tracker.py to provide:
  - Current Kalman estimate + variance for a player/stat pair
  - Season-long ability trajectory (for the player detail chart)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stat whitelist — guard every f-string SQL column interpolation
# ---------------------------------------------------------------------------

VALID_STATS: frozenset[str] = frozenset({
    # Passing (QB)
    "pass_attempts", "completions", "passing_yards", "passing_tds", "interceptions",
    # Rushing
    "carries", "rushing_yards", "rushing_tds",
    # Receiving
    "receptions", "receiving_yards", "receiving_tds", "targets",
    # Cross-position / composite
    "fantasy_ppr", "fumbles", "sacks_taken",
    # Legacy aliases kept for backwards compatibility
    "snap_pct",
})


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class KalmanEstimate:
    """Kalman form estimate for one (player, stat) at a specific week."""
    player_id:  str
    stat:       str
    week:       int
    season:     int
    estimate:   float    # x_k: filtered ability estimate
    variance:   float    # P_k: posterior variance
    std:        float    # sqrt(P_k): uncertainty in same units as stat


@dataclass
class KalmanTrajectory:
    """Season-long Kalman ability trajectory for a player."""
    player_id:  str
    stat:       str
    season:     int
    weeks:      list[int]
    estimates:  list[float]
    variances:  list[float]


# ---------------------------------------------------------------------------
# KalmanService
# ---------------------------------------------------------------------------

class KalmanService:
    """
    Provides Kalman form estimates for the /explain and /predict endpoints.

    Reads pre-computed values from feature_matrix when available;
    re-runs KalmanFeatureEngineer on game_logs as fallback.
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url

    # ------------------------------------------------------------------
    # Current estimate
    # ------------------------------------------------------------------

    def get_estimate(
        self, player_id: str, stat: str, week: int, season: int
    ) -> Optional[KalmanEstimate]:
        """
        Return Kalman estimate + variance for one player/stat/week.

        Reads from feature_matrix.kalman_est_{stat} / kalman_variance_{stat}.
        """
        if stat not in VALID_STATS:
            raise HTTPException(status_code=422, detail=f"Invalid stat: {stat}")
        import psycopg2

        try:
            conn = psycopg2.connect(self._db_url)
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT kalman_est_{stat}, kalman_variance_{stat}
                FROM   feature_matrix
                WHERE  player_id = %s AND week = %s AND season = %s
                LIMIT  1
                """,
                [player_id, week, season],
            )
            row = cur.fetchone()
            conn.close()
        except Exception as exc:
            logger.debug("KalmanService.get_estimate DB error: %s", exc)
            return None

        if row is None or row[0] is None:
            return self._recompute(player_id, stat, week, season)

        var = float(row[1]) if row[1] is not None else 100.0
        return KalmanEstimate(
            player_id=player_id,
            stat=stat,
            week=week,
            season=season,
            estimate=float(row[0]),
            variance=var,
            std=var ** 0.5,
        )

    # ------------------------------------------------------------------
    # Season trajectory
    # ------------------------------------------------------------------

    def get_trajectory(
        self, player_id: str, stat: str, season: int
    ) -> KalmanTrajectory:
        """
        Return the full season-long Kalman ability trajectory for the
        player detail chart.
        """
        if stat not in VALID_STATS:
            raise HTTPException(status_code=422, detail=f"Invalid stat: {stat}")
        import psycopg2

        try:
            conn = psycopg2.connect(self._db_url)
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT week, kalman_est_{stat}, kalman_variance_{stat}
                FROM   feature_matrix
                WHERE  player_id = %s AND season = %s
                ORDER BY week
                """,
                [player_id, season],
            )
            rows = cur.fetchall()
            conn.close()
        except Exception as exc:
            logger.debug("KalmanService.get_trajectory DB error: %s", exc)
            rows = []

        weeks     = [int(r[0]) for r in rows if r[1] is not None]
        estimates = [float(r[1]) for r in rows if r[1] is not None]
        variances = [float(r[2]) if r[2] is not None else 100.0 for r in rows if r[1] is not None]

        return KalmanTrajectory(
            player_id=player_id,
            stat=stat,
            season=season,
            weeks=weeks,
            estimates=estimates,
            variances=variances,
        )

    # ------------------------------------------------------------------
    # Fallback: re-run KalmanFeatureEngineer on game_logs
    # ------------------------------------------------------------------

    def _recompute(
        self, player_id: str, stat: str, week: int, season: int
    ) -> Optional[KalmanEstimate]:
        """
        Re-run the Kalman tracker on raw game_logs when feature_matrix
        is not populated for this player/week.
        """
        if stat not in VALID_STATS:
            return None
        import psycopg2

        try:
            conn = psycopg2.connect(self._db_url)
            cur = conn.cursor()
            # Load prior game rows (all weeks < target week)
            cur.execute(
                f"""
                SELECT {stat}
                FROM   game_logs
                WHERE  player_id = %s
                  AND  season    = %s
                  AND  week      < %s
                ORDER BY week
                """,
                [player_id, season, week],
            )
            rows = cur.fetchall()
            conn.close()
        except Exception as exc:
            logger.debug("KalmanService._recompute DB error: %s", exc)
            return None

        if not rows:
            return None

        from ml.kalman_tracker import STAT_SOURCE_COL, KalmanFeatureEngineer

        src_col = STAT_SOURCE_COL.get(stat, stat)
        prior_rows = [{src_col: float(r[0]) if r[0] is not None else 0.0} for r in rows]

        kfe = KalmanFeatureEngineer()
        feats = kfe.compute_kalman_form(prior_rows, position=None)

        est_key = f"kalman_est_{stat}"
        var_key = f"kalman_variance_{stat}"
        if est_key not in feats:
            return None

        var = float(feats.get(var_key, 100.0))
        return KalmanEstimate(
            player_id=player_id,
            stat=stat,
            week=week,
            season=season,
            estimate=float(feats[est_key]),
            variance=var,
            std=var ** 0.5,
        )
