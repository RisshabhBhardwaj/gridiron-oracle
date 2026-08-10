"""
backend/app/services/projection.py

Projection service — core business logic for /predict and /projections/week.

Execution order:
  1. Query projections table for pre-computed results (happy path).
  2. If no rows found, fall back to PipelineRunner.run() (live inference).
  3. Augment with Kalman form estimates.

The service always returns typed ProjectionResult dataclasses;
the API layer converts them to Pydantic response models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import HTTPException

from backend.app.core.tracing import start_span

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
class PropComparison:
    market_line:            Optional[float] = None
    model_p_over:           Optional[float] = None
    implied_market_p_over:  Optional[float] = None
    edge_pct:               Optional[float] = None
    brier_score_historical: Optional[float] = None


@dataclass
class ProjectionResult:
    """Full projection result for one player / stat / week."""
    player_id:               str
    player_name:             str
    position:                str
    team:                    Optional[str]
    week:                    int
    season:                  int
    stat:                    str
    projection:              float
    floor:                   Optional[float]
    ceiling:                 Optional[float]
    boom_probability:        Optional[float]
    bust_probability:        Optional[float]
    fantasy_projection:      Optional[float]
    fantasy_floor:           Optional[float]
    fantasy_ceiling:         Optional[float]
    kalman_ability_estimate: Optional[float]
    kalman_uncertainty:      Optional[float]
    confidence_score:        Optional[float]
    prop_comparison:         Optional[PropComparison]
    data_freshness:          datetime
    model_version:           str
    interval_method:         str = "unavailable"
    degraded:                bool = False
    pipeline_run_id:         Optional[str] = None


# ---------------------------------------------------------------------------
# ProjectionService
# ---------------------------------------------------------------------------

class ProjectionService:
    """
    Orchestrates projection retrieval for the /predict endpoint.

    Args:
        db_url:        psycopg2 DSN (postgresql://...).
        model_version: Tag written into every response for traceability.
    """

    def __init__(self, db_url: str, model_version: str = "latest") -> None:
        self._db_url = db_url
        self._model_version = model_version

    # ------------------------------------------------------------------
    # Single player projection (for /predict)
    # ------------------------------------------------------------------

    def get_projection(
        self,
        player_name: str,
        week: int,
        season: int,
        stat: str = "receiving_yards",
    ) -> Optional[ProjectionResult]:
        """
        Return the projection for a single player/week/season/stat.

        Resolution order:
          1. Fuzzy-match player_name → player_id via players table.
          2. Load pre-computed row from projections table.
          3. Fall back to PipelineRunner (fast=True) if no row found.
        """
        with start_span(
            "projection.get_projection",
            attributes={
                "player.name_query": player_name,
                "projection.week": week,
                "projection.season": season,
                "projection.stat": stat,
            },
            tracer_name="backend.app.services.projection",
        ):
            player_id, player_name_resolved, position, team = self._resolve_player(player_name)
            if player_id is None:
                logger.warning("Player not found: %s", player_name)
                return None

            row = self._load_projection_row(player_id, week, season, stat)
            if row is None:
                logger.info(
                    "No pre-computed projection for %s wk%d — running pipeline",
                    player_id, week,
                )
                row = self._run_pipeline(player_id, position or "WR", week, season, stat)
            if row is None:
                return None

            kalman_est, kalman_var = self._load_kalman(player_id, week, season, stat)

            return ProjectionResult(
                player_id=player_id,
                player_name=player_name_resolved or player_name,
                position=position or row.get("position", ""),
                team=team,
                week=week,
                season=season,
                stat=stat,
                projection=_f(row.get("projection", 0.0)),
                floor=_interval_value(row, "floor"),
                ceiling=_interval_value(row, "ceiling"),
                interval_method=_interval_method(row),
                degraded=bool(row.get("degraded", False)),
                pipeline_run_id=row.get("pipeline_run_id"),
                boom_probability=_opt(row.get("boom_probability")),
                bust_probability=_opt(row.get("bust_probability")),
                fantasy_projection=_opt(row.get("fantasy_projection")),
                fantasy_floor=_opt(row.get("fantasy_floor")),
                fantasy_ceiling=_opt(row.get("fantasy_ceiling")),
                kalman_ability_estimate=kalman_est,
                kalman_uncertainty=_opt(kalman_var ** 0.5 if kalman_var else None),
                confidence_score=_confidence(kalman_var),
                prop_comparison=None,  # populated by API layer if odds available
                data_freshness=self._data_freshness(player_id, week, season),
                model_version=self._model_version,
            )

    # ------------------------------------------------------------------
    # Batch week projection (for /projections/week/{n})
    # ------------------------------------------------------------------

    def get_week_projections(
        self,
        week: int,
        season: int,
        positions: list[str] | None = None,
        stat: str = "receiving_yards",
    ) -> list[ProjectionResult]:
        """Return all projections for a given week, ordered by projection desc."""
        import psycopg2

        pos_filter = positions or ["WR", "RB", "TE", "QB"]
        placeholders = ",".join(["%s"] * len(pos_filter))

        try:
            conn = psycopg2.connect(self._db_url)
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT p.player_id, p.season, p.week, p.stat,
                       p.position, p.projection, p.floor, p.ceiling,
                       p.boom_probability, p.bust_probability,
                       p.fantasy_projection, p.fantasy_floor, p.fantasy_ceiling,
                       p.posterior_samples, p.pipeline_run_id,
                       pl.full_name, pl.team
                FROM   projections p
                LEFT JOIN players pl ON pl.id = p.player_id
                WHERE  p.week     = %s
                  AND  p.season   = %s
                  AND  p.stat     = %s
                  AND  p.position IN ({placeholders})
                ORDER BY p.projection DESC NULLS LAST
                """,
                [week, season, stat] + pos_filter,
            )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            conn.close()
        except Exception as exc:
            logger.error("get_week_projections DB error: %s", exc)
            return []

        # data_freshness should reflect the materialized feature row used for
        # projection freshness, not request time.
        try:
            conn2 = psycopg2.connect(self._db_url)
            cur2 = conn2.cursor()
            cur2.execute(
                "SELECT MAX(computed_at) FROM feature_matrix "
                "WHERE season = %s AND week = %s",
                (season, week),
            )
            ts_row = cur2.fetchone()
            conn2.close()
            freshness = ts_row[0] if ts_row and ts_row[0] else datetime.now(timezone.utc)
        except Exception:
            freshness = datetime.now(timezone.utc)

        results = []
        for raw in rows:
            r = dict(zip(cols, raw))
            results.append(ProjectionResult(
                player_id=r["player_id"],
                player_name=r.get("full_name") or r["player_id"],
                position=r.get("position") or "",
                team=r.get("team"),
                week=week,
                season=season,
                stat=stat,
                projection=_f(r.get("projection", 0.0)),
                floor=_interval_value(r, "floor"),
                ceiling=_interval_value(r, "ceiling"),
                interval_method=_interval_method(r),
                pipeline_run_id=r.get("pipeline_run_id"),
                boom_probability=_opt(r.get("boom_probability")),
                bust_probability=_opt(r.get("bust_probability")),
                fantasy_projection=_opt(r.get("fantasy_projection")),
                fantasy_floor=_opt(r.get("fantasy_floor")),
                fantasy_ceiling=_opt(r.get("fantasy_ceiling")),
                kalman_ability_estimate=None,
                kalman_uncertainty=None,
                confidence_score=None,
                prop_comparison=None,
                data_freshness=freshness,
                model_version=self._model_version,
            ))
        return results

    # ------------------------------------------------------------------
    # Season projection (for /projections/season/{n})
    # ------------------------------------------------------------------

    def get_season_projections(
        self,
        season: int,
        start_week: int,
        stats: list[str] = ["passing_yards", "rushing_yards", "receiving_yards", "fantasy_ppr"],
        positions: list[str] | None = None,
    ) -> list[dict]:
        """
        Return the rest-of-season projections for all players.
        Uses the C++ engine for fast simulation.
        """
        import psycopg2
        import pandas as pd

        with start_span(
            "projection.get_season_projections",
            attributes={
                "projection.season": season,
                "projection.start_week": start_week,
                "projection.positions": positions or ["WR", "RB", "TE", "QB"],
                "projection.stats": stats,
            },
            tracer_name="backend.app.services.projection",
        ):
            pos_filter = positions or ["WR", "RB", "TE", "QB"]
            placeholders = ",".join(["%s"] * len(pos_filter))

            try:
                conn = psycopg2.connect(self._db_url)
                # Fetch the most recent kalman estimates for players before or on start_week
                query = f"""
                    WITH ranked AS (
                        SELECT player_id, team, position,
                               kalman_est_passing_yards, kalman_variance_passing_yards,
                               kalman_est_rushing_yards, kalman_variance_rushing_yards,
                               kalman_est_receiving_yards, kalman_variance_receiving_yards,
                               kalman_est_fantasy_ppr, kalman_variance_fantasy_ppr,
                               ROW_NUMBER() OVER(PARTITION BY player_id ORDER BY week DESC, computed_at DESC) as rn
                        FROM feature_matrix
                        WHERE season = %s AND week < %s AND position IN ({placeholders})
                    ),
                    players_meta AS (
                        SELECT id as player_id, full_name as player_name FROM players
                    )
                    SELECT r.player_id, pm.player_name, r.team, r.position,
                           r.kalman_est_passing_yards, r.kalman_variance_passing_yards,
                           r.kalman_est_rushing_yards, r.kalman_variance_rushing_yards,
                           r.kalman_est_receiving_yards, r.kalman_variance_receiving_yards,
                           r.kalman_est_fantasy_ppr, r.kalman_variance_fantasy_ppr
                    FROM ranked r
                    JOIN players_meta pm ON pm.player_id = r.player_id
                    WHERE r.rn = 1
                """

                df = pd.read_sql(query, conn, params=[season, start_week] + pos_filter)
                conn.close()
            except Exception as exc:
                logger.error("get_season_projections DB error: %s", exc)
                return []

            if df.empty:
                return []

            try:
                from ml.season_simulator import SeasonSimulator

                sim = SeasonSimulator(
                    season=season,
                    start_week=start_week,
                    end_week=18,
                    n_simulations=500,
                    positions=pos_filter,
                    stats=stats,
                    use_copula=False,
                    use_cpp=True,
                )
                with start_span(
                    "projection.season_cpp_fast",
                    attributes={
                        "season.player_count": len(df),
                        "season.stat_count": len(stats),
                        "season.positions": pos_filter,
                    },
                    tracer_name="backend.app.services.projection",
                ):
                    res = sim._run_cpp_fast(df)

                results = []
                for pid, p_name, pos, team in zip(df["player_id"], df["player_name"], df["position"], df["team"]):
                    player_res = {
                        "player_id": pid,
                        "player_name": p_name,
                        "position": pos,
                        "team": team,
                    }
                    has_any_stat = False
                    for stat in stats:
                        season_totals = res.player_season_totals.get(pid, {}).get(stat)
                        if season_totals and season_totals["mean"] > 1.0:  # filters out bench players
                            player_res[stat] = {
                                "mean": season_totals["mean"],
                                "p10": season_totals["p10"],
                                "p50": season_totals["p50"],
                                "p90": season_totals["p90"],
                            }
                            has_any_stat = True

                    if has_any_stat:
                        results.append(player_res)

                return results

            except Exception as exc:
                logger.error("get_season_projections Simulator error: %s", exc)
                return []

    # ------------------------------------------------------------------
    # Scenario / what-if re-projection (for /scenario)
    # ------------------------------------------------------------------

    def run_scenario(
        self,
        player_id: str,
        week: int,
        season: int,
        stat: str,
        overrides: dict,
    ) -> Optional[ProjectionResult]:
        """
        Re-project with feature overrides for the /scenario endpoint.

        Overrides are applied as multiplicative adjustments on top of
        the Kalman estimate, then passed through Monte Carlo simulation.
        """
        with start_span(
            "projection.run_scenario",
            attributes={
                "player.id": player_id,
                "projection.week": week,
                "projection.season": season,
                "projection.stat": stat,
                "scenario.override_count": len(overrides),
            },
            tracer_name="backend.app.services.projection",
        ):
            kalman_est, kalman_var = self._load_kalman(player_id, week, season, stat)

            base_row = self._load_projection_row(player_id, week, season, stat)
            base_proj = _f((base_row or {}).get("projection", kalman_est or 0.0))

            adjusted = base_proj

            # Apply overrides
            snap = overrides.get("snap_share")
            if snap is not None:
                adjusted *= float(snap) / 0.65  # 65% baseline snap share

            wind = overrides.get("wind_speed_mph")
            if wind is not None and stat in ("receiving_yards", "passing_yards"):
                w = float(wind)
                if w > 20:
                    adjusted *= 0.90
                elif w > 10:
                    adjusted *= 0.96

            temp = overrides.get("temperature_f")
            if temp is not None and stat in ("receiving_yards", "passing_yards"):
                t = float(temp)
                if t < 32:
                    adjusted *= 0.93
                elif t < 50:
                    adjusted *= 0.97

            defender_grade = overrides.get("primary_defender_grade")
            if defender_grade is not None:
                grade = float(defender_grade)
                delta = grade - 70.0
                if stat in ("receiving_yards", "receiving_tds", "targets", "receptions"):
                    adjusted *= np.clip(1.0 - (delta * 0.0025), 0.88, 1.12)
                elif stat in ("passing_yards", "passing_tds", "completions"):
                    adjusted *= np.clip(1.0 - (delta * 0.0018), 0.90, 1.10)
                elif stat in ("fantasy_ppr", "ppr_points"):
                    adjusted *= np.clip(1.0 - (delta * 0.0015), 0.92, 1.08)

            if overrides.get("dome_override"):
                adjusted = base_proj  # dome removes all weather penalties

            # Monte Carlo with adjusted mean
            from ml.monte_carlo import MonteCarloProjector

            rng = np.random.default_rng(42)
            std = max((kalman_var or 100.0) ** 0.5, 1.0)
            samples = rng.normal(adjusted, std, 2000)

            mc = MonteCarloProjector()
            _, position, team = self._player_meta(player_id)
            with start_span(
                "projection.scenario_monte_carlo",
                attributes={
                    "projection.stat": stat,
                    "player.position": position or "WR",
                    "mc.sample_count": len(samples),
                },
                tracer_name="backend.app.services.projection",
            ):
                pr = mc.project(samples, stat=stat, position=position or "WR")

            player_name, position, team = self._player_meta(player_id)

            return ProjectionResult(
                player_id=player_id,
                player_name=player_name or player_id,
                position=position or "",
                team=team,
                week=week,
                season=season,
                stat=stat,
                projection=pr.projection,
                floor=pr.floor,
                ceiling=pr.ceiling,
                interval_method="scenario_monte_carlo",
                pipeline_run_id=None,
                boom_probability=pr.boom_probability,
                bust_probability=pr.bust_probability,
                fantasy_projection=pr.fantasy_projection,
                fantasy_floor=pr.fantasy_floor,
                fantasy_ceiling=pr.fantasy_ceiling,
                kalman_ability_estimate=kalman_est,
                kalman_uncertainty=_opt(std),
                confidence_score=_confidence(kalman_var),
                prop_comparison=None,
                data_freshness=self._data_freshness(player_id, week, season),
                model_version=self._model_version,
            )

    # ------------------------------------------------------------------
    # Feature dict for SHAP (used by /explain)
    # ------------------------------------------------------------------

    def get_feature_dict(
        self, player_id: str, week: int, season: int
    ) -> dict:
        """Return the feature_matrix row as a dict for SHAP computation."""
        import psycopg2

        try:
            conn = psycopg2.connect(self._db_url)
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM feature_matrix WHERE player_id=%s AND week=%s AND season=%s LIMIT 1",
                [player_id, week, season],
            )
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
            conn.close()
            if row:
                return {k: v for k, v in zip(cols, row) if isinstance(v, (int, float))}
        except Exception as exc:
            logger.debug("get_feature_dict DB error: %s", exc)
        return {}

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _resolve_player(
        self, player_name: str
    ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Fuzzy-match player name → (player_id, full_name, position, team)."""
        import psycopg2

        try:
            conn = psycopg2.connect(self._db_url)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, full_name, position, team
                FROM   players
                WHERE  LOWER(full_name) = LOWER(%s)
                   OR  LOWER(full_name) LIKE LOWER(%s)
                ORDER BY full_name
                LIMIT  1
                """,
                [player_name, f"%{player_name}%"],
            )
            row = cur.fetchone()
            conn.close()
            if row:
                return row[0], row[1], row[2], row[3]
        except Exception as exc:
            logger.error("_resolve_player error: %s", exc)
        return None, None, None, None

    def _player_meta(
        self, player_id: str
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Return (full_name, position, team) for a player_id."""
        import psycopg2

        try:
            conn = psycopg2.connect(self._db_url)
            cur = conn.cursor()
            cur.execute(
                "SELECT full_name, position, team FROM players WHERE id = %s",
                [player_id],
            )
            row = cur.fetchone()
            conn.close()
            if row:
                return row[0], row[1], row[2]
        except Exception as exc:
            logger.debug("_player_meta error: %s", exc)
        return None, None, None

    def _load_projection_row(
        self, player_id: str, week: int, season: int, stat: str
    ) -> Optional[dict]:
        """Load the most recent projection row for (player_id, week, season, stat)."""
        import psycopg2

        try:
            conn = psycopg2.connect(self._db_url)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT projection, floor, ceiling,
                       boom_probability, bust_probability,
                       fantasy_projection, fantasy_floor, fantasy_ceiling,
                       posterior_samples, pipeline_run_id, position, created_at
                FROM   projections
                WHERE  player_id = %s AND week = %s AND season = %s AND stat = %s
                ORDER BY created_at DESC
                LIMIT  1
                """,
                [player_id, week, season, stat],
            )
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
            conn.close()
            return dict(zip(cols, row)) if row else None
        except Exception as exc:
            logger.debug("_load_projection_row error: %s", exc)
            return None

    def _load_kalman(
        self, player_id: str, week: int, season: int, stat: str
    ) -> tuple[Optional[float], Optional[float]]:
        """Return (kalman_est_{stat}, kalman_variance_{stat}) from feature_matrix."""
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
            if row:
                return _opt(row[0]), _opt(row[1])
        except Exception as exc:
            logger.debug("_load_kalman error: %s", exc)
        return None, None

    def _run_pipeline(
        self,
        player_id: str,
        position: str,
        week: int,
        season: int,
        stat: str,
    ) -> Optional[dict]:
        """On-the-fly pipeline fallback (fast mode, no MCMC)."""
        try:
            with start_span(
                "projection.pipeline_fallback",
                attributes={
                    "player.id": player_id,
                    "player.position": position,
                    "projection.week": week,
                    "projection.season": season,
                    "projection.stat": stat,
                },
                tracer_name="backend.app.services.projection",
            ):
                from ml.train import PipelineRunner

                runner = PipelineRunner(fast=True)
                df = runner.run(season, week, [position], [stat])
                if df.empty:
                    return None
                mask = df["player_id"] == player_id
                row = df[mask].iloc[0] if mask.any() else df.iloc[0]
                r = row.to_dict()
                return {
                    "projection":         r.get("projection", 0.0),
                    "floor":              r.get("floor", 0.0),
                    "ceiling":            r.get("ceiling", 0.0),
                    "boom_probability":   r.get("boom_probability"),
                    "bust_probability":   r.get("bust_probability"),
                    "fantasy_projection": r.get("fantasy_projection"),
                    "fantasy_floor":      r.get("fantasy_floor"),
                    "fantasy_ceiling":    r.get("fantasy_ceiling"),
                    "posterior_samples":  r.get("posterior_samples"),
                    "degraded":           bool(r.get("degraded", False)),
                    "pipeline_run_id":    r.get("pipeline_run_id"),
                    "position":           position,
                }
        except Exception as exc:
            logger.error("Pipeline fallback error: %s", exc)
            return None

    def _data_freshness(
        self, player_id: str, week: int, season: int
    ) -> datetime:
        """Return computed_at of the feature_matrix row, or now()."""
        import psycopg2

        try:
            conn = psycopg2.connect(self._db_url)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT computed_at FROM feature_matrix
                WHERE  player_id = %s AND week = %s AND season = %s
                LIMIT  1
                """,
                [player_id, week, season],
            )
            row = cur.fetchone()
            conn.close()
            if row and row[0]:
                ts = row[0]
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return ts
        except Exception:
            pass
        return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(v, default: float = 0.0) -> float:
    """Safe float conversion with NaN guard."""
    if v is None:
        return default
    try:
        f = float(v)
        return default if f != f else f
    except (TypeError, ValueError):
        return default


def _opt(v) -> Optional[float]:
    """Safe optional float conversion."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def _interval_method(row: dict) -> str:
    """Only label intervals as percentiles when backed by real posterior draws."""
    samples = row.get("posterior_samples")
    if isinstance(samples, str):
        # psycopg2 normally decodes JSONB, but a JSON string is still usable.
        try:
            import json
            samples = json.loads(samples)
        except (TypeError, ValueError):
            samples = None
    return "posterior_samples" if isinstance(samples, list) and samples else "unavailable"


def _interval_value(row: dict, field: str) -> Optional[float]:
    """Never expose legacy residual bands as p10/p90 values."""
    return _opt(row.get(field)) if _interval_method(row) == "posterior_samples" else None


def _confidence(kalman_variance: Optional[float]) -> Optional[float]:
    """Convert Kalman variance → [0,1] confidence. variance=100 → 0.61."""
    if kalman_variance is None:
        return None
    return round(float(np.exp(-kalman_variance / 200.0)), 3)
