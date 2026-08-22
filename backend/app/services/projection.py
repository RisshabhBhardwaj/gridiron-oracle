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

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import HTTPException

from backend.app.core.tracing import start_span
from ml.served_learner import served_learner as _served_learner

_CONFORMAL_METHODS = frozenset({
    "posterior_samples",
    "causal_oof_conformal_90",
    "conformal",
    "mapie_enbpi",
})

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

SEASON_PROJECTIONS_UNAVAILABLE = (
    "Season / rest-of-season projections failed closed. The uninformative "
    "Kalman prior is never used for this surface."
)

# Ride-along honesty fix (Phase 1, Component F1): the live pipeline fallback
# degrades to a bare kalman_est_{stat} passthrough whenever MLflow is unset
# (ml/train.py), and reports degraded=True while still returning a number.
# Serving that number as if it were a real forecast is worse than serving
# nothing. Disabled by default; only flip on for local/dev debugging.
_PIPELINE_FALLBACK_ENABLED = False


class NoForecastAvailable(Exception):
    """Player resolved but no approved projection row exists for this request.

    Distinct from "player not found" (get_projection returns None for that).
    The API layer maps this to a structured forecast_available=False
    response rather than a degraded number.
    """


def load_approved_pipeline_run_ids(manifest_path: Optional[Path] = None) -> frozenset[str]:
    """Fail closed unless the baseline manifest pins at least one materialization run."""
    if manifest_path is None:
        from backend.app.core.config import settings
        manifest_path = Path(settings.baseline_manifest_path)
    if not manifest_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Baseline manifest missing: {manifest_path}",
        )
    payload = json.loads(manifest_path.read_text())
    raw = (payload.get("projection_policy") or {}).get("approved_pipeline_run_ids") or []
    ids = frozenset(str(item) for item in raw if item)
    if not ids:
        raise HTTPException(
            status_code=503,
            detail=(
                "No approved_pipeline_run_ids pinned in the baseline manifest; "
                "weekly serving is fail-closed."
            ),
        )
    return ids


# ---------------------------------------------------------------------------
# Depth-chart freshness gate (Phase 1, Component E)
# ---------------------------------------------------------------------------
#
# A stale or partially-refreshed depth-chart snapshot doesn't raise — it
# silently under-counts rosters, which is a quieter and harder-to-spot
# failure than serving a retired player. This gate exists to fail loud
# instead, before any depth-chart-derived roster filter is trusted.

# Known-rostered players whose absence from a season's depth-chart snapshot
# signals staleness. Extend as roster reality shifts.
_SPOT_CHECK_PLAYERS: dict[int, tuple[str, ...]] = {
    2026: ("00-0036893",),  # Najee Harris
}


@dataclass
class FreshnessResult:
    ok: bool
    reason: str
    team_counts: dict[str, int]


def check_depth_chart_freshness(conn, season: int) -> FreshnessResult:
    """
    Sanity-check the depth-chart snapshot for `season` before it is trusted
    as a roster-truth source.

    Range is [30, 100] ACT-linked players per team, not the in-season 53+8
    (IR) number: preseason rosters run up to ~90 before final cuts, and this
    gate must not itself fail loud against a legitimately large preseason
    snapshot. The floor of 30 and requiring >=28/32 teams present are what
    actually catch a broken or mid-refresh ingest.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT dc.team, count(DISTINCT dc.player_id)
        FROM   depth_charts dc
        JOIN   players p ON p.id = dc.player_id
        WHERE  dc.season = %s AND p.status = 'ACT'
        GROUP BY dc.team
        """,
        (season,),
    )
    team_counts = dict(cur.fetchall())

    if len(team_counts) < 28:
        return FreshnessResult(
            ok=False,
            reason=f"only {len(team_counts)}/32 teams have depth-chart rows for season {season}",
            team_counts=team_counts,
        )

    bad_teams = {team: n for team, n in team_counts.items() if n < 30 or n > 100}
    if bad_teams:
        return FreshnessResult(
            ok=False,
            reason=f"team ACT counts outside [30, 100]: {bad_teams}",
            team_counts=team_counts,
        )

    spot_check = _SPOT_CHECK_PLAYERS.get(season, ())
    if spot_check:
        cur.execute(
            "SELECT DISTINCT player_id FROM depth_charts WHERE season = %s AND player_id = ANY(%s)",
            (season, list(spot_check)),
        )
        present = {row[0] for row in cur.fetchall()}
        missing = set(spot_check) - present
        if missing:
            return FreshnessResult(
                ok=False,
                reason=f"spot-check players missing from season {season} depth chart: {sorted(missing)}",
                team_counts=team_counts,
            )

    return FreshnessResult(ok=True, reason="ok", team_counts=team_counts)


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
    served_learner:          str = "unknown"


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
            if row is None and _PIPELINE_FALLBACK_ENABLED:
                logger.info(
                    "No pre-computed projection for %s wk%d — running pipeline",
                    player_id, week,
                )
                row = self._run_pipeline(player_id, position or "WR", week, season, stat)
            if row is None:
                raise NoForecastAvailable(
                    f"No approved projection for player_id={player_id} "
                    f"week={week} season={season} stat={stat}"
                )

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
                served_learner=_served_learner(stat, position or row.get("position") or ""),
                boom_probability=_opt(row.get("boom_probability")),
                bust_probability=_opt(row.get("bust_probability")),
                fantasy_projection=_opt(row.get("fantasy_projection")),
                fantasy_floor=_interval_value(row, "fantasy_floor"),
                fantasy_ceiling=_interval_value(row, "fantasy_ceiling"),
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
        approved = list(load_approved_pipeline_run_ids())

        try:
            conn = psycopg2.connect(self._db_url)
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT p.player_id, p.season, p.week, p.stat,
                       p.position, p.projection, p.floor, p.ceiling,
                       p.boom_probability, p.bust_probability,
                       p.fantasy_projection, p.fantasy_floor, p.fantasy_ceiling,
                       p.posterior_samples, p.pipeline_run_id, p.interval_method,
                       pl.full_name, pl.team
                FROM   projections p
                LEFT JOIN players pl ON pl.id = p.player_id
                WHERE  p.week     = %s
                  AND  p.season   = %s
                  AND  p.stat     = %s
                  AND  p.position IN ({placeholders})
                  AND  p.pipeline_run_id = ANY(%s)
                ORDER BY p.projection DESC NULLS LAST
                """,
                [week, season, stat] + pos_filter + [approved],
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
                served_learner=_served_learner(stat, r.get("position") or ""),
                boom_probability=_opt(r.get("boom_probability")),
                bust_probability=_opt(r.get("bust_probability")),
                fantasy_projection=_opt(r.get("fantasy_projection")),
                fantasy_floor=_interval_value(r, "fantasy_floor"),
                fantasy_ceiling=_interval_value(r, "fantasy_ceiling"),
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
        Rest-of-season totals. Prefers a materialized SeasonSimulator run
        (scripts/materialize_season_simulation.py — the real Phase 4 model +
        autoregressive Kalman/Elo/Pickens-coupled Monte Carlo, too slow to run
        synchronously here) for an approved (season, start_week) pipeline run;
        falls back to the flat weekly-stack-rate x SP2-availability path
        otherwise.
        """
        from ml.playing_time import (
            assert_cold_start_qb_not_in_top24,
            attach_playing_time,
            rank_rest_of_season,
            simulate_season_paths,
        )

        if start_week < 1 or start_week > 18:
            raise HTTPException(status_code=400, detail="start_week must be in 1..18")
        remaining = 19 - int(start_week)
        skill = [p.upper() for p in (positions or ["QB", "RB", "WR", "TE"])]
        try:
            approved = load_approved_pipeline_run_ids()
            self._require_depth_chart_fresh(season)
            simulated = self._load_season_simulation_rows(season, start_week, skill, stats, approved)
            if simulated:
                return self._rank_with_cold_start_guard(simulated)
            feature_rows = self._load_season_feature_rows(season, start_week, skill)
            weekly_rates = self._load_weekly_rates(season, start_week, approved)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"{SEASON_PROJECTIONS_UNAVAILABLE} ({exc})",
            ) from exc

        attached = attach_playing_time(feature_rows)
        # Hand-set Gaussian residual scales, not fit from held-out error — tuned to look
        # plausible per stat, not derived from any per-player or per-position residual model.
        residual = {
            "fantasy_ppr": 6.0,
            "passing_yards": 55.0,
            "rushing_yards": 22.0,
            "receiving_yards": 28.0,
        }
        out: list[dict] = []
        for row in attached:
            player_id = str(row["player_id"])
            item: dict = {
                "player_id": player_id,
                "player_name": row.get("player_name") or player_id,
                "position": row.get("position") or "",
                "team": row.get("team"),
                "prior_games": row.get("prior_games"),
                "prior_active_games": row.get("prior_active_games"),
                "depth_rank": row.get("depth_rank"),
                "p_active": row.get("p_active"),
                "degraded": False,
                # simulate_season_paths draws Bernoulli(p_active) x N(rate, residual_scale) —
                # a plain Monte Carlo interval, not EnbPI (conformal) or any bootstrap method.
                "interval_method": "playing_time_gaussian_mc",
            }
            for stat in stats:
                rate = weekly_rates.get((player_id, stat))
                if rate is None:
                    rate = _opt(row.get(f"seas_avg_{stat}")) or 0.0
                paths = simulate_season_paths(
                    float(rate),
                    float(row.get("p_active") or 0.02),
                    remaining,
                    residual_scale=residual.get(stat, 8.0),
                    n_sims=300,
                    rng=0,
                )
                item[stat] = {
                    "mean": paths["mean"],
                    "p10": paths["p10"],
                    "p50": paths["p50"],
                    "p90": paths["p90"],
                }
            item["mean"] = float((item.get("fantasy_ppr") or {}).get("mean") or 0.0)
            out.append(item)
        return self._rank_with_cold_start_guard(out)

    @staticmethod
    def _rank_with_cold_start_guard(rows: list[dict]) -> list[dict]:
        """
        Shared by both the flat-rate path and the materialized-simulation
        path: rank by mean, then enforce SP2's cold-start guard (a QB with no
        prior snaps cannot be a rest-of-season top-24 star — see
        ml.playing_time.assert_cold_start_qb_not_in_top24's docstring and
        the memory note on the old top-24 metric scoring a 9-11 QB roster as
        optimal). On violation, zero and flag the offending rows and re-rank
        rather than 500ing — a cold-start QB is a data-quality signal to
        surface via `degraded`, not a reason to fail the whole board.
        """
        from ml.playing_time import assert_cold_start_qb_not_in_top24, rank_rest_of_season

        ranked = rank_rest_of_season(rows, value_key="mean")
        try:
            assert_cold_start_qb_not_in_top24(ranked)
        except AssertionError:
            for row in ranked:
                if (
                    str(row.get("position") or "").upper() == "QB"
                    and float(row.get("prior_games") or 0) <= 0
                    and float(row.get("prior_active_games") or 0) <= 0
                ):
                    row["degraded"] = True
                    row["mean"] = 0.0
                    if "fantasy_ppr" in row and isinstance(row["fantasy_ppr"], dict):
                        row["fantasy_ppr"] = {k: 0.0 for k in ("mean", "p10", "p50", "p90")}
            ranked = rank_rest_of_season(ranked, value_key="mean")
            assert_cold_start_qb_not_in_top24(ranked)
        return ranked

    def _load_season_simulation_rows(
        self,
        season: int,
        start_week: int,
        positions: list[str],
        stats: list[str],
        approved: frozenset[str],
    ) -> list[dict]:
        """
        Read a materialized SeasonSimulator run (season_simulations table) if
        one exists under an approved pipeline_run_id for (season, start_week).
        Returns [] on no match — the caller falls back to the flat-rate path.
        Pivots the table's one-row-per-(player, stat) shape into the same
        {player_id, ..., <stat>: {mean,p10,p50,p90}, ...} item shape
        get_season_projections's flat-rate path already produces, so
        _rank_with_cold_start_guard and the response model don't need to
        know which path served a given row.
        """
        import psycopg2
        import psycopg2.extras

        conn = psycopg2.connect(self._db_url)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT player_id, player_name, position, team, stat,
                           mean, p10, p50, p90, p_active, prior_games,
                           prior_active_games, depth_rank, degraded, interval_method
                    FROM season_simulations
                    WHERE season = %s AND start_week = %s
                      AND pipeline_run_id = ANY(%s)
                      AND UPPER(COALESCE(position, '')) = ANY(%s)
                    """,
                    (season, start_week, list(approved), positions),
                )
                rows = [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

        by_player: dict[str, dict] = {}
        for r in rows:
            if r["stat"] not in stats:
                continue
            pid = str(r["player_id"])
            item = by_player.setdefault(pid, {
                "player_id": pid,
                "player_name": r.get("player_name") or pid,
                "position": r.get("position") or "",
                "team": r.get("team"),
                "prior_games": r.get("prior_games"),
                "prior_active_games": r.get("prior_active_games"),
                "depth_rank": r.get("depth_rank"),
                "p_active": r.get("p_active"),
                "degraded": bool(r.get("degraded", False)),
                "interval_method": r.get("interval_method") or "season_simulator_mc",
            })
            item[r["stat"]] = {
                "mean": r.get("mean"), "p10": r.get("p10"),
                "p50": r.get("p50"), "p90": r.get("p90"),
            }

        out = list(by_player.values())
        for item in out:
            item["mean"] = float((item.get("fantasy_ppr") or {}).get("mean") or 0.0)
        return out

    def _load_season_feature_rows(
        self, season: int, start_week: int, positions: list[str]
    ) -> list[dict]:
        """
        Roster membership, team, and depth_rank come from depth_charts — the
        same source the weekly serving path already trusts — not from the
        stale `players` table or a feature_matrix row that may be years old.
        `players.status != 'RET'` alone let 7,684 of 7,809 players through
        regardless of whether they're on any real roster (Tom Brady:
        status='ACT', last real season 2022, served 21st on the season
        board at 302.5 PPR before this fix). depth_charts is the roster
        gate; feature_matrix is only consulted afterward, per surviving
        player, for the Kalman/rate priors (prior_snap_share,
        seas_games_played, seas_avg_*) that describe how they've performed.

        As-of rule mirrors feature_engineer._fill_lagged_depth_chart_rank's
        lagged join, except inclusive of the target week itself: a week-1
        preseason snapshot is legitimate roster truth for SERVING week 1
        (there is no same-week leak risk here — that concern is specific to
        TRAINING rows, where a same-week join could leak the outcome back
        into a historical feature). So the depth-chart row used is the most
        recent one at or before (season, start_week) — falling back to the
        prior season's chart only when `season` has no depth_charts rows
        published AT ALL yet, decided once for the whole query, never
        per-player. A per-player fallback ("if this player specifically has
        no row this season, use their last one from any prior season") is
        exactly what let Tom Brady back onto the 2026 board via his 2022 row
        — he genuinely has no 2026 depth-chart row (he's retired), and the
        2026 week-1 chart already exists (3,185 rows), so that absence must
        exclude him, not fall through to whatever season he last appeared in.
        """
        import psycopg2
        import psycopg2.extras

        conn = psycopg2.connect(self._db_url)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (dc.player_id)
                           dc.player_id,
                           COALESCE(pl.full_name, dc.player_id) AS player_name,
                           COALESCE(dc.position, pl.position, '') AS position,
                           dc.team,
                           fm.prior_snap_share,
                           fm.seas_games_played AS prior_games,
                           -- seas_games_played only increments on a week the player
                           -- actually appears in game_logs (verified: a bye week or a
                           -- missed game leaves no row, so it never falls behind by
                           -- counting a "gap" that wasn't really an opportunity) — it IS
                           -- the active-games numerator, so also serves as
                           -- prior_active_games directly.
                           fm.seas_games_played AS prior_active_games,
                           -- Real denominator for p_active_from_priors: how many games
                           -- has this player's CURRENT (depth-chart) team actually played
                           -- (completed, home_score IS NOT NULL), scoped to the TARGET
                           -- (season, start_week) window being served — not fm.season/
                           -- fm.week, which may be a stale prior-season row when a player
                           -- has no current-season feature_matrix history yet (a rookie,
                           -- or an early-season snapshot). seas_games_played resets every
                           -- season, so the denominator must too, or an early-season
                           -- snapshot for a team that played many seasons ago would pull
                           -- in years of unrelated games. Using seas_games_played for both
                           -- numerator and denominator (as an earlier fix attempt did)
                           -- makes them mathematically always equal, which collapses the
                           -- active rate to a function of sample size alone rather than
                           -- actual availability.
                           COALESCE(
                               NULLIF((
                                   SELECT COUNT(*) FROM games g
                                   WHERE (g.home_team = dc.team OR g.away_team = dc.team)
                                     AND g.home_score IS NOT NULL
                                     AND g.season = %(season)s AND g.week < %(start_week)s
                               ), 0),
                               (
                                   SELECT COUNT(*) FROM games g
                                   WHERE (g.home_team = dc.team OR g.away_team = dc.team)
                                     AND g.home_score IS NOT NULL
                                     AND g.season = %(season)s - 1
                               )
                           ) AS team_games_played,
                           fm.seas_avg_fantasy_ppr,
                           fm.seas_avg_passing_yards,
                           fm.seas_avg_rushing_yards,
                           fm.seas_avg_receiving_yards,
                           dc.depth_rank AS depth_rank,
                           fm.season,
                           fm.week
                    FROM depth_charts dc
                    JOIN players pl ON pl.id = dc.player_id
                    LEFT JOIN LATERAL (
                        SELECT fm2.prior_snap_share, fm2.seas_games_played,
                               fm2.seas_avg_fantasy_ppr, fm2.seas_avg_passing_yards,
                               fm2.seas_avg_rushing_yards, fm2.seas_avg_receiving_yards,
                               fm2.season, fm2.week
                        FROM feature_matrix fm2
                        WHERE fm2.player_id = dc.player_id
                          AND (
                                (fm2.season = %(season)s AND fm2.week < %(start_week)s)
                             OR (fm2.season < %(season)s)
                              )
                        ORDER BY fm2.season DESC, fm2.week DESC
                        LIMIT 1
                    ) fm ON true
                    WHERE UPPER(COALESCE(dc.position, pl.position, '')) = ANY(%(positions)s)
                      AND COALESCE(pl.status, 'ACT') != 'RET'
                      AND (
                            CASE WHEN EXISTS (
                                SELECT 1 FROM depth_charts WHERE season = %(season)s
                            )
                            THEN (dc.season = %(season)s AND dc.week <= %(start_week)s)
                            ELSE dc.season = %(season)s - 1
                            END
                          )
                    ORDER BY dc.player_id, dc.season DESC, dc.week DESC
                    """,
                    {"season": season, "start_week": start_week, "positions": positions},
                )
                rows = [dict(item) for item in cur.fetchall()]
        finally:
            conn.close()
        return rows

    def _require_depth_chart_fresh(self, season: int) -> None:
        """
        Blocking staleness check (promoted from the prior log-only
        _warn_if_depth_chart_stale, Phase 1 Component E). _load_season_feature_rows
        now gates roster membership itself on depth_charts, for both the
        materialized-simulation and flat-rate paths — a stale or
        under-populated depth chart no longer just risks quietly
        under-counting a roster, it is the thing standing between the board
        and serving whoever's feature_matrix row happens to survive the
        filter. Fail loud instead.
        """
        import psycopg2

        try:
            conn = psycopg2.connect(self._db_url)
            try:
                result = check_depth_chart_freshness(conn, season)
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("depth-chart freshness check failed to run: %s", exc)
            return
        if not result.ok:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"{SEASON_PROJECTIONS_UNAVAILABLE} Depth chart freshness gate "
                    f"failed for season={season}: {result.reason}"
                ),
            )

    def _load_weekly_rates(
        self, season: int, start_week: int, approved: frozenset[str]
    ) -> dict[tuple[str, str], float]:
        import psycopg2
        import psycopg2.extras

        conn = psycopg2.connect(self._db_url)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (player_id, stat)
                           player_id, stat, projection
                    FROM projections
                    WHERE season = %s
                      AND week = %s
                      AND pipeline_run_id = ANY(%s)
                      AND projection IS NOT NULL
                    ORDER BY player_id, stat, created_at DESC
                    """,
                    (season, start_week, list(approved)),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return {
            (str(row["player_id"]), str(row["stat"])): float(row["projection"])
            for row in rows
            if row.get("projection") is not None
        }

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
        """Load the most recent approved projection row for (player_id, week, season, stat)."""
        import psycopg2

        approved = list(load_approved_pipeline_run_ids())
        try:
            conn = psycopg2.connect(self._db_url)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT projection, floor, ceiling,
                       boom_probability, bust_probability,
                       fantasy_projection, fantasy_floor, fantasy_ceiling,
                       posterior_samples, pipeline_run_id, position, created_at,
                       interval_method
                FROM   projections
                WHERE  player_id = %s AND week = %s AND season = %s AND stat = %s
                  AND  pipeline_run_id = ANY(%s)
                ORDER BY created_at DESC
                LIMIT  1
                """,
                [player_id, week, season, stat, approved],
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
    """Honor a stored conformal method; otherwise require real posterior draws."""
    stored = row.get("interval_method")
    if isinstance(stored, str) and stored.strip() and stored.strip() != "unavailable":
        return stored.strip()
    samples = row.get("posterior_samples")
    if isinstance(samples, str):
        try:
            samples = json.loads(samples)
        except (TypeError, ValueError):
            samples = None
    return "posterior_samples" if isinstance(samples, list) and samples else "unavailable"


def _interval_value(row: dict, field: str) -> Optional[float]:
    """Expose floor/ceiling only when a recognized interval method backs them."""
    method = _interval_method(row)
    if method not in _CONFORMAL_METHODS:
        return None
    value = _opt(row.get(field))
    if value is None:
        return None
    if field in {"floor", "fantasy_floor"}:
        position = str(row.get("position") or "").upper()
        if position and position != "QB":
            value = max(0.0, value)
    return value


def _confidence(kalman_variance: Optional[float]) -> Optional[float]:
    """Convert Kalman variance → [0,1] confidence. variance=100 → 0.61."""
    if kalman_variance is None:
        return None
    return round(float(np.exp(-kalman_variance / 200.0)), 3)
