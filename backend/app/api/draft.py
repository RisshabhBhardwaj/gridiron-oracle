"""Draft board endpoints backed exclusively by causal preseason projections."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["draft"])


class DraftBoardPlayer(BaseModel):
    player_name: str
    position: Optional[str] = None
    team: Optional[str] = None
    adp: float
    player_id: Optional[str] = None
    source: str  # ADP source
    model_rank: Optional[int] = None
    model_fantasy_ppr: Optional[float] = None
    adp_rank: Optional[int] = None
    value_vs_adp: Optional[float] = None


class DraftBoardResponse(BaseModel):
    season: int
    source: str  # ADP source
    projection_source: str
    as_of: date
    scoring: str
    count: int
    players: list[DraftBoardPlayer]
    spearman_rho: Optional[float] = None
    model_source: Optional[str] = None  # compatibility alias for older clients
    note: str = Field(
        default=(
            "Ranks use a pre-draft fixed-universe projection: historical per-game PPR "
            "multiplied by a historical games-played prior. Target-season OOF and actuals are refused."
        )
    )


def _fetch_adp(conn: Any, season: int, source: Optional[str], scoring: str, position: Optional[str]) -> list[dict]:
    sql = """
        SELECT player_name, position, team, adp, player_id, source
        FROM fantasy_adp
        WHERE season = %s AND scoring = %s AND player_id IS NOT NULL
    """
    params: list[Any] = [season, scoring]
    if source:
        sql += " AND source = %s"
        params.append(source)
    if position:
        sql += " AND UPPER(position) = %s"
        params.append(position.upper())
    else:
        # The preseason projection contract has a fixed QB/RB/WR/TE universe.
        # Do not show K/DEF rows with a misleading blank model rank.
        sql += " AND UPPER(position) IN ('QB', 'RB', 'WR', 'TE')"
    sql += " ORDER BY source ASC, adp ASC, player_id ASC"
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def _unique_projection_map(rows: list[dict], *, source: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        player_id = str(row["player_id"])
        if player_id in out:
            raise ValueError(f"{source} returned duplicate preseason projections for player_id={player_id}")
        if row.get("fantasy_ppr") is None:
            continue
        out[player_id] = row
    return out


def _fetch_preseason_projection_ppr(
    conn: Any, season: int, as_of: Optional[date] = None
) -> tuple[dict[str, dict], Optional[str], Optional[date]]:
    """Read an explicitly as-of, materialized preseason projection run only."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            WITH latest AS (
                SELECT MAX(as_of) AS as_of
                FROM draft_preseason_projections
                WHERE season = %s AND as_of <= COALESCE(%s, CURRENT_DATE)
            )
            SELECT player_id, player_name, position, team, projection AS fantasy_ppr,
                   source AS projection_source, as_of
            FROM draft_preseason_projections dp
            JOIN latest USING (as_of)
            WHERE dp.season = %s
            """,
            (season, as_of, season),
        )
        rows = [dict(row) for row in cur.fetchall()]
    if not rows:
        return {}, None, None
    metadata = rows[0]
    return _unique_projection_map(rows, source="draft_preseason_projections"), metadata["projection_source"], metadata["as_of"]


def _fetch_db_projections_ppr(
    conn: Any, season: int
) -> tuple[dict[str, dict], Optional[str], Optional[date]]:
    """Compatibility fallback for materialized *week-0* records only.

    ``projection`` is the schema's point estimate.  Summing it makes this
    source use the same season-total scale as the dedicated preseason table.
    Regular-season weekly rows are deliberately excluded: they are not draft
    inputs, regardless of whether their values originated in an OOF artifact.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT pr.player_id, COALESCE(p.full_name, pr.player_id) AS player_name,
                   COALESCE(pr.position, p.position) AS position,
                   COALESCE(pr.team, p.team) AS team,
                   SUM(pr.projection) AS fantasy_ppr,
                   MAX(pr.created_at)::date AS as_of
            FROM projections pr
            LEFT JOIN players p ON p.id = pr.player_id
            WHERE pr.season = %s AND pr.week = 0 AND pr.stat = 'fantasy_ppr'
              AND pr.projection IS NOT NULL
            GROUP BY pr.player_id, p.full_name, pr.position, p.position, pr.team, p.team
            """,
            (season,),
        )
        rows = [dict(row) for row in cur.fetchall()]
    if not rows:
        return {}, None, None
    return _unique_projection_map(rows, source="projections"), "preseason_projection_rows", rows[0]["as_of"]


def _fetch_model_ppr(conn: Any, season: int, as_of: Optional[date]) -> tuple[dict[str, dict], str, date]:
    try:
        values, source, resolved_as_of = _fetch_preseason_projection_ppr(conn, season, as_of)
    except psycopg2.errors.UndefinedTable:
        # Supports installs awaiting the accompanying migration; still only
        # permits explicit week-0 records, never target-season weekly OOF.
        conn.rollback()
        values, source, resolved_as_of = {}, None, None
    if values and source and resolved_as_of:
        return values, source, resolved_as_of
    values, source, resolved_as_of = _fetch_db_projections_ppr(conn, season)
    if values and source and resolved_as_of:
        return values, source, resolved_as_of
    raise LookupError("No causal preseason projection run is available")


@router.get("/draft/board", response_model=DraftBoardResponse)
def draft_board(
    season: int = Query(..., ge=2010, le=2030),
    source: Optional[str] = Query(None, description="Optional fantasy_adp.source; defaults to an available source"),
    scoring: str = Query("ppr"),
    position: Optional[str] = Query(None, description="QB|RB|WR|TE"),
    as_of: Optional[date] = Query(None, description="Use a preseason run published on or before this ISO date"),
) -> DraftBoardResponse:
    try:
        conn = psycopg2.connect(settings.database_url)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}") from exc
    try:
        adp_rows = _fetch_adp(conn, season, source, scoring, position)
        if not adp_rows:
            raise HTTPException(status_code=404, detail=f"No resolved ADP rows for season={season} source={source or 'any'} scoring={scoring}")
        selected_source = str(adp_rows[0]["source"])
        # An automatic lookup must not merge sources (their ADP scales and
        # sampling methods differ).  Pick the first available source.
        adp_rows = [row for row in adp_rows if row["source"] == selected_source]
        try:
            model_by_id, projection_source, projection_as_of = _fetch_model_ppr(conn, season, as_of)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        matched = [(str(row["player_id"]), float(model_by_id[str(row["player_id"])]["fantasy_ppr"]))
                   for row in adp_rows if str(row["player_id"]) in model_by_id]
        matched.sort(key=lambda item: (-item[1], item[0]))
        model_rank = {player_id: index for index, (player_id, _) in enumerate(matched, start=1)}
        model_value = dict(matched)

        players: list[DraftBoardPlayer] = []
        adp_values: list[float] = []
        rank_values: list[float] = []
        for adp_rank, row in enumerate(adp_rows, start=1):
            player_id = str(row["player_id"])
            rank = model_rank.get(player_id)
            value = float(adp_rank - rank) if rank is not None else None
            if rank is not None:
                adp_values.append(float(adp_rank))
                rank_values.append(float(rank))
            players.append(DraftBoardPlayer(
                player_name=row["player_name"], position=row.get("position"), team=row.get("team"),
                adp=float(row["adp"]), player_id=player_id, source=row.get("source") or selected_source,
                model_rank=rank, model_fantasy_ppr=model_value.get(player_id), adp_rank=adp_rank,
                value_vs_adp=value,
            ))
        rho = None
        if len(adp_values) >= 5:
            from scipy.stats import spearmanr
            rho = float(spearmanr(adp_values, rank_values).correlation)
        return DraftBoardResponse(
            season=season, source=selected_source, scoring=scoring, count=len(players), players=players,
            spearman_rho=rho, projection_source=projection_source, model_source=projection_source,
            as_of=projection_as_of,
        )
    finally:
        conn.close()


@router.get("/adp", response_model=DraftBoardResponse)
def list_adp(**kwargs: Any) -> DraftBoardResponse:
    """Thin alias — retained for clients that used the original endpoint."""
    return draft_board(**kwargs)
