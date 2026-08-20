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
from ml.vor import attach_vor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["draft"])

_K_DST_NOTE = (
    "Ranks use VOR for an 8-team PPR league (1QB / 2RB / 2WR / 1TE / 1FLEX). "
    "K and DST are out of scope. Rookies are ranked from a draft-capital curve fitted on prior seasons; a zero-history player without that basis stays unranked. "
    "Projections are a pre-draft fixed-universe historical PPR prior; "
    "target-season OOF and actuals are refused. "
    "model_rank is the independent board (no market input); blended_rank averages it "
    "with ADP. On the roster-aware walk-forward harness (reports/draft_walkforward.json, "
    "reports/draft_board_experiments.json) the independent board beat ADP in 3 of 6 "
    "seasons and the blend in 5 of 6, so blended_rank is the recommended sort. The "
    "blend consumes market information and its edge over ADP is therefore a weaker "
    "claim than an independent board's would be."
)


class DraftBoardPlayer(BaseModel):
    player_name: str
    position: Optional[str] = None
    team: Optional[str] = None
    adp: float
    player_id: Optional[str] = None
    source: str  # ADP source
    model_rank: Optional[int] = None
    model_fantasy_ppr: Optional[float] = None
    projection_basis: Optional[str] = None
    adp_rank: Optional[int] = None
    value_vs_adp: Optional[float] = None
    blended_rank: Optional[int] = None


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
    recommended_rank_field: str = "blended_rank"
    note: str = Field(default=_K_DST_NOTE)


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


ROOKIE_CURVE_BASIS = "rookie_draft_capital_vacated_opportunity"


def model_ranks_by_vor(
    adp_rows: list[dict],
    model_by_id: dict[str, dict],
) -> dict[str, int]:
    """Rank ADP-matched players by 8-team VOR.

    A player with no prior games is ranked only when his projection carries the
    fitted rookie basis. The blanket exclusion dates from when every rookie got
    one of three constants, so ranking them was noise; the curve in
    ml/rookie_priors.py is fitted from draft capital and ranks rookies better
    than the market does (Spearman 0.53 vs ADP's 0.42 over 127 rookie seasons,
    reports/draft_component_eval.json). Continuing to discard them would throw
    away the one component measured to beat consensus. A zero-history player
    with no such basis still stays out -- that projection is backed by nothing.
    """
    eligible: list[dict] = []
    for row in adp_rows:
        player_id = str(row["player_id"])
        projection = model_by_id.get(player_id)
        if projection is None:
            continue
        historical = projection.get("historical_games")
        if historical is not None and int(historical) == 0:
            if projection.get("projection_basis") != ROOKIE_CURVE_BASIS:
                continue
        eligible.append(
            {
                "player_id": player_id,
                "position": projection.get("position") or row.get("position"),
                "projection": float(projection["fantasy_ppr"]),
            }
        )
    ranked = attach_vor(eligible)
    ranked.sort(key=lambda item: (-(item["vor"] if item.get("vor") is not None else -1e18), item["player_id"]))
    return {str(item["player_id"]): index for index, item in enumerate(ranked, start=1)}


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
                   source AS projection_source, as_of,
                   COALESCE(historical_games, 0) AS historical_games,
                   projection_basis
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


def blended_ranks(players: list["DraftBoardPlayer"]) -> dict[str, int]:
    """Rank the 50/50 average of model and ADP rank, for players holding both.

    Rank-averaging two boards is a variance-reduction move, not a new
    projection: it is expected to beat either input, and it does so only
    because it consumes the market. Callers that need a market-independent
    ranking must use ``model_rank``.
    """
    scored = [
        (player.player_id, 0.5 * float(player.model_rank) + 0.5 * float(player.adp_rank))
        for player in players
        if player.player_id and player.model_rank is not None and player.adp_rank is not None
    ]
    scored.sort(key=lambda item: (item[1], item[0]))
    return {player_id: index for index, (player_id, _score) in enumerate(scored, start=1)}


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
        model_rank = model_ranks_by_vor(adp_rows, model_by_id)
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
                projection_basis=(model_by_id.get(player_id) or {}).get("projection_basis"),
            ))
        blended = blended_ranks(players)
        for player in players:
            player.blended_rank = blended.get(player.player_id)
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
