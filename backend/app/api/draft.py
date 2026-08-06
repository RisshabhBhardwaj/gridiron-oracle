"""
backend/app/api/draft.py

Draft board endpoints: ADP + optional season fantasy_ppr projections joined
for rank comparison.
"""

from __future__ import annotations

import logging
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
    source: str
    model_rank: Optional[int] = None
    model_fantasy_ppr: Optional[float] = None
    adp_rank: Optional[int] = None
    value_vs_adp: Optional[float] = None  # adp_rank - model_rank (>0 = undervalued by ADP)


class DraftBoardResponse(BaseModel):
    season: int
    source: str
    scoring: str
    count: int
    players: list[DraftBoardPlayer]
    spearman_rho: Optional[float] = None
    note: str = Field(
        default="ADP from fantasy_adp; model ranks from season fantasy_ppr when available."
    )


def _normalize_name(name: str) -> str:
    s = (name or "").lower().strip()
    for ch in (".", "'", "-", " jr", " sr", " iii", " ii", " iv"):
        s = s.replace(ch, "")
    return " ".join(s.split())


def _fetch_adp(conn, season: int, source: str, scoring: str, position: Optional[str]) -> list[dict]:
    sql = """
        SELECT player_name, position, team, adp, player_id, source
        FROM fantasy_adp
        WHERE season = %s AND source = %s AND scoring = %s
    """
    params: list[Any] = [season, source, scoring]
    if position:
        sql += " AND UPPER(position) = %s"
        params.append(position.upper())
    sql += " ORDER BY adp ASC"
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def _fetch_model_ppr(conn, season: int) -> dict[str, dict]:
    """Map normalized name → {player_id, fantasy_ppr, position, team}."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'projections'
            """
        )
        cols = {r["column_name"] for r in cur.fetchall()}
        mean_col = "mean" if "mean" in cols else ("projected_mean" if "projected_mean" in cols else None)
        if mean_col and "stat" in cols:
            cur.execute(
                f"""
                SELECT pr.player_id,
                       COALESCE(p.full_name, pr.player_id) AS player_name,
                       COALESCE(pr.position, p.position) AS position,
                       COALESCE(pr.team, p.team) AS team,
                       AVG(pr.{mean_col}) AS fantasy_ppr
                FROM projections pr
                LEFT JOIN players p ON pr.player_id = p.id
                WHERE pr.season = %s AND pr.stat = 'fantasy_ppr'
                GROUP BY pr.player_id, p.full_name, pr.position, p.position, pr.team, p.team
                """,
                (season,),
            )
            rows = [dict(r) for r in cur.fetchall()]
            if rows:
                return {_normalize_name(r["player_name"]): r for r in rows}

        # Fallback: season actuals (useful pre-projection / backtest seasons)
        cur.execute(
            """
            SELECT gl.player_id,
                   COALESCE(p.full_name, gl.player_id) AS player_name,
                   gl.position,
                   MAX(gl.team) AS team,
                   SUM(gl.fantasy_points_ppr) AS fantasy_ppr
            FROM game_logs gl
            LEFT JOIN players p ON gl.player_id = p.id
            WHERE gl.season = %s
            GROUP BY gl.player_id, p.full_name, gl.position
            HAVING SUM(gl.fantasy_points_ppr) IS NOT NULL
            """,
            (season,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        return {_normalize_name(r["player_name"]): r for r in rows}


@router.get("/draft/board", response_model=DraftBoardResponse)
def draft_board(
    season: int = Query(..., ge=2010, le=2030),
    source: str = Query("historical", description="fantasy_adp.source"),
    scoring: str = Query("ppr"),
    position: Optional[str] = Query(None, description="QB|RB|WR|TE"),
) -> DraftBoardResponse:
    try:
        conn = psycopg2.connect(settings.database_url)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}") from exc

    try:
        adp_rows = _fetch_adp(conn, season, source, scoring, position)
        if not adp_rows:
            raise HTTPException(
                status_code=404,
                detail=f"No ADP rows for season={season} source={source} scoring={scoring}",
            )
        model_by_name = _fetch_model_ppr(conn, season)

        # ADP ranks
        for i, row in enumerate(adp_rows, start=1):
            row["adp_rank"] = i

        # Model ranks among ADP pool that match
        matched = []
        for row in adp_rows:
            key = _normalize_name(row["player_name"])
            m = model_by_name.get(key)
            if m and m.get("fantasy_ppr") is not None:
                matched.append((key, float(m["fantasy_ppr"])))
        matched.sort(key=lambda x: -x[1])
        model_rank_map = {k: i + 1 for i, (k, _) in enumerate(matched)}
        model_val_map = {k: v for k, v in matched}

        players: list[DraftBoardPlayer] = []
        pairs_adp: list[float] = []
        pairs_model: list[float] = []
        for row in adp_rows:
            key = _normalize_name(row["player_name"])
            m_rank = model_rank_map.get(key)
            m_val = model_val_map.get(key)
            adp_rank = int(row["adp_rank"])
            value = (adp_rank - m_rank) if m_rank is not None else None
            if m_rank is not None:
                pairs_adp.append(float(adp_rank))
                pairs_model.append(float(m_rank))
            players.append(
                DraftBoardPlayer(
                    player_name=row["player_name"],
                    position=row.get("position"),
                    team=row.get("team"),
                    adp=float(row["adp"]),
                    player_id=row.get("player_id"),
                    source=row.get("source") or source,
                    model_rank=m_rank,
                    model_fantasy_ppr=m_val,
                    adp_rank=adp_rank,
                    value_vs_adp=float(value) if value is not None else None,
                )
            )

        spearman_rho = None
        if len(pairs_adp) >= 5:
            try:
                from scipy.stats import spearmanr

                spearman_rho = float(spearmanr(pairs_adp, pairs_model).correlation)
            except Exception:
                spearman_rho = None

        return DraftBoardResponse(
            season=season,
            source=source,
            scoring=scoring,
            count=len(players),
            players=players,
            spearman_rho=spearman_rho,
        )
    finally:
        conn.close()


@router.get("/adp", response_model=DraftBoardResponse)
def list_adp(
    season: int = Query(..., ge=2010, le=2030),
    source: str = Query("historical"),
    scoring: str = Query("ppr"),
    position: Optional[str] = Query(None),
) -> DraftBoardResponse:
    """Thin alias — same payload as /draft/board."""
    return draft_board(season=season, source=source, scoring=scoring, position=position)
