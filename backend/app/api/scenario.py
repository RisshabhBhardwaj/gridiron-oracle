"""
backend/app/api/scenario.py

POST /scenario — what-if re-projection with custom feature overrides.

CLAUDE.md Frontend Rules:
  "What-If sliders must debounce 300ms before calling /scenario."
  (debounce is implemented client-side; this endpoint is stateless)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from backend.app.services.projection import ProjectionService

router = APIRouter(prefix="", tags=["scenario"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ScenarioOverrides(BaseModel):
    """
    Feature overrides for the what-if studio.

    All fields are optional — omitted fields use the base-case values.
    """
    wind_speed_mph:          Optional[float] = Field(None, ge=0, le=60)
    temperature_f:           Optional[float] = Field(None, ge=-20, le=110)
    snap_share:              Optional[float] = Field(None, ge=0.0, le=1.0)
    primary_defender_grade:  Optional[float] = Field(None, ge=0.0, le=100.0)
    dome_override:           Optional[bool]  = None


class ScenarioRequest(BaseModel):
    player_id: str
    week:      int = Field(..., ge=1, le=22)
    season:    int = Field(..., ge=2019)
    stat:      str = "receiving_yards"
    overrides: ScenarioOverrides


class Percentiles(BaseModel):
    p10: float
    p50: float
    p90: float


class ScenarioResponse(BaseModel):
    player_id:          str
    player_name:        str
    stat:               str
    week:               int
    season:             int
    base_projection:    float
    scenario_projection: float
    delta:              float
    delta_pct:          float
    percentiles:        Percentiles
    boom_probability:   Optional[float] = None
    bust_probability:   Optional[float] = None
    fantasy_projection: Optional[float] = None
    data_freshness:     datetime


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/scenario", response_model=ScenarioResponse)
def scenario(req: ScenarioRequest) -> ScenarioResponse:
    """
    Re-project with custom feature overrides.

    Called by the What-If Studio sliders (debounced 300ms client-side).
    Returns both the base projection and the scenario projection so the UI
    can display the delta.
    """
    svc = ProjectionService(
        db_url=settings.database_url,
        model_version=settings.model_version,
    )

    # Load base projection for comparison
    base_row = svc._load_projection_row(  # noqa: SLF001
        req.player_id, req.week, req.season, req.stat
    )
    base_proj = float((base_row or {}).get("projection") or 0.0)

    # Run scenario with overrides
    overrides_dict = {
        k: v for k, v in req.overrides.model_dump().items() if v is not None
    }
    result = svc.run_scenario(
        player_id=req.player_id,
        week=req.week,
        season=req.season,
        stat=req.stat,
        overrides=overrides_dict,
    )
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No projection found for player_id='{req.player_id}'",
        )

    delta = result.projection - base_proj
    delta_pct = (delta / base_proj * 100.0) if base_proj != 0 else 0.0

    return ScenarioResponse(
        player_id=result.player_id,
        player_name=result.player_name,
        stat=req.stat,
        week=req.week,
        season=req.season,
        base_projection=round(base_proj, 2),
        scenario_projection=round(result.projection, 2),
        delta=round(delta, 2),
        delta_pct=round(delta_pct, 1),
        percentiles=Percentiles(
            p10=result.floor,
            p50=result.projection,
            p90=result.ceiling,
        ),
        boom_probability=result.boom_probability,
        bust_probability=result.bust_probability,
        fantasy_projection=result.fantasy_projection,
        data_freshness=datetime.now(timezone.utc),
    )
