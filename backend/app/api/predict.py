"""
backend/app/api/predict.py

/predict and /projections/week/{n} endpoints.

CLAUDE.md §3 (API Rules):
  - Every endpoint returns a typed Pydantic response model.
  - data_freshness timestamp required on every /predict response.
  - Frontend displays a warning if data_freshness is > 24 hours old.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from backend.app.core.rate_limit import limiter
from backend.app.core.runtime_mode import ArtifactRequiredError
from backend.app.services.projection import ProjectionService, VALID_STATS

router = APIRouter(prefix="", tags=["predict"])


# ---------------------------------------------------------------------------
# Stat whitelist helper — rejects invalid stat values before hitting services
# ---------------------------------------------------------------------------

def _validate_stat(stat: str) -> str:
    if stat not in VALID_STATS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid stat '{stat}'. Must be one of: {sorted(VALID_STATS)}",
        )
    return stat


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class StatProjection(BaseModel):
    """Dynamic projection values — any of the 15 target stats may be present."""
    model_config = {"extra": "allow"}   # allow arbitrary stat keys

    # All 15 target stats (None = not projected for this position)
    pass_attempts:    Optional[float] = None
    completions:      Optional[float] = None
    passing_yards:    Optional[float] = None
    passing_tds:      Optional[float] = None
    interceptions:    Optional[float] = None
    carries:          Optional[float] = None
    rushing_yards:    Optional[float] = None
    rushing_tds:      Optional[float] = None
    receptions:       Optional[float] = None
    receiving_yards:  Optional[float] = None
    receiving_tds:    Optional[float] = None
    targets:          Optional[float] = None
    fantasy_ppr:      Optional[float] = None
    fumbles:          Optional[float] = None
    sacks_taken:      Optional[float] = None
    # Composite fantasy scoring
    ppr_points:       Optional[float] = None


class Percentiles(BaseModel):
    p10: Optional[float] = None
    p50: float
    p90: Optional[float] = None


class SHAPFactor(BaseModel):
    feature: str
    impact:  float
    label:   str


class PredictResponse(BaseModel):
    player:                  str
    player_id:               str
    week:                    int
    season:                  int
    position:                str
    stat:                    str
    projection:              StatProjection
    percentiles:             Percentiles
    interval_method:          str
    degraded:                 bool = False
    pipeline_run_id:          Optional[str] = None
    confidence_score:        Optional[float] = None
    kalman_ability_estimate: Optional[float] = None
    kalman_uncertainty:      Optional[float] = None
    boom_probability:        Optional[float] = None
    bust_probability:        Optional[float] = None
    top_factors:             list[SHAPFactor] = Field(default_factory=list)
    attribution_source:      str = "unavailable"
    model_version:           str
    data_freshness:          datetime


class WeekPlayerProjection(BaseModel):
    """Lightweight projection for the /projections/week batch response."""
    player_id:          str
    player_name:        str
    position:           str
    team:               Optional[str] = None
    stat:               str
    projection:         float
    floor:              Optional[float] = None
    ceiling:            Optional[float] = None
    interval_method:    str
    degraded:           bool = False
    pipeline_run_id:    Optional[str] = None
    p5:                 Optional[float] = None
    p25:                Optional[float] = None
    p75:                Optional[float] = None
    p95:                Optional[float] = None
    boom_probability:   Optional[float] = None
    bust_probability:   Optional[float] = None
    fantasy_projection: Optional[float] = None
    fantasy_floor:      Optional[float] = None
    fantasy_ceiling:    Optional[float] = None
    model_version:      str
    data_freshness:     datetime


class WeekProjectionsResponse(BaseModel):
    week:          int
    season:        int
    stat:          str
    count:         int
    projections:   list[WeekPlayerProjection]
    data_freshness: datetime


class SeasonStatProjection(BaseModel):
    mean: float
    p10:  float
    p50:  float
    p90:  float


class SeasonPlayerProjection(BaseModel):
    player_id:          str
    player_name:        str
    position:           str
    team:               Optional[str] = None
    passing_yards:      Optional[SeasonStatProjection] = None
    rushing_yards:      Optional[SeasonStatProjection] = None
    receiving_yards:    Optional[SeasonStatProjection] = None
    fantasy_ppr:        Optional[SeasonStatProjection] = None


class SeasonProjectionsResponse(BaseModel):
    season:        int
    start_week:    int
    count:         int
    projections:   list[SeasonPlayerProjection]
    data_freshness: datetime


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

def _svc() -> ProjectionService:
    return ProjectionService(
        db_url=settings.database_url,
        model_version=settings.model_version,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/predict", response_model=PredictResponse)
@limiter.limit("60/minute")
def predict(
    request: Request,
    player: str = Query(..., max_length=200, description="Player full name (partial match OK)"),
    week:   int = Query(..., ge=1, le=22, description="NFL week number"),
    season: int = Query(..., ge=2019,     description="NFL season year"),
    stat:   str = Query("receiving_yards", max_length=64, description="Stat to project"),
    svc:    ProjectionService = Depends(_svc),
) -> PredictResponse:
    """
    Full projection for one player / week / stat.

    Returns point estimate, percentile fan (p10/p50/p90), Kalman form
    estimate, confidence score, and SHAP factor attributions.

    data_freshness > 24h triggers a staleness warning in the frontend.
    """
    _validate_stat(stat)
    result = svc.get_projection(player, week, season, stat)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No projection found for player='{player}' week={week} season={season}",
        )

    # Attach SHAP factors
    shap_result = _build_shap_factors(
        svc, result.player_id, week, season, stat, result.position
    )
    if isinstance(shap_result, tuple):
        top_factors, attribution_source = shap_result
    else:
        top_factors = shap_result
        attribution_source = "unavailable"

    # Build stat projection — dynamically set the requested stat field
    stat_proj = StatProjection.model_validate({stat: result.projection, "ppr_points": result.fantasy_projection})

    return PredictResponse(
        player=result.player_name,
        player_id=result.player_id,
        week=result.week,
        season=result.season,
        position=result.position,
        stat=stat,
        projection=stat_proj,
        percentiles=Percentiles(
            p10=result.floor,
            p50=result.projection,
            p90=result.ceiling,
        ),
        interval_method=result.interval_method,
        degraded=result.degraded,
        pipeline_run_id=result.pipeline_run_id,
        confidence_score=result.confidence_score,
        kalman_ability_estimate=result.kalman_ability_estimate,
        kalman_uncertainty=result.kalman_uncertainty,
        boom_probability=result.boom_probability,
        bust_probability=result.bust_probability,
        top_factors=top_factors,
        attribution_source=attribution_source,
        model_version=result.model_version,
        data_freshness=result.data_freshness,
    )


@router.get("/projections/week/{week}", response_model=WeekProjectionsResponse)
@limiter.limit("30/minute")
def projections_for_week(
    request:        Request,
    week:           int,
    season:         int  = Query(..., ge=2019),
    stat:           str  = Query("receiving_yards", max_length=64),
    positions:      list[str] = Query(default=["WR", "RB", "TE", "QB"]),
    min_projection: float = Query(default=0.0, description="Minimum projection to include (e.g. 5.0 filters noise)"),
    svc:            ProjectionService = Depends(_svc),
) -> WeekProjectionsResponse:
    """
    Batch projections for all skill-position players in a given week.

    Used by the Dashboard page to populate the sortable projection table.
    """
    _validate_stat(stat)
    results = svc.get_week_projections(week, season, positions, stat)
    # Apply server-side filter
    if min_projection > 0:
        results = [r for r in results if r.projection >= min_projection]

    items = [
        WeekPlayerProjection(
            player_id=r.player_id,
            player_name=r.player_name,
            position=r.position,
            team=r.team,
            stat=stat,
            projection=r.projection,
            floor=r.floor,
            ceiling=r.ceiling,
            interval_method=r.interval_method,
            degraded=r.degraded,
            pipeline_run_id=r.pipeline_run_id,
            boom_probability=r.boom_probability,
            bust_probability=r.bust_probability,
            fantasy_projection=r.fantasy_projection,
            fantasy_floor=r.fantasy_floor,
            fantasy_ceiling=r.fantasy_ceiling,
            model_version=r.model_version,
            data_freshness=r.data_freshness,
        )
        for r in results
    ]

    return WeekProjectionsResponse(
        week=week,
        season=season,
        stat=stat,
        count=len(items),
        projections=items,
        data_freshness=datetime.now(timezone.utc),
    )


@router.get("/projections/season/{season}", response_model=SeasonProjectionsResponse)
@limiter.limit("20/minute")
def season_projections(
    request:        Request,
    season:         int,
    start_week:     int  = Query(..., ge=1, le=18, description="First week of simulated rest-of-season"),
    positions:      list[str] = Query(default=["WR", "RB", "TE", "QB"]),
    svc:            ProjectionService = Depends(_svc),
) -> SeasonProjectionsResponse:
    """
    Batch rest-of-season projections for all players.
    Uses the C++ autoregressive engine.
    """
    results = svc.get_season_projections(
        season=season,
        start_week=start_week,
        positions=positions,
        stats=["passing_yards", "rushing_yards", "receiving_yards", "fantasy_ppr"]
    )

    items = []
    for r in results:
        kwargs = {
            "player_id": r["player_id"],
            "player_name": r["player_name"],
            "position": r["position"],
            "team": r.get("team"),
        }
        for stat in ["passing_yards", "rushing_yards", "receiving_yards", "fantasy_ppr"]:
            if stat in r:
                kwargs[stat] = SeasonStatProjection(**r[stat])
                
        items.append(SeasonPlayerProjection(**kwargs))

    return SeasonProjectionsResponse(
        season=season,
        start_week=start_week,
        count=len(items),
        projections=items,
        data_freshness=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_shap_factors(
    svc: ProjectionService,
    player_id: str,
    week: int,
    season: int,
    stat: str,
    position: str,
) -> tuple[list[SHAPFactor], str]:
    """Attach SHAP attributions to the prediction response."""
    try:
        from ml.shap_service import SHAPService
        features = svc.get_feature_dict(player_id, week, season)
        shap_svc = SHAPService(mlflow_tracking_uri=settings.mlflow_tracking_uri)
        explanation = shap_svc.explain(features, stat=stat, position=position, top_n=5)
        return [
            SHAPFactor(feature=a.feature, impact=a.impact, label=a.label)
            for a in explanation.attributions
        ], explanation.source
    except ArtifactRequiredError:
        return [], "artifact_required"
    except Exception:
        return [], "unavailable"
