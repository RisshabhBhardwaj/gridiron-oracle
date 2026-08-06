"""
backend/app/api/explain.py

/explain/{player_id} endpoint.

Returns SHAP factor attributions for a player's projection.
Factor labels are plain-English strings from ml/shap_service.FEATURE_LABELS.

CLAUDE.md §3: "SHAP factor labels served by the backend must be plain English
strings, not raw feature names."
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from backend.app.core.runtime_mode import ArtifactRequiredError
from backend.app.services.projection import ProjectionService

router = APIRouter(prefix="", tags=["explain"])


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

class FactorItem(BaseModel):
    feature: str    # raw column name (for frontend filtering)
    label:   str    # plain-English label from FEATURE_LABELS
    impact:  float  # SHAP value (positive = pushes projection up)
    value:   float  # actual feature value for this player


class ExplainResponse(BaseModel):
    player_id:    str
    week:         int
    season:       int
    stat:         str
    position:     str
    base_value:   float               # mean prediction (SHAP base value)
    attribution_source: str
    top_factors:  list[FactorItem] = Field(default_factory=list)
    data_freshness: datetime


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.get("/explain/{player_id}", response_model=ExplainResponse)
def explain(
    player_id: str,
    week:      int = Query(..., ge=1, le=22),
    season:    int = Query(..., ge=2019),
    stat:      str = Query("receiving_yards"),
    top_n:     int = Query(5, ge=1, le=20),
) -> ExplainResponse:
    """
    SHAP force-plot data for a player projection.

    Used by the SHAP bar chart in the Player Detail page.
    top_n controls how many factors to return (default 5, max 20).
    """
    svc = ProjectionService(
        db_url=settings.database_url,
        model_version=settings.model_version,
    )

    # Get features
    features = svc.get_feature_dict(player_id, week, season)

    # Get position (for SHAP model lookup)
    _, position, _ = svc._player_meta(player_id)  # noqa: SLF001
    position = position or "WR"

    # Compute SHAP attributions
    from ml.shap_service import SHAPService
    shap_svc = SHAPService(mlflow_tracking_uri=settings.mlflow_tracking_uri)
    try:
        explanation = shap_svc.explain(features, stat=stat, position=position, top_n=top_n)
    except ArtifactRequiredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Base value: the mean projection (Kalman estimate as proxy)
    kalman_est, _ = svc._load_kalman(player_id, week, season, stat)  # noqa: SLF001
    base_value = kalman_est or sum(a.impact for a in explanation.attributions)

    return ExplainResponse(
        player_id=player_id,
        week=week,
        season=season,
        stat=stat,
        position=position,
        base_value=float(base_value),
        attribution_source=explanation.source,
        top_factors=[
            FactorItem(
                feature=a.feature,
                label=a.label,
                impact=a.impact,
                value=a.value,
            )
            for a in explanation.attributions
        ],
        data_freshness=svc._data_freshness(player_id, week, season),  # noqa: SLF001
    )
