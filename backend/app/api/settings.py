"""
backend/app/api/settings.py

GET /settings and PUT /settings endpoints.

Manages user-configurable feature weights and fantasy scoring presets.
Weights are stored in user_config.json; validated to sum to 100%.

CLAUDE.md §5.1: Weights are fully user-configurable, validated to sum to 100%.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from backend.app.core.config import settings as app_settings
from backend.app.core.rate_limit import limiter

router = APIRouter(prefix="", tags=["settings"])
logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(app_settings.settings_file)


# ---------------------------------------------------------------------------
# Default weight configuration
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = {
    "kalman_form":      35.0,
    "seasonal_baseline": 20.0,
    "matchup":          20.0,
    "weather_venue":    10.0,
    "team_context":      5.0,
    "roster_injury":     5.0,
    "rule_meta":         5.0,
}

_DEFAULT_CONFIG = {
    "weights":              _DEFAULT_WEIGHTS,
    "fantasy_scoring":      "ppr",      # "ppr" | "half_ppr" | "standard"
    "engine_exposure_cap":  0.25,       # max fraction of bankroll per bet
    "weight_preset_name":   "Default",
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class FeatureWeights(BaseModel):
    kalman_form:       float = Field(35.0, ge=0, le=100)
    seasonal_baseline: float = Field(20.0, ge=0, le=100)
    matchup:           float = Field(20.0, ge=0, le=100)
    weather_venue:     float = Field(10.0, ge=0, le=100)
    team_context:      float = Field( 5.0, ge=0, le=100)
    roster_injury:     float = Field( 5.0, ge=0, le=100)
    rule_meta:         float = Field( 5.0, ge=0, le=100)

    @model_validator(mode="after")
    def weights_must_sum_to_100(self) -> "FeatureWeights":
        total = (
            self.kalman_form + self.seasonal_baseline + self.matchup
            + self.weather_venue + self.team_context + self.roster_injury
            + self.rule_meta
        )
        if not (99.9 <= total <= 100.1):
            raise ValueError(
                f"Feature weights must sum to 100 (got {total:.1f}). "
                f"Adjust the values to total exactly 100."
            )
        return self


class SettingsResponse(BaseModel):
    weights:              FeatureWeights
    fantasy_scoring:      str    # "ppr" | "half_ppr" | "standard"
    engine_exposure_cap:  float  # max fraction of bankroll per bet
    weight_preset_name:   str


class SettingsUpdateRequest(BaseModel):
    weights:             Optional[FeatureWeights] = None
    fantasy_scoring:     Optional[str]            = None
    engine_exposure_cap: Optional[float]          = Field(None, ge=0.01, le=1.0)
    weight_preset_name:  Optional[str]            = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/settings", response_model=SettingsResponse)
def get_settings() -> SettingsResponse:
    """Retrieve the current weight configuration."""
    cfg = _load_config()
    return SettingsResponse(
        weights=FeatureWeights(**cfg.get("weights", _DEFAULT_WEIGHTS)),
        fantasy_scoring=cfg.get("fantasy_scoring", "ppr"),
        engine_exposure_cap=cfg.get("engine_exposure_cap", 0.25),
        weight_preset_name=cfg.get("weight_preset_name", "Default"),
    )


@router.put("/settings", response_model=SettingsResponse)
@limiter.limit("10/minute")
def update_settings(request: Request, req: SettingsUpdateRequest) -> SettingsResponse:
    """
    Update the weight configuration.

    Weights must sum to 100%; validated by Pydantic before saving.
    Returns the updated configuration.
    """
    cfg = _load_config()

    body_hash = hashlib.sha256(req.model_dump_json().encode()).hexdigest()[:16]
    logger.info(
        "settings.write request_id=%s path=%s body_hash=%s",
        request.headers.get("X-Request-ID", "-"),
        request.url.path,
        body_hash,
    )

    if req.weights is not None:
        cfg["weights"] = req.weights.model_dump()
    if req.fantasy_scoring is not None:
        valid_scoring = {"ppr", "half_ppr", "standard"}
        if req.fantasy_scoring not in valid_scoring:
            raise HTTPException(
                status_code=422,
                detail=f"fantasy_scoring must be one of {valid_scoring}",
            )
        cfg["fantasy_scoring"] = req.fantasy_scoring
    if req.engine_exposure_cap is not None:
        cfg["engine_exposure_cap"] = req.engine_exposure_cap
    if req.weight_preset_name is not None:
        cfg["weight_preset_name"] = req.weight_preset_name

    _save_config(cfg)

    return SettingsResponse(
        weights=FeatureWeights(**cfg["weights"]),
        fantasy_scoring=cfg.get("fantasy_scoring", "ppr"),
        engine_exposure_cap=cfg.get("engine_exposure_cap", 0.25),
        weight_preset_name=cfg.get("weight_preset_name", "Default"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load user_config.json, returning defaults if file is missing."""
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except Exception as exc:
            logger.warning("Failed to parse %s: %s — using defaults", _CONFIG_PATH, exc)
    return dict(_DEFAULT_CONFIG)


def _save_config(cfg: dict) -> None:
    """Persist configuration to user_config.json."""
    try:
        _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception as exc:
        logger.error("Failed to save %s: %s", _CONFIG_PATH, exc)
        raise HTTPException(status_code=500, detail="Failed to save settings") from exc
