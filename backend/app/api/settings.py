"""
backend/app/api/settings.py

GET /settings and PUT /settings endpoints.

Manages user-configurable feature weights and fantasy scoring presets.
Weights are stored in user_config.json; validated to sum to 100%.

Weights are fully user-configurable, validated to sum to 100%.
"""

from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import json
import logging
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError, model_validator

from backend.app.core.config import settings as app_settings
from backend.app.core.rate_limit import limiter

router = APIRouter(prefix="", tags=["settings"])
logger = logging.getLogger(__name__)

# Optional explicit override. When unset, the path is read from settings at
# call time, so a SETTINGS_FILE change (or a test fixture) takes effect.
_CONFIG_PATH: Optional[Path] = None


def _config_path() -> Path:
    if _CONFIG_PATH is not None:
        return _CONFIG_PATH
    return Path(app_settings.settings_file)


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


SettingsSource = Literal["file", "default_missing", "default_invalid"]


class StoredConfig(BaseModel):
    """Shape of user_config.json. Keys that are absent take the defaults."""
    weights:             FeatureWeights = Field(default_factory=FeatureWeights)
    fantasy_scoring:     Literal["ppr", "half_ppr", "standard"] = "ppr"
    engine_exposure_cap: float = Field(0.25, ge=0.01, le=1.0)
    weight_preset_name:  str = "Default"


class SettingsResponse(BaseModel):
    weights:              FeatureWeights
    fantasy_scoring:      str    # "ppr" | "half_ppr" | "standard"
    engine_exposure_cap:  float  # max fraction of bankroll per bet
    weight_preset_name:   str
    # Where the returned values came from. "default_*" means the stored file
    # was not used and the defaults were returned instead.
    settings_source:      SettingsSource


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
    """Retrieve the current weight configuration and where it came from."""
    cfg, source = _read_config()
    return _to_response(cfg, source)


@router.put("/settings", response_model=SettingsResponse)
@limiter.limit("10/minute")
def update_settings(request: Request, req: SettingsUpdateRequest) -> SettingsResponse:
    """
    Update the weight configuration.

    Weights must sum to 100%; validated by Pydantic before saving.
    Returns the updated configuration. settings_source reports where the
    base the update was merged onto came from (the file, or defaults when
    the file was missing or unusable). After a successful save the file
    holds the result, so the next GET reports "file".
    """
    cfg, source = _read_config()

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

    return _to_response(cfg, source)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_config() -> dict:
    return copy.deepcopy(_DEFAULT_CONFIG)


def _read_config() -> tuple[dict, SettingsSource]:
    """Return the stored config and where it came from.

    Falls back to defaults when the file is missing ("default_missing"), or
    unreadable, not valid JSON, not an object, or fails validation
    ("default_invalid"). Keys the file holds beyond the known ones are kept.
    """
    path = _config_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _default_config(), "default_missing"
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Cannot read %s: %s — using defaults", path, exc)
        return _default_config(), "default_invalid"
    try:
        data = json.loads(raw)
        stored = StoredConfig.model_validate(data)
    except (ValueError, ValidationError) as exc:
        logger.warning("Invalid settings in %s: %s — using defaults", path, exc)
        return _default_config(), "default_invalid"
    return {**data, **stored.model_dump()}, "file"


def _load_config() -> dict:
    """Load user_config.json, returning defaults if the file is missing or invalid."""
    return _read_config()[0]


def _to_response(cfg: dict, source: SettingsSource) -> SettingsResponse:
    return SettingsResponse(
        weights=FeatureWeights(**cfg["weights"]),
        fantasy_scoring=cfg["fantasy_scoring"],
        engine_exposure_cap=cfg["engine_exposure_cap"],
        weight_preset_name=cfg["weight_preset_name"],
        settings_source=source,
    )


def _save_config(cfg: dict) -> None:
    """Persist configuration to user_config.json (atomic, serialized by a lock)."""
    path = _config_path()
    try:
        _write_json_atomic(path, cfg)
    except Exception as exc:
        logger.error("Failed to save %s: %s", path, exc)
        raise HTTPException(status_code=500, detail="Failed to save settings") from exc


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive flock on a sibling lock file for the duration of the block."""
    lock_path = path.with_name(path.name + ".lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the flock


def _fsync_dir(directory: Path) -> None:
    """Best-effort fsync of a directory so a completed rename survives a crash."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # some filesystems do not support directory fsync
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, cfg: dict) -> None:
    """Replace path with cfg's JSON so readers see the old file or the new one, never a partial one."""
    with _exclusive_lock(path):
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except FileNotFoundError:
            mode = 0o644
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp_name, mode)  # mkstemp creates 0600; keep the file's existing mode
            os.replace(tmp_name, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_name)
            raise
        _fsync_dir(path.parent)
