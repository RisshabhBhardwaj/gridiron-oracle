"""
backend/app/api/drive.py

Phase 5 — serves the C++ DriveMCMC engine, loaded once with the fitted
(fp, down, ytg, score_diff_bucket, quarter) transition table from
ml/oof/transitions_by_game_state.csv (see scripts/export_drive_transitions.py
--db). Fails closed (503) if the shared library isn't built or the fitted
artifact hasn't been materialized — this endpoint never fits a model
per-request, only loads pre-computed transition parameters.

The DriveMCMC C++ handle is NOT thread-safe (see engine/include/drive_mcmc.hpp's
ownership contract); FastAPI's sync routes run in a threadpool, so a module-level
lock serializes access to the single cached handle. Each simulate() call is
sub-millisecond, so this isn't a real bottleneck.
"""
from __future__ import annotations

import threading
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="", tags=["drive"])

_ENGINE = None
_ENGINE_LOCK = threading.Lock()


def _get_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            from ml.drive_engine import load_drive_engine

            _ENGINE = load_drive_engine()
        return _ENGINE


class DriveSimulationResponse(BaseModel):
    field_pos: int
    down: int
    yards_to_go: int
    score_differential: int
    quarter: int
    n_simulations: int
    p_touchdown: float
    p_field_goal: float
    expected_yards: float
    expected_pass_rate: float
    drive_value: float
    note: str = (
        "Transition table is fit on pbp_plays (Phase 2/5) with cell-count "
        "shrinkage toward the (field_pos, down, yards_to_go) marginal for "
        "sparse (score, quarter) combinations — see "
        "ml.markov_simulator.DriveMarkovModel._shrink_to_marginal."
    )


@router.get("/drive/simulate", response_model=DriveSimulationResponse)
def simulate_drive(
    field_pos: int = Query(..., ge=0, le=100, description="0=own goal line, 100=opponent goal line"),
    down: int = Query(1, ge=1, le=4),
    yards_to_go: int = Query(10, ge=1, le=30),
    score_differential: int = Query(0, ge=-56, le=56, description="Possessing team's margin, fixed for the drive"),
    quarter: int = Query(1, ge=1, le=5, description="1-4; 5 (OT) is clamped to 4 by the engine"),
    n_simulations: Optional[int] = Query(None, ge=100, le=50000),
) -> DriveSimulationResponse:
    try:
        engine = _get_engine()
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"DriveMCMC engine unavailable: {exc}",
        ) from exc

    if n_simulations is not None and n_simulations != engine.n_simulations:
        from ml.drive_engine import load_drive_engine

        engine = load_drive_engine(n_simulations=n_simulations)

    result = engine.simulate(
        field_pos=field_pos,
        down=down,
        yards_to_go=yards_to_go,
        score_differential=score_differential,
        quarter=quarter,
    )
    return DriveSimulationResponse(
        field_pos=field_pos,
        down=down,
        yards_to_go=yards_to_go,
        score_differential=score_differential,
        quarter=quarter,
        n_simulations=n_simulations or engine.n_simulations,
        **result,
    )
