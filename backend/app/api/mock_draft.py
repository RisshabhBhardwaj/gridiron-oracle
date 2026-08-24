"""
backend/app/api/mock_draft.py

Stateless mock draft API:
- GET /mock-draft/profiles?season=2026: Manager tendency profiles with empirical shrinkage.
- POST /mock-draft/pick: Simulates picks up to the user's turn in snake order.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from ml.mock_draft import build_mock_draft_pool, pick
from scripts.generate_sleeper_draft_profiles import NOTE_CAVEAT, load_profiles

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["mock_draft"])


class ManagerProfile(BaseModel):
    owner_id: str
    display_name: str
    draft_slot: Optional[int] = None
    drafts: int = 0
    picks: int = 0
    first_qb_pick_shrunk: float = 48.0
    first_te_pick_shrunk: float = 64.0
    adp_delta_mean_shrunk: float = 0.0
    adp_delta_sd_shrunk: float = 6.0
    early_rb_share_shrunk: float = 0.5
    early_wr_share_shrunk: float = 0.4


class MockDraftProfilesResponse(BaseModel):
    season: int
    count: int
    profiles: list[ManagerProfile]
    note: str = Field(default=NOTE_CAVEAT)


class PickRequest(BaseModel):
    season: int = 2026
    draft_order: list[str]  # 8 manager owner_ids in slot 1..8 order
    picks_so_far: list[dict[str, Any]] = Field(default_factory=list)  # list of already picked {player_id, owner_id, pick_no, player_name, ...}
    user_slot: int = 1  # 1-indexed slot number of the human user (1..8)
    auto_pick_user: bool = False
    seed: Optional[int] = None


class PickedItem(BaseModel):
    pick_no: int
    round: int
    slot: int
    owner_id: str
    player: dict[str, Any]


class PickResponse(BaseModel):
    season: int
    new_picks: list[PickedItem]
    is_user_turn: bool
    is_draft_complete: bool
    next_turn_slot: Optional[int] = None
    seed: int


@router.get("/mock-draft/profiles", response_model=MockDraftProfilesResponse)
def get_mock_draft_profiles(
    season: int = Query(default=2026, description="Target season"),
) -> MockDraftProfilesResponse:
    """Fetch 8 manager tendency profiles with empirical Bayes shrinkage."""
    try:
        # Profiles describe *historical* manager behaviour, so they must be
        # built from the drafts that already happened — never from `season`,
        # which is the season being mocked and has no draft yet. Passing
        # seasons=[2026] silently produced zero profiles and left the mock
        # draft UI with an empty manager list.
        data = load_profiles(settings.database_url)
        profiles_raw = data.get("profiles", [])
        profiles = []
        for p in profiles_raw:
            profiles.append(
                ManagerProfile(
                    owner_id=p["owner_id"],
                    display_name=p["display_name"],
                    draft_slot=p.get("draft_slot"),
                    drafts=p.get("drafts", 0),
                    picks=p.get("picks", 0),
                    first_qb_pick_shrunk=float(p.get("first_qb_pick_shrunk") or 48.0),
                    first_te_pick_shrunk=float(p.get("first_te_pick_shrunk") or 64.0),
                    adp_delta_mean_shrunk=float(p.get("adp_delta_mean_shrunk") or 0.0),
                    adp_delta_sd_shrunk=float(p.get("adp_delta_sd_shrunk") or 6.0),
                    early_rb_share_shrunk=float(p.get("early_rb_share_shrunk") or 0.5),
                    early_wr_share_shrunk=float(p.get("early_wr_share_shrunk") or 0.4),
                )
            )
        return MockDraftProfilesResponse(
            season=season,
            count=len(profiles),
            profiles=profiles,
            note=data.get("method", {}).get("note", NOTE_CAVEAT),
        )
    except Exception as exc:
        logger.error("Error loading mock draft profiles: %s", exc)
        raise HTTPException(status_code=503, detail=f"Draft profiles unavailable: {exc}") from exc


@router.post("/mock-draft/pick", response_model=PickResponse)
def simulate_mock_draft_picks(req: PickRequest) -> PickResponse:
    """
    Stateless pick engine:
    Simulates consecutive opponent picks in snake order until it's the user's turn
    or the draft finishes (136 picks total = 8 teams × 17 rounds).
    """
    n_teams = len(req.draft_order)
    if n_teams != 8:
        raise HTTPException(status_code=400, detail="draft_order must contain exactly 8 manager IDs.")

    total_rounds = 17
    total_picks = n_teams * total_rounds  # 136 picks
    seed = req.seed if req.seed is not None else int(np.random.randint(0, 1_000_000_000))
    rng = np.random.default_rng(seed)

    # 1. Load profiles map
    try:
        prof_data = load_profiles(settings.database_url, seasons=[req.season])
        profiles_by_id = {p["owner_id"]: p for p in prof_data.get("profiles", [])}
    except Exception:
        profiles_by_id = {}

    # 2. Load available player pool
    try:
        pool = build_mock_draft_pool(settings.database_url, season=req.season)
    except Exception as exc:
        logger.error("Error loading draft pool: %s", exc)
        raise HTTPException(status_code=503, detail=f"Draft pool unavailable: {exc}") from exc

    # Filter out already drafted players
    drafted_pids = {str(p.get("player_id") or p.get("player", {}).get("player_id")) for p in req.picks_so_far}
    available = [p for p in pool if str(p["player_id"]) not in drafted_pids]

    # Reconstruct team rosters
    rosters_by_owner: dict[str, list[dict[str, Any]]] = {oid: [] for oid in req.draft_order}
    for p in req.picks_so_far:
        oid = p.get("owner_id")
        player_obj = p.get("player") or p
        if oid in rosters_by_owner:
            rosters_by_owner[oid].append(player_obj)

    new_picks: list[PickedItem] = []
    current_pick_no = len(req.picks_so_far) + 1

    while current_pick_no <= total_picks and len(available) > 0:
        round_idx = (current_pick_no - 1) // n_teams + 1
        pos_in_round = (current_pick_no - 1) % n_teams  # 0..7

        # Snake order: odd rounds 1..8, even rounds 8..1
        if round_idx % 2 == 1:
            slot = pos_in_round + 1
        else:
            slot = n_teams - pos_in_round

        owner_id = req.draft_order[slot - 1]
        is_user = (slot == req.user_slot)

        if is_user and not req.auto_pick_user and len(new_picks) > 0:
            # Stop right before user turn if we have already simulated opponent picks
            break

        prof = profiles_by_id.get(owner_id, {
            "first_qb_pick_shrunk": 48.0,
            "first_te_pick_shrunk": 64.0,
            "adp_delta_mean_shrunk": 0.0,
            "adp_delta_sd_shrunk": 6.0,
        })
        chosen = pick(
            available=available,
            roster=rosters_by_owner[owner_id],
            profile=prof,
            rng=rng,
            pick_no=current_pick_no,
        )

        item = PickedItem(
            pick_no=current_pick_no,
            round=round_idx,
            slot=slot,
            owner_id=owner_id,
            player=chosen,
        )
        new_picks.append(item)
        rosters_by_owner[owner_id].append(chosen)
        available = [p for p in available if str(p["player_id"]) != str(chosen["player_id"])]
        current_pick_no += 1

        if is_user and not req.auto_pick_user:
            # If user was auto-picked on step 1, continue simulation to next user turn
            pass

    # Determine state after simulation
    next_pick_no = len(req.picks_so_far) + len(new_picks) + 1
    is_complete = next_pick_no > total_picks or len(available) == 0

    if not is_complete:
        round_idx = (next_pick_no - 1) // n_teams + 1
        pos_in_round = (next_pick_no - 1) % n_teams
        next_slot = (pos_in_round + 1) if (round_idx % 2 == 1) else (n_teams - pos_in_round)
        is_user_turn = (next_slot == req.user_slot)
    else:
        next_slot = None
        is_user_turn = False

    return PickResponse(
        season=req.season,
        new_picks=new_picks,
        is_user_turn=is_user_turn,
        is_draft_complete=is_complete,
        next_turn_slot=next_slot,
        seed=seed,
    )
