"""
backend/app/services/weighting.py

Apply the user's Settings to one served projection row (weights-08 stream,
decision 0008 item 6; design in
`.agent-work/gridiron/audit/settings-weights-design.md`).

What each setting does here:

- The seven bucket weights rescale the row's stored per-bucket attributions
  (`ml/bucket_weights.py`). At the defaults the stored row is returned
  unchanged. Where the row carries no stored attributions the numbers are
  returned unchanged with reason ``"no_attribution"`` — stored attributions
  are written at materialization time, which is a rerun-time step (no
  training in this stream), so today that reason is the common one.
- ``fantasy_scoring`` picks the LeagueRules (``ml/scoring.py``). ``ppr``
  returns the stored ``fantasy_projection`` (direct regression) unchanged;
  ``half_ppr`` / ``standard`` re-score the week's component-stat projections
  with the matching rules.
- ``engine_exposure_cap`` maps to the C++ engine's ``kelly_fraction``
  (``engine/src/main.cpp`` reads ``kelly_fraction``, default 0.25 — the same
  default). It is NOT ``max_bet_fraction`` (hard cap 0.05): mapping there
  would loosen the cap 5x at the defaults. See ``engine_kelly_config``.

Every adjusted number is labelled: ``weighting`` is ``"user"`` only when a
weight moved the number, otherwise ``"default"`` with a reason. An adjusted
number is a counterfactual re-weighting, not a retrained model, and has no
backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from ml.bucket_weights import apply_bucket_weights
from ml.scoring import rules_for_scoring, score_fantasy

#: The engine's hard per-bet cap. The Settings value never changes this;
#: it maps to kelly_fraction only.
ENGINE_MAX_BET_FRACTION: float = 0.05


@dataclass
class WeightedRow:
    """One served row after the Settings layer."""

    projection: float
    floor: Optional[float]
    ceiling: Optional[float]
    fantasy_projection: Optional[float]
    weighting: str = "default"
    weights_reason: Optional[str] = None
    fantasy_scoring: str = "ppr"


def load_user_settings() -> dict:
    """Read the Settings store, falling back to defaults (never raises)."""
    from backend.app.api import settings as settings_api

    return settings_api._load_config()


def load_bucket_attributions(
    player_id: str, week: int, season: int, stat: str
) -> Optional[dict[str, float]]:
    """Stored per-bucket attributions for one served row, if published.

    No attribution sidecar exists yet: materialization does not write one
    (that is VPS-rerun work — native CatBoost/LightGBM TreeSHAP per row at
    materialization time, with the residual check from the design note).
    Returns None until that table exists, so callers stay honest: with no
    attributions the canonical numbers are returned unchanged and labelled
    ``no_attribution``.
    """
    _ = (player_id, week, season, stat)
    return None


def apply_settings_to_row(
    *,
    projection: float,
    floor: Optional[float],
    ceiling: Optional[float],
    fantasy_projection: Optional[float],
    component_stats: Optional[Mapping[str, float]],
    player_id: str,
    week: int,
    season: int,
    stat: str,
    settings: Optional[dict] = None,
) -> WeightedRow:
    """Apply the Settings layer to one served row.

    ``component_stats`` maps component-stat name to its projected value for
    this player/week (used only when fantasy_scoring is not ppr). Identity:
    at default weights and ppr scoring every numeric field is returned
    exactly unchanged with weighting ``"default"``.
    """
    cfg = settings if settings is not None else load_user_settings()
    weights = cfg.get("weights") or {}
    scoring = cfg.get("fantasy_scoring") or "ppr"

    attributions = load_bucket_attributions(player_id, week, season, stat)
    non_negative = stat not in ("spread_line",)
    weighted = apply_bucket_weights(
        float(projection), attributions, weights, non_negative=non_negative
    )
    delta = weighted.adjusted - float(projection)

    out = WeightedRow(
        projection=weighted.adjusted,
        floor=None if floor is None else floor + delta,
        ceiling=None if ceiling is None else ceiling + delta,
        fantasy_projection=fantasy_projection,
        weighting=weighted.weighting,
        weights_reason=weighted.reason,
        fantasy_scoring=scoring,
    )

    if scoring != "ppr" and fantasy_projection is not None:
        rescored = _rescore_fantasy(component_stats, scoring)
        if rescored is not None:
            out.fantasy_projection = rescored
    return out


def _rescore_fantasy(
    component_stats: Optional[Mapping[str, float]], scoring: str
) -> Optional[float]:
    """Re-score components under non-PPR rules; None when not possible."""
    if not component_stats:
        return None
    try:
        rules = rules_for_scoring(scoring)
    except KeyError:
        return None
    return score_fantasy(dict(component_stats), rules)


def engine_kelly_config(engine_exposure_cap: float = 0.25) -> dict[str, float]:
    """Map Settings ``engine_exposure_cap`` onto the C++ engine config.

    The value multiplies the full-Kelly fraction (``kelly_fraction``), and
    the engine then hard-caps every bet at ``max_bet_fraction`` — which this
    mapping leaves fixed. At the default 0.25 the generated config equals
    the engine's built-in defaults, so nothing changes. A settings change
    needs an engine restart: the engine reads its config once at startup
    (``engine/src/main.cpp``) and the IPC bridge carries no config message.
    """
    return {
        "kelly_fraction": float(engine_exposure_cap),
        "max_bet_fraction": ENGINE_MAX_BET_FRACTION,
    }

