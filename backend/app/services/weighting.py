"""
backend/app/services/weighting.py

Apply the user's Settings to one served projection row (weights-08 stream,
decision 0008 item 6; design in
`.agent-work/gridiron/audit/settings-weights-design.md`).

What each setting does here:

- The seven bucket weights rescale the row's stored per-bucket attributions
  (`ml/bucket_weights.py`). At the defaults the stored row is returned
  unchanged. Where the row carries no stored attributions the numbers are
  returned unchanged with reason ``"no_attribution"``. Attributions are
  written at materialization time (``projection_attributions``, from the
  SHA-pinned attribution files); rows served from a release that predates
  them have none.
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

import logging
from dataclasses import dataclass
from typing import Mapping, Optional

from ml.bucket_attribution import ATTRIBUTION_TOLERANCE
from ml.bucket_weights import (
    BUCKET_KEYS,
    DEFAULT_BUCKET_WEIGHTS,
    apply_bucket_weights,
    weights_are_default,
)
from ml.scoring import rules_for_scoring, score_fantasy

#: The engine's hard per-bet cap. The Settings value never changes this;
#: it maps to kelly_fraction only.
ENGINE_MAX_BET_FRACTION: float = 0.05

logger = logging.getLogger(__name__)


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
    """Stored per-bucket attributions for the served row, or None.

    Fails closed (decision 0002). The attributions are used only when all of
    these hold, and otherwise the caller returns the canonical number unchanged
    with reason ``no_attribution``:

    - the row is served by an approved pipeline run;
    - the attribution row carries that same run id (so it explains this
      projection and not an older one);
    - ``base + buckets + unweighted`` still equals the served projection to
      within ``ATTRIBUTION_TOLERANCE``.
    """
    import psycopg2

    from backend.app.core.config import settings
    from backend.app.services.projection import load_approved_pipeline_run_ids

    try:
        approved = list(load_approved_pipeline_run_ids())
        conn = psycopg2.connect(settings.database_url)
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT p.projection, a.base, a.unweighted,
                       {", ".join(f"a.{key}" for key in BUCKET_KEYS)}
                FROM   projections p
                JOIN   projection_attributions a
                       ON  a.player_id = p.player_id
                       AND a.game_id = p.game_id
                       AND a.stat = p.stat
                       AND a.pipeline_run_id = p.pipeline_run_id
                WHERE  p.player_id = %s AND p.week = %s AND p.season = %s AND p.stat = %s
                  AND  p.pipeline_run_id = ANY(%s)
                ORDER BY p.created_at DESC
                LIMIT  1
                """,
                [player_id, week, season, stat, approved],
            )
            row = cur.fetchone()
        finally:
            conn.close()
    except Exception as exc:  # no table yet, no database, no manifest: all mean "no attribution"
        logger.debug("load_bucket_attributions: %s", exc)
        return None
    if row is None:
        return None
    projection, base, unweighted, *buckets = (float(value) for value in row)
    if abs(projection - (base + unweighted + sum(buckets))) > ATTRIBUTION_TOLERANCE:
        logger.warning(
            "attributions for %s %s w%s %s do not add up to the served value; ignoring them",
            player_id, season, week, stat,
        )
        return None
    return dict(zip(BUCKET_KEYS, buckets))


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
    weights = {**DEFAULT_BUCKET_WEIGHTS, **(cfg.get("weights") or {})}
    scoring = cfg.get("fantasy_scoring") or "ppr"

    # Defaults never need attributions, so skip the lookup (one query per row).
    attributions = (
        None if weights_are_default(weights)
        else load_bucket_attributions(player_id, week, season, stat)
    )
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

