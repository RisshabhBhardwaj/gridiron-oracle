"""
Explicit evaluation cohort definition.

The cohort is defined once from GameLog + snap filter and held fixed across
all baselines and models. Membership must not depend on Kalman estimates
existing — that silently changed the scored population when baselines changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from ml.utils import MIN_SNAP_PCT


@dataclass(frozen=True)
class CohortSpec:
    """Immutable description of who enters evaluation."""

    min_snap_pct: float = MIN_SNAP_PCT
    require_snap_when_present: bool = True
    # When offense_pct is NULL (seasons without snap data), keep the row.
    keep_null_snap: bool = True

    def describe(self) -> dict[str, Any]:
        return {
            "min_snap_pct": self.min_snap_pct,
            "require_snap_when_present": self.require_snap_when_present,
            "keep_null_snap": self.keep_null_snap,
            "units_expected": "[0, 1] fraction (values > 1.0 are treated as percent and /100)",
        }


DEFAULT_COHORT = CohortSpec()


def normalize_offense_pct(raw: Optional[float]) -> Optional[float]:
    """
    Snap-unit audit helper.

    nflverse historically emits offense_pct on a 0–100 scale for some seasons.
    Internal feature code expects a [0, 1] fraction. Values > 1.0 are divided
    by 100; values in [0, 1] are left unchanged.
    """
    if raw is None:
        return None
    val = float(raw)
    if val > 1.0:
        return val / 100.0
    return val


def passes_snap_filter(
    offense_pct: Optional[float],
    *,
    spec: CohortSpec = DEFAULT_COHORT,
) -> bool:
    pct = normalize_offense_pct(offense_pct)
    if pct is None:
        return spec.keep_null_snap
    if not spec.require_snap_when_present:
        return True
    return pct >= spec.min_snap_pct


def filter_cohort_frame(
    df: pd.DataFrame,
    *,
    snap_col: str = "offense_pct",
    spec: CohortSpec = DEFAULT_COHORT,
) -> pd.DataFrame:
    """Filter a GameLog-like DataFrame to the evaluation cohort."""
    if df.empty:
        return df
    if snap_col not in df.columns:
        raise KeyError(
            f"Cohort filter requires {snap_col!r} column; got {list(df.columns)}"
        )
    mask = df[snap_col].map(lambda v: passes_snap_filter(v, spec=spec))
    return df.loc[mask].copy()
