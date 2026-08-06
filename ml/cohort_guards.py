"""
Training cohort sanity checks for stat/position-specific model training.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

STRICT_POSITION_TARGETS: dict[str, str] = {
    "passing_yards": "QB",
    "pass_attempts": "QB",
    "completions": "QB",
    "passing_tds": "QB",
    "interceptions": "QB",
}

MAX_ZERO_RATE: dict[str, float] = {
    "passing_yards": 0.20,
    "pass_attempts": 0.20,
    "completions": 0.20,
}


def validate_training_cohort(
    df: pd.DataFrame,
    *,
    target: str,
    target_col: str,
    position_filter: str | None,
    source: str,
) -> None:
    """Fail fast when a training cohort looks structurally wrong."""
    if df.empty:
        raise ValueError(f"{source}: training dataframe is empty for target={target}")

    expected_position = STRICT_POSITION_TARGETS.get(target)
    if expected_position:
        if position_filter != expected_position:
            raise ValueError(
                f"{source}: target={target} must be trained with position_filter={expected_position}, "
                f"got {position_filter!r}"
            )
        observed_positions = sorted(df["position"].dropna().astype(str).unique().tolist())
        if observed_positions != [expected_position]:
            raise ValueError(
                f"{source}: target={target} has contaminated positions {observed_positions}; "
                f"expected only {expected_position}"
            )

    zero_threshold = MAX_ZERO_RATE.get(target)
    if zero_threshold is not None:
        zero_rate = float((df[target_col].fillna(0.0) <= 0.0).mean())
        if zero_rate > zero_threshold:
            raise ValueError(
                f"{source}: target={target} zero-rate {zero_rate:.1%} exceeds "
                f"threshold {zero_threshold:.1%}; likely cohort contamination or malformed targets"
            )
        logger.info(
            "%s cohort check: target=%s position=%s zero-rate=%.2f%%",
            source,
            target,
            position_filter,
            zero_rate * 100.0,
        )
