"""
ml/feature_groups.py

Phase 4 A/B harness for discrete feature groups.

Governing rule: every group ships as an experiment with a recorded held-out
delta. Groups stay OFF FEATURE_COLS until MAE improves on a causal holdout.

Usage:
    python -m ml.feature_groups --group opp_adj_usage --holdout-season 2024 \\
        --position WR --target fantasy_ppr

Writes JSON under ml/experiments/feature_groups/.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from ml.utils import FEATURE_COLS, FEATURE_GROUPS, TARGET_COL_MAP, load_feature_matrix
from ml.feature_contract import assert_model_input_columns
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL

logger = logging.getLogger(__name__)

_EXPERIMENT_DIR = Path(__file__).resolve().parent / "experiments" / "feature_groups"


@dataclass
class FeatureGroupResult:
    group: str
    holdout_season: int
    position: str
    target: str
    n_train: int
    n_holdout: int
    baseline_mae: float
    treatment_mae: float
    delta_mae: float  # treatment - baseline; negative = improvement
    promote: bool
    feature_cols_added: list[str]
    evaluation_model: str
    promotion_eligible: bool
    limitation: str
    created_at: str


def resolve_feature_cols(groups: Optional[list[str]] = None) -> list[str]:
    """Return FEATURE_COLS plus any enabled Phase 4 groups."""
    cols = list(FEATURE_COLS)
    for name in groups or []:
        if name not in FEATURE_GROUPS:
            raise KeyError(f"Unknown feature group '{name}'. Known: {sorted(FEATURE_GROUPS)}")
        for c in FEATURE_GROUPS[name]:
            if c not in cols:
                cols.append(c)
    assert_model_input_columns(cols, consumer="feature-group experiment")
    return cols


def _prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    present = [c for c in feature_cols if c in df.columns]
    if target not in df.columns:
        raise KeyError(f"Target '{target}' missing from feature matrix")
    X = df[present].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce")
    mask = y.notna()
    return X.loc[mask].fillna(0.0), y.loc[mask]


def evaluate_feature_group(
    group: str,
    holdout_season: int,
    position: str,
    target: str = "fantasy_ppr",
    min_delta: float = -0.05,
    database_url: str = DEFAULT_HOST_DATABASE_URL,
    train_seasons: Optional[list[int]] = None,
) -> FeatureGroupResult:
    """
    Causal holdout: train on seasons < holdout_season, score on holdout_season.

    Uses Ridge only as an inexpensive screening proxy. Its output is explicitly
    ineligible for promotion because a Ridge delta is not evidence that the
    deployed LGBM/CatBoost stack will improve. A real-stack experiment must
    confirm a proxy win before a group can join the serving feature contract.
    """
    if group not in FEATURE_GROUPS:
        raise KeyError(f"Unknown feature group '{group}'. Known: {sorted(FEATURE_GROUPS)}")

    seasons = train_seasons or list(range(2019, holdout_season + 1))
    added = FEATURE_GROUPS[group]
    requested_cols = resolve_feature_cols([group])
    df = load_feature_matrix(
        database_url,
        seasons,
        position_filter=position.upper(),
        feature_cols=requested_cols,
    )

    if df.empty:
        raise RuntimeError(f"No rows for position={position}")

    target_col = TARGET_COL_MAP.get(target, target)
    if target_col not in df.columns:
        raise KeyError(f"Target '{target_col}' missing from feature matrix")

    train = df[df["season"] < holdout_season]
    hold = df[df["season"] == holdout_season]
    if train.empty or hold.empty:
        raise RuntimeError(
            f"Need train seasons < {holdout_season} and holdout={holdout_season}; "
            f"got train={len(train)} hold={len(hold)}"
        )

    baseline_cols = list(FEATURE_COLS)
    treatment_cols = resolve_feature_cols([group])

    Xb, yb = _prepare_xy(train, baseline_cols, target_col)
    Xh_b, yh = _prepare_xy(hold, baseline_cols, target_col)
    Xt, yt = _prepare_xy(train, treatment_cols, target_col)
    Xh_t, yh_t = _prepare_xy(hold, treatment_cols, target_col)

    common_idx = yh.index.intersection(yh_t.index)
    yh = yh.loc[common_idx]
    Xh_b = Xh_b.loc[common_idx]
    Xh_t = Xh_t.loc[common_idx]

    ridge_b = Ridge(alpha=10.0).fit(Xb, yb)
    ridge_t = Ridge(alpha=10.0).fit(Xt, yt)
    baseline_mae = float(mean_absolute_error(yh, ridge_b.predict(Xh_b)))
    treatment_mae = float(mean_absolute_error(yh, ridge_t.predict(Xh_t)))
    delta = treatment_mae - baseline_mae
    proxy_passed = delta <= min_delta

    return FeatureGroupResult(
        group=group,
        holdout_season=holdout_season,
        position=position.upper(),
        target=target,
        n_train=int(len(yb)),
        n_holdout=int(len(yh)),
        baseline_mae=baseline_mae,
        treatment_mae=treatment_mae,
        delta_mae=delta,
        # A proxy win is useful triage, but it is not a promotion decision.
        promote=False,
        feature_cols_added=added,
        evaluation_model="ridge_proxy",
        promotion_eligible=False,
        limitation=(
            "Ridge proxy result only; rerun with the deployed LGBM/CatBoost "
            "stack before enabling this group. proxy_passed=" + str(proxy_passed)
        ),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def save_result(result: FeatureGroupResult) -> Path:
    _EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = _EXPERIMENT_DIR / f"{result.group}_{result.position}_{result.holdout_season}_{stamp}.json"
    path.write_text(json.dumps(asdict(result), indent=2))
    logger.info("Wrote %s (delta_mae=%.4f promote=%s)", path, result.delta_mae, result.promote)
    return path


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="A/B evaluate a Phase 4 feature group")
    p.add_argument("--group", required=True, choices=sorted(FEATURE_GROUPS))
    p.add_argument("--holdout-season", type=int, required=True)
    p.add_argument("--position", default="WR")
    p.add_argument("--target", default="fantasy_ppr")
    p.add_argument("--min-delta", type=float, default=-0.05,
                   help="Promote if treatment_mae - baseline_mae <= this (default -0.05)")
    args = p.parse_args(argv)

    result = evaluate_feature_group(
        group=args.group,
        holdout_season=args.holdout_season,
        position=args.position,
        target=args.target,
        min_delta=args.min_delta,
    )
    path = save_result(result)
    print(json.dumps(asdict(result), indent=2))
    print(f"saved={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
