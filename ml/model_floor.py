"""
ml/model_floor.py

Phase 5 — single well-regularized model floor.

Before reintroducing XGB/LGBM/CatBoost/TFT stacking, measure a Ridge floor
on causal OOF (train seasons < holdout). Stack layers are only re-added where
they beat this floor on held-out MAE.

Usage:
    python -m ml.model_floor --holdout-season 2024 --position WR --target fantasy_ppr
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

from ml.feature_groups import resolve_feature_cols
from ml.utils import TARGET_COL_MAP, load_feature_matrix
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL

logger = logging.getLogger(__name__)
_OUT = Path(__file__).resolve().parent / "experiments" / "model_floor"


@dataclass
class ModelFloorResult:
    holdout_season: int
    position: str
    target: str
    alpha: float
    n_train: int
    n_holdout: int
    mae: float
    rmse: float
    feature_groups: list[str]
    n_features: int
    created_at: str


def fit_ridge_floor(
    holdout_season: int,
    position: str,
    target: str = "fantasy_ppr",
    alpha: float = 10.0,
    feature_groups: Optional[list[str]] = None,
    database_url: str = DEFAULT_HOST_DATABASE_URL,
) -> ModelFloorResult:
    seasons = list(range(2019, holdout_season + 1))
    from ml import utils as ml_utils

    cols = resolve_feature_cols(feature_groups)
    original = list(ml_utils.FEATURE_COLS)
    try:
        ml_utils.FEATURE_COLS = list(dict.fromkeys(original + cols))
        df = load_feature_matrix(database_url, seasons, position_filter=position.upper())
    finally:
        ml_utils.FEATURE_COLS = original

    present = [c for c in cols if c in df.columns]
    train = df[df["season"] < holdout_season]
    hold = df[df["season"] == holdout_season]
    target_col = TARGET_COL_MAP.get(target, target)
    if target_col not in df.columns:
        raise KeyError(f"Missing target {target_col}")

    def xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = frame[present].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y = pd.to_numeric(frame[target_col], errors="coerce")
        mask = y.notna()
        return X.loc[mask], y.loc[mask]

    Xtr, ytr = xy(train)
    Xh, yh = xy(hold)
    if Xtr.empty or Xh.empty:
        raise RuntimeError(
            f"Empty train/holdout for {position}/{target} holdout={holdout_season} "
            f"(train={len(Xtr)} hold={len(Xh)})"
        )

    model = Ridge(alpha=alpha).fit(Xtr, ytr)
    pred = model.predict(Xh)
    mae = float(mean_absolute_error(yh, pred))
    rmse = float(np.sqrt(np.mean((yh.to_numpy() - pred) ** 2)))
    return ModelFloorResult(
        holdout_season=holdout_season,
        position=position.upper(),
        target=target,
        alpha=alpha,
        n_train=int(len(ytr)),
        n_holdout=int(len(yh)),
        mae=mae,
        rmse=rmse,
        feature_groups=list(feature_groups or []),
        n_features=len(present),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Fit Ridge model floor on causal holdout")
    p.add_argument("--holdout-season", type=int, required=True)
    p.add_argument("--position", default="WR")
    p.add_argument("--target", default="fantasy_ppr")
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--groups", nargs="*", default=[], help="Optional Phase 4 feature groups")
    args = p.parse_args(argv)

    result = fit_ridge_floor(
        holdout_season=args.holdout_season,
        position=args.position,
        target=args.target,
        alpha=args.alpha,
        feature_groups=args.groups or None,
    )
    _OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = _OUT / f"ridge_{result.position}_{result.holdout_season}_{stamp}.json"
    path.write_text(json.dumps(asdict(result), indent=2))
    print(json.dumps(asdict(result), indent=2))
    print(f"saved={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
