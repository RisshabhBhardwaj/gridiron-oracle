"""
CLI parsing and execution for TFT training.

This keeps argparse and command-line side effects out of `ml.tft_training`,
which focuses on programmatic orchestration.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np

from ml.season_constants import assert_not_fitting_incomplete_season
from ml.tft_model import TFTConfig, logger
from ml.utils import TARGET_COL_MAP, _parse_seasons, load_feature_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train TFT base learner with walk-forward CV.",
    )
    parser.add_argument(
        "--seasons", required=True,
        help="Season range or list: '2018-2024' or '2022 2023 2024'.",
    )
    parser.add_argument(
        "--target", default="receiving_yards",
        choices=list(TARGET_COL_MAP),
        help="Prediction target.",
    )
    parser.add_argument(
        "--position", default="WR",
        help="Position filter (e.g. WR, RB, QB, TE). Pass 'all' for no filter.",
    )
    parser.add_argument(
        "--out-dir", default="ml/oof", dest="out_dir",
        help="Directory for OOF CSV output.",
    )
    parser.add_argument(
        "--mlflow-uri", default=None, dest="mlflow_uri",
        help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI env var.",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", dest="no_mlflow",
        help="Disable MLflow logging.",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=64, dest="hidden_size",
        help="TFT hidden size (default: 64). Overridden by Optuna if n-trials > 0.",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=30, dest="max_epochs",
        help="Max training epochs per fold (default: 30).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, dest="n_trials",
        help="Optuna trial count (default: 20). Pass 0 to skip tuning.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Development shortcut: sets --max-epochs=3 and --n-trials=0.",
    )
    parser.add_argument(
        "--checkpoint", default=None, dest="checkpoint",
        help=(
            "Path to a Lightning .ckpt file for incremental fine-tuning. "
            "Applied only to the last (most data-rich) fold. "
            "Example: ml/checkpoints/tft/receiving_yards_fold5/best-epoch=04-val_loss=0.0142.ckpt"
        ),
    )
    parser.add_argument(
        "--db-url", default=None, dest="db_url",
        help="PostgreSQL connection URL. Defaults to DATABASE_URL env var.",
    )
    return parser


def _write_completion_checkpoint(target: str) -> Path:
    ckpt_dir = Path("ml/checkpoints/done")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / f"tft_{target}.done"
    ckpt_file.touch()
    logger.info("TFT checkpoint written: %s (stat %s complete)", ckpt_file, target)
    return ckpt_file


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    max_epochs = 3 if args.fast else args.max_epochs
    n_trials = 0 if args.fast else args.n_trials

    db_url = args.db_url or os.environ.get("DATABASE_URL", "")
    if not db_url:
        parser.error("--db-url or DATABASE_URL env var is required")

    seasons = _parse_seasons(args.seasons)
    # Trainer entrypoint guarantee: an incomplete season must never enter a
    # walk-forward fold. `_parse_seasons` already raises on out-of-range input;
    # this is the explicit assert the pre-Week-1 claim rests on (audit C-28).
    assert_not_fitting_incomplete_season(seasons)
    position = None if args.position.lower() == "all" else args.position
    mlflow_uri = "" if args.no_mlflow else args.mlflow_uri

    cfg = TFTConfig(
        target=args.target,
        hidden_size=args.hidden_size,
        max_epochs=max_epochs,
    )

    if args.fast:
        logger.info("--fast mode: max_epochs=%d  n_optuna_trials=%d", max_epochs, n_trials)

    df = load_feature_matrix(db_url, seasons, position_filter=position)
    if df.empty:
        logger.error("No data loaded — aborting.")
        raise SystemExit(1)

    from ml.tft_training import train

    result = train(
        df=df,
        seasons=seasons,
        target=args.target,
        config=cfg,
        n_optuna_trials=n_trials,
        position_filter=position,
        mlflow_tracking_uri=mlflow_uri,
        out_dir=Path(args.out_dir),
        checkpoint_path=args.checkpoint,
    )

    _write_completion_checkpoint(args.target)

    print(
        f"\nTFT training complete:\n"
        f"  Folds:      {len(result.fold_results)}\n"
        f"  Agg MAE:    {np.mean([fr.mae for fr in result.fold_results]):.3f}\n"
        f"  Agg RMSE:   {np.mean([fr.rmse for fr in result.fold_results]):.3f}\n"
        f"  MLflow ID:  {result.run_id or 'N/A'}\n"
        f"  OOF path:   {result.oof_path or 'N/A'}\n"
    )


__all__ = ["build_parser", "main"]
