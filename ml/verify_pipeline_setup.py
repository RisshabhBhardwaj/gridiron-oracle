#!/usr/bin/env python3
"""
ml/verify_pipeline_setup.py

Verification script for the projection pipeline. Ensures all prerequisites
are in place before running projections with full base-learner inference
(XGB, LGB, CatBoost, TFT) and stacking.

Run:
    python -m ml.verify_pipeline_setup

Checks:
  1. Environment: MLFLOW_TRACKING_URI, DATABASE_URL
  2. MLflow: reachable, experiments exist for each stat
  3. Base models: XGB, LGB, CatBoost, TFT have finished runs per stat
  4. Ridge coefs: ml/oof/ridge_{stat}_coefs.json exists with xgb, lgbm, catboost, tft
  5. feature_matrix: has data for recent season/week
  6. TFT prior rows: can load prior feature_matrix rows (TFT sequence requirement)
  7. Optional: dry-run projection (--smoke-test)

Exit code: 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Stats the pipeline uses (must match train.py _DEFAULT_STATS)
_STATS = [
    "pass_attempts", "completions", "passing_yards", "passing_tds", "interceptions",
    "rushing_yards", "rushing_tds", "carries",
    "receiving_yards", "receiving_tds", "receptions", "targets",
    "fantasy_ppr", "fumbles",
]

_LEARNERS = ["xgb", "lgbm", "catboost", "tft"]
_OOF_DIR = Path(__file__).resolve().parent / "oof"


def _ok(msg: str) -> None:
    logger.info("  ✓ %s", msg)


def _fail(msg: str) -> None:
    logger.error("  ✗ %s", msg)


def _warn(msg: str) -> None:
    logger.warning("  ⚠ %s", msg)


def check_env() -> tuple[bool, str | None, str | None]:
    """Check MLFLOW_TRACKING_URI and DATABASE_URL."""
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    db_url = os.environ.get("DATABASE_URL", "")

    if not mlflow_uri:
        _fail("MLFLOW_TRACKING_URI not set — pipeline will use Kalman proxy for all base learners")
        return False, None, None
    _ok(f"MLFLOW_TRACKING_URI={mlflow_uri}")

    if not db_url:
        _fail("DATABASE_URL not set — cannot load feature_matrix or run projections")
        return False, mlflow_uri, None
    _ok("DATABASE_URL is set")

    return True, mlflow_uri, db_url


def check_mlflow_reachable(mlflow_uri: str) -> bool:
    """Check MLflow server is reachable."""
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.search_experiments(max_results=1)
        _ok("MLflow server reachable")
        return True
    except Exception as e:
        _fail(f"MLflow unreachable: {e}")
        return False


def check_experiments(mlflow_uri: str) -> dict[str, dict[str, bool]]:
    """Check each stat has xgb, lgbm, catboost, tft experiments with finished runs."""
    import mlflow
    mlflow.set_tracking_uri(mlflow_uri)

    results: dict[str, dict[str, bool]] = {}
    for stat in _STATS:
        results[stat] = {}
        for learner in _LEARNERS:
            exp_name = f"{learner}_{stat}"
            try:
                exp = mlflow.get_experiment_by_name(exp_name)
                if exp is None:
                    results[stat][learner] = False
                    continue
                runs = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=1,
                    filter_string="status = 'FINISHED'",
                )
                results[stat][learner] = not runs.empty
            except Exception:
                results[stat][learner] = False
    return results


def check_ridge_coefs() -> dict[str, bool]:
    """Check ridge_{stat}_coefs.json exists and has xgb, lgbm, catboost, tft."""
    results: dict[str, bool] = {}
    for stat in _STATS:
        path = _OOF_DIR / f"ridge_{stat}_coefs.json"
        if not path.exists():
            results[stat] = False
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            has_all = all(k in data for k in _LEARNERS) and "intercept" in data
            results[stat] = has_all
        except Exception:
            results[stat] = False
    return results


def check_feature_matrix(db_url: str) -> bool:
    """Check feature_matrix has data for a recent season."""
    try:
        import psycopg2
        from scraper.adapters.nflreadpy_adapter import _psycopg2_dsn

        dsn = _psycopg2_dsn(db_url)
        conn = psycopg2.connect(dsn)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT season, week, COUNT(*) FROM feature_matrix "
                "GROUP BY season, week ORDER BY season DESC, week DESC LIMIT 5"
            )
            rows = cur.fetchall()
            if not rows:
                _fail("feature_matrix is empty")
                return False
            _ok(f"feature_matrix has data: {rows[0][2]} rows for S{rows[0][0]}W{rows[0][1]}")
            return True
        finally:
            conn.close()
    except Exception as e:
        _fail(f"feature_matrix check failed: {e}")
        return False


def check_tft_prior_rows(db_url: str) -> bool:
    """Check we can load prior feature_matrix rows (required for TFT inference)."""
    try:
        from ml.utils import load_feature_matrix_prior

        # Use a sample query: prior rows for season=2025 week=1
        load_feature_matrix_prior(
            db_url, target_season=2025, target_week=1,
            player_ids=["00-0031234"],  # placeholder; may return empty
            position_filter="WR",
        )
        _ok("load_feature_matrix_prior works (TFT sequence loader)")
        return True
    except Exception as e:
        _fail(f"load_feature_matrix_prior failed: {e}")
        return False


def run_smoke_test() -> bool:
    """Run a quick dry-run projection to verify the full pipeline."""
    try:
        from ml.train import PipelineRunner

        runner = PipelineRunner(
            fast=True,
            mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", ""),
        )
        df = runner.dry_run(2025, 1, ["WR"], ["receiving_yards"])
        if df.empty:
            _warn("Dry-run returned empty DataFrame (synthetic data path)")
        else:
            _ok(f"Dry-run produced {len(df)} projections")
        return True
    except Exception as e:
        _fail(f"Dry-run failed: {e}")
        return False


def main() -> int:
    smoke_test = "--smoke-test" in sys.argv

    logger.info("═══ Pipeline Setup Verification ═══\n")

    # 1. Environment
    logger.info("1. Environment")
    env_ok, mlflow_uri, db_url = check_env()
    if not env_ok:
        logger.info("\nFix: export MLFLOW_TRACKING_URI and DATABASE_URL")
        return 1
    logger.info("")

    # 2. MLflow reachable
    logger.info("2. MLflow")
    if not check_mlflow_reachable(mlflow_uri):
        logger.info("\nFix: start MLflow (mlflow ui) or set correct MLFLOW_TRACKING_URI")
        return 1
    logger.info("")

    # 3. Experiments + runs
    logger.info("3. Base models (XGB, LGB, CatBoost, TFT) per stat")
    exp_results = check_experiments(mlflow_uri)
    missing: list[str] = []
    for stat in _STATS:
        bad = [learner for learner in _LEARNERS if not exp_results[stat].get(learner, False)]
        if bad:
            missing.append(f"{stat}: {', '.join(bad)}")
    if missing:
        for m in missing[:5]:
            _warn(m)
        if len(missing) > 5:
            _warn(f"... and {len(missing) - 5} more")
        logger.info("")
        logger.info("Fix: train missing models (ml/train_all_models.sh or per-stat)")
    else:
        _ok(f"All {len(_STATS)} stats have XGB, LGB, CatBoost, TFT models")
    logger.info("")

    # 4. Ridge coefs
    logger.info("4. Ridge stacking coefs (ml/oof/ridge_*_coefs.json)")
    ridge_results = check_ridge_coefs()
    ridge_missing = [s for s in _STATS if not ridge_results.get(s, False)]
    if ridge_missing:
        _warn(f"Missing or incomplete for: {', '.join(ridge_missing[:5])}")
        if len(ridge_missing) > 5:
            _warn(f"... and {len(ridge_missing) - 5} more")
        logger.info("")
        logger.info("Fix: run stacking_ensemble with TFT OOF included")
    else:
        _ok(f"All {len(_STATS)} stats have ridge coefs with xgb, lgbm, catboost, tft")
    logger.info("")

    # 5. feature_matrix
    logger.info("5. feature_matrix (DB)")
    if not check_feature_matrix(db_url):
        logger.info("\nFix: run make ingest or pipeline/orchestrator")
        return 1
    logger.info("")

    # 6. TFT prior rows
    logger.info("6. TFT prior rows (load_feature_matrix_prior)")
    if not check_tft_prior_rows(db_url):
        return 1
    logger.info("")

    # 7. Optional smoke test
    if smoke_test:
        logger.info("7. Smoke test (dry-run projection)")
        if not run_smoke_test():
            return 1
        logger.info("")

    logger.info("═══ Verification complete ═══")
    if missing or ridge_missing:
        logger.info("")
        logger.info("Some checks had warnings. Projections may still work if you only")
        logger.info("need stats/positions that passed. Re-run stacking if TFT was added")
        logger.info("after the last stacking run.")
        return 0  # warnings, not hard fail
    logger.info("")
    logger.info("All checks passed. Run projections with:")
    logger.info("  python -m ml.train --season 2025 --week 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
