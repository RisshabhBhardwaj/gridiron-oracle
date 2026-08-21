#!/usr/bin/env python3
"""Run strict model inference for one pre-built forward feature snapshot."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.config import settings
from ml.train import PipelineRunner
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--week", required=True, type=int)
    parser.add_argument("--feature-run-id", required=True)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_HOST_DATABASE_URL))
    parser.add_argument("--fast", action="store_true", help="Use Gaussian uncertainty approximation")
    args = parser.parse_args()
    # PipelineRunner's DB loader intentionally reads this process environment.
    # Make the explicit CLI argument authoritative for this invocation.
    os.environ["DATABASE_URL"] = args.database_url
    if settings.product_mode != "artifact_backed":
        raise SystemExit("Forward materialization requires PRODUCT_MODE=artifact_backed")
    with psycopg2.connect(normalize_dsn(args.database_url)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT as_of, feature_rows FROM forecast_runs WHERE pipeline_run_id = %s "
                "AND season = %s AND week = %s",
                (args.feature_run_id, args.season, args.week),
            )
            snapshot = cur.fetchone()
    if not snapshot:
        raise SystemExit("No matching forward feature snapshot, refusing inference")

    runner = PipelineRunner(
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
        fast=args.fast,
    )
    frame = runner.run(season=args.season, week=args.week)
    if frame.empty:
        raise SystemExit("Forward inference returned no rows")
    if frame["degraded"].fillna(True).any():
        raise SystemExit("Forward inference produced degraded rows, refusing materialization")
    run_ids = {str(value) for value in frame["pipeline_run_id"].dropna().unique()}
    if len(run_ids) != 1:
        raise SystemExit(f"Expected one projection pipeline run id, got {sorted(run_ids)}")
    model_run_id = run_ids.pop()
    summary = {
        "kind": "forward_projection_materialization",
        "feature_pipeline_run_id": args.feature_run_id,
        "feature_as_of": snapshot[0].isoformat(),
        "feature_rows": snapshot[1],
        "projection_rows": int(len(frame)),
    }
    with psycopg2.connect(normalize_dsn(args.database_url)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO forecast_runs
                     (pipeline_run_id, season, week, as_of, feature_rows, source_summary)
                     VALUES (%s, %s, %s, %s, %s, %s)""",
                (model_run_id, args.season, args.week, snapshot[0], len(frame), json.dumps(summary)),
            )
        conn.commit()
    print(json.dumps({"pipeline_run_id": model_run_id, **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
