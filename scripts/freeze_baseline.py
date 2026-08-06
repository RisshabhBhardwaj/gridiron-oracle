#!/usr/bin/env python3
"""
Create a baseline manifest for the currently verified runtime.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.config import settings
from backend.app.services.runtime_status import RuntimeStatusService


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> int:
    root = ROOT
    baselines_dir = root / "releases" / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    report = RuntimeStatusService(
        db_url=settings.database_url,
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
        model_version=settings.model_version,
    ).build_report()

    if report.get("overall_status") == "blocked" and os.environ.get("ALLOW_BLOCKED_BASELINE") != "1":
        print(json.dumps(report, indent=2, sort_keys=True))
        print(
            "Refusing to freeze a blocked runtime baseline. "
            "Fix readiness checks first, or set ALLOW_BLOCKED_BASELINE=1 for an explicit emergency override.",
            file=sys.stderr,
        )
        return 2

    approved_pipeline_run_ids = [
        run_id.strip()
        for run_id in os.environ.get("APPROVED_PIPELINE_RUN_IDS", "").split(",")
        if run_id.strip()
    ]

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "model_version": settings.model_version,
        "product_mode": settings.product_mode,
        "mlflow_tracking_uri": settings.mlflow_tracking_uri,
        "baseline_manifest_path": settings.baseline_manifest_path,
        "integrity": report,
        "verification": {
            "local_verify_command": "make verify-local",
            "readiness_verify_command": "python scripts/verify_release_readiness.py",
        },
        "environment": {
            "DATABASE_URL": os.environ.get("DATABASE_URL", settings.database_url),
            "BASELINE_MANIFEST_PATH": settings.baseline_manifest_path,
            "ARTIFACT_INVALIDATION_PATH": settings.artifact_invalidation_path,
        },
        "projection_policy": {
            "approved_pipeline_run_ids": approved_pipeline_run_ids,
            "require_posterior_samples": True,
            "require_interval_columns": True,
        },
    }

    slug = settings.model_version.replace("/", "_").replace(":", "_")
    manifest_path = baselines_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{slug}.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    current_path = root / settings.baseline_manifest_path
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"Wrote baseline manifest: {manifest_path}")
    print(f"Updated current baseline: {current_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
