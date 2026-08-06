#!/usr/bin/env python3
"""
Fail fast when the runtime is not ready for artifact-backed deployment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.config import settings
from backend.app.services.runtime_status import RuntimeStatusService


def main() -> int:
    report = RuntimeStatusService(
        db_url=settings.database_url,
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
        model_version=settings.model_version,
    ).build_report()

    print(json.dumps(report, indent=2, sort_keys=True))

    if settings.product_mode == "artifact_backed":
        return 0 if report["overall_status"] == "ok" else 1

    return 0 if report["overall_status"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
