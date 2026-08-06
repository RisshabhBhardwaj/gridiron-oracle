#!/usr/bin/env python3
"""
Append invalidation rules for known-bad projection history.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.config import settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-run-id", action="append", default=[])
    parser.add_argument("--position")
    parser.add_argument("--stat")
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    path = ROOT / settings.artifact_invalidation_path
    payload = {"invalid_pipeline_run_ids": [], "invalid_projection_targets": []}
    if path.exists():
        payload = json.loads(path.read_text())

    run_ids = payload.setdefault("invalid_pipeline_run_ids", [])
    for run_id in args.pipeline_run_id:
        run_id = run_id.strip()
        if run_id and run_id not in run_ids:
            run_ids.append(run_id)

    if args.position and args.stat:
        rule = {
            "position": args.position.strip(),
            "stat": args.stat.strip(),
            "reason": args.reason.strip(),
        }
        targets = payload.setdefault("invalid_projection_targets", [])
        if rule not in targets:
            targets.append(rule)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Updated invalidation manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
