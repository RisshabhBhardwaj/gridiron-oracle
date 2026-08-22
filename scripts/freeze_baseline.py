#!/usr/bin/env python3
"""
Create a baseline manifest for the currently verified runtime.
"""

from __future__ import annotations

import hashlib
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
from ml.artifact_manifest import REQUIRED_SERVING_CELLS


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _carry_forward_artifact_pins(payload: dict, current_path: Path) -> None:
    """
    Preserve the ``artifacts`` list and recompute ``artifact_digests``.

    Artifact selection is manifest-pinned and fails closed (see
    ``ml/artifact_manifest.py``), so a manifest written without these keys would
    break every reader: draft, materialize, ADP eval and conformal calibration
    all resolve their inputs through them. This function never *invents* pins —
    it carries the existing artifact list forward and re-hashes what is on disk,
    so a re-freeze cannot silently drop the pins, and a changed artifact is
    recorded rather than hidden.

    Re-pointing the release at different artifacts remains a deliberate edit to
    the ``artifacts`` block.
    """
    if not current_path.exists():
        return
    try:
        previous = json.loads(current_path.read_text())
    except json.JSONDecodeError:
        return

    artifacts = previous.get("artifacts")
    if not artifacts:
        return
    payload["artifacts"] = artifacts
    for key in ("stack_policy", "known_issues", "reprojection_gates"):
        if key in previous and key not in payload:
            payload[key] = previous[key]

    def _walk(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from _walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from _walk(item)

    digests: dict[str, str] = {}
    missing: list[str] = []
    for relpath in sorted(set(_walk(artifacts))):
        path = ROOT / relpath
        if not path.is_file():
            missing.append(relpath)
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        digests[relpath] = digest.hexdigest()

    payload["artifact_digests"] = digests
    if missing:
        # Recorded rather than swallowed: a pinned artifact that is not on disk
        # is a release problem the operator has to see.
        payload["artifact_pins_missing"] = missing
        print(
            "WARNING: manifest lists artifacts that are not on disk: "
            + ", ".join(missing),
            file=sys.stderr,
        )


def _materialize_run_id() -> str:
    """Read the just-completed, full-matrix materialization provenance."""
    report_path = ROOT / "reports" / "materialize_stack_projections.json"
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Materialization report is required and unreadable: {exc}") from exc

    cells = {
        (str(cell.get("stat")), str(cell.get("position")))
        for cell in report.get("cells", [])
    }
    run_id = str(report.get("pipeline_run_id") or "").strip()
    if (
        report.get("dry_run")
        or report.get("git_commit") != _git(["rev-parse", "HEAD"])
        or cells != set(REQUIRED_SERVING_CELLS)
        or not run_id
    ):
        raise RuntimeError(
            "Materialization report is not a complete run for this HEAD; "
            "materialize all manifest-pinned cells before freezing (see "
            "ml.artifact_manifest.REQUIRED_SERVING_CELLS)."
        )
    return run_id


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
    if not approved_pipeline_run_ids:
        try:
            approved_pipeline_run_ids = [_materialize_run_id()]
        except RuntimeError as exc:
            print(f"Refusing to freeze baseline: {exc}", file=sys.stderr)
            return 2

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
            # Stack OOF materialization intentionally leaves percentile fields
            # null until a real per-player posterior is available.
            "require_posterior_samples": False,
            "require_interval_columns": True,
        },
    }

    current_path = root / settings.baseline_manifest_path
    _carry_forward_artifact_pins(payload, current_path)

    slug = settings.model_version.replace("/", "_").replace(":", "_")
    manifest_path = baselines_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{slug}.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"Wrote baseline manifest: {manifest_path}")
    print(f"Updated current baseline: {current_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
