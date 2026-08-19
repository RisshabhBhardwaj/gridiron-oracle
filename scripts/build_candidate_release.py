#!/usr/bin/env python3
"""Build a SHA-pinned, non-promoted release manifest for causal artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.artifact_manifest import REQUIRED_SERVING_CELLS, sha256_of


def _one(directory: Path, pattern: str) -> Path:
    found = sorted(directory.glob(pattern))
    if len(found) != 1:
        raise ValueError(f"Expected exactly one {pattern!r} in {directory}; found {[p.name for p in found]}")
    return found[0]


def build(artifact_dir: Path, output: Path, release_status: str) -> dict[str, object]:
    artifact_dir = artifact_dir.resolve()
    entries: dict[str, list[str]] = {"stacks": [], "ridge_coefficients": [], "serving_parity": []}
    for stat, position in REQUIRED_SERVING_CELLS:
        stack = _one(artifact_dir, f"stack_{stat}_{position}_*.csv")
        coef = _one(artifact_dir, f"ridge_{stat}_{position}_coefs.json")
        parity = _one(artifact_dir, f"serving_divergence_{stat}_{position}.json")
        for kind, path in (("stacks", stack), ("ridge_coefficients", coef), ("serving_parity", parity)):
            entries[kind].append(path.relative_to(ROOT).as_posix())
    manifest = {
        "candidate": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_rebuild": "ml/oof/rebuild_20260809T231500",
        "artifacts": entries,
        "artifact_digests": {
            relpath: sha256_of(ROOT / relpath)
            for values in entries.values() for relpath in values
        },
        "release_status": release_status,
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--release-status",
        default="candidate_only_pending_wave_04_promotion_and_materialization",
        help="Explicit non-promoted candidate status recorded in the manifest.",
    )
    args = parser.parse_args()
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else ROOT / args.artifact_dir
    output = args.output if args.output.is_absolute() else ROOT / args.output
    print(json.dumps(build(artifact_dir, output, args.release_status), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
