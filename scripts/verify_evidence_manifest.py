#!/usr/bin/env python3
"""
Verify shipped evidence against `releases/artifacts/MANIFEST.json` — audit C-29.

`.gitignore` ignores `reports/`, `ml/oof/` and `ml/experiments/` while ~20
artifacts are force-added into them. The consequence is that re-running an eval
produces no `git status` signal for anything it newly creates, so evidence
drifts silently — which already happened once (`gate_*.json` written at 15:46,
`current_baseline.json` frozen at 17:14, describing different states).

This script is the signal that was missing. It hashes every manifest entry and
reports content drift, missing files, and untracked evidence sitting in the
ignored directories.

    python scripts/verify_evidence_manifest.py            # verify
    python scripts/verify_evidence_manifest.py --update   # re-hash entries
    python scripts/verify_evidence_manifest.py --strict   # provisional drift fails too

Exit codes: 0 clean · 1 drift in a `frozen` entry (or any entry under
`--strict`) · 2 manifest itself is malformed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "releases" / "artifacts" / "MANIFEST.json"

# Directories that .gitignore hides, so new files in them are invisible to
# `git status`. Evidence living here must be manifest-listed.
WATCHED_DIRS = ("reports", "ml/experiments")

VALID_STATUSES = {"frozen", "provisional"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: missing manifest {MANIFEST_PATH}", file=sys.stderr)
        raise SystemExit(2)
    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
    except json.JSONDecodeError as exc:
        print(f"ERROR: {MANIFEST_PATH} is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if not isinstance(manifest.get("artifacts"), list):
        print("ERROR: manifest must have an 'artifacts' list", file=sys.stderr)
        raise SystemExit(2)
    return manifest


def validate_schema(manifest: dict) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for i, entry in enumerate(manifest["artifacts"]):
        where = f"artifacts[{i}]"
        for field in ("path", "sha256", "status", "produced_by", "description"):
            if not entry.get(field):
                problems.append(f"{where}: missing '{field}'")
        status = entry.get("status")
        if status and status not in VALID_STATUSES:
            problems.append(f"{where}: status {status!r} not in {sorted(VALID_STATUSES)}")
        path = entry.get("path")
        if path:
            if path in seen:
                problems.append(f"{where}: duplicate path {path!r}")
            seen.add(path)
    return problems


def verify(*, strict: bool) -> int:
    manifest = load_manifest()
    problems = validate_schema(manifest)
    if problems:
        for p in problems:
            print(f"MANIFEST SCHEMA: {p}", file=sys.stderr)
        return 2

    listed: set[str] = set()
    failures = 0
    warnings = 0

    for entry in manifest["artifacts"]:
        rel = entry["path"]
        listed.add(rel)
        path = REPO_ROOT / rel
        blocking = strict or entry["status"] == "frozen"

        if not path.exists():
            print(f"MISSING   {rel} (listed in manifest, not on disk)", file=sys.stderr)
            failures += blocking
            warnings += not blocking
            continue

        actual = sha256(path)
        if actual != entry["sha256"]:
            label = "DRIFT" if blocking else "drift"
            print(
                f"{label:9} {rel}\n"
                f"            expected {entry['sha256']}\n"
                f"            actual   {actual}",
                file=sys.stderr,
            )
            failures += blocking
            warnings += not blocking

    # Evidence sitting in an ignored directory but absent from the manifest is
    # exactly the invisible-drift case this exists to surface.
    unlisted: list[str] = []
    for watched in WATCHED_DIRS:
        base = REPO_ROOT / watched
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel not in listed:
                unlisted.append(rel)
    for rel in unlisted:
        print(f"UNLISTED  {rel} (in an ignored directory, not in the manifest)", file=sys.stderr)
    if strict:
        failures += len(unlisted)
    else:
        warnings += len(unlisted)

    total = len(manifest["artifacts"])
    if failures:
        print(
            f"\nEvidence manifest FAILED: {failures} blocking problem(s), "
            f"{warnings} non-blocking, {total} entries.\n"
            "Re-hash deliberate changes with:\n"
            "  python scripts/verify_evidence_manifest.py --update",
            file=sys.stderr,
        )
        return 1

    print(f"Evidence manifest OK: {total} entries, {warnings} non-blocking note(s).")
    return 0


def update() -> int:
    manifest = load_manifest()
    problems = validate_schema(manifest)
    if problems:
        for p in problems:
            print(f"MANIFEST SCHEMA: {p}", file=sys.stderr)
        return 2

    changed = 0
    for entry in manifest["artifacts"]:
        path = REPO_ROOT / entry["path"]
        if not path.exists():
            print(f"SKIP {entry['path']} (not on disk)", file=sys.stderr)
            continue
        actual = sha256(path)
        if actual != entry["sha256"]:
            entry["sha256"] = actual
            entry["bytes"] = path.stat().st_size
            changed += 1
            print(f"updated {entry['path']}")

    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{changed} entr{'y' if changed == 1 else 'ies'} re-hashed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="re-hash listed entries")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat provisional drift and unlisted evidence as failures",
    )
    args = parser.parse_args()
    return update() if args.update else verify(strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
