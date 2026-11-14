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
    python scripts/verify_evidence_manifest.py --register 'reports/eval_causal_*.csv' \\
        --produced-by scripts/run_causal_evals.py --description 'Per-cell causal eval'

Exit codes: 0 clean · 1 drift in a `frozen` entry (or any entry under
`--strict`) · 2 manifest itself is malformed.

Relationship to `releases/current_baseline.json`
------------------------------------------------
There are two manifests and they have different jobs. Two registries claiming
the same job is how a repo ends up with two digests for one file and no rule for
which wins, so the split is enforced here rather than described in prose:

* **`releases/current_baseline.json`** is authoritative for **serving
  artifacts** — the stacks under `ml/oof/` that `/draft`, materialization, ADP
  eval and conformal calibration resolve through `ml/artifact_manifest.py`.
  Selection there is SHA-pinned and fails closed.
* **This manifest** is authoritative for **evidence** — gate records, eval
  reports, ADP reports. It detects drift; it does not select anything.

Enforced consequences:

1. A path under a delegated directory (`ml/oof/`) may not appear here at all.
2. A path already pinned by the serving manifest may appear here only with
   ``"sha256_authority": "releases/current_baseline.json"`` in place of a literal
   ``sha256`` — one digest, stored once, so the two can never disagree.
3. Verification also asserts the serving manifest loads and every cell it pins
   resolves, so a single command covers both registries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANIFEST_PATH = REPO_ROOT / "releases" / "artifacts" / "MANIFEST.json"
SERVING_MANIFEST_RELPATH = "releases/current_baseline.json"
#: Check a candidate serving manifest (for example the state after a re-freeze,
#: built in a temp dir) instead of the committed one. The repo-relative name used
#: in messages and in ``sha256_authority`` does not change.
SERVING_MANIFEST_ENV = "EVIDENCE_SERVING_MANIFEST"


def serving_manifest_path() -> Path:
    override = os.environ.get(SERVING_MANIFEST_ENV)
    return Path(override) if override else REPO_ROOT / SERVING_MANIFEST_RELPATH

# Directories that .gitignore hides, so new files in them are invisible to
# `git status`. Evidence living here must be manifest-listed.
WATCHED_DIRS = ("reports", "ml/experiments")

# Directories owned by the serving manifest. Evidence entries must not reach into
# these; `ml/oof/` is pinned by releases/current_baseline.json and selected
# through ml/artifact_manifest.py.
DELEGATED_DIRS = ("ml/oof",)

VALID_STATUSES = {"frozen", "provisional"}

#: Sentinel a path may use instead of a literal digest when the serving manifest
#: already pins it.
AUTHORITY_KEY = "sha256_authority"


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


def serving_digests() -> dict[str, str]:
    """
    Digests from the serving manifest — the other registry's authority.

    Fails closed: if the serving manifest cannot be loaded, this verifier must
    not silently fall back to treating evidence digests as the only truth.
    """
    from ml.artifact_manifest import ManifestError, load_manifest

    try:
        manifest = load_manifest(serving_manifest_path(), root=REPO_ROOT)
    except ManifestError as exc:
        print(f"ERROR: serving manifest unreadable: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    return {k: v.lower() for k, v in manifest.digests.items()}


def verify_serving_manifest() -> int:
    """
    Assert the serving manifest resolves every cell it pins.

    Run from here so `make verify-evidence` covers both registries; a green
    evidence check beside a broken serving manifest would be misleading.
    """
    from ml.artifact_manifest import ManifestError, clear_cache, load_manifest

    clear_cache()
    try:
        manifest = load_manifest(serving_manifest_path(), root=REPO_ROOT)
        for stat, position in sorted(manifest.stack_index):
            manifest.resolve_stack(stat, position)
    except (ManifestError, FileNotFoundError) as exc:
        print(f"SERVING MANIFEST FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        f"Serving manifest OK: {len(manifest.stack_index)} pinned cells resolve "
        f"and match their SHA-256 ({SERVING_MANIFEST_RELPATH})."
    )
    # The causal evals back /backtest and /season. Until the re-freeze adds the
    # pin this is a note, not a failure; once pinned, every file must verify.
    if "causal_evals" not in (manifest.raw.get("artifacts") or {}):
        print("NOTE: 'causal_evals' is not pinned yet; /backtest fails closed until the re-freeze adds it.")
        return 0
    try:
        files = manifest.resolve_group("causal_evals")
    except (ManifestError, FileNotFoundError) as exc:
        print(f"SERVING MANIFEST FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"Causal evals OK: {len(files)} pinned files match their SHA-256.")
    return 0


def validate_schema(manifest: dict, pinned: dict[str, str] | None = None) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    pinned = pinned or {}
    for i, entry in enumerate(manifest["artifacts"]):
        where = f"artifacts[{i}]"
        path_value = entry.get("path") or ""

        # Rule 1: serving artifacts are not evidence entries.
        if any(
            path_value == d or path_value.startswith(d.rstrip("/") + "/")
            for d in DELEGATED_DIRS
        ):
            problems.append(
                f"{where}: {path_value!r} lives under a serving-manifest directory "
                f"({', '.join(DELEGATED_DIRS)}). Serving artifacts are pinned in "
                f"{SERVING_MANIFEST_RELPATH}; remove this entry rather than "
                "maintaining a parallel list."
            )

        # Rule 2: a serving-pinned path defers its digest instead of copying it.
        authority = entry.get(AUTHORITY_KEY)
        if path_value in pinned:
            if entry.get("sha256"):
                problems.append(
                    f"{where}: {path_value!r} is already pinned by "
                    f"{SERVING_MANIFEST_RELPATH}, so it must not carry its own "
                    f"'sha256'. Replace it with "
                    f'"{AUTHORITY_KEY}": "{SERVING_MANIFEST_RELPATH}".'
                )
            elif authority != SERVING_MANIFEST_RELPATH:
                problems.append(
                    f"{where}: {path_value!r} must set "
                    f'"{AUTHORITY_KEY}": "{SERVING_MANIFEST_RELPATH}".'
                )
        elif authority:
            problems.append(
                f"{where}: {path_value!r} sets {AUTHORITY_KEY} but "
                f"{SERVING_MANIFEST_RELPATH} does not pin it."
            )

        required = ("path", "status", "produced_by", "description")
        for field in required:
            if not entry.get(field):
                problems.append(f"{where}: missing '{field}'")
        if not entry.get("sha256") and not authority:
            problems.append(f"{where}: missing 'sha256' (or {AUTHORITY_KEY})")

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
    serving_rc = verify_serving_manifest()
    pinned = serving_digests()

    manifest = load_manifest()
    problems = validate_schema(manifest, pinned)
    if problems:
        for p in problems:
            print(f"MANIFEST SCHEMA: {p}", file=sys.stderr)
        return 2

    listed: set[str] = set()
    failures = serving_rc
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

        # A deferred entry is checked against the serving manifest's digest, so
        # there is exactly one recorded digest per file.
        if entry.get(AUTHORITY_KEY):
            expected = pinned[rel]
            source = f" (authority: {SERVING_MANIFEST_RELPATH})"
        else:
            expected = entry["sha256"]
            source = ""

        actual = sha256(path)
        if actual != expected:
            label = "DRIFT" if blocking else "drift"
            print(
                f"{label:9} {rel}{source}\n"
                f"            expected {expected}\n"
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
    pinned = serving_digests()
    manifest = load_manifest()
    problems = validate_schema(manifest, pinned)
    if problems:
        for p in problems:
            print(f"MANIFEST SCHEMA: {p}", file=sys.stderr)
        return 2

    changed = 0
    for entry in manifest["artifacts"]:
        path = REPO_ROOT / entry["path"]
        if entry.get(AUTHORITY_KEY):
            # Not ours to re-hash: re-freezing a serving artifact goes through
            # scripts/freeze_baseline.py.
            print(f"DEFER {entry['path']} (pinned by {SERVING_MANIFEST_RELPATH})")
            continue
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


def register(
    manifest: dict,
    patterns: list[str],
    *,
    produced_by: str,
    description: str,
    root: Path = REPO_ROOT,
) -> list[str]:
    """Add provisional entries for unlisted files matching ``patterns``.

    Only files under a watched directory (``reports/``, ``ml/experiments/``) are
    eligible: serving artifacts belong to the serving manifest, and anything
    else is not evidence this registry tracks. Already-listed paths are left
    alone. Returns the paths added.
    """
    if not produced_by or not description:
        raise ValueError("produced_by and description are required")
    listed = {entry["path"] for entry in manifest["artifacts"]}
    added: list[str] = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if rel in listed:
                continue
            if not any(rel.startswith(d.rstrip("/") + "/") for d in WATCHED_DIRS):
                raise ValueError(f"{rel} is not under a watched directory ({', '.join(WATCHED_DIRS)})")
            manifest["artifacts"].append(
                {
                    "path": rel,
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "status": "provisional",
                    "produced_by": produced_by,
                    "description": description,
                }
            )
            listed.add(rel)
            added.append(rel)
    return added


def register_cli(patterns: list[str], produced_by: str, description: str) -> int:
    manifest = load_manifest()
    try:
        added = register(manifest, patterns, produced_by=produced_by, description=description)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    manifest["artifacts"].sort(key=lambda entry: entry["path"])
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    for rel in added:
        print(f"registered {rel}")
    print(f"{len(added)} entr{'y' if len(added) == 1 else 'ies'} added (status: provisional).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="re-hash listed entries")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat provisional drift and unlisted evidence as failures",
    )
    parser.add_argument("--register", nargs="+", metavar="GLOB", help="add unlisted evidence files as provisional entries")
    parser.add_argument("--produced-by", help="with --register: the script that produced the files")
    parser.add_argument("--description", help="with --register: what the files are")
    args = parser.parse_args()
    if args.register:
        if not args.produced_by or not args.description:
            parser.error("--register needs --produced-by and --description")
        return register_cli(args.register, args.produced_by, args.description)
    return update() if args.update else verify(strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
