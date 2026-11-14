#!/usr/bin/env python3
"""
Add a new artifact group to ``releases/current_baseline.json``, with digests.

Use this before ``make freeze-baseline`` when a release gains a new group of
pinned files (for example ``causal_evals``, or ``base_models`` from decision
0012). ``freeze_baseline.py`` refuses a blocked runtime, and a required check
that reads the new group fails until the group has digests. Adding the paths
without digests would therefore make the freeze refuse; typing digests by hand
is error-prone. This script computes them from the files on disk.

Rules (each one fails closed):
  - the group must not exist yet (re-pointing a group is a different, deliberate
    change: see ``scripts/promote_rebuild_candidate.py``);
  - every path is repo-relative and under ``releases/candidates/``;
  - every file exists, is committed and unchanged in git (so the freeze's
    lineage check holds), and is not a Git LFS pointer.

It writes only ``artifacts[<group>]`` and the new paths' ``artifact_digests``.
The freeze then carries the block forward and re-hashes everything.

    python scripts/pin_artifact_group.py --group causal_evals \\
        releases/candidates/causal_20261114/evals/*
    --dry-run   validate and print the plan; write nothing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.artifact_manifest import sha256_of  # noqa: E402

CANDIDATES = "releases/candidates/"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class PinError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def _check_committed(root: Path, relpaths: list[str]) -> None:
    tracked = set(_git(root, "ls-files", "--", *relpaths).splitlines())
    untracked = [p for p in relpaths if p not in tracked]
    if untracked:
        raise PinError(f"not committed: {', '.join(untracked)}")
    dirty = _git(root, "status", "--porcelain", "--", *relpaths).strip()
    if dirty:
        raise PinError(f"uncommitted changes in the files to pin:\n{dirty}")


def pin_group(
    group: str,
    paths: list[str],
    *,
    root: Path = REPO_ROOT,
    dry_run: bool = False,
    check_git: bool = True,
) -> dict:
    if not group or not group.replace("_", "").isalnum():
        raise PinError(f"bad group name {group!r}")
    relpaths: list[str] = []
    for raw in paths:
        p = Path(raw)
        absolute = (p if p.is_absolute() else root / p).resolve()
        try:
            rel = absolute.relative_to(root.resolve()).as_posix()
        except ValueError:
            raise PinError(f"{raw} is outside the repository") from None
        if not rel.startswith(CANDIDATES):
            raise PinError(f"{rel} is not under {CANDIDATES}")
        if not absolute.is_file():
            raise PinError(f"{rel} is not a file")
        with absolute.open("rb") as fh:
            if fh.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
                raise PinError(f"{rel} is a Git LFS pointer; run `git lfs pull` first")
        relpaths.append(rel)
    relpaths = sorted(set(relpaths))
    if not relpaths:
        raise PinError("no files to pin")
    if check_git:
        _check_committed(root, relpaths)

    manifest_file = root / "releases" / "current_baseline.json"
    payload = json.loads(manifest_file.read_text())
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise PinError("current_baseline.json has no artifacts block; promote a candidate first")
    if group in artifacts:
        raise PinError(f"group {group!r} is already pinned; this script never re-points a group")
    digests = payload.setdefault("artifact_digests", {})
    clash = [p for p in relpaths if p in digests]
    if clash:
        raise PinError(f"already pinned under another group: {', '.join(clash)}")

    new_digests = {rel: sha256_of(root / rel) for rel in relpaths}
    summary = {"group": group, "files": len(relpaths), "digests": new_digests}
    if dry_run:
        return summary
    artifacts[group] = relpaths
    digests.update(new_digests)
    manifest_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    try:
        summary = pin_group(args.group, args.paths, dry_run=args.dry_run)
    except PinError as exc:
        print(f"pin_artifact_group: {exc}", file=sys.stderr)
        return 1
    verb = "would pin" if args.dry_run else "pinned"
    print(f"{verb} {summary['files']} files under artifacts.{summary['group']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
