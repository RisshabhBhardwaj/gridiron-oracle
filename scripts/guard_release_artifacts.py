#!/usr/bin/env python3
"""
Refuse to delete artifacts the release manifest pins.

``ml/train_all_models.sh`` used to open a normal (non-``--resume``) run with

    rm -f "$OOF_DIR"/*.csv

which deletes the *serving* artifact set — every file
``releases/current_baseline.json`` points at, including the tracked stacks that
back the shipped projections. "Clean start" and "destroy the release" were the
same command.

Clean-slate behaviour is now opt-in, and even then it routes through this guard:

    python scripts/guard_release_artifacts.py --check-purge ml/oof

exits non-zero and names the offending files if a purge of that directory would
remove anything pinned. Callers may pass ``--allow-unpinned-purge`` to delete
only the unpinned files.

Exit codes:
    0 — nothing pinned would be destroyed
    3 — the purge would destroy pinned release artifacts (refused)
    4 — the manifest could not be loaded (refused; fail closed)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.artifact_manifest import ManifestError, load_manifest  # noqa: E402

EXIT_REFUSED = 3
EXIT_NO_MANIFEST = 4


def protected_in(directory: Path, *, manifest_path: Path | None = None) -> list[str]:
    """Repo-relative paths under *directory* that the manifest pins."""
    manifest = load_manifest(manifest_path)
    prefix = directory.resolve()
    protected: list[str] = []
    for relpath in sorted(manifest.protected_relpaths()):
        absolute = (manifest.root / relpath).resolve()
        if absolute == prefix or prefix in absolute.parents:
            protected.append(relpath)
    return protected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-purge",
        metavar="DIR",
        default=None,
        help="Directory a caller intends to purge.",
    )
    parser.add_argument(
        "--list-cells",
        action="store_true",
        help="Print the release cell matrix as 'stat:POSITION' lines and exit. "
             "This is the authoritative list of cells that must have a stack "
             "artifact — trainers iterate a wider stat×position cross product "
             "in which some combinations legitimately produce nothing.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest path (defaults to releases/current_baseline.json).",
    )
    parser.add_argument(
        "--list-protected",
        action="store_true",
        help="Print the protected paths and exit 0 without judging the purge.",
    )
    args = parser.parse_args()

    manifest_arg = Path(args.manifest) if args.manifest else None

    if args.list_cells:
        try:
            manifest = load_manifest(manifest_arg)
        except ManifestError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return EXIT_NO_MANIFEST
        for stat, position in sorted(manifest.stack_index):
            print(f"{stat}:{position}")
        return 0

    if not args.check_purge:
        parser.error("one of --check-purge or --list-cells is required")

    directory = Path(args.check_purge)
    if not directory.is_absolute():
        directory = ROOT / directory

    try:
        protected = protected_in(directory, manifest_path=manifest_arg)
    except ManifestError as exc:
        print(f"REFUSED: cannot verify release artifacts — {exc}", file=sys.stderr)
        return EXIT_NO_MANIFEST

    if args.list_protected:
        for relpath in protected:
            print(relpath)
        return 0

    if protected:
        print(
            f"REFUSED: purging {directory} would delete "
            f"{len(protected)} release artifact(s) pinned by the manifest:",
            file=sys.stderr,
        )
        for relpath in protected:
            print(f"  {relpath}", file=sys.stderr)
        print(
            "\nThese back the currently shipped projections. Restore-from-archive "
            "is the only recovery path — they must never be regenerated in place. "
            "If you genuinely want a clean slate, move the release artifacts aside "
            "and re-freeze the manifest deliberately.",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    print(f"OK: no manifest-pinned artifacts under {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
