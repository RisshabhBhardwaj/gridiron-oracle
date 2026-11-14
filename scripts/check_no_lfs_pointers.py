#!/usr/bin/env python3
"""Fail if any pinned model file is a Git LFS pointer instead of the model bytes.

Standard library only, so it runs on the box with plain ``python3`` after the code sync.
Checks every file under ``releases/candidates/*/models/``. Exit 0: all real bytes.
Exit 1: a pointer was found, or there are no model files at all (an empty tree must not pass).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def find_pointers(root: Path) -> tuple[list[Path], int]:
    """Return (pointer files, number of model files checked) under ``root``."""
    pointers: list[Path] = []
    checked = 0
    for path in sorted(root.glob("releases/candidates/*/models/*")):
        if not path.is_file():
            continue
        checked += 1
        with path.open("rb") as handle:
            if handle.read(len(POINTER_PREFIX)) == POINTER_PREFIX:
                pointers.append(path)
    return pointers, checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    pointers, checked = find_pointers(args.root)
    if checked == 0:
        print(f"no model files found under {args.root}/releases/candidates/*/models/", file=sys.stderr)
        return 1
    if pointers:
        for path in pointers:
            print(f"Git LFS pointer, not model bytes: {path.relative_to(args.root)}", file=sys.stderr)
        print("On the Mac run `git lfs pull`, check `git lfs ls-files`, then sync again.", file=sys.stderr)
        return 1
    print(f"ok: {checked} model files, none is an LFS pointer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
