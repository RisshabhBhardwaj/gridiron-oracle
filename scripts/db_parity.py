#!/usr/bin/env python3
"""
Row counts and content checksums for the public schema, for database parity checks.

    DATABASE_URL=postgresql://... python scripts/db_parity.py snapshot --out FILE
    python scripts/db_parity.py diff FILE_A FILE_B

The snapshot is read in one read-only REPEATABLE READ transaction. The session is
pinned to UTC and ISO dates, so the same rows give the same text on any server.
Checksums are md5 over the row text, taken only for CHECKSUM_TABLES. Each one
builds the whole table as a single string in server memory, so it is costly on
small hosted tiers.

DATABASE_URL is read from the environment. It is never printed or written, and
failures print the exception class name only. A SQLAlchemy-style driver suffix
(postgresql+asyncpg://) is dropped so psycopg2 accepts the URL.

Exit codes. snapshot: 0 written, 2 failed. diff: 0 match, 1 difference, 2 bad input.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from psycopg2 import sql

SCHEMA_VERSION = 1
CHECKSUM_TABLES = ("feature_matrix", "projections")
SESSION_SETUP = (
    "SET LOCAL TIME ZONE 'UTC'",
    "SET LOCAL DateStyle = 'ISO, YMD'",
    "SET LOCAL extra_float_digits = 1",
)
_CHECKSUM_SQL = "SELECT md5(string_agg(t::text, '|' ORDER BY t::text)) FROM {} t"
_DRIVER_SUFFIX = re.compile(r"^postgresql\+[A-Za-z0-9_]+://")


def _psycopg2_dsn(dsn: str) -> str:
    return _DRIVER_SUFFIX.sub("postgresql://", dsn, count=1)


def _public_tables(cur: Any) -> list[str]:
    cur.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY table_name"
    )
    return [row[0] for row in cur.fetchall()]


def _row_count(cur: Any, table: str) -> int:
    cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table)))
    return int(cur.fetchone()[0])


def _checksum(cur: Any, table: str) -> str | None:
    cur.execute(sql.SQL(_CHECKSUM_SQL).format(sql.Identifier(table)))
    value = cur.fetchone()[0]
    return None if value is None else str(value)


def build_snapshot(conn: Any) -> dict[str, Any]:
    """Return row counts for every public base table and checksums for CHECKSUM_TABLES.

    A checksum is null when its table is absent or empty.
    """
    conn.set_session(readonly=True, isolation_level="REPEATABLE READ", autocommit=False)
    cur = conn.cursor()
    for statement in SESSION_SETUP:
        cur.execute(statement)
    tables = _public_tables(cur)
    row_counts = {name: _row_count(cur, name) for name in tables}
    checksums = {
        name: (_checksum(cur, name) if name in tables else None)
        for name in CHECKSUM_TABLES
    }
    return {"schema_version": SCHEMA_VERSION, "row_counts": row_counts, "checksums": checksums}


def diff_snapshots(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """Return one line per difference. An empty list means the snapshots match."""
    diffs: list[str] = []
    if left.get("schema_version") != right.get("schema_version"):
        diffs.append(f"schema_version: {left.get('schema_version')!r} != {right.get('schema_version')!r}")
    for section in ("row_counts", "checksums"):
        a = left.get(section, {})
        b = right.get(section, {})
        for key in sorted(set(a) | set(b)):
            label = f"{section}.{key}"
            if key not in a:
                diffs.append(f"{label}: only in B")
            elif key not in b:
                diffs.append(f"{label}: only in A")
            elif a[key] != b[key]:
                diffs.append(f"{label}: {a[key]!r} != {b[key]!r}")
    return diffs


def _connect(dsn: str) -> Any:
    import psycopg2

    return psycopg2.connect(_psycopg2_dsn(dsn))


def _run_snapshot(out: Path) -> int:
    dsn = os.environ.get("DATABASE_URL", "").strip()
    if not dsn:
        print("error: DATABASE_URL is not set", file=sys.stderr)
        return 2
    try:
        conn = _connect(dsn)
    except Exception as exc:
        print(f"error: connect failed ({type(exc).__name__})", file=sys.stderr)
        return 2
    try:
        snapshot = build_snapshot(conn)
    except Exception as exc:
        print(f"error: snapshot failed ({type(exc).__name__})", file=sys.stderr)
        return 2
    finally:
        conn.close()
    out.write_text(json.dumps(snapshot, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


def _run_diff(file_a: Path, file_b: Path) -> int:
    try:
        left = json.loads(file_a.read_text(encoding="utf-8"))
        right = json.loads(file_b.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read snapshot ({type(exc).__name__})", file=sys.stderr)
        return 2
    diffs = diff_snapshots(left, right)
    if not diffs:
        print("MATCH")
        return 0
    for line in diffs:
        print(f"DIFF  {line}")
    print(f"MISMATCH: {len(diffs)} difference(s)")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Row counts and checksums for DB parity checks.")
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="Write a snapshot of DATABASE_URL to a JSON file.")
    snap.add_argument("--out", required=True, type=Path)
    diff = sub.add_parser("diff", help="Compare two snapshot files.")
    diff.add_argument("file_a", type=Path)
    diff.add_argument("file_b", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "snapshot":
        return _run_snapshot(args.out)
    return _run_diff(args.file_a, args.file_b)


if __name__ == "__main__":
    raise SystemExit(main())
