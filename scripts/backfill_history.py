#!/usr/bin/env python3
"""
Backfill pre-cut history into the production database from a read-only copy.

Why: the box's `oracle` database was cut to 2024 onward, so training rows for
2019-2023 are absent and the Elo ratings computed on 2024+ rows start from a
truncated history (measured 2026-11-14: 66 of 72 model features have identical
2024 averages in `oracle` and the full-history copy; only the six Elo columns
differ, e.g. team_off_elo 1563 vs 1832). The fix is to restore the missing
history, not to impute it, and then recompute Elo over the full history.

The source copy also predates audit C1 (commit 52e8cde, 2026-10-10):
`elo_implied_win_prob` there was built from end-of-fit ratings, which carry
information from games after the row's. The feature contract does not catch
that, so recomputing Elo for every season is a required step, and ``--verify``
fails if that column still equals the source's values.

What it does, per table in ``BACKFILL_TABLES`` (parents first):
  * copies source rows from seasons strictly before the target's earliest season
    for that table (the whole range when the target table is empty), never past
    ``--max-season`` (default LAST_COMPLETE_SEASON). Rows the target already has
    are never changed: ``ON CONFLICT DO NOTHING``.
  * leaves serial ``id`` columns to the target's sequence (they would collide).
  * copies only columns present on both sides and reports the rest.
  * refuses to commit if any ``feature_matrix.snap_pct_off`` (target-game snap
    data, the August 2026 leak) is non-null.

It runs in one transaction. Without ``--apply`` it does all the work, prints the
counts, and rolls back. The source is opened read-only.

    python scripts/backfill_history.py --source-url ... --target-url ...            # dry run
    python scripts/backfill_history.py --source-url ... --target-url ... --apply    # commit
    python -m pipeline.enrich_elo_embeddings                                      # Elo over full history
    python scripts/backfill_history.py --source-url ... --target-url ... --verify   # compare features
    python scripts/verify_feature_contract.py --seasons 2019 2020 2021 2022 2023 2024 2025

URLs may come from BACKFILL_SOURCE_URL and DATABASE_URL. They are never printed.
Take a backup of the target before ``--apply``.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class TableSpec:
    name: str
    season_col: Optional[str] = "season"


#: Parents first. Tables the feature pipeline and Elo fit read, plus the
#: feature matrix itself. Projections, simulations and staging are not history.
BACKFILL_TABLES: tuple[TableSpec, ...] = (
    TableSpec("players", season_col=None),
    TableSpec("games"),
    TableSpec("game_logs"),
    TableSpec("team_game_stats"),
    TableSpec("depth_charts"),
    TableSpec("participation_player_game"),
    TableSpec("nextgen_stats"),
    TableSpec("pbp_plays"),
    TableSpec("pbp_features"),
    TableSpec("pbp_matchups"),
    TableSpec("player_season_profiles", season_col="effective_season"),
    TableSpec("injury_history"),
    TableSpec("ftn_player_game"),
    TableSpec("ftn_play"),
    TableSpec("feature_matrix"),
)

LEAK_COLUMN = "snap_pct_off"

#: Columns the Elo recompute must change in the source's history (audit C1).
#: ``--verify`` does not require them to match; it requires them to differ.
RECOMPUTED_COLUMNS: tuple[str, ...] = ("elo_implied_win_prob",)


class BackfillError(RuntimeError):
    """The backfill cannot proceed or would not be safe to commit."""


@dataclass
class TableReport:
    table: str
    target_min_season: Optional[int] = None
    staged: int = 0
    inserted: int = 0
    skipped_existing: int = 0
    source_only_columns: list[str] = field(default_factory=list)
    target_only_columns: list[str] = field(default_factory=list)
    note: str = ""


def describe_url(url: str) -> tuple[str, int, str]:
    """(host, port, dbname) for same-database checks and safe logging."""
    parsed = urlparse(url.replace("postgresql+asyncpg://", "postgresql://"))
    return (parsed.hostname or "", parsed.port or 5432, (parsed.path or "").lstrip("/"))


def assert_distinct_databases(source_url: str, target_url: str) -> None:
    if describe_url(source_url) == describe_url(target_url):
        raise BackfillError("source and target are the same database")


def choose_columns(
    source_cols: Sequence[str],
    target_cols: Sequence[str],
    serial_cols: Sequence[str],
    required_cols: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    """Return (columns to copy, source-only columns, target-only columns).

    ``serial_cols`` are left to the target's sequence. Raises when a target
    column that is NOT NULL without a default would not be filled.
    """
    serial = set(serial_cols)
    copy = [c for c in target_cols if c in set(source_cols) and c not in serial]
    unfilled = [c for c in required_cols if c not in set(copy) and c not in serial]
    if unfilled:
        raise BackfillError(f"required target columns have no source value: {unfilled}")
    source_only = [c for c in source_cols if c not in set(target_cols)]
    target_only = [c for c in target_cols if c not in set(source_cols) and c not in serial]
    return copy, source_only, target_only


def season_bounds(season_col: Optional[str], max_season: int, target_min: Optional[int]) -> list[tuple[str, int]]:
    """``(operator, season)`` conditions on the season column; empty means 'all rows'.

    Never past ``max_season``, and strictly before the target's earliest season
    so rows the target already holds (and may have recomputed) are not refilled.
    """
    if season_col is None:
        return []
    bounds = [("<=", int(max_season))]
    if target_min is not None:
        bounds.append(("<", int(target_min)))
    return bounds


def _columns(cur: Any, table: str) -> list[dict]:
    cur.execute(
        """
        SELECT column_name, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """,
        (table,),
    )
    return [{"name": r[0], "nullable": r[1] == "YES", "default": r[2]} for r in cur.fetchall()]


def _backfill_table(src: Any, tgt: Any, spec: TableSpec, max_season: int) -> TableReport:
    from psycopg2 import sql

    report = TableReport(table=spec.name)
    scur, tcur = src.cursor(), tgt.cursor()
    src_cols, tgt_cols = _columns(scur, spec.name), _columns(tcur, spec.name)
    if not tgt_cols:
        raise BackfillError(f"target has no table {spec.name}")
    if not src_cols:
        report.note = "absent in source; skipped"
        return report

    serial = [c["name"] for c in tgt_cols if (c["default"] or "").startswith("nextval(")]
    required = [c["name"] for c in tgt_cols if not c["nullable"] and c["default"] is None]
    cols, report.source_only_columns, report.target_only_columns = choose_columns(
        [c["name"] for c in src_cols], [c["name"] for c in tgt_cols], serial, required
    )
    col_list = sql.SQL(", ").join(sql.Identifier(c) for c in cols)
    table = sql.Identifier(spec.name)

    target_min: Optional[int] = None
    if spec.season_col:
        tcur.execute(sql.SQL("SELECT min({c}) FROM {t}").format(c=sql.Identifier(spec.season_col), t=table))
        target_min = tcur.fetchone()[0]
    report.target_min_season = target_min

    where = sql.SQL("")
    bounds = season_bounds(spec.season_col, max_season, target_min)
    if bounds:
        conditions = [
            sql.SQL("{c} {op} {v}").format(c=sql.Identifier(spec.season_col), op=sql.SQL(op), v=sql.Literal(value))
            for op, value in bounds
        ]
        where = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions)

    stage = sql.Identifier(f"_bf_{spec.name}")
    tcur.execute(
        sql.SQL("CREATE TEMP TABLE {s} ON COMMIT DROP AS SELECT {cols} FROM {t} WITH NO DATA").format(
            s=stage, cols=col_list, t=table
        )
    )
    copy_out = sql.SQL("COPY (SELECT {cols} FROM {t}{where}) TO STDOUT").format(cols=col_list, t=table, where=where)
    copy_in = sql.SQL("COPY {s} ({cols}) FROM STDIN").format(s=stage, cols=col_list)
    with tempfile.SpooledTemporaryFile(max_size=64 << 20, mode="w+b") as buffer:
        scur.copy_expert(copy_out.as_string(src), buffer)
        buffer.seek(0)
        tcur.copy_expert(copy_in.as_string(tgt), buffer)
    tcur.execute(sql.SQL("SELECT count(*) FROM {s}").format(s=stage))
    report.staged = tcur.fetchone()[0]
    tcur.execute(
        sql.SQL("INSERT INTO {t} ({cols}) SELECT {cols} FROM {s} ON CONFLICT DO NOTHING").format(
            t=table, cols=col_list, s=stage
        )
    )
    report.inserted = tcur.rowcount
    report.skipped_existing = report.staged - report.inserted
    return report


def _assert_no_snap_leak(tgt: Any) -> None:
    cur = tgt.cursor()
    cur.execute(f"SELECT count(*) FROM feature_matrix WHERE {LEAK_COLUMN} IS NOT NULL")
    leaked = cur.fetchone()[0]
    if leaked:
        raise BackfillError(
            f"feature_matrix has {leaked} rows with non-null {LEAK_COLUMN} (target-game snap data, the "
            "August 2026 leak). Refusing to commit; the source matrix predates the fix."
        )


def backfill(
    source_url: str,
    target_url: str,
    *,
    max_season: Optional[int] = None,
    apply: bool = False,
    tables: Optional[Sequence[str]] = None,
) -> list[TableReport]:
    """Copy missing history. Returns one report per table; commits only when ``apply``."""
    import psycopg2

    from ml.season_constants import LAST_COMPLETE_SEASON
    from scraper.adapters.nflreadpy_adapter import _psycopg2_dsn

    assert_distinct_databases(source_url, target_url)
    cap = LAST_COMPLETE_SEASON if max_season is None else int(max_season)
    specs = [s for s in BACKFILL_TABLES if tables is None or s.name in set(tables)]
    if tables and len(specs) != len(set(tables)):
        raise BackfillError(f"unknown tables: {sorted(set(tables) - {s.name for s in BACKFILL_TABLES})}")

    src = psycopg2.connect(_psycopg2_dsn(source_url))
    tgt = psycopg2.connect(_psycopg2_dsn(target_url))
    try:
        src.set_session(readonly=True)
        reports = [_backfill_table(src, tgt, spec, cap) for spec in specs]
        if any(s.name == "feature_matrix" for s in specs):
            _assert_no_snap_leak(tgt)
        if apply:
            tgt.commit()
        else:
            tgt.rollback()
        return reports
    except Exception:
        tgt.rollback()
        raise
    finally:
        src.close()
        tgt.close()


def _feature_averages(conn: Any, max_season: int) -> dict[int, dict[str, float]]:
    from ml.utils import FEATURE_COLS

    averages = ", ".join(f"avg({c}::float8) AS {c}" for c in FEATURE_COLS)
    cur = conn.cursor()
    cur.execute(
        f"SELECT season, count(*) AS n, {averages} FROM feature_matrix WHERE season <= %s GROUP BY season ORDER BY season",
        (max_season,),
    )
    names = ["season", "n", *FEATURE_COLS]
    return {int(row[0]): dict(zip(names[1:], row[1:])) for row in cur.fetchall()}


def compare_averages(
    source: dict[int, dict[str, float]],
    target: dict[int, dict[str, float]],
    *,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-6,
    recomputed: Sequence[str] = RECOMPUTED_COLUMNS,
) -> list[str]:
    """Differences between per-season row counts and feature averages, as messages.

    Columns in ``recomputed`` are expected to differ from the source (it holds
    pre-fix values); they are reported only if they are still identical in every
    season, which means the Elo recompute has not run.
    """
    problems: list[str] = []
    still_source: dict[str, int] = {c: 0 for c in recomputed}
    compared = 0
    for season in sorted(set(source) | set(target)):
        if season not in source or season not in target:
            problems.append(f"{season}: present in only one database")
            continue
        compared += 1
        s, t = source[season], target[season]
        if s["n"] != t["n"]:
            problems.append(f"{season}: row count source {s['n']} vs target {t['n']}")
        for col in s:
            if col == "n":
                continue
            a, b = s[col], t.get(col)
            if col in still_source:
                if a is not None and b is not None and abs(a - b) <= 1e-9:
                    still_source[col] += 1
                continue
            if a is None and b is None:
                continue
            if a is None or b is None:
                problems.append(f"{season}: {col} null in one database only")
            elif abs(a - b) > max(abs_tol, rel_tol * max(abs(a), abs(b))):
                problems.append(f"{season}: {col} source {a:.6g} vs target {b:.6g}")
    for col, count in still_source.items():
        if compared and count == compared:
            problems.append(
                f"{col} is identical to the source in every season; the source holds pre-fix values. "
                "Run the Elo recompute (pipeline.enrich_elo_embeddings.run_elo_enrichment)."
            )
    return problems


def verify(source_url: str, target_url: str, *, max_season: Optional[int] = None) -> list[str]:
    import psycopg2

    from ml.season_constants import LAST_COMPLETE_SEASON
    from scraper.adapters.nflreadpy_adapter import _psycopg2_dsn

    assert_distinct_databases(source_url, target_url)
    cap = LAST_COMPLETE_SEASON if max_season is None else int(max_season)
    src = psycopg2.connect(_psycopg2_dsn(source_url))
    tgt = psycopg2.connect(_psycopg2_dsn(target_url))
    try:
        src.set_session(readonly=True)
        return compare_averages(_feature_averages(src, cap), _feature_averages(tgt, cap))
    finally:
        src.close()
        tgt.close()


def _print_reports(reports: Sequence[TableReport], *, applied: bool) -> None:
    print(f"{'table':28} {'target min':>10} {'staged':>9} {'inserted':>9} {'existing':>9}")
    for r in reports:
        print(f"{r.table:28} {str(r.target_min_season or '-'):>10} {r.staged:>9} {r.inserted:>9} {r.skipped_existing:>9}  {r.note}")
        if r.source_only_columns:
            print(f"    source-only columns not copied: {', '.join(r.source_only_columns)}")
        if r.target_only_columns:
            print(f"    target-only columns left default/NULL: {', '.join(r.target_only_columns)}")
    print("COMMITTED" if applied else "DRY RUN: rolled back, nothing written")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-url", default=os.environ.get("BACKFILL_SOURCE_URL"))
    parser.add_argument("--target-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--max-season", type=int, default=None)
    parser.add_argument("--tables", nargs="*", help="limit to these tables (default: all history tables)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="commit (default is a rolled-back dry run)")
    mode.add_argument("--verify", action="store_true", help="compare per-season feature averages, source vs target")
    args = parser.parse_args(argv)
    if not args.source_url or not args.target_url:
        print("--source-url (or BACKFILL_SOURCE_URL) and --target-url (or DATABASE_URL) are required", file=sys.stderr)
        return 2
    try:
        if args.verify:
            problems = verify(args.source_url, args.target_url, max_season=args.max_season)
            for problem in problems:
                print(f"DIFF {problem}", file=sys.stderr)
            print("verified: feature averages match" if not problems else f"{len(problems)} difference(s)")
            return 1 if problems else 0
        reports = backfill(
            args.source_url, args.target_url, max_season=args.max_season, apply=args.apply, tables=args.tables or None
        )
    except BackfillError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    _print_reports(reports, applied=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
