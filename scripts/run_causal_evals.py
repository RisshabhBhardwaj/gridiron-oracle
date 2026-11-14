#!/usr/bin/env python3
"""
Run the causal evaluation for every pinned serving cell, then summarize.

For each of the 32 cells in ``ml.artifact_manifest.REQUIRED_SERVING_CELLS``:
  1. resolve the stack OOF pinned by ``releases/current_baseline.json``
     (digest-checked; a missing or altered file stops the run),
  2. build the baseline history from ``feature_matrix`` actuals,
  3. score it with ``ml.eval_causal.score_oof_against_baselines``,
  4. write ``reports/eval_causal_<stat>_<position>.csv``.
Finally ``scripts/summarize_causal_evals.py`` aggregates only the files this
run wrote. Stale ``eval_causal_*.csv`` files in the output directory are
refused, so the count can never include a previous run's rows.

Run after ``make freeze-baseline``:

    DATABASE_URL=... python scripts/run_causal_evals.py
    python scripts/summarize_causal_evals.py --check

A cell with no scored rows fails the run (exit 1) rather than being skipped:
a headline count over fewer than 32 cells would be misleading.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.eval_causal import score_oof_against_baselines  # noqa: E402
from ml.utils import TARGET_COL_MAP  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "reports"
HISTORY_KEYS = ("player_id", "season", "week")


class EvalRunError(RuntimeError):
    """A cell could not be evaluated honestly."""


def eval_filename(stat: str, position: str) -> str:
    return f"eval_causal_{stat}_{position}.csv"


def load_history(db_url: str, stats: Iterable[str], max_season: int) -> pd.DataFrame:
    """One frame of per-game actuals for baseline construction.

    Columns: player_id, season, week, then one column per stat named by the
    model stat (what ``ml.eval_causal`` expects). Seasons after ``max_season``
    are excluded so a stale partial season cannot enter a baseline.
    """
    import psycopg2
    from scraper.adapters.nflreadpy_adapter import _psycopg2_dsn

    stats = sorted(set(stats))
    columns = [TARGET_COL_MAP[s] for s in stats]
    select = ", ".join(["player_id", "season", "week", *columns])
    conn = psycopg2.connect(_psycopg2_dsn(db_url))
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT {select} FROM feature_matrix WHERE season <= %s", (max_season,))
        frame = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    finally:
        conn.close()
    return frame.rename(columns={TARGET_COL_MAP[s]: s for s in stats})


def evaluate_cell(
    stat: str,
    position: str,
    stack_path: Path,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Score one cell. Raises EvalRunError if nothing could be scored."""
    oof = pd.read_csv(stack_path)
    table = score_oof_against_baselines(oof, history, stat=stat, position=position)
    if table.empty:
        raise EvalRunError(f"{stat}/{position}: no evaluable rows in {stack_path.name}")
    if table["beats_naive"].isna().any() or table["beats_trailing3"].isna().any():
        raise EvalRunError(
            f"{stat}/{position}: a baseline score is undefined for some season; "
            "the history does not cover the evaluated rows"
        )
    if not table["max_train_season_ok"].all():
        raise EvalRunError(f"{stat}/{position}: OOF contains non-causal rows")
    return table


def run(
    *,
    out_dir: Path,
    history: pd.DataFrame,
    resolve_stack: Callable[[str, str], Path],
    cells: Iterable[tuple[str, str]],
    only: Optional[set[tuple[str, str]]] = None,
    allow_stale: bool = False,
) -> list[Path]:
    """Evaluate ``cells`` and return the files written.

    Refuses to start when ``out_dir`` already holds ``eval_causal_*.csv`` files
    (unless ``allow_stale``), because the summary counts every such file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("eval_causal_*.csv"))
    if existing and not allow_stale:
        raise EvalRunError(
            f"{len(existing)} eval_causal_*.csv already in {out_dir} "
            f"(first: {existing[0].name}). Move them away or use a new --out-dir; "
            "the summary counts every file there."
        )
    written: list[Path] = []
    for stat, position in sorted(cells):
        if only is not None and (stat, position) not in only:
            continue
        table = evaluate_cell(stat, position, resolve_stack(stat, position), history)
        path = out_dir / eval_filename(stat, position)
        table.to_csv(path, index=False)
        beat = int((table["beats_naive"].astype(bool) & table["beats_trailing3"].astype(bool)).sum())
        print(f"{stat}/{position}: beat both in {beat}/{len(table)} seasons -> {path.name}", flush=True)
        written.append(path)
    return written


def _parse_only(values: Optional[list[str]]) -> Optional[set[tuple[str, str]]]:
    if not values:
        return None
    out = set()
    for value in values:
        stat, _, position = value.partition(":")
        if not stat or not position:
            raise SystemExit(f"--only expects stat:POS, got {value!r}")
        out.add((stat, position))
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--only", nargs="*", help="limit to stat:POS cells (dry runs; the full count needs all 32)")
    parser.add_argument("--no-summary", action="store_true", help="skip the summary step")
    args = parser.parse_args(argv)

    from ml.artifact_manifest import REQUIRED_SERVING_CELLS, get_manifest
    from ml.season_constants import LAST_COMPLETE_SEASON

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL is required (feature_matrix actuals build the baselines)", file=sys.stderr)
        return 2

    only = _parse_only(args.only)
    cells = sorted(REQUIRED_SERVING_CELLS)
    wanted = [c for c in cells if only is None or c in only]
    if only and len(wanted) != len(only):
        print(f"unknown cells: {sorted(only - set(cells))}", file=sys.stderr)
        return 2

    manifest = get_manifest()
    history = load_history(db_url, {s for s, _ in wanted}, LAST_COMPLETE_SEASON)
    try:
        written = run(
            out_dir=args.out_dir,
            history=history,
            resolve_stack=lambda s, p: manifest.resolve_stack(s, p),
            cells=cells,
            only=only,
        )
    except EvalRunError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    if only is None and len(written) != len(REQUIRED_SERVING_CELLS):
        print(f"FAILED: wrote {len(written)} of {len(REQUIRED_SERVING_CELLS)} cells", file=sys.stderr)
        return 1
    if args.no_summary:
        return 0
    from scripts.summarize_causal_evals import main as summarize

    return summarize(["--eval-dir", str(args.out_dir)])


if __name__ == "__main__":
    raise SystemExit(main())
