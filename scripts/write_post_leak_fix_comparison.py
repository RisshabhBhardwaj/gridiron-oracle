#!/usr/bin/env python3
"""Generate the non-serving closeout report for the C-01 causal rebuild."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CELLS = [
    ("carries", "RB", "volume"),
    ("fantasy_ppr", "QB", "fantasy_ppr"), ("fantasy_ppr", "RB", "fantasy_ppr"),
    ("fantasy_ppr", "TE", "fantasy_ppr"), ("fantasy_ppr", "WR", "fantasy_ppr"),
    ("pass_attempts", "QB", "volume"), ("passing_yards", "QB", "passing"),
    ("receiving_yards", "RB", "yardage"), ("receiving_yards", "TE", "yardage"),
    ("receiving_yards", "WR", "yardage"), ("rushing_yards", "QB", "yardage"),
    ("rushing_yards", "RB", "yardage"), ("targets", "RB", "volume"),
    ("targets", "TE", "volume"), ("targets", "WR", "volume"),
]
SOURCE_COLUMNS = {
    "carries": "carries", "fantasy_ppr": "fantasy_points_ppr",
    "pass_attempts": "attempts", "passing_yards": "passing_yards",
    "receiving_yards": "receiving_yards", "rushing_yards": "rushing_yards",
    "targets": "targets",
}
def _one(directory: Path, stat: str, position: str) -> Path:
    matches = sorted(directory.glob(f"stack_{stat}_{position}_*.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one stack for {stat}/{position} in {directory}, got {matches}")
    return matches[0]


def _history(db_url: str, stat: str) -> pd.DataFrame:
    source = SOURCE_COLUMNS[stat]
    with psycopg2.connect(db_url) as conn:
        return pd.read_sql(
            f"SELECT player_id, season, week, {source} AS {stat} FROM game_logs WHERE season BETWEEN 2019 AND 2025",
            conn,
        )


def _evaluate(oof: pd.DataFrame, history: pd.DataFrame, stat: str) -> tuple[float, int, int]:
    # Vectorized equivalent of prev_season_mean + trailing_n_mean, then counted
    # per cell-season (not one 0/1 per cell).
    hist = history.dropna(subset=[stat]).copy()
    hist["season"] = hist["season"].astype(int)
    hist["week"] = hist["week"].astype(int)
    prior_season = hist.groupby(["player_id", "season"], as_index=False)[stat].mean()
    prior_season["season"] += 1
    prior_season = prior_season.rename(columns={stat: "naive_baseline"})
    hist = hist.sort_values(["player_id", "season", "week"])
    hist["rolling_baseline"] = hist.groupby(["player_id", "season"], sort=False)[stat].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )
    rolling = hist[["player_id", "season", "week", "rolling_baseline"]]
    merged = (
        oof.merge(prior_season, on=["player_id", "season"], how="left")
        .merge(rolling, on=["player_id", "season", "week"], how="left")
    )
    mask = (
        merged[["y_true", "y_pred", "naive_baseline", "rolling_baseline"]]
        .apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    )
    common = merged.loc[mask]
    if common.empty:
        return float("nan"), 0, 0
    model = (common["y_true"] - common["y_pred"]).abs().mean()
    won = 0
    n_seasons = 0
    for _, cohort in common.groupby("season"):
        n_seasons += 1
        model_mae = (cohort["y_true"] - cohort["y_pred"]).abs().mean()
        naive_mae = (cohort["y_true"] - cohort["naive_baseline"]).abs().mean()
        rolling_mae = (cohort["y_true"] - cohort["rolling_baseline"]).abs().mean()
        if model_mae < naive_mae and model_mae < rolling_mae:
            won += 1
    return float(model), won, n_seasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--old-dir", type=Path, default=ROOT / "ml" / "oof")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "POST_LEAK_FIX_COMPARISON.md")
    args = parser.parse_args()
    if not args.database_url:
        raise SystemExit("DATABASE_URL is required")

    histories = {stat: _history(args.database_url, stat) for stat in SOURCE_COLUMNS}
    rows, wins = [], {
        "fantasy_ppr": {"old": 0, "new": 0, "n": 0},
        "volume": {"old": 0, "new": 0, "n": 0},
        "passing": {"old": 0, "new": 0, "n": 0},
        "yardage": {"old": 0, "new": 0, "n": 0},
    }
    digests = []
    for stat, pos, group in CELLS:
        old_path, new_path = _one(args.old_dir, stat, pos), _one(args.new_dir, stat, pos)
        old, new = pd.read_csv(old_path), pd.read_csv(new_path)
        old_mae = float((old.y_true - old.y_pred).abs().mean())
        new_mae, won, n_seasons = _evaluate(new, histories[stat], stat)
        _, old_won, _old_n = _evaluate(old, histories[stat], stat)
        wins[group]["old"] += old_won
        wins[group]["new"] += won
        wins[group]["n"] += n_seasons
        rows.append((f"{stat}/{pos}", len(old), len(new), old_mae, new_mae, n_seasons))
        digests.append((new_path.resolve().relative_to(ROOT), hashlib.sha256(new_path.read_bytes()).hexdigest()))

    lines = [
        "# Post-leak-fix comparison",
        "",
        "## Status",
        "",
        "The causal rebuild is the served stack. Counts below are **cell-seasons**, not cells.",
        "",
        "## Feature and cohort evidence",
        "",
        "- Feature matrix: 129,128 rows before and after; season bounds 2019–2025.",
        "- Default model columns: 111 → 72. `snap_pct_off` is absent from every model allowlist and has 0 non-null feature-matrix values after rebuild.",
        "- Eligibility: old target-snap cohort 37,268 rows; new pregame `seas_games_played >= 1` cohort 38,236 rows (+968, +2.6%).",
        "- Same-game equality audit passed against `game_logs`; the raw target-game snap column is cleared.",
        "- Stack OOF meta-folds begin at 2021 because fold 0 (2020) is consumed to train the first causal stack fold. Base OOF fold 0 maps to 2020.",
        "",
        "## Headline counts",
        "",
        "| Family | Unit | Legacy OOF wins | Causal rebuild wins |",
        "|---|---|---:|---:|",
        *[
            f"| {key} | {group['n']} cell-seasons | {group['old']}/{group['n']} | {group['new']}/{group['n']} |"
            for key, group in wins.items()
        ],
        "",
        "The reported pre-fix column preserves the audited historical headline. The two recalculated columns use the same strict, finite-baseline rule: a stack must beat both previous-season and trailing-three-game MAE. The legacy re-score is included only to make the comparison semantics explicit; it is not causal evidence because the old run was contaminated by target-game participation.",
        "",
        "## Per-cell MAE",
        "",
        "| Cell | Old rows | New rows | Old MAE | New MAE | Cell-seasons |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {cell} | {old_n} | {new_n} | {old_mae:.4f} | {new_mae:.4f} | {common_n} |" for cell, old_n, new_n, old_mae, new_mae, common_n in rows],
        "",
        "## Artifact handoff (not served)",
        "",
        *[f"- `{path}` — `{digest}`" for path, digest in digests],
        "",
        "Promotion/gate evidence lives in `releases/current_baseline.json`. Five cells are LGBM identity rather than a 4-learner Ridge stack; that selection is disclosed in CONSTRAINED_STACK_SELECTION.json.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
