#!/usr/bin/env python3
"""Score the rest-of-season surface against realized points (design spec SP3.4).

The 2026-08-19 audit found this surface inverted: its entire top-20 was backup
quarterbacks and nominal 80% intervals covered 10.5% of outcomes. This script
re-runs that exact measurement so the claim that SP3 fixed it is evidence
rather than assertion.

Run: ``python scripts/verify_season_acceptance.py --season 2025 --start-week 10``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _pinned_cells() -> list[tuple[str, str]]:
    """Every (stat, position) cell the pinned release serves, from the manifest."""
    from ml.artifact_manifest import get_manifest

    return sorted(get_manifest().stack_index)


def verify_forward_week_structural(
    dsn: str,
    *,
    season: int,
    start_week: int,
    pipeline_run_id: str,
    approved_run_id: str | None = None,
    cells: list[tuple[str, str]] | None = None,
) -> dict:
    """Structural gate for a forward (unplayed) week (audit C-1, decision 0012).

    The statistical gate below scores pending projections against realized
    points, which do not exist before kickoff, so for a forward week that join is
    always empty. This check asks the defined question instead: did the run
    produce served rows for the target week, for every pinned (stat, position)
    cell and every position, and rows on the other two surfaces?

    Rows count when they carry the pending id or, if given, the approved id: a
    re-run after promotion writes nothing new (the producer skips rows an approved
    run already serves), and that week is ``already_current``, not missing.
    Everything is scoped to ``start_week``; rows for other weeks never count.
    """
    import psycopg2

    from pipeline.availability import served_predicate

    run_ids = [pipeline_run_id] + ([approved_run_id] if approved_run_id else [])
    required = list(cells) if cells is not None else _pinned_cells()

    projection_sql = (
        "SELECT stat, position, pipeline_run_id, COUNT(*) FROM projections "
        "WHERE season = %s AND week = %s AND pipeline_run_id = ANY(%s) "
        f"AND {served_predicate()} GROUP BY stat, position, pipeline_run_id"
    )
    surface_sql = {
        "season_simulations":
            "SELECT COUNT(*) FROM season_simulations "
            "WHERE season = %s AND start_week = %s AND pipeline_run_id = ANY(%s)",
        "team_game_predictions":
            "SELECT COUNT(*) FROM team_game_predictions "
            "WHERE season = %s AND week = %s AND model_run_id = ANY(%s)",
    }
    params = (season, start_week, run_ids)
    by_cell: dict[tuple[str, str], dict[str, int]] = {}
    counts: dict[str, int] = {}
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(projection_sql, params)
            for stat, position, run_id, n in cur.fetchall():
                by_cell.setdefault((stat, position), {})[run_id] = int(n)
            for table, sql in surface_sql.items():
                cur.execute(sql, params)
                row = cur.fetchone()
                counts[table] = int(row[0]) if row and row[0] is not None else 0
    counts = {"projections": sum(sum(v.values()) for v in by_cell.values()), **counts}

    missing_cells = [list(c) for c in required if not by_cell.get(c)]
    required_positions = sorted({pos for _, pos in required})
    served_positions = {pos for (_, pos), v in by_cell.items() if v}
    missing_positions = [pos for pos in required_positions if pos not in served_positions]
    missing_surfaces = sorted(table for table, n in counts.items() if n == 0)
    already_current = bool(
        approved_run_id and required
        and all(by_cell.get(c, {}).get(approved_run_id) for c in required)
    )
    met = not missing_surfaces and not missing_cells and not missing_positions
    return {
        "mode": "structural",
        "season": season,
        "start_week": start_week,
        "pipeline_run_id": pipeline_run_id,
        "approved_run_id": approved_run_id,
        "row_counts": counts,
        "acceptance": {"every_surface_has_target_week_rows": met},
        "missing_surfaces": missing_surfaces,
        "missing_cells": missing_cells,
        "missing_positions": missing_positions,
        "already_current": already_current,
        "diagnosis": (
            f"Week {start_week} has served rows for every pinned cell and position, and rows "
            f"on the other surfaces, under {' or '.join(run_ids)}."
            if met else
            f"Week {start_week} under {' or '.join(run_ids)}: surfaces without rows "
            f"{missing_surfaces}; positions without rows {missing_positions}; "
            f"{len(missing_cells)} pinned cells without rows. Step 4b or steps 5-6 did not "
            "write what they should; do not promote."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--start-week", type=int, default=10)
    parser.add_argument("--end-week", type=int, default=18)
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "season_acceptance.json")
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="Forward-week structural gate (audit C-1): assert the pending run id has "
             "rows for the target week instead of scoring against realized points, "
             "which do not exist before kickoff.",
    )
    parser.add_argument(
        "--pipeline-run-id",
        default=None,
        help="Pending run id to check in structural mode (e.g. weekly_auto_2026_pending)",
    )
    parser.add_argument(
        "--approved-run-id",
        default=None,
        help="Approved run id whose rows also count in structural mode (a re-run of a promoted week)",
    )
    from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL

    parser.add_argument(
        "--database-url",
        default=DEFAULT_HOST_DATABASE_URL,
        help="Database to read projections and realized points from (the refresh passes its own DSN)",
    )
    args = parser.parse_args()

    if args.structural_only:
        if not args.pipeline_run_id:
            parser.error("--structural-only requires --pipeline-run-id")
        from pipeline.schema import normalize_dsn

        payload = verify_forward_week_structural(
            normalize_dsn(args.database_url),
            season=args.season,
            start_week=args.start_week,
            pipeline_run_id=args.pipeline_run_id,
            approved_run_id=args.approved_run_id,
            cells=_pinned_cells(),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return 0 if payload["acceptance"]["every_surface_has_target_week_rows"] else 1

    import pandas as pd
    import psycopg2
    from scipy.stats import spearmanr

    from backend.app.services.projection import ProjectionService
    from pipeline.schema import normalize_dsn

    dsn = normalize_dsn(args.database_url)
    projected = ProjectionService(dsn).get_season_projections(args.season, args.start_week)
    frame = pd.DataFrame([
        {
            "player_id": row["player_id"],
            "player_name": row["player_name"],
            "position": row["position"],
            "degraded": row.get("degraded"),
            "mean": float((row.get("fantasy_ppr") or {}).get("mean") or 0.0),
            "p10": float((row.get("fantasy_ppr") or {}).get("p10") or 0.0),
            "p90": float((row.get("fantasy_ppr") or {}).get("p90") or 0.0),
        }
        for row in projected
    ])

    with psycopg2.connect(dsn) as conn:
        realized = pd.read_sql(
            "SELECT player_id, SUM(fantasy_points_ppr) AS realized FROM game_logs "
            "WHERE season = %s AND week BETWEEN %s AND %s GROUP BY player_id",
            conn, params=(args.season, args.start_week, args.end_week),
        )

    merged = frame.merge(realized, on="player_id", how="inner")
    merged["realized"] = pd.to_numeric(merged["realized"], errors="coerce").fillna(0.0)
    if len(merged) < args.top_n:
        print(json.dumps({"status": "too_few_matched", "n": int(len(merged))}, indent=2))
        return 1

    covered = ((merged["realized"] >= merged["p10"]) & (merged["realized"] <= merged["p90"])).mean()
    top = merged.nlargest(args.top_n, "mean")
    rho_all = float(spearmanr(merged["mean"], merged["realized"]).correlation)
    rho_top = float(spearmanr(top["mean"], top["realized"]).correlation)

    payload = {
        "season": args.season,
        "window": f"weeks {args.start_week}-{args.end_week}",
        "n_matched": int(len(merged)),
        "interval_coverage_80": round(float(covered), 4),
        "coverage_in_band": bool(0.75 <= covered <= 0.85),
        "median_interval_width": round(float((merged["p90"] - merged["p10"]).median()), 2),
        "median_abs_error": round(float((merged["realized"] - merged["mean"]).abs().median()), 2),
        "spearman_all": round(rho_all, 4),
        f"spearman_top{args.top_n}": round(rho_top, 4),
        f"top{args.top_n}_positively_correlated": bool(rho_top > 0),
        f"top{args.top_n}_position_mix": {
            str(k): int(v) for k, v in top["position"].value_counts().items()
        },
        f"top{args.top_n}_mean_realized": round(float(top["realized"].mean()), 2),
        "field_mean_realized": round(float(merged["realized"].mean()), 2),
        "top10_board": [
            {
                "player": row.player_name,
                "position": row.position,
                "projected": round(float(row.mean), 1),
                "realized": round(float(row.realized), 1),
            }
            for row in top.nlargest(10, "mean").itertuples(index=False)
        ],
        "acceptance": {
            "top_n_positively_correlated": bool(rho_top > 0),
            "coverage_80_in_75_85": bool(0.75 <= covered <= 0.85),
        },
    }
    payload["acceptance"]["met"] = all(payload["acceptance"].values())
    payload["diagnosis"] = (
        "The rate fed to the season path is a single week's stack projection "
        "(_load_weekly_rates reads week = start_week) extrapolated across every "
        "remaining week. Nothing models whether the player keeps that role, so "
        "thin-sample players inherit a starter's rate for nine weeks. That, not "
        "the interval arithmetic, is what holds coverage below the 75-85% band. "
        "Shrinking the rate toward a positional median was tried and rejected: "
        "it anchors a backup quarterback to the median quarterback, which raised "
        "rank correlation while making the board materially worse. The fix is a "
        "role-persistence model (SP2), not a wider interval."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    # The report is written either way; an unmet acceptance must still fail the caller.
    return 0 if payload["acceptance"]["met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
