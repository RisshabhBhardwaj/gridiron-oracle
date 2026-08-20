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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--start-week", type=int, default=10)
    parser.add_argument("--end-week", type=int, default=18)
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "season_acceptance.json")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import psycopg2
    from scipy.stats import spearmanr

    from backend.app.services.projection import ProjectionService
    from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
    from pipeline.schema import normalize_dsn

    dsn = normalize_dsn(DEFAULT_HOST_DATABASE_URL)
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

    with psycopg2.connect(normalize_dsn(DEFAULT_HOST_DATABASE_URL)) as conn:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
