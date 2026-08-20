#!/usr/bin/env python3
"""Decide draft-board components on evidence that has the power to decide them.

The snake-draft harness produces six numbers a year apart with a season-to-season
sd of 170.6; its mean gap against ADP carries an SE near 66. Component questions
cannot be answered there. This script answers them over ~1000 player-seasons with
paired bootstrap confidence intervals, and reports ADP as the opponent throughout.

Run: ``python scripts/draft_component_eval.py``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Mirrors the shipped defaults in ml.draft_projection.build_preseason_projections.
# Each ablation removes exactly one component from production.
CONFIGS = {
    "production": {"use_rookie_curve": True, "use_age_curve": False, "use_calibrated_games": True},
    "no_rookie_curve": {"use_rookie_curve": False, "use_age_curve": False, "use_calibrated_games": True},
    "no_games_calibration": {"use_rookie_curve": True, "use_age_curve": False, "use_calibrated_games": False},
    "with_age_curve": {"use_rookie_curve": True, "use_age_curve": True, "use_calibrated_games": True},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2020)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "draft_component_eval.json")
    args = parser.parse_args()

    import warnings

    warnings.filterwarnings("ignore")
    import pandas as pd
    import psycopg2

    from ml.draft_eval import (
        calibration_ratio,
        within_position_precision,
        evaluate_component,
        paired_difference,
        pooled_spearman,
        top_n_precision,
    )
    from ml.draft_projection import build_preseason_projections
    from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
    from pipeline.schema import normalize_dsn

    dsn = normalize_dsn(DEFAULT_HOST_DATABASE_URL)
    with psycopg2.connect(dsn) as conn:
        players = pd.read_sql(
            "SELECT id, full_name, position, team, entry_year, draft_round, draft_number, birth_date "
            "FROM players", conn
        )
        logs = pd.read_sql(
            "SELECT player_id, season, week, season_type, position, team, fantasy_points_ppr "
            "FROM game_logs WHERE week BETWEEN 1 AND 18", conn
        )
        adp = pd.read_sql(
            "SELECT season, source, player_id, adp FROM fantasy_adp "
            "WHERE player_id IS NOT NULL AND scoring = 'ppr'", conn
        )

    pooled = []
    for season in range(args.start_season, args.end_season + 1):
        market = adp[adp["season"] == season]
        if market.empty:
            continue
        market = market[market["source"] == str(market["source"].iloc[0])]
        realized = (
            logs[logs["season"] == season]
            .groupby("player_id", as_index=False)["fantasy_points_ppr"]
            .sum()
            .rename(columns={"fantasy_points_ppr": "realized"})
        )
        frame = None
        for name, flags in CONFIGS.items():
            built = build_preseason_projections(players, logs, season=season, **flags)
            column = built[["player_id", "projection", "position", "projection_basis"]].rename(
                columns={"projection": name}
            )
            frame = column if frame is None else frame.merge(
                column[["player_id", name]], on="player_id", how="outer"
            )
        merged = market.merge(frame, on="player_id", how="inner").merge(
            realized, on="player_id", how="left"
        )
        merged["realized"] = pd.to_numeric(merged["realized"], errors="coerce").fillna(0.0)
        merged["season"] = season
        merged["is_rookie"] = merged["projection_basis"].eq(
            "rookie_draft_capital_vacated_opportunity"
        )
        pooled.append(merged)

    data = pd.concat(pooled, ignore_index=True)
    payload: dict = {
        "n_player_seasons": int(len(data)),
        "n_players": int(data["player_id"].nunique()),
        "n_seasons": int(data["season"].nunique()),
        "n_rookies": int(data["is_rookie"].sum()),
        "why_this_exists": (
            "the snake-draft harness has a season-to-season sd of 170.6 and an SE near 66 on "
            "six seasons; it cannot resolve component differences and must not be used to pick them"
        ),
        "components": {},
        "decisions": {},
    }

    for name in CONFIGS:
        payload["components"][name] = evaluate_component(data, name, n_boot=args.n_boot)
    payload["components"]["adp"] = evaluate_component(
        data, "adp", higher_is_better=False, n_boot=args.n_boot
    )

    for label, a, b in (
        ("rookie_curve_vs_none", "production", "no_rookie_curve"),
        ("age_curve_vs_none", "with_age_curve", "production"),
        ("games_calibration_vs_none", "production", "no_games_calibration"),
        ("production_vs_ADP", "production", "adp"),
    ):
        # ADP ranks low-is-better; compare on within-season ranks so the two
        # orientations are on one scale.
        if b == "adp":
            data = data.assign(_adp_desc=-data["adp"])
            b = "_adp_desc"
        payload["decisions"][label] = {
            "spearman": paired_difference(data, a, b, pooled_spearman, n_boot=args.n_boot),
            "within_position_precision": paired_difference(
                data, a, b, within_position_precision, n_boot=args.n_boot, by_season=True
            ),
        }
        if not label.endswith("ADP"):
            payload["decisions"][label]["calibration_ratio_abs_error"] = paired_difference(
                data, a, b, lambda f, c: -abs(calibration_ratio(f, c) - 1.0), n_boot=args.n_boot
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
