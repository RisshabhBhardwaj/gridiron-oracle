#!/usr/bin/env python3
"""Measure pre-declared draft-board variants on the validated roster-aware harness.

Variants are fixed here before any is scored, and *every* variant is reported
whether it wins or loses. Selecting the best of N and quoting only that number
is the multiple-testing failure the design spec forbids, so the output carries
``effective_n_trials``.

  production           the served board: Marcel rate x multi-season expected
                       games x walk-forward age curve
  no_multiseason_games production with expected_games_from_history removed
  no_age_curve         production with the fitted age multiplier removed
  no_rookie_curve      production with the draft-capital rookie prior removed
  market_blend         production rank-blended 50/50 with ADP  (CONSUMES MARKET
                       INFO -- its "beats ADP" is a weaker claim than an
                       independent board's and is reported under a separate key)

Ablations, not additions: every refinement ships, so each variant removes one.

READ THIS BEFORE ACTING ON THE NUMBERS. This harness resolves nothing at the
component level: six seasons, season-to-season sd ~170, SE ~66. Two components
were once switched on and off here on readings inside that noise. Component
decisions belong to scripts/draft_component_eval.py, which measures the same
projections over ~1030 player-seasons with paired bootstrap CIs. What this
harness is for is the end-to-end board question, reported with its interval.

Every variant is ranked by VOR before drafting, so the comparison isolates
projection quality rather than crediting one variant for the VOR translation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INDEPENDENT = ("production", "no_multiseason_games", "no_age_curve", "no_rookie_curve")
MARKET_AWARE = ("market_blend",)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--start-season", type=int, default=2020)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "draft_board_experiments.json")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import psycopg2

    from ml.baselines import attach_marcel
    from ml.draft_projection import build_preseason_projections
    from ml.draft_sim import evaluate_board
    from ml.multiplicity import effective_n_trials
    from ml.playing_time import expected_games_from_history
    from ml.vor import attach_vor
    from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
    from pipeline.schema import normalize_dsn

    dsn = normalize_dsn(args.database_url or DEFAULT_HOST_DATABASE_URL)
    with psycopg2.connect(dsn) as conn:
        players = pd.read_sql(
            "SELECT id, full_name, position, team, entry_year, draft_round, draft_number, birth_date FROM players", conn
        )
        logs = pd.read_sql(
            "SELECT player_id, season, week, season_type, position, team, fantasy_points_ppr "
            "FROM game_logs WHERE week BETWEEN 1 AND 18",
            conn,
        )
        adp = pd.read_sql(
            "SELECT season, source, scoring, player_id, position, adp FROM fantasy_adp "
            "WHERE player_id IS NOT NULL AND scoring = 'ppr'",
            conn,
        )

    rows = []
    for season in range(args.start_season, args.end_season + 1):
        market = adp[adp["season"] == season]
        if market.empty:
            rows.append({"season": season, "status": "no_adp"})
            continue
        source = str(market["source"].iloc[0])
        market = market[market["source"] == source]
        base = build_preseason_projections(players, logs, season=season)
        if base.empty:
            rows.append({"season": season, "status": "no_projections"})
            continue

        variants = {
            "production": base,
            "no_multiseason_games": build_preseason_projections(
                players, logs, season=season, use_multiseason_games=False
            ),
            "no_age_curve": build_preseason_projections(
                players, logs, season=season, use_age_curve=False
            ),
            "no_rookie_curve": build_preseason_projections(
                players, logs, season=season, use_rookie_curve=False
            ),
        }
        scored: dict[str, pd.DataFrame] = {}
        for name, frame in variants.items():
            scored[name] = pd.DataFrame(
                attach_vor(frame[["player_id", "position", "projection"]].to_dict("records"))
            ).rename(columns={"vor": name})

        merged = market.rename(columns={"position": "adp_position"}).merge(
            base[["player_id", "position"]], on="player_id", how="inner"
        )
        for name, frame in scored.items():
            merged = merged.merge(frame[["player_id", name]], on="player_id", how="left")
        actuals = (
            logs[logs["season"] == season]
            .groupby("player_id", as_index=False)["fantasy_points_ppr"]
            .sum()
            .rename(columns={"fantasy_points_ppr": "realized"})
        )
        merged = merged.merge(actuals, on="player_id", how="left")
        merged["realized"] = pd.to_numeric(merged["realized"], errors="coerce").fillna(0.0)
        if len(merged) < 24:
            rows.append({"season": season, "status": "too_few_matched", "n": int(len(merged))})
            continue

        # Rank-average of the production board with the market.
        model_rank = merged["production"].rank(ascending=False, method="average")
        market_rank = merged["adp"].rank(ascending=True, method="average")
        merged["market_blend"] = -(0.5 * model_rank + 0.5 * market_rank)

        history = logs[logs["season"] < season].rename(columns={"fantasy_points_ppr": "fantasy_ppr"})
        eval_rows = merged[["player_id", "position"]].copy()
        eval_rows["season"] = season
        eval_rows["week"] = 1
        marcel = attach_marcel(eval_rows, history, stat_col="fantasy_ppr")
        merged = merged.merge(marcel[["player_id", "marcel_baseline"]], on="player_id", how="left")

        positions = merged["position"].to_numpy()
        realized = merged["realized"]
        boards = {name: merged[name] for name in INDEPENDENT + MARKET_AWARE}
        boards["adp"] = -merged["adp"]
        result = {
            name: round(
                evaluate_board(values, merged["adp"], positions, realized)["mean_starter_points"], 2
            )
            for name, values in boards.items()
        }
        rows.append({
            "season": season,
            "source": source,
            "n": int(len(merged)),
            "starter_points": result,
            "beats_adp": {
                name: bool(result[name] > result["adp"]) for name in INDEPENDENT + MARKET_AWARE
            },
        })

    measured = [row for row in rows if "starter_points" in row]
    payload = {
        "league": "8-team PPR 1QB/2RB/2WR/1TE/1FLEX",
        "metric": "mean starting-lineup points, snake draft rotated through all 8 slots",
        "effective_n_trials": effective_n_trials(
            n_stats=1, n_positions=1, n_learners=len(INDEPENDENT) + len(MARKET_AWARE)
        ),
        "multiplicity_note": (
            "several boards were scored on the same six seasons; a single winning variant "
            "is not evidence at face value and must survive the promotion gate"
        ),
        "power_warning": (
            "differences below roughly 130 mean starting-lineup points are inside this "
            "metric's noise on six seasons; use reports/draft_component_eval.json to "
            "choose components"
        ),
        "seasons": rows,
        "beats_adp_seasons": {
            name: f"{sum(1 for row in measured if row['beats_adp'][name])}/{len(measured)}"
            for name in INDEPENDENT + MARKET_AWARE
        },
        "mean_starter_points": {
            name: round(float(np.mean([row["starter_points"][name] for row in measured])), 2)
            for name in tuple(INDEPENDENT) + MARKET_AWARE + ("adp",)
        } if measured else {},
        "market_aware_variants": list(MARKET_AWARE),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({k: payload[k] for k in ("beats_adp_seasons", "mean_starter_points", "effective_n_trials")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
