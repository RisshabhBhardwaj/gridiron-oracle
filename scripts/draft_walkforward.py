#!/usr/bin/env python3
"""Walk-forward 8-team PPR draft evaluation vs ADP and Marcel, seasons 2019-2025.

Uses only ``game_logs`` with ``season < target``. Target-season OOF is refused.

Primary metric is roster-aware (:mod:`ml.draft_sim`): run a snake draft, field a
legal 1QB/2RB/2WR/1TE/1FLEX lineup, count what it scored. The previous
cross-position ``points_lost_vs_optimal`` metric is retained under
``legacy_points_lost`` for continuity, but it is *not* an acceptance metric --
its oracle is the top 24 raw scorers, which in this league is 9-11 quarterbacks
and an illegal roster.

The new metric is validated before it is used to judge the model: ADP must beat
a naive Marcel rate board in a majority of seasons. That comparison has a known
answer and does not involve the model, so a metric that fails it is broken
regardless of what it says about the board under test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BOARDS = ("model_vor", "adp", "marcel_vor", "marcel_raw")


def _season_row(season: int, source: str, merged, positions, realized) -> dict:
    from ml.consensus_baseline import points_lost_vs_optimal, top_n_hit_rate
    from ml.draft_sim import evaluate_board, positional_points_lost

    scores = {
        "model_vor": merged["vor"],
        "adp": -merged["adp"],
        "marcel_vor": merged["marcel_vor"],
        "marcel_raw": merged["marcel_baseline"],
    }
    lineup = {
        name: evaluate_board(values, merged["adp"], positions, realized)
        for name, values in scores.items()
    }
    legacy = {
        name: points_lost_vs_optimal(
            values.fillna(0).to_numpy(), realized.to_numpy(), slots=24
        )
        for name, values in scores.items()
    }
    return {
        "season": season,
        "source": source,
        "n": int(len(merged)),
        "n_zero_realized": int((realized <= 0).sum()),
        "starter_points": {name: round(lineup[name]["mean_starter_points"], 2) for name in BOARDS},
        "starter_points_detail": {name: lineup[name] for name in BOARDS},
        "legacy_points_lost": {name: legacy[name] for name in BOARDS},
        "positional_points_lost": {
            "model_vor": positional_points_lost(merged, "projection"),
            "adp": positional_points_lost(merged, "adp", ascending=True),
            "marcel_vor": positional_points_lost(merged, "marcel_season"),
        },
        "top24_hit_rate_vs_adp": top_n_hit_rate(
            merged["vor"].rank(ascending=False, method="average"),
            merged["adp"].rank(ascending=True, method="average"),
            n=24,
        ),
        "beats_adp": bool(
            lineup["model_vor"]["mean_starter_points"] > lineup["adp"]["mean_starter_points"]
        ),
        "beats_marcel": bool(
            lineup["model_vor"]["mean_starter_points"] > lineup["marcel_vor"]["mean_starter_points"]
        ),
        "adp_beats_marcel_raw": bool(
            lineup["adp"]["mean_starter_points"] > lineup["marcel_raw"]["mean_starter_points"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--start-season", type=int, default=2019)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--source", default=None, help="fantasy_adp.source; default = first available")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "draft_walkforward.json")
    args = parser.parse_args()

    import pandas as pd
    import psycopg2

    from ml.baselines import attach_marcel
    from ml.draft_projection import build_preseason_projections
    from ml.vor import attach_vor
    from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
    from pipeline.schema import normalize_dsn

    dsn = normalize_dsn(args.database_url or DEFAULT_HOST_DATABASE_URL)
    with psycopg2.connect(dsn) as conn:
        players = pd.read_sql("SELECT id, full_name, position, team, entry_year, draft_round, draft_number, birth_date FROM players", conn)
        logs = pd.read_sql(
            """
            SELECT player_id, season, week, season_type, position, team, fantasy_points_ppr
            FROM game_logs
            WHERE week BETWEEN 1 AND 18
            """,
            conn,
        )
        adp = pd.read_sql(
            """
            SELECT season, source, scoring, player_id, position, adp
            FROM fantasy_adp
            WHERE player_id IS NOT NULL AND scoring = 'ppr'
            """,
            conn,
        )

    rows = []
    for season in range(args.start_season, args.end_season + 1):
        seasonal_adp = adp[adp["season"] == season]
        if logs[logs["season"] < season].empty:
            rows.append({"season": season, "status": "insufficient_history"})
            continue
        if seasonal_adp.empty:
            rows.append({"season": season, "status": "no_adp"})
            continue
        source = args.source or str(seasonal_adp["source"].iloc[0])
        market = seasonal_adp[seasonal_adp["source"] == source]
        if market.empty:
            rows.append({"season": season, "status": f"no_adp_source_{source}"})
            continue
        projections = build_preseason_projections(players, logs, season=season)
        if projections.empty:
            rows.append({"season": season, "status": "no_projections"})
            continue
        vor = pd.DataFrame(attach_vor(projections.to_dict("records")))
        actuals = (
            logs[logs["season"] == season]
            .groupby("player_id", as_index=False)["fantasy_points_ppr"]
            .sum()
            .rename(columns={"fantasy_points_ppr": "realized"})
        )
        merged = market.rename(columns={"position": "adp_position"}).merge(vor, on="player_id", how="inner")
        # Left join, not inner: a drafted player who never took a snap scored
        # zero, and a board that correctly faded him must be credited for it.
        # An inner join silently deletes exactly the busts.
        merged = merged.merge(actuals, on="player_id", how="left")
        merged["realized"] = pd.to_numeric(merged["realized"], errors="coerce").fillna(0.0)
        if len(merged) < 24:
            rows.append({"season": season, "source": source, "status": "too_few_matched", "n": int(len(merged))})
            continue

        history = logs[logs["season"] < season].rename(columns={"fantasy_points_ppr": "fantasy_ppr"})
        eval_rows = merged[["player_id", "position"]].copy()
        eval_rows["season"] = season
        eval_rows["week"] = 1
        marcel = attach_marcel(eval_rows, history, stat_col="fantasy_ppr")
        merged = merged.merge(marcel[["player_id", "marcel_baseline"]], on="player_id", how="left")
        # Marcel is a per-game rate. Give it the same season-total and
        # replacement-level treatment the model gets, so the comparison
        # isolates projection quality rather than crediting the model for VOR.
        merged["marcel_season"] = pd.to_numeric(merged["marcel_baseline"], errors="coerce") * pd.to_numeric(
            merged["games_played_prior"], errors="coerce"
        )
        marcel_vor = pd.DataFrame(
            attach_vor(
                merged.assign(projection=merged["marcel_season"])[
                    ["player_id", "position", "projection"]
                ].to_dict("records")
            )
        ).rename(columns={"vor": "marcel_vor"})
        merged = merged.merge(marcel_vor[["player_id", "marcel_vor"]], on="player_id", how="left")

        rows.append(_season_row(season, source, merged, merged["position"].to_numpy(), merged["realized"]))

    measured = [row for row in rows if "beats_adp" in row]
    adp_over_marcel = sum(1 for row in measured if row["adp_beats_marcel_raw"])
    mean_points = {
        name: round(sum(row["starter_points"][name] for row in measured) / len(measured), 2)
        for name in BOARDS
    } if measured else {}
    # Season counts over six seasons are close to a coin flip; the mean gap is
    # the statistic that decides the acceptance bar, so it goes in the headline.
    if measured:
        import statistics

        diffs = [row["starter_points"]["model_vor"] - row["starter_points"]["adp"] for row in measured]
        gap = sum(diffs) / len(diffs)
        sd = statistics.stdev(diffs) if len(diffs) > 1 else float("nan")
        se = sd / (len(diffs) ** 0.5) if len(diffs) > 1 else float("nan")
        # 21 seasons would be needed to resolve a gap this size at 95%, and only
        # six seasons of resolved ADP exist. Reporting a verdict here would be
        # asserting something the data cannot support in either direction.
        underpowered = not (se == se) or abs(gap) < 1.96 * se
        power = {
            "mean_gap_vs_adp": round(gap, 2),
            "sd_across_seasons": round(sd, 2),
            "standard_error": round(se, 2),
            "ci95": [round(gap - 1.96 * se, 2), round(gap + 1.96 * se, 2)],
            "seasons_needed_for_95pct": (
                round((1.96 * sd / abs(gap)) ** 2) if gap and sd == sd else None
            ),
            "underpowered": bool(underpowered),
        }
        headline = (
            f"UNDERPOWERED: board {mean_points['model_vor']} vs ADP {mean_points['adp']}, "
            f"gap {gap:+.1f} +/- {1.96 * se:.1f} (95%). Six seasons of ADP exist and about "
            f"{power['seasons_needed_for_95pct']} would be needed. Neither 'beats' nor 'loses to' "
            f"ADP is supportable from this metric -- see reports/draft_component_eval.json, "
            f"which resolves component questions over ~1030 player-seasons."
            if underpowered else
            f"ACCEPTANCE {'MET' if gap > 0 else 'NOT MET'}: gap {gap:+.1f} +/- {1.96 * se:.1f} (95%)"
        )
    else:
        headline, power = "no seasons measured", {}
    metric_valid = bool(measured) and adp_over_marcel > len(measured) / 2
    payload = {
        "league": "8-team PPR 1QB/2RB/2WR/1TE/1FLEX",
        "primary_metric": "mean starting-lineup points, snake draft rotated through all 8 slots",
        "metric_validation": {
            "check": "a correct roster-aware metric ranks ADP above a naive Marcel rate board",
            "adp_beats_marcel_raw_seasons": f"{adp_over_marcel}/{len(measured)}",
            "valid": metric_valid,
            "note": "computed without reference to the model board; a failure invalidates the metric, not the model",
        },
        "adp_caveat": (
            "fantasy_adp source is generic historical ADP, not FFC teams=8; "
            "the 8-team bar in the design spec is not yet fully met by the opponent"
        ),
        "known_provenance_gap": (
            "players.team is the current team, so rookie vacated-opportunity shares in "
            "back-seasons are computed against present-day rosters"
        ),
        "seasons": rows,
        "n_seasons_measured": len(measured),
        "beats_adp": int(sum(1 for row in measured if row["beats_adp"])),
        "beats_marcel": int(sum(1 for row in measured if row["beats_marcel"])),
        "mean_starter_points": mean_points,
        "mean_gap_vs_adp": (
            round(mean_points["model_vor"] - mean_points["adp"], 2) if mean_points else None
        ),
        "headline": headline,
        "power": power,
        "beats_marcel_caveat": (
            "the board's veteran per-game rate IS the Marcel rate (ml.draft_projection."
            "apply_marcel_playing_time), so this counts the VOR translation, rookie priors "
            "and the games multiplier -- not a better rate projection"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps(
        {
            "headline": payload["headline"],
            "mean_starter_points": payload["mean_starter_points"],
            "metric_validation": payload["metric_validation"],
            "n_seasons_measured": payload["n_seasons_measured"],
            "beats_adp": payload["beats_adp"],
            "beats_marcel": payload["beats_marcel"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
