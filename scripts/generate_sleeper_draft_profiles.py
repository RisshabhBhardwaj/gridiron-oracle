#!/usr/bin/env python3
"""
Generate transparent per-manager profiles from historical Sleeper picks.

Estimates per-manager drafting tendencies (QB/TE timing, early RB/WR share, ADP reach/fall)
with empirical Bayes shrinkage toward the league mean (k=30).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import psycopg2
import psycopg2.extras

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn

EARLY_PICK_CUTOFF = 24  # first three rounds in this eight-team league
SHRINKAGE_K = 30.0      # Empirical Bayes shrinkage prior weight

NOTE_CAVEAT = (
    "With 8 owners × ~68 picks each, positional-need-conditional behavior is not estimable. "
    "Per-manager parameters (first-QB/TE timing, early RB/WR share, ADP delta) are estimated "
    "with heavy empirical Bayes shrinkage toward the league mean (k=30). Negative ADP delta "
    "indicates habitual reaching."
)


def _as_float(values: list[float | int]) -> float | None:
    return round(float(mean(values)), 2) if values else None


def _shrink(sample_val: float | None, league_val: float, n: int, k: float = SHRINKAGE_K) -> float:
    """Empirical Bayes shrinkage toward league mean: weight = n / (n + k)."""
    if sample_val is None or math.isnan(sample_val):
        return round(float(league_val), 2)
    weight = n / (n + k) if (n + k) > 0 else 0.0
    return round(float(weight * sample_val + (1.0 - weight) * league_val), 2)


def build_profiles(rows: list[dict[str, Any]], names: dict[str, str]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("owner_id"):
            grouped[str(row["owner_id"])].append(row)

    # 1. First pass: extract raw stats per owner and pool for league averages
    raw_profiles = []
    all_first_qbs: list[float] = []
    all_first_tes: list[float] = []
    all_adp_deltas: list[float] = []
    all_early_rb_shares: list[float] = []
    all_early_wr_shares: list[float] = []

    for owner_id, picks in grouped.items():
        by_draft: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pick in picks:
            by_draft[str(pick.get("draft_id", "default"))].append(pick)

        position_counts = Counter(str(pick.get("position") or "Unknown") for pick in picks)
        early = [pick for pick in picks if int(pick["pick_no"]) <= EARLY_PICK_CUTOFF]
        early_positions = Counter(str(pick.get("position") or "Unknown") for pick in early)

        first_qb = [min(int(p["pick_no"]) for p in draft if p.get("position") == "QB")
                    for draft in by_draft.values() if any(p.get("position") == "QB" for p in draft)]
        first_te = [min(int(p["pick_no"]) for p in draft if p.get("position") == "TE")
                    for draft in by_draft.values() if any(p.get("position") == "TE" for p in draft)]

        adp_deltas = [
            float(p["adp_delta"])
            for p in picks
            if p.get("adp_delta") is not None and not math.isnan(float(p["adp_delta"]))
        ]

        rb_share = (sum(1 for pick in early if pick.get("position") == "RB") / len(early)) if early else None
        wr_share = (sum(1 for pick in early if pick.get("position") == "WR") / len(early)) if early else None

        if first_qb:
            all_first_qbs.extend(first_qb)
        if first_te:
            all_first_tes.extend(first_te)
        if adp_deltas:
            all_adp_deltas.extend(adp_deltas)
        if rb_share is not None:
            all_early_rb_shares.append(rb_share)
        if wr_share is not None:
            all_early_wr_shares.append(wr_share)

        raw_profiles.append({
            "owner_id": owner_id,
            "display_name": names.get(owner_id, owner_id),
            "draft_slot": picks[0].get("draft_slot") if picks else None,
            "drafts": len(by_draft),
            "picks": len(picks),
            "positions": dict(sorted(position_counts.items())),
            "early_pick_positions": dict(sorted(early_positions.items())),
            "rb_wr_share_first_24": round(sum(1 for pick in early if pick.get("position") in {"RB", "WR"}) / len(early), 3) if early else None,
            "first_qb": first_qb,
            "first_te": first_te,
            "adp_deltas": adp_deltas,
            "early_rb_share": rb_share,
            "early_wr_share": wr_share,
        })

    # League defaults if empty
    league_qb_mean = mean(all_first_qbs) if all_first_qbs else 48.0
    league_te_mean = mean(all_first_tes) if all_first_tes else 64.0
    league_adp_delta_mean = mean(all_adp_deltas) if all_adp_deltas else 0.0
    league_adp_delta_sd = stdev(all_adp_deltas) if len(all_adp_deltas) > 1 else 10.0
    league_rb_share = mean(all_early_rb_shares) if all_early_rb_shares else 0.5
    league_wr_share = mean(all_early_wr_shares) if all_early_wr_shares else 0.4

    # 2. Second pass: apply shrinkage
    profiles: list[dict[str, Any]] = []
    for raw in raw_profiles:
        n_picks = raw["picks"]
        n_adp = len(raw["adp_deltas"])
        qb_mean = mean(raw["first_qb"]) if raw["first_qb"] else None
        te_mean = mean(raw["first_te"]) if raw["first_te"] else None
        adp_mean = mean(raw["adp_deltas"]) if raw["adp_deltas"] else None
        adp_sd = stdev(raw["adp_deltas"]) if len(raw["adp_deltas"]) > 1 else (5.0 if raw["adp_deltas"] else None)

        profiles.append({
            "owner_id": raw["owner_id"],
            "display_name": raw["display_name"],
            "draft_slot": raw["draft_slot"],
            "drafts": raw["drafts"],
            "picks": n_picks,
            "positions": raw["positions"],
            "early_pick_positions": raw["early_pick_positions"],
            "rb_wr_share_first_24": raw["rb_wr_share_first_24"],
            "first_qb_pick_average": _as_float(raw["first_qb"]),
            "first_te_pick_average": _as_float(raw["first_te"]),
            "first_qb_pick_median": float(median(raw["first_qb"])) if raw["first_qb"] else None,
            "first_te_pick_median": float(median(raw["first_te"])) if raw["first_te"] else None,
            "first_qb_pick_shrunk": _shrink(qb_mean, league_qb_mean, n_picks),
            "first_te_pick_shrunk": _shrink(te_mean, league_te_mean, n_picks),
            "n_adp_matched": n_adp,
            "adp_delta_mean": round(float(adp_mean), 2) if adp_mean is not None else None,
            "adp_delta_sd": round(float(adp_sd), 2) if adp_sd is not None else None,
            "adp_delta_mean_shrunk": _shrink(adp_mean, league_adp_delta_mean, n_adp),
            "adp_delta_sd_shrunk": _shrink(adp_sd, league_adp_delta_sd, n_adp),
            "early_rb_share_shrunk": _shrink(raw["early_rb_share"], league_rb_share, n_picks),
            "early_wr_share_shrunk": _shrink(raw["early_wr_share"], league_wr_share, n_picks),
        })

    return {
        "method": {
            "early_pick_cutoff": EARLY_PICK_CUTOFF,
            "shrinkage_k": SHRINKAGE_K,
            "note": NOTE_CAVEAT,
        },
        "profiles": sorted(profiles, key=lambda p: (p["display_name"].lower(), p["owner_id"])),
    }


def load_profiles(
    database_url: str,
    seasons: list[int] | None = None,
    league_id: str | None = None,
) -> dict[str, Any]:
    with psycopg2.connect(normalize_dsn(database_url)) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            query = """
                SELECT p.draft_id, p.season, p.league_id, p.owner_id, p.pick_no, p.draft_slot,
                       p.position, p.player_id, a.adp, (p.pick_no - a.adp) AS adp_delta
                FROM sleeper_league_draft_picks p
                LEFT JOIN fantasy_player_ids f ON f.sleeper_id = p.player_id
                LEFT JOIN fantasy_adp a ON a.player_id = f.gsis_id AND a.season = p.season AND a.scoring = 'ppr'
                WHERE 1=1
            """
            params: list[Any] = []
            if seasons:
                query += " AND p.season = ANY(%s)"
                params.append(seasons)
            if league_id:
                query += " AND p.league_id = %s"
                params.append(league_id)
            query += " ORDER BY p.draft_id, p.pick_no"

            cur.execute(query, params)
            rows = [dict(row) for row in cur.fetchall()]

            cur.execute("""
                SELECT DISTINCT ON (owner_id) owner_id, COALESCE(display_name, username, owner_id) AS name
                FROM sleeper_league_members ORDER BY owner_id, imported_at DESC
            """)
            names = {str(row["owner_id"]): str(row["name"]) for row in cur.fetchall()}
    return build_profiles(rows, names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    parser.add_argument("--seasons", type=int, nargs="*", default=None, help="Filter to specific seasons")
    parser.add_argument("--league-id", default=None, help="Filter to specific league ID")
    parser.add_argument("--output", type=Path, default=Path("reports/sleeper_league_profiles.json"))
    args = parser.parse_args()

    payload = load_profiles(args.database_url, seasons=args.seasons, league_id=args.league_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
