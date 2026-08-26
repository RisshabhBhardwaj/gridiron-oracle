"""
scripts/compute_position_priors.py

Derive ml/kalman_tracker.py's POSITION_PRIORS from game_logs.

POSITION_PRIORS is the Kalman filter's x_0 -- the expected value of ONE
game's box score for a player with no history at all. Its docstring claimed
this script existed and that the numbers were "2019-2025 season averages per
game for each position"; the script did not exist, and that description
describes the wrong population anyway.

The bug it hid: a per-game average taken over every game a position played is
dominated by starters, because starters play most of the games. Applying that
number to a player with no track record says a rostered fourth-string tight
end is expected to post a starting tight end's line. Measured against the
served 2026 board, every prior was 1.5-2.3x the realized debut-season rate:

    WR receiving_yards  55.0 prior vs 24.9 realized
    RB rushing_yards    60.0        vs 26.9
    TE receiving_yards  35.0        vs 19.1
    QB passing_yards   240.0        vs 156.9

That matters more than a cold-start estimate normally would, because
SeasonSimulator._apply_volume_budget allocates a team's yardage budget in
proportion to these rates and the budget is FIXED. Every yard handed to a
player who has never taken a snap is subtracted from that team's actual
starter. On the 2026 roster 294 of 808 players (36%) had zero prior game rows,
including 137 WRs and 76 TEs, which is why the board came out with no player
standing out from any other.

POPULATION: a player's DEBUT season -- the first season they record any
regular-season game log. That is exactly the cold-start case (no prior rows
whatever), rather than a proxy like "few games this year", which still admits
established veterans returning from a missed season.

DENOMINATOR: games the player actually appeared in, NOT roster weeks. x_0 is
the expected observation on a game the player PLAYS; whether he dresses at all
is p_active's job (ml/playing_time.py), and dividing by roster weeks here
would double-count it. Debut seasons where a player appeared and did nothing
are in the sample at 0, which is what pulls these below the starter averages.

Usage:
    python -m scripts.compute_position_priors                  # print the dict
    python -m scripts.compute_position_priors --check          # diff vs current, exit 1 on drift
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import psycopg2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.kalman_tracker import (  # noqa: E402
    KALMAN_STATS,
    POSITION_PRIORS,
    STAT_MULTI_SOURCE_COLS,
    STAT_SOURCE_COL,
)
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL  # noqa: E402

POSITIONS = ("QB", "RB", "WR", "TE")
# Debuts are taken from 2020 on: 2019 is the first season in game_logs, so
# every 2019 player looks like a debut when most of them are veterans whose
# earlier seasons simply are not in the table.
DEFAULT_MIN_DEBUT_SEASON = 2020
# Stats where a tiny non-zero mean is noise, not signal (a QB "receiving"
# yard is a lateral). Rounding to this many places also keeps the emitted
# dict readable.
_DECIMALS = {"passing_yards": 1, "rushing_yards": 1, "receiving_yards": 1, "fantasy_ppr": 1}


def _source_columns() -> list[str]:
    cols: set[str] = set()
    for stat in KALMAN_STATS:
        src = STAT_SOURCE_COL[stat]
        if src in STAT_MULTI_SOURCE_COLS:
            cols.update(STAT_MULTI_SOURCE_COLS[src])
        else:
            cols.add(src)
    return sorted(cols)


def load_debut_games(
    database_url: str, min_debut_season: int = DEFAULT_MIN_DEBUT_SEASON
) -> pd.DataFrame:
    """Every regular-season game row from a player's debut season."""
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'game_logs'"
            )
            available = {r[0] for r in cur.fetchall()}
        # target_share / air_yards_share and friends are computed downstream,
        # not stored per game. Ask only for what the table has; compute_priors
        # leaves the hand-set prior in place for anything missing.
        cols = ", ".join(f'"{c}"' for c in _source_columns() if c in available)
        frame = pd.read_sql(
            f"SELECT player_id, season, position, {cols} "
            "FROM game_logs WHERE season_type = 'REG'",
            conn,
        )
    finally:
        conn.close()
    if frame.empty:
        raise ValueError("game_logs returned no regular-season rows")
    debut = frame.groupby("player_id")["season"].transform("min")
    return frame[(frame["season"] == debut) & (debut >= min_debut_season)].fillna(0.0)


def compute_priors(debut_games: pd.DataFrame) -> dict[str, dict[str, float]]:
    priors: dict[str, dict[str, float]] = {}
    for position in POSITIONS:
        rows = debut_games[debut_games["position"] == position]
        if rows.empty:
            raise ValueError(f"no debut-season rows for position {position!r}")
        stats: dict[str, float] = {}
        for stat in KALMAN_STATS:
            src = STAT_SOURCE_COL[stat]
            if src in STAT_MULTI_SOURCE_COLS:
                present = [c for c in STAT_MULTI_SOURCE_COLS[src] if c in rows.columns]
                value = rows[present].sum(axis=1).mean() if present else 0.0
            elif src in rows.columns:
                value = rows[src].mean()
            else:
                # A stat game_logs does not carry (target_share and friends are
                # computed downstream). Leave the hand-set value alone rather
                # than zeroing a prior this script cannot speak to.
                stats[stat] = float(POSITION_PRIORS.get(position, {}).get(stat, 0.0))
                continue
            # Laterals give QBs a receiving line of about -0.03 yards. A
            # negative expected box score is not a thing; floor at 0.
            stats[stat] = round(max(float(value), 0.0), _DECIMALS.get(stat, 2))
        priors[position] = stats
    return _zero_cross_position_noise(priors)


# A derived mean below this fraction of the largest value any position posts
# for the same stat is a position doing something it essentially never does:
# a receiver's 0.1 passing yards per game is laterals and one trick play in
# six seasons, not a passing rate.
_CROSS_POSITION_NOISE_FLOOR = 0.01


def _zero_cross_position_noise(
    priors: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    """Zero a position's prior for a stat it effectively never records.

    Not cosmetic. SeasonSimulator._draw_stat_samples gates on
    `est <= _ZERO_RATE_EPS` to keep a player out of a stat's pool entirely,
    and _apply_volume_budget then rescales whoever IS in the pool up to the
    team's budget. A 0.1 yd/game passing prior therefore does not stay worth
    0.1 yards -- it puts all 137 cold-start receivers into the passing pool
    and the rescale multiplies them, which took non-QB passing from 0.9% of
    the league to 6.9%. The hand-set priors this script replaces used exact
    zeros for those cells, and that exactness was load-bearing.

    Within-position ratios are left alone: a quarterback really does rush for
    14.8 yards a game (55% of a running back's) and a receiver really does
    take 0.2 carries (3% of a back's), so neither is noise.
    """
    for stat in {s for pos in priors.values() for s in pos}:
        peak = max(priors[pos].get(stat, 0.0) for pos in priors)
        if peak <= 0:
            continue
        for pos in priors:
            if 0.0 < priors[pos].get(stat, 0.0) < peak * _CROSS_POSITION_NOISE_FLOOR:
                priors[pos][stat] = 0.0
    return priors


def format_priors(priors: dict[str, dict[str, float]]) -> str:
    out = ["POSITION_PRIORS: dict[str, dict[str, float]] = {"]
    width = max(len(s) for s in KALMAN_STATS) + 3
    for position in POSITIONS:
        out.append(f'    "{position}": {{')
        for stat in KALMAN_STATS:
            key = f'"{stat}":'
            out.append(f"        {key:<{width}} {priors[position][stat]},")
        out.append("    },")
    out.append("}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    parser.add_argument("--min-debut-season", type=int, default=DEFAULT_MIN_DEBUT_SEASON)
    parser.add_argument(
        "--check", action="store_true",
        help="Compare against the values currently in ml/kalman_tracker.py and "
             "exit 1 if any differ by more than 10%%.",
    )
    args = parser.parse_args()

    debut_games = load_debut_games(args.database_url, args.min_debut_season)
    n_players = debut_games["player_id"].nunique()
    print(
        f"# debut seasons >= {args.min_debut_season}: "
        f"{n_players} players / {len(debut_games)} games",
        file=sys.stderr,
    )
    priors = compute_priors(debut_games)

    if args.check:
        drifted = []
        for position in POSITIONS:
            for stat in KALMAN_STATS:
                current = float(POSITION_PRIORS.get(position, {}).get(stat, 0.0))
                derived = priors[position][stat]
                scale = max(abs(current), abs(derived))
                if scale > 0.05 and abs(current - derived) / scale > 0.10:
                    drifted.append((position, stat, current, derived))
        if drifted:
            print("POSITION_PRIORS drift vs game_logs:")
            for position, stat, current, derived in drifted:
                print(f"  {position:<3} {stat:<22} current {current:>8.2f}  derived {derived:>8.2f}")
            return 1
        print("POSITION_PRIORS match game_logs within 10%.")
        return 0

    print(format_priors(priors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
