"""pipeline/line_snapshots.py

Causal line snapshots (decision 0008, item 3).

The ``games`` row carries only the latest-seen line: every normalize run
overwrites ``total_line``/``spread_line`` with the newest feed value, so the
history of what the line was at any earlier moment is lost. A training or
forward row built later then reads a line that may not have existed before
kickoff.

``game_line_snapshots`` keeps that history. Each normalize run appends one
row per game whose feed carries a line (never updates, never deletes):

    (game_id, captured_at, spread_line, total_line, source)

Readers use :func:`get_line_as_of`, which returns the latest snapshot
captured strictly before ``as_of`` — and, when the kickoff time is known,
strictly before kickoff too. A snapshot captured at or after either bound is
never returned.

When no snapshot qualifies, :func:`elo_implied_line` derives a line from the
pregame Elo ratings only (``TeamEloSystem.get_snapshot``, which is the state
heading into that week — never the current ratings, which would leak later
results). The Elo-to-points conversion is a fallback-only heuristic, not a
market model; it exists so a missing line degrades to a matchup-aware prior
instead of a fixed league median.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Optional

TABLE = "game_line_snapshots"

SNAPSHOT_SOURCE_SCHEDULE = "nflreadpy_schedule"

# INSERT used by pipeline/normalize.py (which owns the execute_values import
# its tests patch). Keeps the SQL text in one place.
SNAPSHOT_INSERT_SQL = """
INSERT INTO game_line_snapshots (
    game_id, spread_line, total_line, captured_at, source
) VALUES %s
ON CONFLICT (game_id, captured_at) DO NOTHING
"""

# Std of the NFL victory margin, used only to invert an Elo win probability
# into a point spread (Stern 1991 reports ~13.45 for the NFL). Fallback-only.
_MARGIN_STD = 13.45

# League-average total. Only the anchor: the Elo term moves it off this value
# for any matchup whose pregame ratings differ from average.
_LEAGUE_AVG_TOTAL = 45.0

# Points of total per 400 Elo points of net offensive advantage
# (offenses above average raise the total, defenses above average lower it).
# Fallback-only heuristic; magnitude chosen so a 100-Elo net edge moves the
# total by a little over a point.
_TOTAL_POINTS_PER_400_ELO = 5.0

_ELO_PAR = 1500.0


@dataclass(frozen=True)
class GameLine:
    """A (spread, total) pair. Spread follows the nflreadpy convention:
    negative means the home team is favored."""

    spread_line: Optional[float]
    total_line: Optional[float]


def snapshot_batch(
    game_dicts: list[dict[str, Any]],
    captured_at: Any,
    source: str = SNAPSHOT_SOURCE_SCHEDULE,
) -> list[tuple[Any, ...]]:
    """Build append rows for games whose feed carries at least one line.

    Games with neither line observed contribute nothing — there is no history
    to preserve. Pure function (no DB) so normalize stays unit-testable.
    """
    rows: list[tuple[Any, ...]] = []
    for g in game_dicts:
        spread = g.get("spread_line")
        total = g.get("total_line")
        if spread is None and total is None:
            continue
        rows.append((g.get("id"), spread, total, captured_at, source))
    return rows


def get_line_as_of(
    conn: Any,
    game_id: str,
    as_of: Any,
    *,
    kickoff_at: Any = None,
) -> Optional[GameLine]:
    """Latest line snapshot captured strictly before ``as_of``.

    When ``kickoff_at`` is given, snapshots captured at or after kickoff are
    excluded as well, so a post-kickoff line move can never leak into a
    pre-kickoff row. Returns None when no snapshot qualifies.
    """
    sql = (
        "SELECT spread_line, total_line FROM game_line_snapshots "
        "WHERE game_id = %s AND captured_at < %s"
    )
    params: list[Any] = [game_id, as_of]
    if kickoff_at is not None:
        sql += " AND captured_at < %s"
        params.append(kickoff_at)
    sql += " ORDER BY captured_at DESC LIMIT 1"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    if row is None:
        return None
    if isinstance(row, dict):
        return GameLine(
            spread_line=row.get("spread_line"),
            total_line=row.get("total_line"),
        )
    return GameLine(spread_line=row[0], total_line=row[1])


def elo_implied_line(
    home_team: str,
    away_team: str,
    season: int,
    week: int,
    elo: Any,
) -> GameLine:
    """Line implied by pregame Elo ratings only.

    ``elo`` is a fitted ``TeamEloSystem``. Only ``get_snapshot`` (the state
    heading into that week) is read — never current ratings — so results from
    this week or later cannot leak in. The win probability comes from the
    existing Elo math (``ml.team_elo._expected_score``, including the home
    edge); only the probability-to-points inversion below is new, and it is
    fallback-only.
    """
    from ml.team_elo import _expected_score

    home_snap = elo.get_snapshot(home_team, season, week)
    away_snap = elo.get_snapshot(away_team, season, week)
    home_off = home_snap.off_elo if home_snap else _ELO_PAR
    home_def = home_snap.def_elo if home_snap else _ELO_PAR
    away_off = away_snap.off_elo if away_snap else _ELO_PAR
    away_def = away_snap.def_elo if away_snap else _ELO_PAR

    home_combined = (home_off + home_def) / 2.0
    away_combined = (away_off + away_def) / 2.0
    prob_home = _expected_score(home_combined, away_combined, a_is_home=True)
    # Clamp: inv_cdf is undefined at exactly 0/1 (unreachable from the Elo
    # logistic anyway, but a degenerate Elo state must not crash the build).
    prob_home = min(max(prob_home, 1e-6), 1.0 - 1e-6)
    spread_home = -NormalDist().inv_cdf(prob_home) * _MARGIN_STD

    net_offense = (home_off - _ELO_PAR) + (away_off - _ELO_PAR) - (
        (home_def - _ELO_PAR) + (away_def - _ELO_PAR)
    )
    total = _LEAGUE_AVG_TOTAL + _TOTAL_POINTS_PER_400_ELO * net_offense / 400.0
    return GameLine(spread_line=spread_home, total_line=total)


def causal_line_for(
    conn: Any,
    *,
    game_id: str,
    as_of: Any,
    home_team: str,
    away_team: str,
    season: int,
    week: int,
    kickoff_at: Any = None,
    elo: Any = None,
) -> GameLine:
    """Snapshot line if one was known before ``as_of`` (and kickoff), else Elo.

    A snapshot leg that is present is used as-is; a leg the snapshot feed
    never carried falls back to the Elo-implied value, so the result never
    carries a None both paths downstream must special-case. With no snapshot
    at all the whole line is Elo-implied — never a fixed median.
    """
    if elo is None:
        from ml.team_elo import get_elo_system

        elo = get_elo_system()
    fallback = elo_implied_line(home_team, away_team, season, week, elo)
    snap = get_line_as_of(conn, game_id, as_of, kickoff_at=kickoff_at)
    if snap is None:
        return fallback
    spread = snap.spread_line if snap.spread_line is not None else fallback.spread_line
    total = snap.total_line if snap.total_line is not None else fallback.total_line
    return GameLine(spread_line=spread, total_line=total)
