"""Rookie priors from draft capital and vacated landing-team opportunity."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

ELIGIBLE = frozenset({"QB", "RB", "WR", "TE"})


def vacated_opportunity_share(
    prior_logs: pd.DataFrame,
    roster: pd.DataFrame,
    *,
    season: int,
) -> dict[tuple[str, str], float]:
    """Share of prior-season positional PPR that left the landing team.

    ``roster`` is the *target* season roster (player_id, team, position).
    Opportunity from players not on that roster is vacated.
    """
    if prior_logs.empty or roster.empty:
        return {}
    logs = prior_logs.copy()
    logs = logs[logs["season"] == season - 1] if "season" in logs.columns else logs
    logs["position"] = logs["position"].astype(str).str.upper()
    logs["team"] = logs["team"].astype(str).str.upper()
    logs["fantasy_points_ppr"] = pd.to_numeric(logs["fantasy_points_ppr"], errors="coerce")
    logs = logs.dropna(subset=["fantasy_points_ppr"])
    returning = set(roster["player_id"].astype(str))
    team_pos = logs.groupby(["team", "position"], as_index=False)["fantasy_points_ppr"].sum()
    gone = logs[~logs["player_id"].astype(str).isin(returning)]
    vacated = gone.groupby(["team", "position"], as_index=False)["fantasy_points_ppr"].sum()
    merged = team_pos.merge(
        vacated, on=["team", "position"], how="left", suffixes=("_all", "_gone")
    )
    merged["fantasy_points_ppr_gone"] = merged["fantasy_points_ppr_gone"].fillna(0.0)
    shares: dict[tuple[str, str], float] = {}
    for row in merged.itertuples(index=False):
        total = float(row.fantasy_points_ppr_all)
        if total <= 0:
            continue
        shares[(str(row.team), str(row.position))] = float(row.fantasy_points_ppr_gone) / total
    return shares


def rookie_multiplier(draft_round: object, vacated_share: float) -> float:
    try:
        rnd = int(draft_round) if draft_round is not None else 7
    except (TypeError, ValueError):
        rnd = 7
    rnd = min(max(rnd, 1), 8)
    capital = 0.18 + 0.14 * ((8 - rnd) / 7.0)
    return float(capital + 0.55 * max(0.0, min(1.0, vacated_share)))


def label_projection_basis(*, historical_games: int, is_rookie: bool) -> str:
    if is_rookie or historical_games == 0:
        return "rookie_draft_capital_vacated_opportunity"
    return "historical_ppr_x_games_prior"


# Rookie fantasy production is strongly and monotonically predicted by draft
# capital. Measured over 2019-2025 rookie seasons, mean season PPR by overall
# pick: 1-15 -> 194, 16-32 -> 165, 33-64 -> 113, 65-105 -> 67, 106-160 -> 54,
# 161-262 -> 40, undrafted -> 18. The previous prior (a position-mean rate times
# a 0.18-0.87 multiplier) projected the average drafted rookie at 33 points
# against 140 realized -- 24% of the truth -- and underranked rookies by 57
# board positions where the market missed by 14. These are fitted, not chosen.
CAPITAL_BIN_EDGES: tuple[int, ...] = (15, 32, 64, 105, 160, 262)
CAPITAL_BIN_LABELS: tuple[str, ...] = (
    "1-15", "16-32", "33-64", "65-105", "106-160", "161-262", "undrafted"
)
ROOKIE_SHRINK_ROWS = 8.0
VACATED_LOW, VACATED_HIGH = 0.85, 1.15


def capital_bin(draft_number: object) -> str:
    """Bucket an overall draft pick. Missing capital is treated as undrafted."""
    try:
        pick = float(draft_number)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return CAPITAL_BIN_LABELS[-1]
    if pick != pick or pick <= 0:
        return CAPITAL_BIN_LABELS[-1]
    for label, edge in zip(CAPITAL_BIN_LABELS, CAPITAL_BIN_EDGES):
        if pick <= edge:
            return label
    return CAPITAL_BIN_LABELS[-1]


def effective_pick(players: pd.DataFrame) -> pd.Series:
    """Overall draft pick per player, falling back to the round's midpoint.

    ``draft_number`` is the better signal but is not present in every caller's
    projection of the players table, and requiring it turned a missing column
    into a KeyError. A player with only ``draft_round`` is placed at the middle
    of that round; a player with neither is undrafted, which is its own bin.
    """
    index = players["id"] if "id" in players.columns else players["player_id"]
    number = (
        pd.to_numeric(players["draft_number"], errors="coerce")
        if "draft_number" in players.columns
        else pd.Series(np.nan, index=players.index)
    )
    if "draft_round" in players.columns:
        rounds = pd.to_numeric(players["draft_round"], errors="coerce")
        midpoint = (rounds - 1.0) * 32.0 + 16.0
        number = number.fillna(midpoint)
    return pd.Series(number.to_numpy(), index=index.to_numpy())


def fit_rookie_curve(
    game_logs: pd.DataFrame,
    players: pd.DataFrame,
    *,
    max_season: int,
) -> dict:
    """Fit rookie per-game rate and games played from draft capital.

    A rookie season is one where ``season == entry_year``. Only seasons
    ``<= max_season`` are used; later rows raise rather than leak.
    """
    logs = game_logs.copy()
    logs["season"] = pd.to_numeric(logs["season"], errors="coerce")
    logs["fantasy_points_ppr"] = pd.to_numeric(logs["fantasy_points_ppr"], errors="coerce")
    logs = logs.dropna(subset=["season", "fantasy_points_ppr"])
    if not logs.empty and logs["season"].max() > max_season:
        raise ValueError(
            f"rookie curve for max_season={max_season} was handed rows from "
            f"season {int(logs['season'].max())}; that is target-season leakage"
        )

    roster = players.rename(columns={"id": "player_id"}).copy()
    roster["position"] = roster["position"].astype(str).str.upper()
    roster = roster[roster["position"].isin(ELIGIBLE)]
    roster["entry_year"] = pd.to_numeric(roster.get("entry_year"), errors="coerce")
    roster["draft_number"] = roster["player_id"].map(effective_pick(players))

    seasons = (
        logs.groupby(["player_id", "season"], as_index=False)
        .agg(points=("fantasy_points_ppr", "sum"), games=("fantasy_points_ppr", "size"))
    )
    frame = seasons.merge(
        roster[["player_id", "position", "entry_year", "draft_number"]], on="player_id", how="inner"
    )
    frame = frame[frame["entry_year"] == frame["season"]]
    if frame.empty:
        return {"by_pos_bin": {}, "by_bin": {}, "overall": (0.0, 0.0), "n": 0}
    frame["ppg"] = frame["points"] / frame["games"].clip(lower=1)
    frame["bin"] = [capital_bin(value) for value in frame["draft_number"]]

    def _agg(keys: list[str]) -> dict:
        grouped = frame.groupby(keys).agg(ppg=("ppg", "mean"), games=("games", "mean"), n=("ppg", "size"))
        return {
            (key if isinstance(key, tuple) else (key,)): (
                float(row.ppg), float(row.games), int(row.n)
            )
            for key, row in grouped.iterrows()
        }

    overall = (float(frame["ppg"].mean()), float(frame["games"].mean()))
    return {
        "by_pos_bin": _agg(["position", "bin"]),
        "by_bin": _agg(["bin"]),
        "overall": overall,
        "n": int(len(frame)),
    }


def rookie_rate_and_games(
    curve: dict,
    position: str,
    draft_number: object,
    *,
    vacated_share: float = 0.0,
) -> tuple[float, float]:
    """Expected (per-game PPR, games played) for a rookie, shrunk by cell size.

    The position x capital cell is shrunk toward the capital cell, which is far
    better populated: a first-round tight end in a three-season training window
    may be a single player.
    """
    if not curve or not curve.get("n"):
        return 0.0, 14.0
    bucket = capital_bin(draft_number)
    pos = str(position).upper()
    fallback_ppg, fallback_games = curve["overall"]
    bin_ppg, bin_games, _bin_n = curve["by_bin"].get((bucket,), (fallback_ppg, fallback_games, 0))
    cell = curve["by_pos_bin"].get((pos, bucket))
    if cell is None:
        ppg, games = bin_ppg, bin_games
    else:
        cell_ppg, cell_games, cell_n = cell
        weight = cell_n / (cell_n + ROOKIE_SHRINK_ROWS)
        ppg = weight * cell_ppg + (1.0 - weight) * bin_ppg
        games = weight * cell_games + (1.0 - weight) * bin_games
    share = max(0.0, min(1.0, float(vacated_share or 0.0)))
    return float(ppg) * (VACATED_LOW + (VACATED_HIGH - VACATED_LOW) * share), float(games)
