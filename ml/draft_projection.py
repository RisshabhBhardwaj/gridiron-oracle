"""Causal, preseason full-season PPR projection construction for draft boards."""

from __future__ import annotations

import pandas as pd

ELIGIBLE_POSITIONS = ("QB", "RB", "WR", "TE")


def build_preseason_projections(
    players: pd.DataFrame,
    game_logs: pd.DataFrame,
    *,
    season: int,
) -> pd.DataFrame:
    """Build fixed-universe annual projections from information before ``season``.

    The annual estimate is a recency-weighted PPR per-game prior multiplied by
    a shrunk *historical* games-played prior.  It intentionally receives no
    target-season rows, OOF predictions, schedules, or realized availability.
    """
    required_players = {"id", "full_name", "position"}
    required_logs = {"player_id", "season", "fantasy_points_ppr"}
    if required_players - set(players.columns) or required_logs - set(game_logs.columns):
        raise ValueError("players and game_logs are missing required draft projection columns")

    universe = players.copy()
    universe["position"] = universe["position"].astype(str).str.upper()
    universe = universe[universe["position"].isin(ELIGIBLE_POSITIONS)]
    if "entry_year" in universe:
        universe = universe[universe["entry_year"].isna() | (universe["entry_year"] <= season)]
    universe = universe.drop_duplicates("id").rename(columns={"id": "player_id", "full_name": "player_name"})

    history = game_logs.copy()
    history = history[history["season"] < season]
    if "week" in history:
        history = history[(history["week"] >= 1) & (history["week"] <= 18)]
    if "season_type" in history:
        history = history[history["season_type"].fillna("REG").eq("REG")]
    history["fantasy_points_ppr"] = pd.to_numeric(history["fantasy_points_ppr"], errors="coerce")
    history = history.dropna(subset=["fantasy_points_ppr"])
    history["recency_weight"] = 0.70 ** (season - 1 - history["season"])

    # Position-level fallback ensures every eligible player in the fixed universe
    # receives a projection, including rookies and players without a prior game.
    position_lookup = universe[["player_id", "position"]]
    history_with_pos = history.merge(position_lookup, on="player_id", how="left")
    position_means = history_with_pos.groupby("position").apply(
        lambda frame: (frame["fantasy_points_ppr"] * frame["recency_weight"]).sum() / frame["recency_weight"].sum(),
        include_groups=False,
    ).to_dict()
    league_mean = float(history["fantasy_points_ppr"].mean()) if not history.empty else 0.0

    grouped = history.groupby("player_id")
    rows: list[dict] = []
    for player in universe.to_dict("records"):
        frame = grouped.get_group(player["player_id"]) if player["player_id"] in grouped.groups else pd.DataFrame()
        fallback = float(position_means.get(player["position"], league_mean))
        if frame.empty:
            per_game_mean, historical_games = fallback, 0
        else:
            per_game_mean = float((frame["fantasy_points_ppr"] * frame["recency_weight"]).sum() / frame["recency_weight"].sum())
            historical_games = int(len(frame))
        recent_games = int((frame["season"] == season - 1).sum()) if not frame.empty else 0
        # Availability prior is deliberately shrunk.  It can reflect prior
        # opportunity but cannot reproduce target-season appearances.
        games_played_prior = float(min(17.0, max(8.0, 12.0 + 0.30 * (recent_games - 12))))
        rows.append({
            "player_id": player["player_id"], "player_name": player["player_name"],
            "position": player["position"], "team": player.get("team"),
            "per_game_mean": per_game_mean, "games_played_prior": games_played_prior,
            "historical_games": historical_games,
            "projection": per_game_mean * games_played_prior,
        })
    return pd.DataFrame(rows)
