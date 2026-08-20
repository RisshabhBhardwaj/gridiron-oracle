"""Causal, preseason full-season PPR projection construction for draft boards."""

from __future__ import annotations

import pandas as pd

from ml.age_curve import age_multiplier, build_observations, fit_age_curve, season_age
from ml.baselines import attach_marcel
from ml.playing_time import availability_anchor, expected_games_from_history
from ml.rookie_priors import (
    effective_pick,
    fit_rookie_curve,
    label_projection_basis,
    rookie_multiplier,
    rookie_rate_and_games,
    vacated_opportunity_share,
)

ELIGIBLE_POSITIONS = ("QB", "RB", "WR", "TE")


def _is_declared_rookie(player: dict, season: int) -> bool:
    year = player.get("entry_year")
    if year is None or pd.isna(year):
        return False
    try:
        return int(year) == season
    except (TypeError, ValueError):
        return False


def build_preseason_projections(
    players: pd.DataFrame,
    game_logs: pd.DataFrame,
    *,
    season: int,
    use_multiseason_games: bool = True,
    use_age_curve: bool = False,
    use_rookie_curve: bool = True,
    use_calibrated_games: bool = True,
) -> pd.DataFrame:
    """Build causal annual projections: Marcel rate × SP2 expected games.

    Rookies use draft capital and vacated opportunity. Target-season rows,
    OOF, and realized availability are refused.
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
    if "position" in history.columns:
        history = history.drop(columns=["position"])
    position_lookup = universe[["player_id", "position"]]
    history_with_pos = history.merge(position_lookup, on="player_id", how="left")
    position_means = history_with_pos.groupby("position").apply(
        lambda frame: (frame["fantasy_points_ppr"] * frame["recency_weight"]).sum() / frame["recency_weight"].sum(),
        include_groups=False,
    ).to_dict()
    league_mean = float(history["fantasy_points_ppr"].mean()) if not history.empty else 0.0

    vacated_logs = history_with_pos.copy()
    if "team" not in vacated_logs.columns:
        vacated_logs["team"] = ""
    vacated = vacated_opportunity_share(
        vacated_logs,
        universe[["player_id", "team", "position"]] if "team" in universe.columns else pd.DataFrame(),
        season=season,
    )

    # Rookie outcomes are fitted from draft capital on prior seasons only. The
    # old position-mean-times-multiplier prior projected the average drafted
    # rookie at 24% of what he scored; see ml/rookie_priors.py.
    rookie_curve = {}
    if use_rookie_curve:
        rookie_curve = fit_rookie_curve(history, players, max_season=season - 1)
    draft_numbers = effective_pick(players).to_dict()

    grouped = history.groupby("player_id")
    rows: list[dict] = []
    for player in universe.to_dict("records"):
        frame = grouped.get_group(player["player_id"]) if player["player_id"] in grouped.groups else pd.DataFrame()
        fallback = float(position_means.get(player["position"], league_mean))
        if frame.empty:
            if not _is_declared_rookie(player, season):
                continue
            per_game_mean, historical_games = fallback, 0
        else:
            per_game_mean = float((frame["fantasy_points_ppr"] * frame["recency_weight"]).sum() / frame["recency_weight"].sum())
            historical_games = int(len(frame))
        recent_games = int((frame["season"] == season - 1).sum()) if not frame.empty else 0
        last_season = int(frame["season"].max()) if not frame.empty else None
        is_rookie = historical_games == 0 or _is_declared_rookie(player, season)
        if last_season is not None and last_season < season - 1 and not is_rookie:
            continue
        if is_rookie or historical_games == 0:
            team = str(player.get("team") or "").upper()
            vacated_share = vacated.get((team, str(player["position"])), 0.0)
            if rookie_curve.get("n"):
                per_game_mean, games_played_prior = rookie_rate_and_games(
                    rookie_curve,
                    player["position"],
                    draft_numbers.get(player["player_id"]),
                    vacated_share=vacated_share,
                )
            else:
                per_game_mean = fallback * rookie_multiplier(player.get("draft_round"), vacated_share)
                games_played_prior = 14.0
        else:
            # Superseded for veterans by apply_multiseason_games below; this
            # value only survives when the multi-season estimate is unavailable.
            # p_active is not used here: its prior_games argument means "games
            # observed in the availability window", not career games.
            games_played_prior = float(min(17.0, max(1.0, float(recent_games))))
        rows.append({
            "player_id": player["player_id"], "player_name": player["player_name"],
            "position": player["position"], "team": player.get("team"),
            "per_game_mean": per_game_mean, "games_played_prior": games_played_prior,
            "historical_games": historical_games,
            "projection": per_game_mean * games_played_prior,
            "projection_basis": label_projection_basis(
                historical_games=historical_games, is_rookie=is_rookie
            ),
        })
    frame = apply_marcel_playing_time(pd.DataFrame(rows), history_with_pos, season=season)
    # Every toggle here defaults to what scripts/draft_component_eval.py measured
    # over ~1030 player-seasons with paired bootstrap CIs -- NOT what the snake
    # draft harness scored. That harness has a season-to-season sd of 170.6 over
    # six seasons; differences under ~130 points are inside its noise, and the
    # age curve was once switched off on a 14.2-point reading that meant nothing.
    #
    #   rookie curve   spearman    +0.0421 [+0.0258, +0.0612]  significant
    #   games calib    calibration +0.2521 [+0.2116, +0.2949]  significant
    #   age curve      nothing significant on any metric        -> off
    #
    # The age curve's only measured benefit was to calibration, and the games
    # calibration above subsumes it: with both on, calibration drifts slightly
    # further from 1.0 (1.036 vs 1.006) and rank correlation moves +0.003, ns.
    # It is off because an unresolvable component should lose to the simpler
    # system, not because a number moved. It flipped more than once during
    # 2026-08-19/20 as the measuring instrument improved; this call is the first
    # one made on the powered metric with the calibration fix in place.
    #   rookie curve   spearman    +0.0434 [+0.0132, +0.0768]  significant
    #                  calibration +0.0490 [+0.0397, +0.0594]  significant
    if use_multiseason_games:
        frame = apply_multiseason_games(frame, history, players, season=season)
    if use_calibrated_games:
        frame = apply_games_calibration(frame, history, season=season)
    if use_age_curve:
        frame = apply_age_curve(frame, game_logs, players, season=season)
    return frame


def apply_multiseason_games(
    projections: pd.DataFrame,
    history: pd.DataFrame,
    players: pd.DataFrame,
    *,
    season: int,
) -> pd.DataFrame:
    """Replace the single-season availability heuristic with a 3-season estimate.

    Measured on the roster-aware walk-forward harness: +22.7 mean starting-lineup
    points per season over the single-season prior (reports/draft_board_experiments.json).
    Rookies keep their rookie games prior -- they have no availability history.
    """
    if projections.empty:
        return projections
    expected = expected_games_from_history(history, players, season=season)
    out = projections.copy()
    veterans = out["historical_games"].fillna(0).astype(int) > 0
    mapped = out["player_id"].map(expected)
    use = veterans & mapped.notna()
    out.loc[use, "games_played_prior"] = mapped[use].astype(float)
    out.loc[use, "projection"] = out.loc[use, "per_game_mean"] * out.loc[use, "games_played_prior"]
    return out


def apply_age_curve(
    projections: pd.DataFrame,
    game_logs: pd.DataFrame,
    players: pd.DataFrame,
    *,
    season: int,
) -> pd.DataFrame:
    """Scale veteran projections by an age curve fitted on seasons < ``season``.

    A no-op when ``players`` carries no birth_date. The curve is refit per target
    season; target-season rows would raise inside :func:`ml.age_curve.fit_age_curve`.
    """
    if projections.empty or "birth_date" not in players.columns:
        return projections
    curve = fit_age_curve(
        build_observations(game_logs, players, max_season=season - 1), max_season=season - 1
    )
    if not curve:
        return projections
    births = players.rename(columns={"id": "player_id"})[["player_id", "birth_date"]]
    out = projections.merge(births, on="player_id", how="left")
    ages = season_age(out["birth_date"], pd.Series(season, index=out.index))
    veterans = out["historical_games"].fillna(0).astype(int) > 0
    factors = pd.Series(
        [age_multiplier(curve, pos, age) for pos, age in zip(out["position"], ages)],
        index=out.index,
    )
    factors = factors.where(veterans, 1.0)
    out["projection"] = pd.to_numeric(out["projection"], errors="coerce") * factors
    out["age"] = ages
    return out.drop(columns=["birth_date"])


def apply_marcel_playing_time(
    projections: pd.DataFrame,
    history: pd.DataFrame,
    *,
    season: int,
) -> pd.DataFrame:
    """Veterans: Marcel per-game rate × already-attached SP2 expected games."""
    if projections.empty:
        return projections
    hist = history.copy()
    if "fantasy_ppr" not in hist.columns:
        hist = hist.rename(columns={"fantasy_points_ppr": "fantasy_ppr"})
    eval_rows = projections[["player_id", "position"]].copy()
    eval_rows["season"] = season
    eval_rows["week"] = 1
    scored = attach_marcel(eval_rows, hist, stat_col="fantasy_ppr")
    out = projections.merge(scored[["player_id", "marcel_baseline"]], on="player_id", how="left")
    veterans = out["historical_games"].fillna(0).astype(int) > 0
    use = veterans & out["marcel_baseline"].notna()
    out.loc[use, "per_game_mean"] = out.loc[use, "marcel_baseline"].astype(float)
    out.loc[use, "projection"] = out.loc[use, "per_game_mean"] * out.loc[use, "games_played_prior"]
    out.loc[use, "projection_basis"] = "marcel_rate_x_playing_time"
    return out.drop(columns=["marcel_baseline"])


def apply_prior_season_stack_rate(
    projections: pd.DataFrame,
    stack_oof: pd.DataFrame,
    *,
    season: int,
) -> pd.DataFrame:
    """Replace veteran per-game rates with prior-season stack OOF means.

    Target-season OOF is refused. Rookies keep vacated-opportunity rates.
    """
    if projections.empty or stack_oof.empty:
        return projections
    oof = stack_oof.copy()
    if "season" not in oof.columns or "player_id" not in oof.columns:
        raise ValueError("stack OOF must include player_id and season")
    oof = oof[pd.to_numeric(oof["season"], errors="coerce") == season - 1]
    pred_col = "y_pred" if "y_pred" in oof.columns else "predicted"
    if pred_col not in oof.columns:
        raise ValueError("stack OOF missing y_pred")
    rates = (
        oof.groupby("player_id")[pred_col]
        .mean()
        .rename("stack_rate")
    )
    out = projections.merge(rates, left_on="player_id", right_index=True, how="left")
    veterans = out["historical_games"].fillna(0).astype(int) > 0
    replaced = veterans & out["stack_rate"].notna()
    out.loc[replaced, "per_game_mean"] = out.loc[replaced, "stack_rate"]
    out.loc[replaced, "projection"] = out.loc[replaced, "per_game_mean"] * out.loc[replaced, "games_played_prior"]
    out.loc[replaced, "projection_basis"] = "prior_season_stack_oof_x_playing_time"
    return out.drop(columns=["stack_rate"])


def apply_games_calibration(
    projections: pd.DataFrame,
    history: pd.DataFrame,
    *,
    season: int,
) -> pd.DataFrame:
    """Scale veteran expected games so the universe mean matches the causal anchor.

    ``expected_games_from_history`` ranks availability usefully but sits about 23%
    low in level, which propagated straight into every season total: the projection
    universe read 0.71 projected points per realized point. A single walk-forward
    scale factor fixes the level without disturbing the ordering.

    Measured over 1030 player-seasons with paired bootstrap CIs
    (scripts/draft_component_eval.py): calibration ratio 0.712 -> 1.011, a
    significant gain, with no significant change in rank correlation
    (-0.005, ns). This was once rejected on the six-season board simulation,
    where the effect sat well inside a standard error.
    """
    if projections.empty:
        return projections
    out = projections.copy()
    veterans = out["historical_games"].fillna(0).astype(int) > 0
    current = out.loc[veterans, "games_played_prior"].astype(float)
    if current.empty or current.mean() <= 0:
        return out
    scale = availability_anchor(history, season=season) / current.mean()
    out.loc[veterans, "games_played_prior"] = (current * scale).clip(0.0, 17.0)
    out.loc[veterans, "projection"] = (
        out.loc[veterans, "per_game_mean"] * out.loc[veterans, "games_played_prior"]
    )
    return out
