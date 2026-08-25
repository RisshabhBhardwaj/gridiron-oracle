"""
pipeline/team_game_features.py

Phase 4 (Coherent Prediction Hierarchy) — team-game grain feature assembly.

Builds one row per (season, week, team) with:
  - Team-level Elo, as-of the target week (feature_matrix.team_off_elo /
    team_def_elo are already lagged per player-game; distinct per
    (season, week, team) gives the team-level rating for free).
  - Coach-fingerprinted neutral-script tendency: trailing pass rate and pace,
    keyed on COACH identity (not team), spanning strictly prior games across
    any team that coach has been on. This is what makes a tendency travel
    with a coach when they change teams — the entire point of Phase 4's
    fingerprinting requirement.
  - Rest, home/away, Vegas lines when present (spread_line/total_line — only
    ~40% filled for 2026; the model must not depend on them).
  - Target: the team's own points scored in that game.

team_game_stats has no game_id and its `team` column uses nflreadpy's
current-franchise codes (e.g. "LV" for all seasons), while `games` uses the
abbreviation contemporaneous with that season (e.g. "OAK" for 2019). Both
sides are normalized through ml.team_elo.normalize_team_abbr before joining,
or the 2019-2020 Raiders rows silently drop out of the training set.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from ml.team_elo import normalize_team_abbr

logger = logging.getLogger(__name__)

# "Neutral script" — not obviously garbage time / hurry-up, so the play call
# reflects the coach's actual tendency rather than the score dictating it.
_NEUTRAL_SCORE_MARGIN = 8
_NEUTRAL_MIN_SECONDS_REMAINING = 120
# Below this many trailing neutral-script snaps, a coach's pass rate is
# mostly small-sample noise (e.g. one half of one game) — null it instead.
_MIN_PLAYS_FOR_CONFIDENCE = 50


def _load_games(conn, seasons: list[int]) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id AS game_id, season, week, home_team, away_team,
                   home_score, away_score, home_rest, away_rest,
                   spread_line, total_line, roof, surface,
                   home_coach, away_coach, kickoff_at
            FROM games
            WHERE season = ANY(%s) AND home_score IS NOT NULL AND away_score IS NOT NULL
            """,
            (seasons,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    return pd.DataFrame(rows)


def _load_team_game_stats(conn, seasons: list[int]) -> pd.DataFrame:
    # total_plays/total_yards are now real (backfilled from nfl.load_team_stats()'s
    # attempts/carries/passing_yards/rushing_yards via TeamStatsRow's
    # @computed_field derivations — see scraper/adapters/nflreadpy_adapter.py).
    # COALESCE against the stored value in case a row predates the backfill.
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT team, season, week, pass_attempts, rush_attempts,
                   COALESCE(total_plays, pass_attempts + rush_attempts) AS total_plays,
                   total_yards
            FROM team_game_stats
            WHERE season = ANY(%s)
            """,
            (seasons,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    return pd.DataFrame(rows)


def _load_team_elo(conn, seasons: list[int]) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT DISTINCT season, week, team, team_off_elo, team_def_elo
            FROM feature_matrix
            WHERE season = ANY(%s) AND team_off_elo IS NOT NULL
            """,
            (seasons,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    return pd.DataFrame(rows)


def _load_coach_plays(conn, seasons: list[int]) -> pd.DataFrame:
    """One row per neutral-script offensive snap, tagged with the play-caller's coach."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT pp.season, pp.week, pp.play_type,
                   CASE WHEN pp.posteam = g.home_team THEN g.home_coach
                        WHEN pp.posteam = g.away_team THEN g.away_coach
                   END AS coach
            FROM pbp_plays pp
            JOIN games g ON g.id = pp.game_id
            WHERE pp.season = ANY(%s)
              AND pp.posteam IS NOT NULL
              AND pp.play_type IN ('pass', 'run')
              AND ABS(COALESCE(pp.score_differential, 0)) <= %s
              AND pp.game_seconds_remaining >= %s
            """,
            (seasons, _NEUTRAL_SCORE_MARGIN, _NEUTRAL_MIN_SECONDS_REMAINING),
        )
        rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        # No pbp_plays rows for this window. An empty DataFrame has no columns
        # at all, so df["coach"] used to raise a bare KeyError from deep inside
        # pandas, ~60s into a materialization run, with nothing pointing at the
        # actual cause (a serving-only database whose pbp_plays was dropped for
        # the storage cap). _build_coach_tendency already handles an empty frame
        # and the downstream merge is a LEFT join, so return the right SHAPE and
        # let the caller's own training-window guard decide whether to proceed.
        return pd.DataFrame(columns=["season", "week", "play_type", "coach"])
    df = pd.DataFrame(rows)
    return df[df["coach"].notna()].copy()


def _build_coach_tendency(coach_plays: pd.DataFrame) -> pd.DataFrame:
    """
    Per (coach, season, week): neutral-script pass rate and play count that
    week, PLUS the trailing (strictly prior, any team) average of both —
    the actual model input. Returns one row per (coach, season, week) that
    appears in coach_plays, with prior_* columns.
    """
    if coach_plays.empty:
        return pd.DataFrame(columns=["coach", "season", "week", "prior_coach_pass_rate", "prior_coach_neutral_plays"])

    weekly = (
        coach_plays.groupby(["coach", "season", "week"])
        .agg(
            n_plays=("play_type", "size"),
            n_pass=("play_type", lambda s: (s == "pass").sum()),
        )
        .reset_index()
    )
    weekly["pass_rate"] = weekly["n_pass"] / weekly["n_plays"]
    weekly = weekly.sort_values(["coach", "season", "week"]).reset_index(drop=True)

    out_rows = []
    for coach, grp in weekly.groupby("coach"):
        grp = grp.sort_values(["season", "week"]).reset_index(drop=True)
        cum_pass = 0
        cum_plays = 0
        for _, row in grp.iterrows():
            # Below the confidence floor, a coach's early-tenure rate is
            # mostly sample noise — null it rather than feed a shaky ratio to
            # the model. prior_coach_neutral_plays stays as a diagnostic
            # column, but the model doesn't use it directly: it's a raw
            # cumulative count that grows across a tenure and reads mostly
            # as "how deep into the dataset are we", not coach signal.
            prior_rate = (
                cum_pass / cum_plays
                if cum_plays >= _MIN_PLAYS_FOR_CONFIDENCE else None
            )
            prior_neutral_plays = cum_plays if cum_plays > 0 else None
            out_rows.append({
                "coach": coach, "season": row["season"], "week": row["week"],
                "prior_coach_pass_rate": prior_rate,
                "prior_coach_neutral_plays": prior_neutral_plays,
            })
            cum_pass += row["n_pass"]
            cum_plays += row["n_plays"]
    return pd.DataFrame(out_rows)


def _load_forward_games(conn, season: int, week: int) -> pd.DataFrame:
    """Games for a specific (season, week) regardless of whether they've been played."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id AS game_id, season, week, home_team, away_team,
                   home_score, away_score, home_rest, away_rest,
                   spread_line, total_line, roof, surface,
                   home_coach, away_coach, kickoff_at
            FROM games
            WHERE season = %s AND week = %s
            """,
            (season, week),
        )
        rows = [dict(r) for r in cur.fetchall()]
    return pd.DataFrame(rows)


def _load_latest_team_elo(conn, season: int, week: int) -> pd.DataFrame:
    """Each team's most recent known Elo strictly before (season, week)."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (team) team, team_off_elo, team_def_elo
            FROM feature_matrix
            WHERE team_off_elo IS NOT NULL
              AND (season < %s OR (season = %s AND week < %s))
            ORDER BY team, season DESC, week DESC
            """,
            (season, season, week),
        )
        rows = [dict(r) for r in cur.fetchall()]
    return pd.DataFrame(rows)


def build_team_game_forward_frame(db_url: str, season: int, week: int) -> pd.DataFrame:
    """
    Same feature set as build_team_game_frame's inputs (Elo, coach tendency,
    rest, venue), for a game that hasn't been played yet — no
    team_game_stats/points requirement, since those are the outputs, not
    inputs. Elo and coach tendency use the most recent known values as of
    strictly before the target week rather than requiring a feature_matrix
    row for that exact week (which won't exist for a future week).
    """
    conn = psycopg2.connect(db_url)
    try:
        games = _load_forward_games(conn, season, week)
        elo = _load_latest_team_elo(conn, season, week)
        coach_plays = _load_coach_plays(conn, list(range(2019, season + 1)))
    finally:
        conn.close()

    if games.empty:
        return pd.DataFrame()

    games["home_team_n"] = games["home_team"].map(normalize_team_abbr)
    games["away_team_n"] = games["away_team"].map(normalize_team_abbr)

    home = games.rename(columns={
        "home_team_n": "team_n", "away_team_n": "opponent_n",
        "home_rest": "rest", "away_rest": "opp_rest", "home_coach": "coach",
    }).copy()
    home["is_home"] = 1
    away = games.rename(columns={
        "away_team_n": "team_n", "home_team_n": "opponent_n",
        "away_rest": "rest", "home_rest": "opp_rest", "away_coach": "coach",
    }).copy()
    away["is_home"] = 0

    keep = [
        "game_id", "season", "week", "team_n", "opponent_n", "rest", "opp_rest",
        "coach", "is_home", "spread_line", "total_line", "roof", "surface", "kickoff_at",
    ]
    frame = pd.concat([home[keep], away[keep]], ignore_index=True)

    elo["team_n"] = elo["team"].map(normalize_team_abbr)
    frame = frame.merge(
        elo[["team_n", "team_off_elo", "team_def_elo"]], on="team_n", how="left",
    )
    opp_elo = elo.rename(columns={
        "team_n": "opponent_n", "team_off_elo": "opp_off_elo", "team_def_elo": "opp_def_elo",
    })
    frame = frame.merge(
        opp_elo[["opponent_n", "opp_off_elo", "opp_def_elo"]], on="opponent_n", how="left",
    )

    tendency = _build_coach_tendency(coach_plays)
    # Most recent known tendency per coach, strictly before the target week —
    # same lag discipline as the historical path, just resolved to "latest
    # known" rather than an exact (season, week) match that won't exist yet.
    tendency = tendency[
        (tendency["season"] < season) | ((tendency["season"] == season) & (tendency["week"] < week))
    ]
    latest_tendency = (
        tendency.sort_values(["coach", "season", "week"])
        .groupby("coach", as_index=False)
        .last()[["coach", "prior_coach_pass_rate", "prior_coach_neutral_plays"]]
    )
    frame = frame.merge(latest_tendency, on="coach", how="left")

    frame["is_dome"] = frame["roof"].isin(["dome", "closed"]).astype(int)
    frame["is_turf"] = (frame["surface"].fillna("").str.lower() == "turf").astype(int)
    frame = frame.rename(columns={"team_n": "team", "opponent_n": "opponent"})
    return frame.sort_values(["team"]).reset_index(drop=True)


def build_team_game_frame(db_url: str, seasons: list[int]) -> pd.DataFrame:
    """
    Assemble the Phase 4 team-game training/serving frame.

    One row per (season, week, team) with the team's own points as the
    target and features that are all knowable strictly before kickoff:
    prior-week Elo, the coach's trailing neutral-script tendency (indexed by
    coach identity, so it travels across team changes), rest, home/away, and
    Vegas lines when present (optional anchor, not required).
    """
    conn = psycopg2.connect(db_url)
    try:
        games = _load_games(conn, seasons)
        tgs = _load_team_game_stats(conn, seasons)
        elo = _load_team_elo(conn, seasons)
        coach_plays = _load_coach_plays(conn, seasons)
    finally:
        conn.close()

    if games.empty or tgs.empty:
        return pd.DataFrame()

    games["home_team_n"] = games["home_team"].map(normalize_team_abbr)
    games["away_team_n"] = games["away_team"].map(normalize_team_abbr)
    tgs["team_n"] = tgs["team"].map(normalize_team_abbr)

    home = games.rename(columns={
        "home_team_n": "team_n", "away_team_n": "opponent_n",
        "home_score": "points", "away_score": "opp_points",
        "home_rest": "rest", "away_rest": "opp_rest", "home_coach": "coach",
    }).copy()
    home["is_home"] = 1

    away = games.rename(columns={
        "away_team_n": "team_n", "home_team_n": "opponent_n",
        "away_score": "points", "home_score": "opp_points",
        "away_rest": "rest", "home_rest": "opp_rest", "away_coach": "coach",
    }).copy()
    away["is_home"] = 0

    keep = [
        "game_id", "season", "week", "team_n", "opponent_n", "points",
        "opp_points", "rest", "opp_rest", "coach", "is_home",
        "spread_line", "total_line", "roof", "surface", "kickoff_at",
    ]
    frame = pd.concat([home[keep], away[keep]], ignore_index=True)

    frame = frame.merge(
        tgs[["team_n", "season", "week", "pass_attempts", "rush_attempts", "total_plays", "total_yards"]],
        on=["team_n", "season", "week"], how="inner",
    )

    elo["team_n"] = elo["team"].map(normalize_team_abbr)
    frame = frame.merge(
        elo[["team_n", "season", "week", "team_off_elo", "team_def_elo"]],
        on=["team_n", "season", "week"], how="left",
    )
    opp_elo = elo.rename(columns={
        "team_n": "opponent_n", "team_off_elo": "opp_off_elo", "team_def_elo": "opp_def_elo",
    })
    frame = frame.merge(
        opp_elo[["opponent_n", "season", "week", "opp_off_elo", "opp_def_elo"]],
        on=["opponent_n", "season", "week"], how="left",
    )

    tendency = _build_coach_tendency(coach_plays)
    frame = frame.merge(tendency, on=["coach", "season", "week"], how="left")

    frame["is_dome"] = frame["roof"].isin(["dome", "closed"]).astype(int)
    frame["is_turf"] = (frame["surface"].fillna("").str.lower() == "turf").astype(int)
    frame["pass_rate"] = np.where(
        frame["total_plays"] > 0, frame["pass_attempts"] / frame["total_plays"], np.nan
    )
    frame["win"] = (frame["points"] > frame["opp_points"]).astype(int)
    frame = frame.rename(columns={"team_n": "team", "opponent_n": "opponent"})
    return frame.sort_values(["season", "week", "team"]).reset_index(drop=True)
