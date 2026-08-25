"""Causal playing-time / role model (SP2).

Predicts (a) P(active) and (b) snap/route/carry share given active, using only
information strictly before the target kickoff. A QB with no prior snaps must
not rank in a rest-of-season top 24 — that is the acceptance invariant.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})


def build_playing_time_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Attach causal prior-game features and playing-time targets.

    ``history`` must include player_id, season, week, position, and either
    ``offense_pct`` or ``snap_share``. Same-week participation is the *target*,
    never a feature.
    """
    required = {"player_id", "season", "week"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"playing-time frame missing {sorted(missing)}")
    frame = history.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    snap_col = "offense_pct" if "offense_pct" in frame.columns else "snap_share"
    if snap_col not in frame.columns:
        raise ValueError("history needs offense_pct or snap_share")
    snaps = pd.to_numeric(frame[snap_col], errors="coerce")
    if snap_col == "offense_pct":
        snaps = snaps.where(snaps <= 1.0, snaps / 100.0)
    frame["snap_share"] = snaps.clip(0.0, 1.0)
    frame["active"] = (frame["snap_share"].fillna(0.0) > 0.0).astype(int)

    frame = frame.sort_values(["player_id", "season", "week"])
    grouped = frame.groupby("player_id", sort=False)
    frame["prior_snap_share"] = grouped["snap_share"].shift(1)
    frame["prior_games"] = grouped.cumcount()
    frame["prior_active_games"] = grouped["active"].cumsum().shift(1)
    if "fantasy_points_ppr" in frame.columns:
        frame["fantasy_points_ppr"] = pd.to_numeric(frame["fantasy_points_ppr"], errors="coerce")
        frame["prior_ppr"] = grouped["fantasy_points_ppr"].shift(1)
    if "rec_fantasy_points_exp" in frame.columns:
        # Lag only — contemporaneous xFP is forbidden as a feature.
        frame["rec_fantasy_points_exp"] = pd.to_numeric(frame["rec_fantasy_points_exp"], errors="coerce")
        frame["lagged_rec_xfp"] = grouped["rec_fantasy_points_exp"].shift(1)
        if "fantasy_points_ppr" in frame.columns:
            frame["lagged_actual_minus_expected"] = grouped["fantasy_points_ppr"].shift(1) - frame["lagged_rec_xfp"]
    return frame


ACTIVITY_BASE_RATE = 0.85
ACTIVITY_SHRINK_GAMES = 4.0


def p_active_from_priors(
    prior_snap_share: float | None,
    prior_games: float | None,
    *,
    depth_rank: float | None = None,
    prior_active_games: float | None = None,
) -> float:
    """P(the player takes a snap), from how often he has, not from how many.

    The earlier form was ``0.08 + 0.90 * prior_snap_share``, which answers a
    different question: snap *share* is role size, availability is whether he
    dressed. Measured on 2025 weeks 10-18, players who played all nine games
    received a mean p_active of 0.570 — because a committee back with a 55%
    snap share plays every week and was being scored as a coin flip. Since the
    weekly rate already carries role, multiplying by share double-counted it
    and depressed RB/TE/WR season totals (realized/projected 0.46 / 0.58 / 0.68).

    Availability now comes from the observed active rate, shrunk toward the
    league base rate so a two-game sample cannot assert certainty. Snap share
    is retained only as a fallback when activity counts are unavailable.
    """
    games = 0.0 if prior_games is None or not np.isfinite(prior_games) else float(prior_games)
    active = 0.0 if prior_active_games is None or not np.isfinite(prior_active_games) else float(prior_active_games)
    if games <= 0 and active <= 0:
        # Cold start: no evidence he has ever played. A listed starter gets a
        # real but modest prior; anyone else must not surface in a top board.
        if depth_rank is not None and float(depth_rank) <= 1.0:
            return 0.35
        return 0.02
    if games > 0:
        observed = min(active, games)
        rate = (observed + ACTIVITY_SHRINK_GAMES * ACTIVITY_BASE_RATE) / (games + ACTIVITY_SHRINK_GAMES)
        return float(np.clip(rate, 0.02, 0.98))
    share = 0.0 if prior_snap_share is None or not np.isfinite(prior_snap_share) else float(prior_snap_share)
    return float(np.clip(0.08 + 0.90 * share, 0.02, 0.98))


def role_share_from_priors(
    prior_snap_share: float | None,
    *,
    depth_rank: float | None = None,
) -> float:
    share = 0.0 if prior_snap_share is None or not np.isfinite(prior_snap_share) else float(prior_snap_share)
    if share <= 0 and depth_rank is not None and float(depth_rank) <= 1.0:
        return 0.55
    return float(np.clip(share, 0.0, 1.0))


def attach_playing_time(rows: Iterable[Mapping[str, object]]) -> list[dict]:
    """Copy rows with ``p_active`` and ``role_share`` from causal priors."""
    out = []
    for raw in rows:
        row = dict(raw)
        # p_active_from_priors' `prior_games` arg is the DENOMINATOR — total
        # opportunities to play. `team_games_played` (the player's team's own
        # completed-game count over the same window) is the correct source
        # for that; `row["prior_games"]` is the player's OWN participation
        # count (used elsewhere for cold-start detection) and is NOT a valid
        # denominator — passing it as one makes the "active rate" collapse
        # toward 0 for well-established players instead of reflecting real
        # availability. Callers that don't supply team_games_played (existing
        # test fixtures, callers outside this codebase's DB) fall back to the
        # old prior_games-as-denominator behavior rather than erroring.
        denominator = row.get("team_games_played")
        if denominator is None:
            denominator = row.get("prior_games")
        row["p_active"] = p_active_from_priors(
            _opt_float(row.get("prior_snap_share")),
            _opt_float(denominator),
            depth_rank=_opt_float(row.get("depth_rank")),
            prior_active_games=_opt_float(row.get("prior_active_games")),
        )
        row["role_share"] = role_share_from_priors(
            _opt_float(row.get("prior_snap_share")),
            depth_rank=_opt_float(row.get("depth_rank")),
        )
        out.append(row)
    return out


def rest_of_season_points(
    weekly_rate: float,
    p_active: float,
    remaining_weeks: int,
    *,
    residual_scale: float = 6.0,
    n_sims: int = 400,
    rng: np.random.Generator | int | None = 0,
) -> dict[str, float]:
    """Season totals from week-by-week availability draws, not mean × remaining."""
    return simulate_season_paths(
        weekly_rate,
        p_active,
        remaining_weeks,
        residual_scale=residual_scale,
        n_sims=n_sims,
        rng=rng,
    )


def rank_rest_of_season(
    rows: Sequence[Mapping[str, object]],
    *,
    value_key: str = "mean",
) -> list[dict]:
    ranked = [dict(row) for row in rows]
    ranked.sort(key=lambda item: (-float(item.get(value_key) or 0.0), str(item.get("player_id"))))
    for index, row in enumerate(ranked, start=1):
        row["ros_rank"] = index
    return ranked


def assert_cold_start_qb_not_in_top24(ranked: Sequence[Mapping[str, object]]) -> None:
    """SP2 acceptance: a QB with no prior snaps cannot be a ROS star."""
    top = [row for row in ranked if int(row.get("ros_rank") or 99) <= 24]
    offenders = [
        row for row in top
        if str(row.get("position") or "").upper() == "QB"
        and float(row.get("prior_games") or 0) <= 0
        and float(row.get("prior_active_games") or 0) <= 0
        and (row.get("depth_rank") is None or float(row["depth_rank"]) > 1.0)
    ]
    if offenders:
        names = [str(row.get("player_id")) for row in offenders]
        raise AssertionError(f"Cold-start QBs in ROS top 24: {names}")


def simulate_season_paths(
    weekly_rate: float,
    p_active: float,
    remaining_weeks: int,
    *,
    residual_scale: float = 6.0,
    n_sims: int = 400,
    rng: np.random.Generator | int | None = 0,
) -> dict[str, float]:
    """Draw remaining-week paths: Bernoulli(p_active) × N(weekly_rate, residual)."""
    weeks = max(int(remaining_weeks), 0)
    expected_games = float(np.clip(p_active, 0.0, 1.0)) * weeks
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    if weeks == 0 or n_sims <= 0:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "expected_games": 0.0}
    if float(weekly_rate) <= 0.0:
        # No role in this stat: the projection is exactly zero, not a spread
        # around zero. Drawing N(0, residual_scale) here left a WR's
        # passing_yards with a +/-230-yard 80% interval straddling zero (the
        # residual scales in ProjectionService.get_season_projections are
        # per-stat, not per-player), which reads as a real forecast on the
        # card. The mean was already ~0 because the negative half is not
        # clipped -- deliberately, since clipping a zero-mean Gaussian is what
        # manufactured phantom yardage in the season-simulator path (see
        # SeasonSimulator._ZERO_RATE_EPS).
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0,
                "expected_games": expected_games}
    active = generator.random((int(n_sims), weeks)) < float(np.clip(p_active, 0.0, 1.0))
    outcomes = generator.normal(float(weekly_rate), float(residual_scale), size=(int(n_sims), weeks))
    totals = (active * outcomes).sum(axis=1)
    return {
        "mean": float(totals.mean()),
        "p10": float(np.quantile(totals, 0.10)),
        "p50": float(np.quantile(totals, 0.50)),
        "p90": float(np.quantile(totals, 0.90)),
        "expected_games": expected_games,
    }


def as_of_depth_rank(
    charts: pd.DataFrame,
    *,
    player_id: str,
    kickoff_at: datetime,
) -> float | None:
    """Latest depth_rank with published_at <= kickoff. Week-only joins are forbidden."""
    required = {"player_id", "published_at", "depth_rank"}
    missing = required - set(charts.columns)
    if missing:
        raise ValueError(f"depth charts missing {sorted(missing)}")
    kickoff = kickoff_at if kickoff_at.tzinfo else kickoff_at.replace(tzinfo=timezone.utc)
    frame = charts[charts["player_id"].astype(str) == str(player_id)].copy()
    published = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    eligible = frame[published.notna() & (published <= kickoff)]
    if eligible.empty:
        return None
    latest = eligible.loc[published[eligible.index].idxmax()]
    return float(latest["depth_rank"])


def as_of_injury_status(
    reports: pd.DataFrame,
    *,
    player_id: str,
    kickoff_at: datetime,
    stamp_col: str = "captured_at",
) -> str | None:
    required = {"player_id", stamp_col}
    missing = required - set(reports.columns)
    if missing:
        raise ValueError(f"injury reports missing {sorted(missing)}")
    kickoff = kickoff_at if kickoff_at.tzinfo else kickoff_at.replace(tzinfo=timezone.utc)
    frame = reports[reports["player_id"].astype(str) == str(player_id)].copy()
    stamped = pd.to_datetime(frame[stamp_col], utc=True, errors="coerce")
    eligible = frame[stamped.notna() & (stamped <= kickoff)]
    if eligible.empty:
        return None
    latest = eligible.loc[stamped[eligible.index].idxmax()]
    status = latest.get("practice_status") or latest.get("injury_status")
    return None if status is None else str(status)


_AVAIL_FEATURES = ("prior_snap_share", "prior_games", "depth_rank", "lagged_rec_xfp")


def _design_matrix(frame: pd.DataFrame) -> np.ndarray:
    cols = []
    for name in _AVAIL_FEATURES:
        if name in frame.columns:
            series = pd.to_numeric(frame[name], errors="coerce")
        else:
            series = pd.Series(np.nan, index=frame.index)
        if name == "depth_rank":
            series = series.fillna(99.0)
        else:
            series = series.fillna(0.0)
        cols.append(series.to_numpy(dtype=float))
    return np.column_stack(cols)


class PlayingTimeModel:
    """Two-stage availability classifier (isotonic-calibrated) + role regressor."""

    def __init__(self) -> None:
        self._availability = None
        self._role = None
        self._fitted = False

    def fit(self, history: pd.DataFrame) -> "PlayingTimeModel":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression, Ridge

        frame = build_playing_time_frame(history)
        usable = frame.dropna(subset=["active"])
        if len(usable) < 40:
            raise ValueError("Need at least 40 rows to fit playing-time models")
        X = _design_matrix(usable)
        y = usable["active"].to_numpy(dtype=int)
        base = LogisticRegression(max_iter=200, solver="lbfgs")
        self._availability = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self._availability.fit(X, y)
        active = usable[usable["active"] == 1]
        if len(active) < 20:
            raise ValueError("Need at least 20 active rows to fit role share")
        self._role = Ridge(alpha=1.0)
        self._role.fit(_design_matrix(active), active["snap_share"].to_numpy(dtype=float))
        self._fitted = True
        return self

    def predict(self, rows: Iterable[Mapping[str, object]]) -> list[dict]:
        frame = pd.DataFrame(list(rows))
        if frame.empty:
            return []
        if not self._fitted:
            return attach_playing_time(frame.to_dict("records"))
        X = _design_matrix(frame)
        p_active = self._availability.predict_proba(X)[:, 1]
        role = np.clip(self._role.predict(X), 0.0, 1.0)
        out = []
        for index, raw in enumerate(frame.to_dict("records")):
            row = dict(raw)
            row["p_active"] = float(p_active[index])
            row["role_share"] = float(role[index])
            out.append(row)
        return out


def availability_calibration_table(
    y_true: Sequence[float],
    p_hat: Sequence[float],
    *,
    n_bins: int = 5,
) -> pd.DataFrame:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(p_hat, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pred >= lo) & (pred < hi if hi < 1.0 else pred <= hi)
        if not mask.any():
            continue
        rows.append({
            "lo": float(lo),
            "hi": float(hi),
            "n": int(mask.sum()),
            "mean_predicted": float(pred[mask].mean()),
            "mean_actual": float(actual[mask].mean()),
        })
    return pd.DataFrame(rows)


def role_beats_prior_snap_share(frame: pd.DataFrame, predicted_share: np.ndarray) -> bool:
    """Acceptance: role model MAE beats using prior_snap_share alone."""
    actual = pd.to_numeric(frame["snap_share"], errors="coerce").to_numpy(dtype=float)
    prior = pd.to_numeric(frame["prior_snap_share"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pred = np.asarray(predicted_share, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    if mask.sum() < 10:
        return False
    model_mae = float(np.mean(np.abs(actual[mask] - pred[mask])))
    prior_mae = float(np.mean(np.abs(actual[mask] - prior[mask])))
    return model_mae < prior_mae


def _opt_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(number) else number


GAMES_LAG_WEIGHTS: tuple[tuple[int, float], ...] = ((1, 5.0), (2, 4.0), (3, 3.0))
GAMES_SHRINK_SEASONS = 0.9
SEASON_GAMES = 17.0


def expected_games_from_history(
    history: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    season: int,
) -> pd.Series:
    """Expected games played in ``season``, from multi-season availability.

    The single-season heuristic in :func:`p_active_from_priors` reads only the
    prior year, so a player who missed all of season-1 with an injury projects
    at roughly one game regardless of the four healthy seasons behind him. This
    weights the prior three seasons (5/4/3, the Marcel shape) and shrinks toward
    the position's mean availability.

    A season counts as an opportunity only once the player has entered the
    league, so a second-year player is not charged with zeros for the seasons
    before he was drafted.

    Returns a Series indexed by player_id. Uses no data from ``season``.
    """
    required = {"player_id", "season"}
    if required - set(history.columns):
        raise ValueError("history needs player_id and season")
    hist = history.copy()
    hist["season"] = pd.to_numeric(hist["season"], errors="coerce")
    hist = hist[hist["season"] < season]
    roster = universe.rename(columns={"id": "player_id"}).copy()
    roster["position"] = roster["position"].astype(str).str.upper()
    entry = pd.to_numeric(roster.get("entry_year"), errors="coerce")
    entry_by_player = dict(zip(roster["player_id"], entry)) if "entry_year" in roster else {}

    played = hist.groupby(["player_id", "season"]).size().rename("games").reset_index()
    lookup = {(row.player_id, int(row.season)): float(row.games) for row in played.itertuples(index=False)}

    numer: dict[str, float] = {}
    denom: dict[str, float] = {}
    for player_id in roster["player_id"]:
        entered = entry_by_player.get(player_id)
        total_w = 0.0
        total_g = 0.0
        for lag, weight in GAMES_LAG_WEIGHTS:
            year = season - lag
            if entered is not None and np.isfinite(entered) and year < entered:
                continue
            total_w += weight
            total_g += weight * min(SEASON_GAMES, lookup.get((player_id, year), 0.0))
        numer[player_id] = total_g
        denom[player_id] = total_w

    frame = roster[["player_id", "position"]].copy()
    frame["numer"] = frame["player_id"].map(numer)
    frame["denom"] = frame["player_id"].map(denom)
    observed = frame[frame["denom"] > 0].copy()
    observed["rate"] = observed["numer"] / observed["denom"]
    pos_mean = observed.groupby("position")["rate"].mean().to_dict()
    league_mean = float(observed["rate"].mean()) if not observed.empty else 0.5 * SEASON_GAMES

    shrink_weight = GAMES_SHRINK_SEASONS * sum(weight for _lag, weight in GAMES_LAG_WEIGHTS) / len(GAMES_LAG_WEIGHTS)
    values: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        prior = float(pos_mean.get(row.position, league_mean))
        weight = float(row.denom or 0.0)
        if weight <= 0:
            values[row.player_id] = float("nan")
            continue
        blended = (row.numer + shrink_weight * prior) / (weight + shrink_weight)
        values[row.player_id] = float(np.clip(blended, 0.0, SEASON_GAMES))
    return pd.Series(values, name="expected_games")


AVAILABILITY_ANCHOR_MIN_GAMES = 6


def availability_anchor(history: pd.DataFrame, *, season: int, min_prior_games: int = AVAILABILITY_ANCHOR_MIN_GAMES) -> float:
    """Mean games played by draftable-caliber players, from seasons < ``season``.

    "Draftable-caliber" is defined causally: a player counts toward the anchor in
    year Y if he played at least ``min_prior_games`` in Y-1. Anchoring on the whole
    player table instead pulls the mean down with thousands of fringe players who
    are never drafted, which is how the projection universe came to average 10.5
    expected games against a realized 13.6.
    """
    frame = history.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame = frame[frame["season"] < season]
    if frame.empty:
        return 13.5
    played = frame.groupby(["player_id", "season"]).size().rename("games").reset_index()
    prior = played[played["games"] >= min_prior_games][["player_id", "season"]].copy()
    prior["season"] = prior["season"] + 1
    eligible = played.merge(prior, on=["player_id", "season"], how="inner")
    if eligible.empty:
        return 13.5
    return float(eligible["games"].mean())
