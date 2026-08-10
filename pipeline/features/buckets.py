"""
pipeline/features/buckets.py

Pure computation functions for each feature bucket.
No DB access, no side effects — fully unit-testable.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Surface classification — anything in TURF_SURFACES → surface_turf=1
TURF_SURFACES = {
    "fieldturf", "astroturf", "a_turf", "sportturf", "matrixturf",
    "astroplay", "dd grassmaster", "fieldturf 360",
}
GRASS_SURFACES = {"grass", "dessograss", "natural grass", "bermuda grass"}

DOME_ROOFS = {"dome", "closed", "retractable"}   # treat retractable as dome
OUTDOOR_ROOFS = {"outdoors", "open", "outdoor"}


# ── Helper functions ──────────────────────────────────────────────────────────

def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _safe_float(val) -> Optional[float]:
    """Convert to float, preserving None for truly missing data and 0 for zero."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _sum_fumbles(row: dict) -> Optional[float]:
    """
    Sum all fumble types from a game_log row. Returns None only if ALL sources are None.

    Uses only the three count columns (rushing, receiving, sack). Do not add *_fumbles_lost
    — those are a subset of the totals and would double-count if summed together.
    """
    cols = ["rushing_fumbles", "receiving_fumbles", "sack_fumbles"]
    vals = [row.get(c) for c in cols]
    non_none = [v for v in vals if v is not None]
    if not non_none:
        return None
    return sum(float(v) for v in non_none)


# ── Bucket 2: Season Baseline ─────────────────────────────────────────────────

def compute_season_baseline(prior_rows: list[dict]) -> dict[str, Optional[float]]:
    """
    Bucket 2 — season-to-date cumulative averages.

    Args:
        prior_rows: all of the player's completed game rows this season,
                    sorted oldest-first. Includes all weeks < target_week.
    """
    n = len(prior_rows)
    if n == 0:
        return {
            "seas_games_played":          0,
            "seas_avg_receiving_yards":   None,
            "seas_avg_targets":           None,
            "seas_avg_receptions":        None,
            "seas_avg_target_share":      None,
            "seas_avg_fantasy_ppr":       None,
            "seas_yards_per_target":      None,
            "seas_yards_per_reception":   None,
            "seas_avg_carries":           None,
            "seas_avg_rushing_yards":     None,
            "seas_yards_per_carry":       None,
            "seas_avg_attempts":          None,
            "seas_avg_passing_yards":     None,
            "seas_completion_pct":        None,
            "seas_avg_passing_cpoe":      None,
            "seas_avg_passing_epa":       None,
            "seas_avg_receiving_epa":     None,
            "seas_avg_rushing_epa":      None,
            "seas_avg_racr":             None,
            "seas_avg_wopr":             None,
            "seas_avg_receiving_yac":    None,
        }

    def _avg(key: str) -> Optional[float]:
        vals = [r.get(key) for r in prior_rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def _total(key: str) -> float:
        return sum(r.get(key) or 0 for r in prior_rows)

    avg_rec_yards = _avg("receiving_yards")
    avg_targets   = _avg("targets")
    avg_recs      = _avg("receptions")
    avg_carries   = _avg("carries")
    avg_rush_yds  = _avg("rushing_yards")
    avg_attempts  = _avg("attempts")
    avg_pass_yds  = _avg("passing_yards")
    _avg("completions")

    total_rec_yds = _total("receiving_yards")
    total_targets = _total("targets")
    total_recs    = _total("receptions")
    total_carries = _total("carries")
    total_rush    = _total("rushing_yards")
    total_atts    = _total("attempts")
    total_comps   = _total("completions")

    return {
        "seas_games_played":          n,
        "seas_avg_receiving_yards":   avg_rec_yards,
        "seas_avg_targets":           avg_targets,
        "seas_avg_receptions":        avg_recs,
        "seas_avg_target_share":      _avg("target_share"),
        "seas_avg_fantasy_ppr":       _avg("fantasy_points_ppr"),
        "seas_yards_per_target":      _safe_div(total_rec_yds, total_targets),
        "seas_yards_per_reception":   _safe_div(total_rec_yds, total_recs),
        "seas_avg_carries":           avg_carries,
        "seas_avg_rushing_yards":     avg_rush_yds,
        "seas_yards_per_carry":       _safe_div(total_rush, total_carries),
        "seas_avg_attempts":          avg_attempts,
        "seas_avg_passing_yards":     avg_pass_yds,
        "seas_completion_pct":        _safe_div(total_comps, total_atts),
        "seas_avg_passing_cpoe":      _avg("passing_cpoe"),
        "seas_avg_passing_epa":       _avg("passing_epa"),
        "seas_avg_receiving_epa":     _avg("receiving_epa"),
        "seas_avg_rushing_epa":       _avg("rushing_epa"),
        "seas_avg_racr":              _avg("racr"),
        "seas_avg_wopr":              _avg("wopr"),
        "seas_avg_receiving_yac":     _safe_div(
            _total("receiving_yards_after_catch"), total_recs
        ) if total_recs and total_recs > 0 else None,
    }


# ── Bucket 3: Matchup Metrics ─────────────────────────────────────────────────

def compute_matchup_stats(
    opponent_team: str,
    position: Optional[str],
    all_season_rows: list[dict],
    target_week: int,
) -> dict[str, Optional[float]]:
    """
    Bucket 3 — what the opponent has allowed to this position group this season.

    Groups by game_id first (sum multiple players from same team/game) then
    computes per-game averages. This avoids inflating stats by counting
    individual players rather than team totals.

    Args:
        opponent_team: abbreviation of the defensive team (e.g. "GB").
        position:      target player's position (e.g. "WR"). Pass None to
                       include all positions.
        all_season_rows: every PlayerStatsRow dict for this season.
        target_week:   only look at games before this week.
    """
    # Rows where the opponent_team played defense (i.e., player's team faced them)
    opp_rows = [
        r for r in all_season_rows
        if r.get("opponent_team") == opponent_team
        and (position is None or r.get("position") == position)
        and r.get("week") is not None
        and int(r["week"]) < target_week
    ]

    if not opp_rows:
        return {
            "opp_avg_receiving_yards_allowed":  None,
            "opp_avg_targets_allowed":           None,
            "opp_avg_tds_allowed":               None,
            "opp_avg_fantasy_ppr_allowed":       None,
            "opp_avg_rushing_yards_allowed":     None,
            "opp_avg_carries_allowed":           None,
            "opp_avg_pass_attempts_allowed":     None,
            "opp_zone_pct":                      None,
            "opp_man_pct":                       None,
            "opp_blitz_rate":                    None,
            "opp_pressure_rate":                 None,
        }

    # Grab the most recent populated trailing tendencies from past games
    # (These were written by pbp_pipeline.py to previous weeks' feature_matrix rows)
    recent_zone, recent_man, recent_blitz, recent_press = None, None, None, None
    for r in sorted(opp_rows, key=lambda x: x.get("week", 0), reverse=True):
        if r.get("opp_zone_pct") is not None:
            recent_zone = r.get("opp_zone_pct")
            recent_man = r.get("opp_man_pct")
            recent_blitz = r.get("opp_blitz_rate")
            recent_press = r.get("opp_pressure_rate_pbp") or r.get("opp_pressure_rate")
            break

    # Aggregate by game to get per-game team totals.
    # Primary key: game_id. Fallback: (team, week) when game_id is absent
    # (nflreadpy 2024 player_stats omits game_id at weekly summary level).
    game_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "rec_yards": 0.0, "targets": 0.0, "rec_tds": 0.0,
            "fantasy_ppr": 0.0, "rush_yards": 0.0,
            "carries": 0.0, "completions": 0.0, "pass_attempts": 0.0,
        }
    )
    for r in opp_rows:
        gid = (
            r.get("game_id")
            or f"{r.get('team', 'X')}@{r.get('opponent_team', 'Y')}_{r.get('week', 0)}"
        )
        game_totals[gid]["rec_yards"]     += r.get("receiving_yards") or 0
        game_totals[gid]["targets"]       += r.get("targets") or 0
        game_totals[gid]["rec_tds"]       += r.get("receiving_tds") or 0
        game_totals[gid]["fantasy_ppr"]   += r.get("fantasy_points_ppr") or 0
        game_totals[gid]["rush_yards"]    += r.get("rushing_yards") or 0
        game_totals[gid]["carries"]       += r.get("carries") or 0
        game_totals[gid]["completions"]   += r.get("completions") or 0
        game_totals[gid]["pass_attempts"] += r.get("attempts") or r.get("pass_attempts") or 0

    n_games = len(game_totals)
    totals = list(game_totals.values())

    return {
        "opp_avg_receiving_yards_allowed": sum(g["rec_yards"]     for g in totals) / n_games,
        "opp_avg_targets_allowed":         sum(g["targets"]       for g in totals) / n_games,
        "opp_avg_tds_allowed":             sum(g["rec_tds"]       for g in totals) / n_games,
        "opp_avg_fantasy_ppr_allowed":     sum(g["fantasy_ppr"]   for g in totals) / n_games,
        "opp_avg_rushing_yards_allowed":   sum(g["rush_yards"]    for g in totals) / n_games,
        "opp_avg_carries_allowed":         sum(g["carries"]       for g in totals) / n_games,
        "opp_avg_completions_allowed":     sum(g["completions"]   for g in totals) / n_games,
        "opp_avg_pass_attempts_allowed":   sum(g["pass_attempts"] for g in totals) / n_games,
        "opp_zone_pct":                    recent_zone,
        "opp_man_pct":                     recent_man,
        "opp_blitz_rate":                  recent_blitz,
        "opp_pressure_rate":               recent_press,
    }


# ── Positional depth signal ───────────────────────────────────────────────────

def compute_pos_rank(
    player_id: str,
    team: Optional[str],
    position: Optional[str],
    target_week: int,
    all_season_rows: list[dict],
) -> Optional[float]:
    """
    Bucket 11 proxy? No, Bucket 9/10 positional depth signal.
    Dynamically infers whether a player is WR1, WR2, RB1, RB2, TE1, etc.
    by ranking their season-to-date opportunity volume against teammates
    at the same position.

    Uses targets for WR/TE, carries for RB, pass attempts for QB.
    Returns 1.0 for the highest volume player, 2.0 for second, etc.
    """
    if not team or not position:
        return None

    teammate_stats: dict[str, float] = defaultdict(float)
    for r in all_season_rows:
        if r.get("team") == team and r.get("position") == position:
            w = r.get("week")
            if w is not None and w < target_week:
                pid = str(r["player_id"])

                # Determine which primary volume stat to rank
                if position in ("WR", "TE"):
                    val = float(r.get("targets") or 0.0)
                elif position == "RB":
                    val = float(r.get("carries") or 0.0)
                elif position == "QB":
                    val = float(r.get("attempts") or r.get("pass_attempts") or 0.0)
                else:
                    val = float(r.get("fantasy_points_ppr") or 0.0)

                teammate_stats[pid] += val

    if not teammate_stats:
        return 1.0  # Default to depth chart 1 if first week or no data

    # Sort DESC by volume
    sorted_teammates = sorted(teammate_stats.items(), key=lambda x: x[1], reverse=True)

    for rank_idx, (pid, _) in enumerate(sorted_teammates):
        if pid == player_id:
            return float(rank_idx + 1)

    # If player hasn't played yet this season but is active, rank them after current players
    return float(len(sorted_teammates) + 1)


# ── Bucket 4: Weather & Venue ─────────────────────────────────────────────────

def compute_venue_features(game: dict) -> dict[str, Optional[Any]]:
    """
    Bucket 4 — weather and venue context.

    precipitation_bucket remains None until OpenWeatherMap API integration
    (Phase 4). red_zone_target_share populates via PBP pipeline (Phase 4).

    Args:
        game: ScheduleRow.model_dump() or equivalent dict with roof, surface,
              temp, wind keys.
    """
    roof    = (game.get("roof") or "").lower()
    surface = (game.get("surface") or "").lower()
    temp    = game.get("temp")      # °F, None for dome games
    wind    = game.get("wind")      # mph, None for dome games

    is_dome = 1 if any(r in roof for r in DOME_ROOFS) else 0

    # If dome, climate is controlled — treat as mild/calm
    eff_temp = temp if temp is not None else (68.0 if is_dome else None)
    eff_wind = wind if wind is not None else (0.0  if is_dome else None)

    if eff_temp is None:
        temp_bucket = None
    elif eff_temp < 32:
        temp_bucket = 0   # cold
    elif eff_temp < 50:
        temp_bucket = 1   # cool
    else:
        temp_bucket = 2   # mild/warm

    if eff_wind is None:
        wind_bucket = None
    elif eff_wind < 10:
        wind_bucket = 0   # calm
    elif eff_wind < 20:
        wind_bucket = 1   # breezy
    else:
        wind_bucket = 2   # windy

    surface_turf = 1 if any(s in surface for s in TURF_SURFACES) else 0
    # Dome = no precipitation effect; outdoor uses OpenWeatherMap value (0/1/2)
    precip = 0 if is_dome else game.get("precipitation_bucket")

    return {
        "temp_f":              eff_temp,
        "wind_mph":            eff_wind,
        "is_dome":             is_dome,
        "surface_turf":        surface_turf,
        "temp_bucket":         temp_bucket,
        "wind_bucket":         wind_bucket,
        "precipitation_bucket": precip,
    }


# ── Bucket 6: Rest ────────────────────────────────────────────────────────────

def compute_rest_features(game: dict, is_home: bool) -> dict[str, Optional[int]]:
    """
    Bucket 6 — rest days and schedule context.

    Uses game.home_rest / game.away_rest (days since last game, from nflreadpy).
    These already account for bye weeks (a team coming off bye shows ~14 days).

    Args:
        game:    ScheduleRow.model_dump() with home_rest / away_rest fields.
        is_home: whether the target player's team is the home team.
    """
    raw_rest = game.get("home_rest") if is_home else game.get("away_rest")
    days_rest: Optional[int] = int(raw_rest) if raw_rest is not None else None

    is_short_week = int(days_rest < 6)  if days_rest is not None else None
    is_bye_prior  = int(days_rest >= 12) if days_rest is not None else None

    return {
        "days_rest":      days_rest,
        "is_short_week":  is_short_week,
        "is_bye_prior":   is_bye_prior,
    }


# ── Bucket 5: Team Context ────────────────────────────────────────────────────

def compute_team_context(game: dict, player_team: str) -> dict[str, Optional[float]]:
    """
    Bucket 5 — game-level team context.

    spread_line convention from nflreadpy: negative = home team favored.
    is_home: 1 if player_team is home team.
    """
    is_home = 1 if player_team == game.get("home_team") else 0

    return {
        "game_total_line": game.get("total_line"),
        "spread_line":     game.get("spread_line"),
        "is_home":         is_home,
    }


# ── Bucket 8: Injury / Availability ──────────────────────────────────────────

def compute_injury_features(
    player_id: str,
    target_week: int,
    season: int,
    prior_rows: list[dict],
    injury_df: Optional[Any] = None,
) -> dict[str, Optional[int]]:
    """
    Bucket 8 — player availability / injury status.

    injury_status_encoded:
        Numeric encoding from the ESPN injury report for this player/week.
        0=out/dnp, 1=doubtful, 2=questionable, 3=limited, 4=full.
        None if the player is absent from the injury report (assumed healthy).

    games_missed_streak:
        Consecutive weeks (going back from target_week-1) in the current
        season where the player has no game_log entry. A player coming off
        a 2-week injury has streak=2. Resets to 0 on first played week found.
        This is a pure function of prior_rows — no ESPN data required.

    Args:
        player_id:   gsis_id for the player.
        target_week: The week being projected (1-based).
        season:      NFL season year (used to filter prior_rows to this season).
        prior_rows:  All completed game rows for this player, this season,
                     week < target_week, sorted oldest-first.
        injury_df:   Optional pd.DataFrame from EspnAdapter.fetch_injury_report()
                     with columns [player_id, practice_status, week, season].
                     Pass None to skip the ESPN status lookup.
    """
    # ── games_missed_streak ───────────────────────────────────────────────────
    # Build the set of weeks played in this season (from prior_rows).
    weeks_played = {
        int(r["week"]) for r in prior_rows
        if r.get("season") == season and r.get("week") is not None
    }
    streak = 0
    for w in range(target_week - 1, 0, -1):
        if w not in weeks_played:
            streak += 1
        else:
            break   # found a played week — stop counting

    # ── injury_status_encoded ─────────────────────────────────────────────────
    encoded: Optional[int] = None
    if injury_df is not None:
        try:
            if not injury_df.empty and "player_id" in injury_df.columns:
                mask = (
                    (injury_df["player_id"] == player_id)
                    & (injury_df.get("season", season) == season)
                    & (injury_df.get("week",   target_week) == target_week)
                )
                row = injury_df[mask]
                if not row.empty:
                    from scraper.adapters.espn_adapter import STATUS_ENCODING
                    status = str(row.iloc[0].get("practice_status", "")).lower()
                    encoded = STATUS_ENCODING.get(status)
        except Exception as exc:
            logger.debug("compute_injury_features lookup failed: %s", exc)

    return {
        "injury_status_encoded": encoded,
        "games_missed_streak":   streak,
    }


# ── Bucket 7: Rule Coefficients ───────────────────────────────────────────────

def compute_rule_features(season: int) -> dict[str, Optional[float]]:
    """
    Bucket 7 — NFL rule change coefficient.

    Encodes significant rule changes that affect offensive volume and field
    position.  Two implemented anchors:

      2023  (season == 2023): New kickoff touchback rules reduced touchback rates,
            slightly increasing return opportunities. Coefficient = 0.5 (partial
            impact — rule was new and teams were still adapting).

      2024+ (season >= 2024): Revised kickoff formation rules (kick from 40yd
            line, blockers pre-positioned). Significantly increased average return
            yardage and starting field position, boosting offensive efficiency.
            Coefficient = 1.0.

      < 2023: Baseline — no relevant rule change captured. Coefficient = 0.0.
    """
    if season >= 2024:
        rule_coeff: float = 1.0   # 2024+ kickoff formation rule: full effect
    elif season == 2023:
        rule_coeff = 0.5          # 2023 kickoff rule: partial / adaptation phase
    else:
        rule_coeff = 0.0          # pre-2023 baseline
    return {"rule_coeff": rule_coeff}


# ── Bucket 9: Scheme Interactions ────────────────────────────────────────────

def compute_scheme_interactions(
    kalman_form: dict,
    matchup_stats: dict,
) -> dict[str, Optional[float]]:
    """
    Bucket 9 — scheme interaction features.

    Explicit cross-terms of a player's route-depth profile against the opposing
    coverage shell. Without these, XGBoost/LightGBM must discover the interaction
    from raw independently-varying columns, which is much harder (requires a high
    interaction_constraints depth).

    ── Defensive tendency fields ─────────────────────────────────────────────
    Currently populated from game_logs.opp_zone_pct / opp_man_pct … if present.
    Phase 4 enhancement: wire nflreadpy play-by-play coverage shell data so
    these are real game-week values. Until then they default to None.

    ── Interaction terms ─────────────────────────────────────────────────────
    deep_matchup_score:
        kalman_est_air_yards_share × (1 − opp_zone_pct)
        High when a deep-route receiver faces a man-heavy defense (no safety help).
        Drives score UP for field-stretchers vs single-high man.

    coverage_matchup_score:
        kalman_est_target_share × opp_man_pct
        High when a high-volume receiver faces man coverage.
        Rewards separators who win 1-on-1.

    separation_demand_score:
        kalman_est_target_share × opp_man_pct × (1 − opp_zone_pct)
        Three-way interaction. Non-zero only when all three signal simultaneously:
        high target share + man coverage + no zone shell.
    """
    air_share   = kalman_form.get("kalman_est_air_yards_share")
    tgt_share   = kalman_form.get("kalman_est_target_share")
    zone_pct    = matchup_stats.get("opp_zone_pct")
    man_pct     = matchup_stats.get("opp_man_pct")

    def _safe_mul(*vals: Optional[float]) -> Optional[float]:
        """Return product of vals; None if any val is None."""
        if any(v is None for v in vals):
            return None
        result = 1.0
        for v in vals:
            result *= v  # type: ignore[operator]
        return result

    def _safe_mul_sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
        """a × (1 − b); None if either is None."""
        if a is None or b is None:
            return None
        return a * (1.0 - b)

    deep_matchup = _safe_mul_sub(air_share, zone_pct)
    coverage_matchup = _safe_mul(tgt_share, man_pct)
    sep_demand = _safe_mul_sub(
        _safe_mul(tgt_share, man_pct),
        zone_pct,
    )

    return {
        "deep_matchup_score":      deep_matchup,
        "coverage_matchup_score":  coverage_matchup,
        # A target-game snap share used to feed this interaction.  Retain the
        # storage column for rebuild compatibility, but keep it null until a
        # separately registered causal interaction is designed.
        "blitz_exposure":          None,
        "separation_demand_score": sep_demand,
    }


# ── Phase 4 feature groups (A/B-gated) ────────────────────────────────────────

# Reused from ml.volume_redistribution (keep coefficients identical — do not diverge).
_SPREAD_PASS_COEFFICIENT = 0.35
_TOTAL_PASS_COEFFICIENT = 0.25
_NEUTRAL_SPREAD_ABS = 3.0  # |spread| ≤ 3 → neutral script
_GLOBAL_PASS_AVG = 35.0
_GLOBAL_RUSH_AVG = 24.0


@dataclass
class SeasonFeatureContext:
    """Precomputed causal aggregates for one season — build once, reuse per row."""

    team_carries_by_game: dict[tuple[str, str], float]
    team_pass_by_game: dict[tuple[str, str], float]
    team_rush_by_game: dict[tuple[str, str], float]
    league_ts: dict[int, dict[str, float]]
    league_snap: dict[int, dict[str, float]]
    league_carry: dict[int, dict[str, float]]
    league_tgt_allowed: dict[int, float]
    team_game_volume: dict[str, list[tuple[int, float, float]]]


def build_season_feature_context(all_season_rows: list[dict]) -> SeasonFeatureContext:
    """Scan season rows once; return lookups keyed by week / game / team."""
    team_carries_by_game: dict[tuple[str, str], float] = defaultdict(float)
    team_pass_by_game: dict[tuple[str, str], float] = defaultdict(float)
    team_rush_by_game: dict[tuple[str, str], float] = defaultdict(float)
    team_targets_by_game: dict[tuple[str, str], float] = defaultdict(float)
    game_week: dict[tuple[str, str], int] = {}
    by_player: dict[str, list[tuple]] = defaultdict(list)
    player_pos: dict[str, str] = {}

    for r in all_season_rows:
        week = int(r.get("week") or 0)
        gid = r.get("game_id")
        team = r.get("team") or ""
        pid = str(r.get("player_id") or "")
        pos = (r.get("position") or "").upper()
        if gid and team:
            key = (str(gid), str(team))
            team_carries_by_game[key] += float(r.get("carries") or 0)
            team_pass_by_game[key] += float(r.get("attempts") or r.get("pass_attempts") or 0)
            team_rush_by_game[key] += float(r.get("carries") or 0)
            team_targets_by_game[key] += float(r.get("targets") or 0)
            game_week[key] = week
        if pid:
            player_pos[pid] = pos
            snap_raw = r.get("offense_pct")
            snap = None
            if snap_raw is not None:
                v = float(snap_raw)
                snap = v / 100.0 if v > 1.0 else v
            ts = r.get("target_share")
            by_player[pid].append(
                (week, float(ts) if ts is not None else None, snap, gid, float(r.get("carries") or 0), team)
            )

    team_game_volume: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    seen_team_game: set[tuple[str, str]] = set()
    for (gid, team), week in game_week.items():
        key = (team, gid)
        if key in seen_team_game:
            continue
        seen_team_game.add(key)
        team_game_volume[team].append(
            (week, team_pass_by_game.get((gid, team), 0.0), team_rush_by_game.get((gid, team), 0.0))
        )
    for team in team_game_volume:
        team_game_volume[team].sort(key=lambda x: x[0])

    max_week = max(game_week.values()) if game_week else 18
    league_ts: dict[int, dict[str, float]] = {w: {} for w in range(1, max_week + 2)}
    league_snap: dict[int, dict[str, float]] = {w: {} for w in range(1, max_week + 2)}
    league_carry: dict[int, dict[str, float]] = {w: {} for w in range(1, max_week + 2)}
    league_tgt_allowed: dict[int, float] = {}

    for week in range(1, max_week + 2):
        pos_ts: dict[str, list[float]] = defaultdict(list)
        pos_snap: dict[str, list[float]] = defaultdict(list)
        pos_carry: dict[str, list[float]] = defaultdict(list)
        for pid, entries in by_player.items():
            pos = player_pos.get(pid, "")
            ts_vals = [e[1] for e in entries if e[0] < week and e[1] is not None]
            snap_vals = [e[2] for e in entries if e[0] < week and e[2] is not None]
            cs_vals = []
            for e in entries:
                if e[0] >= week:
                    continue
                gid = e[3]
                tc = team_carries_by_game.get((str(gid or ""), str(e[5] or "")), 0.0)
                if tc > 0:
                    cs_vals.append(e[4] / tc)
            if ts_vals:
                pos_ts[pos].append(sum(ts_vals) / len(ts_vals))
            if snap_vals:
                pos_snap[pos].append(sum(snap_vals) / len(snap_vals))
            if cs_vals:
                pos_carry[pos].append(sum(cs_vals) / len(cs_vals))
        for pos, vals in pos_ts.items():
            if vals:
                league_ts[week][pos] = sum(vals) / len(vals)
        for pos, vals in pos_snap.items():
            if vals:
                league_snap[week][pos] = sum(vals) / len(vals)
        for pos, vals in pos_carry.items():
            if vals:
                league_carry[week][pos] = sum(vals) / len(vals)
        tgt_vals = [team_targets_by_game[key] for key, w in game_week.items() if w < week]
        if tgt_vals:
            league_tgt_allowed[week] = sum(tgt_vals) / len(tgt_vals)

    return SeasonFeatureContext(
        team_carries_by_game=dict(team_carries_by_game),
        team_pass_by_game=dict(team_pass_by_game),
        team_rush_by_game=dict(team_rush_by_game),
        league_ts=league_ts,
        league_snap=league_snap,
        league_carry=league_carry,
        league_tgt_allowed=league_tgt_allowed,
        team_game_volume=dict(team_game_volume),
    )


def compute_usage_shares(
    prior_rows: list[dict],
    all_season_rows: list[dict],
    player_id: str,
    team: str,
    week: int,
    position: Optional[str],
    ctx: Optional[SeasonFeatureContext] = None,
) -> dict[str, Optional[float]]:
    empty = {"carry_share": None, "snap_share_trailing": None, "snap_share_trend": None}
    if not prior_rows or not team:
        return empty

    if ctx is not None:
        team_carries_by_game = ctx.team_carries_by_game
    else:
        team_carries_by_game: dict[tuple[str, str], float] = defaultdict(float)
        for r in all_season_rows:
            if r.get("team") == team and (r.get("week") or 0) < week:
                gid = r.get("game_id")
                if gid:
                    team_carries_by_game[(str(gid), str(team))] += float(r.get("carries") or 0)

    player_carry_shares: list[float] = []
    for r in prior_rows:
        gid = r.get("game_id")
        tc = team_carries_by_game.get((str(gid or ""), str(team)), 0.0)
        if tc > 0:
            player_carry_shares.append(float(r.get("carries") or 0) / tc)
    carry_share = (
        sum(player_carry_shares) / len(player_carry_shares) if player_carry_shares else None
    )

    def _snap_frac(r: dict) -> Optional[float]:
        raw = r.get("offense_pct")
        if raw is None:
            return None
        v = float(raw)
        return v / 100.0 if v > 1.0 else v

    snaps = [s for s in (_snap_frac(r) for r in prior_rows) if s is not None]
    snap_share_trailing = sum(snaps) / len(snaps) if snaps else None
    snap_share_trend = (snaps[-1] - snaps[0]) if len(snaps) >= 2 else None
    return {
        "carry_share": carry_share,
        "snap_share_trailing": snap_share_trailing,
        "snap_share_trend": snap_share_trend,
    }


def compute_opp_adj_usage(
    seas: dict[str, Optional[float]],
    matchup: dict[str, Optional[float]],
    usage: dict[str, Optional[float]],
    all_season_rows: list[dict],
    week: int,
    position: Optional[str],
    ctx: Optional[SeasonFeatureContext] = None,
) -> dict[str, Optional[float]]:
    if ctx is None:
        ctx = build_season_feature_context(all_season_rows)
    pos = (position or "").upper()
    league_ts = ctx.league_ts.get(week, {}).get(pos)
    league_snap = ctx.league_snap.get(week, {}).get(pos)
    league_carry = ctx.league_carry.get(week, {}).get(pos)
    league_tgt_allowed = ctx.league_tgt_allowed.get(week)

    player_ts = seas.get("seas_avg_target_share")
    snap_trailing = usage.get("snap_share_trailing")
    carry_share = usage.get("carry_share")
    opp_targets = matchup.get("opp_avg_targets_allowed")
    opp_adj = None
    if player_ts is not None and opp_targets is not None and league_tgt_allowed:
        opp_adj = player_ts * (opp_targets / league_tgt_allowed)

    return {
        "ts_vs_league": _safe_div(player_ts, league_ts),
        "rz_ts_vs_league": None,
        "snap_vs_pos_avg": _safe_div(snap_trailing, league_snap),
        "carry_share_vs_league": _safe_div(carry_share, league_carry),
        "opp_adj_target_share": opp_adj,
        "carry_share": carry_share,
        "snap_share_trailing": snap_trailing,
        "snap_share_trend": usage.get("snap_share_trend"),
    }


def compute_pace_script(
    team: str,
    week: int,
    game: dict,
    all_season_rows: list[dict],
    is_home: Optional[int],
    ctx: Optional[SeasonFeatureContext] = None,
) -> dict[str, Optional[float]]:
    empty = {
        "team_pace": None, "team_pass_rate": None,
        "expected_pass_attempts": None, "expected_pass_rate": None,
        "neutral_script_flag": None,
    }
    if not team:
        return empty

    team_pace = team_pass_rate = None
    base_pass, base_rush = _GLOBAL_PASS_AVG, _GLOBAL_RUSH_AVG

    if ctx is not None:
        prior_games = [g for g in ctx.team_game_volume.get(team, []) if g[0] < week]
        if prior_games:
            plays = [p + r for _, p, r in prior_games]
            passes = [p for _, p, _ in prior_games]
            rushes = [r for _, _, r in prior_games]
            team_pace = sum(plays) / len(plays)
            total_plays = sum(plays)
            team_pass_rate = (sum(passes) / total_plays) if total_plays > 0 else None
            base_pass = sum(passes) / len(passes)
            base_rush = sum(rushes) / len(rushes)
    else:
        by_game: dict[str, dict[str, float]] = defaultdict(lambda: {"pass": 0.0, "rush": 0.0})
        for r in all_season_rows:
            if r.get("team") != team or (r.get("week") or 0) >= week:
                continue
            gid = r.get("game_id")
            if not gid:
                continue
            by_game[gid]["pass"] += float(r.get("attempts") or r.get("pass_attempts") or 0)
            by_game[gid]["rush"] += float(r.get("carries") or 0)
        if by_game:
            plays = [g["pass"] + g["rush"] for g in by_game.values()]
            passes = [g["pass"] for g in by_game.values()]
            rushes = [g["rush"] for g in by_game.values()]
            team_pace = sum(plays) / len(plays)
            total_plays = sum(plays)
            team_pass_rate = (sum(passes) / total_plays) if total_plays > 0 else None
            base_pass = sum(passes) / len(passes)
            base_rush = sum(rushes) / len(rushes)

    spread = game.get("spread_line")
    total = game.get("total_line")
    home_flag = int(is_home) if is_home is not None else (
        1 if team == game.get("home_team") else 0
    )
    expected_pass_attempts = expected_pass_rate = None
    if spread is not None or total is not None:
        spread_f = float(spread or 0.0)
        total_f = float(total or 45.0)
        effective_spread = spread_f if home_flag else -spread_f
        expected_pass_attempts = max(
            base_pass + _SPREAD_PASS_COEFFICIENT * effective_spread
            + _TOTAL_PASS_COEFFICIENT * (total_f - 45.0),
            1.0,
        )
        expected_rush = max(base_rush - _SPREAD_PASS_COEFFICIENT * effective_spread * 0.5, 1.0)
        expected_pass_rate = expected_pass_attempts / (expected_pass_attempts + expected_rush)

    neutral_script_flag = None
    if spread is not None:
        neutral_script_flag = 1.0 if abs(float(spread)) <= _NEUTRAL_SPREAD_ABS else 0.0

    return {
        "team_pace": team_pace,
        "team_pass_rate": team_pass_rate,
        "expected_pass_attempts": expected_pass_attempts,
        "expected_pass_rate": expected_pass_rate,
        "neutral_script_flag": neutral_script_flag,
    }


def compute_progression_priors(
    target_row: dict,
    prior_rows: list[dict],
    season: int,
) -> dict[str, Optional[float]]:
    # `players.years_exp` is a mutable present-day snapshot.  Entry year is an
    # immutable draft/roster fact, so season - entry_year is the only legal
    # as-of experience estimate for the opt-in progression group.
    years_exp = None
    try:
        entry_year = target_row.get("entry_year")
        years_exp_f = max(float(season - int(entry_year)), 0.0) if entry_year is not None else None
    except (TypeError, ValueError):
        years_exp_f = None

    age: Optional[float] = None
    birth = target_row.get("birth_date") or target_row.get("age")
    if birth is not None and not isinstance(birth, (int, float)):
        try:
            year = getattr(birth, "year", None)
            if year is None:
                year = int(str(birth)[:4])
            age = float(season - int(year))
        except (TypeError, ValueError):
            age = None
    elif isinstance(birth, (int, float)):
        age = float(birth)

    draft_round = target_row.get("draft_round")
    try:
        draft_round_f = float(draft_round) if draft_round is not None else None
    except (TypeError, ValueError):
        draft_round_f = None

    career_games = float(len(prior_rows)) if prior_rows is not None else 0.0
    exp_bucket: Optional[float] = None
    if years_exp_f is not None:
        ye = int(years_exp_f)
        if ye <= 0:
            exp_bucket = 0.0
        elif ye == 1:
            exp_bucket = 1.0
        elif ye <= 7:
            exp_bucket = 2.0
        else:
            exp_bucket = 3.0

    return {
        "years_exp": years_exp_f,
        "age": age,
        "career_games": career_games,
        "exp_bucket": exp_bucket,
        "draft_round_progression": draft_round_f,
    }
