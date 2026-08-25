"""
ml/season_simulator.py

Autoregressive Season Simulator for Gridiron Oracle.

OVERVIEW
--------
Projects a full NFL season (weeks 1-18) sequentially, feeding each week's
simulated game outcomes as updated priors into the next week's projection.

This is the "rest of season" simulator used to answer:
  - "What are Travis Kelce's season-end totals given current week 8 form?"
  - "What happens to Jayden Reed's value if Romeo Doubs misses the final 6 weeks?"
  - "Which team's skill players gain the most value after the trade deadline?"

AUTOREGRESSIVE STRUCTURE
------------------------
Week 1 → simulate → update Kalman states → Week 2 → simulate → ...

Each week:
  1. Resolve each scheduled game's expected points via the real Phase 4
     team-game model (ml.team_game_model), using this simulation's own
     carried-forward Elo, and draw a per-path team score from it
     (_draw_team_score_paths).
  2. Pull current Kalman estimates for all rostered players.
  3. Apply Volume Redistribution (injury report for that projected week).
  4. Draw joint samples via Copula Layer (correlated within-team/game), then
     shift each player's samples toward their OWN team's SAME-path score
     from step 1 (_apply_team_script_coupling) — so an outlier stat game and
     a team's blowout loss can't land on the same path independently of
     each other by coincidence; they're coupled by construction.
  5. Run Monte Carlo projection to produce week simulation results.
  6. Feed simulated game stats BACK into the Kalman filter as if observed.
  7. Update Team Elo ratings from the same resolved points predictions used
     in step 1.
  8. Advance to next week.

The feedback loop means early-season performance shifts later-season
projections — exactly as the real Kalman filter would with actual stats.

OUTPUT
------
    SeasonSimulation:
        - player_season_totals:  {player_id: {stat: {mean, p10, p50, p90}}}
        - team_win_totals:       {team: {wins_mean, wins_p10, wins_p90}}
        - week_by_week:          list[dict] — per-week snapshot for every player
        - n_simulations:         int — number of Monte Carlo season paths
        - metadata:              season, start_week, end_week, positions, stats

USAGE
-----
    from ml.season_simulator import SeasonSimulator

    sim = SeasonSimulator(
        season=2025,
        start_week=9,        # Current week (simulate rest of season)
        end_week=18,
        n_simulations=500,   # Season paths (500 is good; 2000 is precise)
        positions=["WR", "TE", "RB"],
        stats=["receiving_yards", "rushing_yards", "fantasy_ppr"],
    )
    results = sim.run(
        players_df=active_roster_df,
        prior_game_rows=prior_rows_by_player,
        schedule_df=schedule_df,           # {home_team, away_team, week}
        injury_projections=injury_map,     # {week: {player_id: status}}
    )

    # Access season totals
    kelce = results.player_season_totals["00-0033106"]
    print(f"Kelce season receiving: {kelce['receiving_yards']['p50']:.0f} yds")
    # → "Kelce season receiving: 1084 yds"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

_NFL_REGULAR_SEASON_WEEKS = list(range(1, 19))   # Weeks 1–18
_DEFAULT_STATS   = ["receiving_yards", "rushing_yards", "passing_yards", "fantasy_ppr"]
_DEFAULT_POSITIONS = ["QB", "RB", "WR", "TE"]

# How many samples to draw per player per week in simulation.
# 500 is a good balance: fast enough to run 500 season paths, precise enough
# for p10/p90 estimation. Increase to 2000 for final production reports.
_WEEK_SAMPLES = 200

# Conference membership for playoff probability (7 berths per conference).
_AFC = frozenset({
    "BUF", "MIA", "NE", "NYJ",
    "BAL", "CIN", "CLE", "PIT",
    "HOU", "IND", "JAX", "TEN",
    "DEN", "KC", "LV", "LAC",
})
_NFC = frozenset({
    "DAL", "NYG", "PHI", "WAS",
    "CHI", "DET", "GB", "MIN",
    "ATL", "CAR", "NO", "TB",
    "ARI", "LAR", "SF", "SEA",
})
_PLAYOFF_BERTHS = 7
# Ridge alpha for the Phase 4 team-game points model used to resolve game
# outcomes — matches ml.team_game_model.train()'s and
# scripts/materialize_team_game_predictions.py's default.
_TEAM_GAME_MODEL_ALPHA = 10.0

# Within-player Pearson r between a stat and that player's own team_score,
# measured from real game_logs joined to games (2019-2025, season >= 8 games
# per player, team-level score not opponent-adjusted). See the Phase 7
# investigation: `git log` for "Pickens coherence" for the exact query.
# Volume stats (targets/receptions/completions) came back ~0 or slightly
# negative (trailing teams often throw MORE, not less) — coupling those would
# be fabricating a relationship the data doesn't support, so they're left at
# the conservative default of 0.0 rather than guessed.
_TEAM_SCRIPT_COUPLING: dict[str, float] = {
    "receiving_yards": 0.10,
    "rushing_yards":   0.18,
    "passing_yards":   0.22,
    "fantasy_ppr":     0.20,
}
_DEFAULT_TEAM_SCRIPT_COUPLING = 0.0

# Volume-budget stats (Phase 7 fix): _apply_team_script_coupling only shifts
# each player's draw by coupling * player_std * z(team_score_path) — z(.) is
# standardised (mean zero), so the shift moves samples along a path but
# leaves the player's EXPECTATION unchanged by construction. That's fine as
# a correlation device but cannot make summed player yards match the team's
# own predicted yards, which is why they diverged by 1.66x before this fix.
# These three stats are instead allocated as a real budget (see
# _apply_volume_budget): the team's own predicted yards, drawn once per
# path via the Phase 4 yards model, split passing-vs-rushing by the league
# split measured from game_logs 2022-2025 (530,441 passing yards vs 268,373
# rushing = 66.4% passing), then divided among that team's players in each
# stat's group in proportion to their own raw simulated share — so the
# group sums to the budget exactly, on every path, and passing_yards and
# receiving_yards share the SAME budget (verified: receiving_yards is
# 99.9998% of passing_yards league-wide over the same window), so they
# agree by construction rather than converging only in expectation.
_VOLUME_BUDGET_STATS = frozenset({"passing_yards", "rushing_yards", "receiving_yards"})
_PASS_YARDS_SHARE = 0.6640356828998991
# Ceiling on _apply_volume_budget's per-path budget/group_sum rescale. Set
# well clear of ordinary operation (a healthy group's rescale sits near 1-3x,
# reaching ~5x in the tails) so the exact group-sum identity still holds on
# every normal path; this only bites the degenerate case where essentially
# nobody with a claim to the stat is available and a sliver of leftover signal
# would otherwise be multiplied ~150x into a full team's yardage.
_MAX_BUDGET_SCALE = 25.0


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class WeekSnapshot:
    """Per-player, per-week simulation result (one path)."""
    player_id: str
    week: int
    stat: str
    simulated_value: float      # This path's simulated stat for this week


@dataclass
class SeasonSimulation:
    """
    Complete output of a SeasonSimulator.run() call.

    All statistics are computed across n_simulations season paths,
    giving full distributional outputs rather than point estimates.
    """
    season: int
    start_week: int
    end_week: int
    n_simulations: int
    positions: list[str]
    stats: list[str]

    # {player_id: {stat: {"mean": float, "p10": float, "p50": float, "p90": float}}}
    player_season_totals: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)

    # {team: {"wins_mean": float, "wins_p10": float, "wins_p90": float}}
    team_win_totals: dict[str, dict[str, float]] = field(default_factory=dict)

    # {team: float} — fraction of season paths where team makes the playoffs
    # (top 7 by wins within conference; ties broken randomly per path).
    playoff_probs: dict[str, float] = field(default_factory=dict)

    # list of {player_id, week, stat, mean, p10, p50, p90} — per-week breakdown
    week_by_week: list[dict[str, Any]] = field(default_factory=list)

    # Raw simulation paths: shape (n_simulations, n_players, n_weeks, n_stats)
    # Large — only stored when store_raw=True in SeasonSimulator
    raw_paths: Optional[np.ndarray] = None

    def top_players(self, stat: str, position: Optional[str] = None, n: int = 10) -> pd.DataFrame:
        """Return top-N players by median season total for a stat."""
        rows = []
        for pid, stat_dict in self.player_season_totals.items():
            if stat not in stat_dict:
                continue
            d = stat_dict[stat]
            rows.append({
                "player_id": pid,
                "stat": stat,
                "mean":  round(d["mean"], 1),
                "p10":   round(d["p10"], 1),
                "p50":   round(d["p50"], 1),
                "p90":   round(d["p90"], 1),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.nlargest(n, "p50").reset_index(drop=True)

    def injury_scenario(
        self,
        injured_player_id: str,
        stat: str,
        beneficiary_player_id: str,
    ) -> dict[str, float]:
        """
        Compare beneficiary's season total with and without the injured player.
        Returns the projected uplift from the injury scenario.
        """
        baseline = self.player_season_totals.get(beneficiary_player_id, {}).get(stat, {})
        return {
            "beneficiary_p50_baseline": baseline.get("p50", 0.0),
            "note": f"Run simulation with {injured_player_id} marked OUT for full uplift comparison.",
        }


# ── Simulator ─────────────────────────────────────────────────────────────────

class SeasonSimulator:
    """
    Autoregressive Season Simulator.

    Runs n_simulations full-season Monte Carlo paths. Each path:
        - Iterates weeks start_week → end_week
        - Simulates each week using Kalman-updated priors
        - Updates Kalman states from simulated outcomes
        - Updates team Elo from simulated game scores
        - Applies projected injury reports week-by-week
        - Writes week results into the accumulator

    Args:
        season:        NFL season year.
        start_week:    First week to simulate. Use current week + 1 for
                       "rest of season" projections.
        end_week:      Last week to simulate (default: 18).
        n_simulations: Number of Monte Carlo season paths (default: 500).
        positions:     Positions to project (default: all 4).
        stats:         Stats to simulate (default: 4 core stats).
        use_copula:    If True, draw joint correlated samples within each team.
                       If False, draw independent samples per player.
                       Default: True.
        store_raw:     If True, store n_simulations × n_players × n_weeks × n_stats
                       raw paths array. Memory-intensive; default False.
    """

    def __init__(
        self,
        season: int,
        start_week: int = 1,
        end_week: int = 18,
        n_simulations: int = 500,
        positions: Optional[list[str]] = None,
        stats: Optional[list[str]] = None,
        use_copula: bool = True,
        use_cpp: bool = False,
        store_raw: bool = False,
        database_url: Optional[str] = None,
    ) -> None:
        self.season         = season
        self.start_week     = start_week
        self.end_week       = end_week
        self.n_simulations  = n_simulations
        self.positions      = positions or _DEFAULT_POSITIONS
        self.stats          = stats or _DEFAULT_STATS
        self.use_copula     = use_copula
        self.use_cpp        = use_cpp
        self.store_raw      = store_raw
        self._database_url  = database_url
        self._sim_elo       = None  # TeamEloSystem carried forward across simulated weeks
        # Phase 4 team-game points model (Ridge), fit lazily on first use —
        # see _ensure_points_model. Game resolution (wins, Elo updates) uses
        # this model's predictions, not a fantasy-points proxy.
        self._points_model        = None
        self._points_fill         = None
        self._points_residual_std = None
        # Phase 4 team-game yards model (Ridge), fit lazily the same way as
        # _points_model — the real per-path team volume budget behind
        # _apply_volume_budget, not a second independent guess.
        self._yards_model         = None
        self._yards_fill          = None
        self._yards_residual_std  = None
        # Per-week build_team_game_forward_frame cache — _resolve_week_games
        # is normally called once per week by run(), but callers that probe
        # a week more than once (retries, tests) shouldn't pay for a second
        # DB round trip for data that can't have changed within a run.
        self._forward_frame_cache: dict[int, pd.DataFrame] = {}

    # ── Main Entry Point ─────────────────────────────────────────────────────

    def run(
        self,
        players_df: pd.DataFrame,
        prior_game_rows: dict[str, list[dict]],
        schedule_df: Optional[pd.DataFrame] = None,
        injury_projections: Optional[dict[int, dict[str, str]]] = None,
        player_active_prob: Optional[dict[str, float]] = None,
        rng_seed: Optional[int] = None,
    ) -> SeasonSimulation:
        """
        Simulate the rest of the season across multiple Monte Carlo paths.

        Args:
            players_df:          DataFrame with [player_id, position, team, ...].
                                 All rostered players eligible for projection.
            prior_game_rows:     {player_id: [dict of game rows]} — Kalman history
                                 as of start_week-1. From normalize.py / DB load.
            schedule_df:         Optional DataFrame [week, home_team, away_team].
                                 Non-empty rows for a week gate real game resolution
                                 for that week: team wins and Elo updates are then
                                 computed from the Phase 4 points model against the
                                 real schedule pulled from the `games` table for
                                 (self.season, week) — schedule_df itself only
                                 signals which weeks to resolve, its home_team/
                                 away_team values are not used for scoring. If None,
                                 team wins/Elo/playoff_probs are omitted; player
                                 projections still work.
            injury_projections:  Optional {week: {player_id: injury_status}}.
                                 Projected injuries per week (e.g. from ESPN IR list
                                 or user-defined scenario) — a KNOWN status for a
                                 SPECIFIC week, discounting volume via
                                 _apply_injury_to_kalman.
            player_active_prob:  Optional {player_id: p_active}, p_active in [0, 1]
                                 — the season-long probability a player suits up
                                 in a given week (ml.playing_time.
                                 p_active_from_priors). Unlike injury_projections
                                 (a known per-week status), this is genuine
                                 uncertainty: each simulated week, each path draws
                                 its own independent Bernoulli(p_active) "did this
                                 player play" outcome, shared across all stats for
                                 that player so a path where they're inactive
                                 zeroes every stat together rather than each stat
                                 independently deciding. Without this, a retired
                                 or seldom-active player's low p_active is stored
                                 as metadata but never actually discounts their
                                 simulated volume — see
                                 test_season_simulator_availability_gating.py.
            rng_seed:            Optional seed for reproducibility.

        Returns:
            SeasonSimulation with season totals, team wins, and week-by-week data.
        """
        rng = np.random.default_rng(rng_seed)
        injury_projections = injury_projections or {}
        player_active_prob = player_active_prob or {}

        # Accumulated stats per player per stat: (n_simulations, n_weeks) → list
        # Key: (player_id, stat) → list of per-simulation season totals
        accum: dict[tuple[str, str], np.ndarray] = {
            (str(row["player_id"]), stat): np.zeros(self.n_simulations)
            for _, row in players_df.iterrows()
            for stat in self.stats
        }

        # Team win accumulator: {team: np.ndarray(n_simulations)}
        team_win_accum: dict[str, np.ndarray] = {}
        if schedule_df is not None and "home_team" in schedule_df.columns:
            all_teams = set(schedule_df["home_team"].tolist() + schedule_df["away_team"].tolist())
            for team in all_teams:
                team_win_accum[team] = np.zeros(self.n_simulations)

        week_by_week_rows: list[dict] = []
        weeks = list(range(self.start_week, self.end_week + 1))

        # Initialise Kalman state from prior rows (shared across simulations
        # as starting point — each simulation builds independent forward paths)
        initial_kalman_df = self._compute_initial_kalman(players_df, prior_game_rows)

        logger.info(
            "SeasonSimulator: S%d W%d-W%d | %d players | %d sims | copula=%s | cpp=%s",
            self.season, self.start_week, self.end_week,
            len(players_df), self.n_simulations, self.use_copula, self.use_cpp,
        )

        if self.use_cpp:
            try:
                return self._run_cpp_fast(initial_kalman_df)
            except Exception as e:
                logger.warning("C++ Engine failed (%s). Falling back to Python simulator.", e)

        # Season-level role multiplier: one draw per player per PATH, held for
        # the whole season (see _SEASON_ROLE_LOG_SD). Mean-1 by construction,
        # so this widens season totals without shifting them.
        role_paths: dict[str, np.ndarray] = {}
        if self._SEASON_ROLE_LOG_SD > 0:
            sd = float(self._SEASON_ROLE_LOG_SD)
            for pid in initial_kalman_df["player_id"].astype(str).unique():
                role_paths[pid] = np.exp(
                    rng.normal(-0.5 * sd * sd, sd, self.n_simulations)
                )

        # ── Main simulation loop: iterate per WEEK ──────────────────────────
        for week in weeks:
            week_injury_report = injury_projections.get(week, {})
            week_schedule = (
                schedule_df[schedule_df["week"] == week]
                if schedule_df is not None and "week" in schedule_df.columns
                else pd.DataFrame()
            )

            # Resolve real game outcomes (Phase 4 points model) FIRST, so each
            # path's team score can be threaded into that same path's player
            # stat draws below (_simulate_week's team_score_paths) — this is
            # the Pickens-test link: a path where the team scored low can't
            # independently also be the path with that team's outlier
            # receiving line, because both come from the same per-path draw.
            resolved = pd.DataFrame()
            team_score_paths: dict[str, np.ndarray] = {}
            # Teams actually scheduled this week, per the real games table —
            # None when no schedule was supplied (can't determine byes, so
            # don't gate at all, matching every caller that never passed
            # schedule_df). When resolved IS available, any team absent from
            # it is on a bye: without this, a uniform 1/18 (5.6%) extra week
            # of volume is added to every player's season total, invisible
            # to the "summed weekly equals season" identity since it
            # inflates both sides equally.
            playing_teams: Optional[set[str]] = None
            team_yards_paths: dict[str, np.ndarray] = {}
            if not week_schedule.empty:
                resolved = self._resolve_week_games(week)
                if not resolved.empty:
                    team_score_paths = self._draw_team_score_paths(resolved, self.n_simulations, rng)
                    team_yards_paths = self._draw_team_yards_paths(resolved, self.n_simulations, rng)
                    playing_teams = set(resolved["team"].astype(str))

            # Draw weekly samples: shape (n_simulations, n_players, n_stats)
            week_raw: dict[tuple[str, str], np.ndarray] = {}
            week_paths = self._simulate_week(
                kalman_df=initial_kalman_df,
                week=week,
                injury_report=week_injury_report,
                n_simulations=self.n_simulations,
                rng=rng,
                team_score_paths=team_score_paths,
                player_active_prob=player_active_prob,
                playing_teams=playing_teams,
                raw_out=week_raw,
                role_paths=role_paths,
            )
            # week_paths: dict[(player_id, stat)] → np.ndarray(n_simulations,)

            # Real volume budget (Phase 7 fix): rescale passing/rushing/
            # receiving yards so each team's group total matches its own
            # per-path yards budget exactly — see _apply_volume_budget and
            # _VOLUME_BUDGET_STATS's module docstring for why this replaces
            # the additive team-script coupling shift for these three stats.
            # No-op when no schedule was supplied (team_yards_paths empty).
            if team_yards_paths:
                team_by_pid = dict(zip(
                    initial_kalman_df["player_id"].astype(str),
                    initial_kalman_df["team"] if "team" in initial_kalman_df.columns
                    else [None] * len(initial_kalman_df),
                ))
                self._apply_volume_budget(
                    week_paths, team_by_pid, team_yards_paths, raw_unmasked=week_raw
                )

            # Accumulate season totals
            for key, sims in week_paths.items():
                if key in accum:
                    accum[key] += sims

            # Week-by-week stats (collapsed across simulations)
            for player_id, stat in week_paths:
                sims = week_paths[(player_id, stat)]
                week_by_week_rows.append({
                    "player_id": player_id,
                    "week":      week,
                    "stat":      stat,
                    "mean":      float(sims.mean()),
                    "p10":       float(np.percentile(sims, 10)),
                    "p50":       float(np.percentile(sims, 50)),
                    "p90":       float(np.percentile(sims, 90)),
                })

            # Accumulate per-path team wins from the same score paths used
            # above for player-stat coupling, and update Elo from the model's
            # point predictions.
            if not resolved.empty:
                self._accumulate_week_wins(resolved, team_score_paths, team_win_accum, self.n_simulations)
                self._update_elo_from_week(resolved, week)

            # The Kalman state is deliberately NOT advanced from this week's
            # simulated output. It used to be: each week fed
            # np.median(week_paths) back in as if it were an observation
            # (gain 0.30), which is autoregressive feedback on the model's
            # own draws, not filtering — no new information arrives during a
            # forward simulation, so there is nothing to update on. Three
            # things went wrong at once:
            #
            #   1. week_paths has already been multiplied by the per-path
            #      Bernoulli(p_active) availability mask. For any player with
            #      p_active < 0.5 the median across paths is EXACTLY 0, so
            #      every week did new_est = 0.70 * prior_est. Over 18 weeks
            #      that is 0.70^18 ~= 0.0016 — total annihilation of the rate.
            #      72% of the production roster (581 of 808 players) sits at
            #      p_active <= 0.4, which is what collapsed fantasy_ppr (league
            #      weekly mean fell 3.74 -> 1.31 across the simulated season).
            #   2. week_paths has ALSO already been rescaled in place by
            #      _apply_volume_budget, so the team-total renormalization
            #      compounded week over week instead of applying once.
            #   3. Because the budget pins each team's yardage total, the volume
            #      vacated by (1) was handed to whoever still had signal —
            #      the rich-get-richer drift behind WR receiving_yards peaking
            #      at 83.8/wk in W1 and 158.9/wk in W18, and behind a team whose
            #      whole QB room had decayed to ~0 spraying its passing budget
            #      over 25 WR/TE noise draws.
            #
            # A rest-of-season projection is stationary by construction: every
            # simulated week is an i.i.d. draw from the same per-game
            # distribution, and horizon uncertainty emerges from summing those
            # independent weeks rather than from inflating the per-week
            # variance. _advance_kalman is retained for callers that have a
            # REAL observation to fold in; it must not be fed simulator output.

            logger.debug(
                "SeasonSimulator: completed W%d (%d player-stat pairs simulated).",
                week, len(week_paths),
            )

        # ── Assemble results ─────────────────────────────────────────────────
        player_season_totals: dict[str, dict[str, dict[str, float]]] = {}
        for (player_id, stat), sims in accum.items():
            if player_id not in player_season_totals:
                player_season_totals[player_id] = {}
            player_season_totals[player_id][stat] = {
                "mean": float(sims.mean()),
                "p10":  float(np.percentile(sims, 10)),
                "p50":  float(np.percentile(sims, 50)),
                "p90":  float(np.percentile(sims, 90)),
            }

        team_win_totals: dict[str, dict[str, float]] = {}
        for team, wins in team_win_accum.items():
            team_win_totals[team] = {
                "wins_mean": float(wins.mean()),
                "wins_p10":  float(np.percentile(wins, 10)),
                "wins_p90":  float(np.percentile(wins, 90)),
            }

        playoff_probs = self._playoff_probs_from_wins(team_win_accum, rng)

        result = SeasonSimulation(
            season=self.season,
            start_week=self.start_week,
            end_week=self.end_week,
            n_simulations=self.n_simulations,
            positions=self.positions,
            stats=self.stats,
            player_season_totals=player_season_totals,
            team_win_totals=team_win_totals,
            playoff_probs=playoff_probs,
            week_by_week=week_by_week_rows,
        )

        logger.info(
            "SeasonSimulator complete: %d weeks simulated, %d players, %d stats.",
            len(weeks), len(player_season_totals), len(self.stats),
        )
        return result

    def _run_cpp_fast(self, kalman_df: pd.DataFrame) -> SeasonSimulation:
        """
        Runs the full season projection natively in C++ via SIMD arrays.
        Bypasses slow Python week-by-week Elo and injury updates.
        """
        from engine.python_bindings import CppSeasonSimulator, PLAYER_DTYPE
        
        cpp_sim = CppSeasonSimulator(self.n_simulations)
        player_season_totals: dict[str, dict[str, dict[str, float]]] = {}
        
        n_players = len(kalman_df)
        pids = kalman_df["player_id"].astype(str).tolist()
        for p in pids:
            player_season_totals[p] = {}

        for stat in self.stats:
            # Build structured array for C++ interface
            p_arr = np.zeros(n_players, dtype=PLAYER_DTYPE)
            for i, (_, row) in enumerate(kalman_df.iterrows()):
                p_arr[i]["player_id"] = str(row["player_id"])[:31].encode()
                p_arr[i]["position"] = str(row.get("position", ""))[:3].encode()
                p_arr[i]["team"] = str(row.get("team", ""))[:3].encode()
                
                est = float(row.get(f"kalman_est_{stat}", 0.0))
                var = float(row.get(f"kalman_variance_{stat}", max(est * 0.5, 5.0) ** 2))
                
                p_arr[i]["kalman_mean"] = est
                p_arr[i]["kalman_variance"] = var
                p_arr[i]["injury_multiplier"] = 1.0 # Static injury assumption in Fast Mode
                p_arr[i]["snap_share"] = 1.0

            # Correlation matrix
            corr_mat = None
            if self.use_copula:
                try:
                    from ml.copula_layer import get_copula
                    copula = get_copula()
                    corr_mat = copula.build_correlation_matrix([(pid, stat) for pid in pids])
                except Exception as e:
                    logger.debug("Copula error building correlation matrix: %s. Using independent.", e)

            # C++ Native full-season Autoregressive loop
            res = cpp_sim.sim_full_season(p_arr, corr_mat, self.start_week, self.end_week)
            
            for i, p in enumerate(pids):
                player_season_totals[p][stat] = {
                    "mean": float(res["mean"][i]),
                    "p10":  float(res["p10"][i]),
                    "p50":  float(res["p50"][i]),
                    "p90":  float(res["p90"][i]),
                }

        return SeasonSimulation(
            season=self.season,
            start_week=self.start_week,
            end_week=self.end_week,
            n_simulations=self.n_simulations,
            positions=self.positions,
            stats=self.stats,
            player_season_totals=player_season_totals,
            team_win_totals={},
            week_by_week=[],
        )

    # ── Per-Week Simulation ──────────────────────────────────────────────────

    # Below this per-game rate a player has no role in the stat at all, so the
    # correct simulated value is exactly zero rather than a draw. Drawing
    # N(0, std) and then clipping at zero — which is what every draw site used
    # to do — is not a harmless no-op: the clipped half-normal has mean
    # 0.4*std, so a WR whose passing_yards prior is exactly 0.0 (see
    # ml.kalman_tracker.POSITION_PRIORS) still manufactured ~2 passing yards a
    # week out of pure noise. Multiply that by ~25 non-QBs per roster and
    # _apply_volume_budget, which splits the team's passing budget in
    # proportion to these raw draws, hands a real slice of every team's
    # passing yards to players who have never thrown a pass.
    _ZERO_RATE_EPS = 1e-6

    # Weekly coefficient of variation assumed for a player with no usable
    # history, used only when observation variance cannot be measured.
    _COLD_START_CV = 0.5

    # Season-level (role) dispersion: log-sd of a per-path, per-player
    # multiplier drawn ONCE and held across every simulated week.
    #
    # Without it the 18 weeks are i.i.d., so a season total's relative spread
    # shrinks by sqrt(n_games) and lands near p90/mean = 1.09 no matter how
    # volatile the player is. Real rest-of-season uncertainty does not shrink
    # that way, because the thing you are unsure about is not this Sunday's
    # bounce -- it is the player's ROLE for the year (target share, snap count,
    # scheme, committee split), which persists week to week. That is a
    # season-level random effect, and it is what the interval was missing.
    #
    # HAND-SET, not fitted: 0.13 log-sd puts a full-season starter's 80% band
    # near p90/mean ~ 1.20, matching how wide published rest-of-season ranges
    # actually are. It should be calibrated against realized season-total
    # coverage once a season of held-out outcomes exists -- the same caveat
    # that already applies to ProjectionService's hand-set residual scales.
    _SEASON_ROLE_LOG_SD = 0.13

    @classmethod
    def _draw_stat_samples(
        cls,
        row,
        stat: str,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        One player's raw (pre-availability-mask, pre-volume-budget) draws for
        `stat`, from their Kalman posterior. Returns exact zeros when the
        posterior mean is zero — see _ZERO_RATE_EPS.
        """
        est = float(row.get(f"kalman_est_{stat}") or 0.0)
        if est <= cls._ZERO_RATE_EPS:
            return np.zeros(size)
        # Predictive variance = uncertainty in the RATE (Kalman posterior)
        #                     + week-to-week spread of the stat itself.
        # Only the first term used to be applied, so a player with a long,
        # consistent history drew almost the same number every week.
        variance = float(row.get(f"kalman_variance_{stat}") or max(est * 0.5, 5.0) ** 2)
        obs_var = row.get(f"kalman_obs_var_{stat}")
        if obs_var is None or not np.isfinite(obs_var):
            # No usable history: fall back to a typical weekly coefficient of
            # variation for NFL box-score stats rather than asserting a
            # confident forecast for a player we know nothing about.
            obs_var = (est * cls._COLD_START_CV) ** 2
        variance += max(float(obs_var), 0.0)
        std = max(float(np.sqrt(variance)), 0.1)
        return np.maximum(rng.normal(est, std, size=size), 0.0)

    def _simulate_week(
        self,
        kalman_df: pd.DataFrame,
        week: int,
        injury_report: dict[str, str],
        n_simulations: int,
        rng: np.random.Generator,
        team_score_paths: Optional[dict[str, np.ndarray]] = None,
        player_active_prob: Optional[dict[str, float]] = None,
        playing_teams: Optional[set[str]] = None,
        raw_out: Optional[dict[tuple[str, str], np.ndarray]] = None,
        role_paths: Optional[dict[str, np.ndarray]] = None,
    ) -> dict[tuple[str, str], np.ndarray]:
        """
        Simulate one week's stats for all players across n_simulations paths.

        Uses:
          - Kalman estimates as the mean of a Normal distribution prior.
          - Volume Redistribution for injured player shares.
          - Copula for within-team/game correlated draws (if use_copula=True).
          - team_score_paths (if given): shifts each player's samples toward
            their OWN team's same-path score deviation, at a per-stat
            strength measured from real data (_TEAM_SCRIPT_COUPLING). This is
            what ties a player's simulated outlier game to their team's
            simulated result on that same path, instead of the two being
            independent draws — see _apply_team_script_coupling.
          - player_active_prob (if given): draws one Bernoulli(p_active)
            "played this week" outcome per player per path (shared across all
            stats for that player-path so they zero out together, not
            independently per stat), and zeroes that player's samples on
            paths where they didn't play. Drawn fresh here since this method
            runs once per week, so a path where a player sits out week 6 can
            still have them active in week 7.
          - raw_out (if given): populated with each player's PRE-availability
            draws. _apply_volume_budget needs these: a team's yardage budget
            is a property of the team playing the game, not of which
            individuals dressed, so when availability zeroes a whole stat
            group the budget must still be allocated by who would have played.
          - playing_teams (if given): teams with NO game this week (a bye)
            are zeroed deterministically on every path — this is a schedule
            fact, not a probability, unlike player_active_prob. None means
            "no schedule was supplied, can't determine byes" and no player is
            gated on this basis, which is the case for every caller that
            doesn't pass schedule_df to run().

        Returns:
            {(player_id, stat): np.ndarray(n_simulations,)} — per-sim weekly totals.
        """
        results: dict[tuple[str, str], np.ndarray] = {}
        team_score_paths = team_score_paths or {}
        player_active_prob = player_active_prob or {}

        team_by_pid = dict(zip(
            kalman_df["player_id"].astype(str),
            kalman_df["team"] if "team" in kalman_df.columns else [None] * len(kalman_df),
        ))

        active_masks: dict[str, np.ndarray] = {}
        for pid in kalman_df["player_id"].astype(str).unique():
            if playing_teams is not None and str(team_by_pid.get(pid)) not in playing_teams:
                active_masks[pid] = np.zeros(n_simulations)
                continue
            p = player_active_prob.get(pid)
            if p is None:
                continue
            p = min(max(float(p), 0.0), 1.0)
            active_masks[pid] = (rng.random(n_simulations) < p).astype(float)

        for stat in self.stats:
            # Volume-budget stats get their team-level total fixed by
            # _apply_volume_budget after this method returns — the additive
            # coupling shift only correlated a player's draw with team
            # script without ever making the group sum honest, so it's
            # skipped here rather than left to distort the raw shares
            # _apply_volume_budget allocates from.
            coupling = (
                0.0 if stat in _VOLUME_BUDGET_STATS
                else _TEAM_SCRIPT_COUPLING.get(stat, _DEFAULT_TEAM_SCRIPT_COUPLING)
            )

            # Apply injury redistribution
            kalman_with_injury = self._apply_injury_to_kalman(
                kalman_df=kalman_df,
                injury_report=injury_report,
                stat=stat,
            )

            if self.use_copula:
                # Group players by team for correlated within-team draws
                team_groups = (
                    kalman_with_injury.groupby("team")
                    if "team" in kalman_with_injury.columns
                    else [("ALL", kalman_with_injury)]
                )
                for team, team_df in team_groups:
                    team_results = self._simulate_team_week_copula(
                        team_df=team_df,
                        stat=stat,
                        n_simulations=n_simulations,
                        rng=rng,
                    )
                    if coupling and team in team_score_paths:
                        team_results = self._apply_team_script_coupling(
                            team_results, team_score_paths[team], coupling
                        )
                    for key, samples in team_results.items():
                        if role_paths is not None and key[0] in role_paths:
                            samples = samples * role_paths[key[0]]
                        if raw_out is not None:
                            raw_out[key] = samples.copy()
                        mask = active_masks.get(key[0])
                        if mask is not None:
                            samples = samples * mask
                        team_results[key] = samples
                    results.update(team_results)
            else:
                # Independent draws per player
                for _, row in kalman_with_injury.iterrows():
                    pid = str(row["player_id"])
                    samples = self._draw_stat_samples(row, stat, n_simulations, rng)
                    team = row.get("team")
                    if coupling and team in team_score_paths and samples.any():
                        samples = self._apply_team_script_coupling(
                            {(pid, stat): samples}, team_score_paths[team], coupling
                        )[(pid, stat)]
                    if role_paths is not None and pid in role_paths:
                        samples = samples * role_paths[pid]
                    if raw_out is not None:
                        raw_out[(pid, stat)] = samples.copy()
                    mask = active_masks.get(pid)
                    if mask is not None:
                        samples = samples * mask
                    results[(pid, stat)] = samples

        return results

    def _simulate_team_week_copula(
        self,
        team_df: pd.DataFrame,
        stat: str,
        n_simulations: int,
        rng: np.random.Generator,
    ) -> dict[tuple[str, str], np.ndarray]:
        """
        Simulate one team's weekly stat using the Gaussian copula.
        """
        try:
            from ml.copula_layer import get_copula
        except ImportError:
            logger.warning("copula_layer not available; falling back to independent draws.")
            return self._simulate_independent(team_df, stat, n_simulations, rng)

        player_ids = team_df["player_id"].astype(str).tolist()
        if not player_ids:
            return {}

        # Build marginal distributions (normal approximation from Kalman)
        marginal_samples: dict[str, np.ndarray] = {}
        for _, row in team_df.iterrows():
            pid = str(row["player_id"])
            # Reference marginal (more draws here for quantile accuracy).
            # All-zero for a player with no role in this stat, which
            # _quantile_transform then maps back to zeros.
            marginal_samples[pid] = self._draw_stat_samples(
                row, stat, _WEEK_SAMPLES * 5, rng
            )

        copula = get_copula()
        # CopulaLayer expects nodes as (player_id, stat) tuples and
        # marginal_samples keyed by the same tuples.
        nodes = [(pid, stat) for pid in player_ids]
        keyed_marginals = {(pid, stat): marginal_samples[pid] for pid in player_ids}
        joint_draws = copula.draw_joint_samples(
            nodes=nodes,
            marginal_samples=keyed_marginals,
            n_samples=n_simulations,
            rng=rng,
        )
        # joint_draws shape: (n_simulations, n_players)
        joint_draws = np.maximum(joint_draws, 0.0)

        return {
            (pid, stat): joint_draws[:, col]
            for col, pid in enumerate(player_ids)
        }

    def _simulate_independent(
        self,
        team_df: pd.DataFrame,
        stat: str,
        n_simulations: int,
        rng: np.random.Generator,
    ) -> dict[tuple[str, str], np.ndarray]:
        """Fallback: independent normal draws per player."""
        results = {}
        for _, row in team_df.iterrows():
            pid = str(row["player_id"])
            results[(pid, stat)] = self._draw_stat_samples(row, stat, n_simulations, rng)
        return results

    @staticmethod
    def _apply_team_script_coupling(
        team_results: dict[tuple[str, str], np.ndarray],
        team_score_path: np.ndarray,
        coupling: float,
    ) -> dict[tuple[str, str], np.ndarray]:
        """
        Shift each player's per-path samples toward their team's SAME-path
        score deviation, at strength `coupling` (a within-player Pearson r
        measured from real game_logs — see _TEAM_SCRIPT_COUPLING).

        adjusted = samples + coupling * player_std * z(team_score_path)

        This is an additive approximation, not an exact correlation-preserving
        decomposition: it shifts the mean without rescaling variance, so the
        realized correlation is slightly below `coupling` and player variance
        inflates by a factor of ~sqrt(1 + coupling^2) (a few percent at the
        coupling magnitudes measured here, 0.10-0.22). Good enough to make a
        low-scoring loss path and an outlier stat path for that team
        correlate in the right direction; not a precision instrument.
        """
        std = float(np.std(team_score_path))
        if std < 1e-6:
            return team_results
        z = (team_score_path - float(np.mean(team_score_path))) / std
        out = {}
        for key, samples in team_results.items():
            player_std = max(float(np.std(samples)), 0.1)
            out[key] = np.maximum(samples + coupling * player_std * z, 0.0)
        return out

    # ── Kalman Integration ───────────────────────────────────────────────────

    def _compute_initial_kalman(
        self,
        players_df: pd.DataFrame,
        prior_game_rows: dict[str, list[dict]],
    ) -> pd.DataFrame:
        """
        Compute Kalman estimates from prior game history as the simulation starting point.
        """
        try:
            from ml.kalman_tracker import KalmanFeatureEngineer
            from ml.kalman_tracker import STAT_SOURCE_COL

            kfe = KalmanFeatureEngineer()
            augmented_rows = []
            for _, row in players_df.iterrows():
                pid = str(row["player_id"])
                pos = str(row.get("position", ""))
                prior = prior_game_rows.get(pid, [])
                kalman_feats = kfe.compute_kalman_form(prior, position=pos or None)
                # Game-to-game (observation) variance, per stat, from the same
                # prior rows. compute_kalman_form returns only the posterior
                # variance of the RATE, which shrinks as evidence accumulates —
                # a consistent starter converges toward ~0 spread. Drawing a
                # week from that alone is the wrong distribution: it answers
                # "how sure are we of his average?" when the question is "what
                # might he do on Sunday?". That is why 80% season intervals
                # came out at p90/mean ~= 1.08 when a real rest-of-season band
                # is nearer 1.20. Attached here rather than in
                # compute_kalman_form so the shared feature contract that
                # feature_engineer.py and the served models depend on is
                # untouched. See _draw_stat_samples for how the two combine.
                for stat in self.stats:
                    col = STAT_SOURCE_COL.get(stat, stat)
                    vals = [
                        float(r[col]) for r in prior
                        if isinstance(r, dict) and r.get(col) is not None
                    ]
                    kalman_feats[f"kalman_obs_var_{stat}"] = (
                        float(np.var(vals, ddof=1)) if len(vals) >= 2 else None
                    )
                augmented_rows.append({**row.to_dict(), **kalman_feats})
            return pd.DataFrame(augmented_rows)
        except Exception as exc:
            logger.warning("_compute_initial_kalman failed (%s); using players_df as-is.", exc)
            return players_df.copy()

    def _advance_kalman(
        self,
        kalman_df: pd.DataFrame,
        week_means: dict[tuple[str, str], float],
    ) -> pd.DataFrame:
        """
        Update Kalman estimates after observing a simulated week.

        For each player and each stat, bump their Kalman estimate toward the
        simulated mean using a simple exponential decay (Kalman gain ≈ 0.3).

        In production, the full KalmanFeatureEngineer.update() would be called.
        This approximation is fast enough for 500+ season simulations.

        IMPORTANT: `week_means` must come from REAL observations. run() no
        longer calls this, because feeding the simulator's own draws back in
        as observations is autoregressive feedback, not filtering — see the
        block comment at run()'s former call site for the three failure modes
        it produced in production.
        """
        KALMAN_GAIN = 0.30   # How much we trust the new observation vs. prior

        df = kalman_df.copy()
        for stat in self.stats:
            col_est = f"kalman_est_{stat}"
            col_var = f"kalman_variance_{stat}"
            if col_est not in df.columns:
                continue

            for idx, row in df.iterrows():
                pid = str(row["player_id"])
                obs = week_means.get((pid, stat))
                if obs is None:
                    continue
                prior_est = float(row.get(col_est) or 0.0)
                prior_var = float(row.get(col_var) or 100.0)

                # Simple Kalman-style update
                new_est = prior_est + KALMAN_GAIN * (obs - prior_est)
                new_var = (1 - KALMAN_GAIN) * prior_var

                df.at[idx, col_est] = new_est
                df.at[idx, col_var] = new_var

        return df

    # ── Injury Redistribution for Simulator ────────────────────────────────

    def _apply_injury_to_kalman(
        self,
        kalman_df: pd.DataFrame,
        injury_report: dict[str, str],
        stat: str,
    ) -> pd.DataFrame:
        """
        Apply injury-adjusted target shares to kalman_df for this week's simulation.
        Uses VolumeRedistributor if available; otherwise zeroes out injured players.
        """
        if not injury_report:
            return kalman_df

        try:
            from ml.volume_redistribution import _injury_multiplier
            
            df = kalman_df.copy()
            df["_injury_status"] = df["player_id"].map(injury_report)

            est_col = f"kalman_est_{stat}"
            if est_col not in df.columns:
                return kalman_df

            for idx, row in df.iterrows():
                status = row["_injury_status"]
                if pd.isna(status) or not status:
                    continue
                    
                # Base multiplier from current status string
                base_mult = _injury_multiplier(status)
                
                # If they are out/doubtful/questionable, estimate probability 
                # they return this simulated week (placeholder for future weeks).
                # For Week 1 of simulation, it's just the base multiplier.
                # If we simulate future weeks, the injury_report should be dynamic.
                # Assuming injury_report is static to the start_week, we discount:
                # In a full integration, `week` would be compared to `start_week`.
                df.at[idx, est_col] = float(row[est_col]) * base_mult

            df = df.drop(columns=["_injury_status"], errors="ignore")
            return df

        except Exception as exc:
            logger.debug("_apply_injury_to_kalman: error (%s); returning unchanged.", exc)
            return kalman_df

    # ── Real game resolution (Phase 4 team-game model) ──────────────────────

    def _db_url(self) -> str:
        if self._database_url:
            return self._database_url
        from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
        return DEFAULT_HOST_DATABASE_URL

    def _ensure_points_model(self) -> None:
        """
        Lazily fit the Phase 4 points Ridge model on every season strictly
        before self.season — same training window and alpha as
        ml.team_game_model.train() and scripts/materialize_team_game_predictions.py.
        Fails loudly: a game-resolution engine that silently falls back to a
        proxy on a training-data gap defeats the reason this exists.
        """
        if self._points_model is not None:
            return
        from sklearn.linear_model import Ridge

        from ml.team_game_model import FEATURE_COLS, _prepare_x, compute_oof_residual_std
        from pipeline.team_game_features import build_team_game_frame

        train_df = build_team_game_frame(self._db_url(), list(range(2019, self.season)))
        points_df = train_df[train_df["points"].notna()] if not train_df.empty else train_df
        if points_df.empty:
            raise RuntimeError(
                f"SeasonSimulator: no team-game training rows with a points target for "
                f"seasons < {self.season}; cannot resolve real game outcomes."
            )
        fill_values = points_df[FEATURE_COLS].median(numeric_only=True)
        X = _prepare_x(points_df, fill_values)
        y = points_df["points"].astype(float).values
        self._points_model = Ridge(alpha=_TEAM_GAME_MODEL_ALPHA).fit(X, y)
        self._points_fill = fill_values
        self._points_residual_std = compute_oof_residual_std(
            points_df, _TEAM_GAME_MODEL_ALPHA, target_col="points"
        )

    def _ensure_yards_model(self) -> None:
        """
        Lazily fit the Phase 4 yards Ridge model — same training window,
        alpha, and feature set as _ensure_points_model, just target_col=
        "yards". This is the real per-path team volume budget behind
        _apply_volume_budget (Phase 7 fix: replaces the additive
        team-script coupling shift, which could correlate a player's draw
        with their team's score but never move the team-level total).
        """
        if self._yards_model is not None:
            return
        from sklearn.linear_model import Ridge

        from ml.team_game_model import FEATURE_COLS, _prepare_x, compute_oof_residual_std
        from pipeline.team_game_features import build_team_game_frame

        train_df = build_team_game_frame(self._db_url(), list(range(2019, self.season)))
        yards_df = train_df[train_df["total_yards"].notna()] if not train_df.empty else train_df
        if yards_df.empty:
            raise RuntimeError(
                f"SeasonSimulator: no team-game training rows with a yards target for "
                f"seasons < {self.season}; cannot allocate a real volume budget."
            )
        fill_values = yards_df[FEATURE_COLS].median(numeric_only=True)
        X = _prepare_x(yards_df, fill_values)
        y = yards_df["total_yards"].astype(float).values
        self._yards_model = Ridge(alpha=_TEAM_GAME_MODEL_ALPHA).fit(X, y)
        self._yards_fill = fill_values
        self._yards_residual_std = compute_oof_residual_std(
            yards_df, _TEAM_GAME_MODEL_ALPHA, target_col="total_yards"
        )

    def _ensure_sim_elo(self) -> None:
        """
        Elo carried forward across simulated weeks, seeded from every REAL
        completed game through the start of this simulation (load_elo_from_db
        only fits rows with a non-null home_score, so future/unplayed games
        are naturally excluded). Must be loaded with a db_url — the module
        singleton get_elo_system() with no db_url returns an unfit, flat-1500
        system, which would silently zero out team-quality signal.
        """
        if self._sim_elo is not None:
            return
        from ml.team_elo import load_elo_from_db
        self._sim_elo = load_elo_from_db(self._db_url(), seasons=list(range(2019, self.season + 1)))

    def _resolve_week_games(self, week: int) -> pd.DataFrame:
        """
        Predict each team's expected points AND yards for `week` via the
        real Phase 4 models, using this simulation's own carried-forward
        Elo (not the DB's real Elo, which has no rows for weeks that
        haven't been played) for the team_off_elo/team_def_elo/opp_off_elo/
        opp_def_elo features. Returns one row per (game_id, team) with
        points_mean and yards_mean columns, or an empty frame if nothing is
        scheduled that week.
        """
        from ml.team_game_model import _prepare_x
        from pipeline.team_game_features import build_team_game_forward_frame

        self._ensure_points_model()
        self._ensure_yards_model()
        self._ensure_sim_elo()
        if week not in self._forward_frame_cache:
            self._forward_frame_cache[week] = build_team_game_forward_frame(
                self._db_url(), self.season, week
            )
        static_frame = self._forward_frame_cache[week]
        if static_frame.empty:
            return static_frame
        frame = static_frame.copy()
        teams = pd.concat([frame["team"], frame["opponent"]]).unique()
        off_elo = {}
        def_elo = {}
        for t in teams:
            off_elo[t], def_elo[t] = self._sim_elo.get_current_ratings(t)
        frame["team_off_elo"] = frame["team"].map(off_elo)
        frame["team_def_elo"] = frame["team"].map(def_elo)
        frame["opp_off_elo"] = frame["opponent"].map(off_elo)
        frame["opp_def_elo"] = frame["opponent"].map(def_elo)
        X_points = _prepare_x(frame, self._points_fill)
        frame["points_mean"] = self._points_model.predict(X_points)
        X_yards = _prepare_x(frame, self._yards_fill)
        frame["yards_mean"] = self._yards_model.predict(X_yards)
        return frame

    def _draw_team_score_paths(
        self,
        resolved: pd.DataFrame,
        n_simulations: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """
        One score path per team per simulation, drawn from the Phase 4
        points model's prediction ± its out-of-fold residual std
        (_ensure_points_model). Drawn ONCE per week and shared by
        _simulate_week (player-stat coupling), _accumulate_week_wins, so a
        team's "path 37" score is the same number everywhere it's used —
        that sharing is what makes a player's outlier game and their team's
        result on the same path arithmetically consistent instead of two
        independent coin flips that happen to both appear in the output.
        """
        paths: dict[str, np.ndarray] = {}
        for _, row in resolved.iterrows():
            paths[str(row["team"])] = rng.normal(
                float(row["points_mean"]), self._points_residual_std, n_simulations
            )
        return paths

    def _draw_team_yards_paths(
        self,
        resolved: pd.DataFrame,
        n_simulations: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """
        One total-offensive-yards path per team per simulation, drawn from
        the Phase 4 yards model's prediction ± its out-of-fold residual std
        — the real per-path volume budget _apply_volume_budget allocates
        among that team's players (Phase 7 fix). Floored at 0: the Gaussian
        residual can occasionally go negative for a low-yardage prediction,
        and a negative budget has no meaning to allocate.
        """
        paths: dict[str, np.ndarray] = {}
        for _, row in resolved.iterrows():
            paths[str(row["team"])] = np.maximum(
                rng.normal(float(row["yards_mean"]), self._yards_residual_std, n_simulations),
                0.0,
            )
        return paths

    def _apply_volume_budget(
        self,
        results: dict[tuple[str, str], np.ndarray],
        team_by_pid: dict[str, str],
        team_yards_paths: dict[str, np.ndarray],
        raw_unmasked: Optional[dict[tuple[str, str], np.ndarray]] = None,
    ) -> None:
        """
        Rescale passing_yards/rushing_yards/receiving_yards in place so
        each team's group total matches its own per-path yards budget
        (see _draw_team_yards_paths) exactly, split passing-vs-rushing by
        _PASS_YARDS_SHARE — replaces the additive team-script coupling
        shift for these three stats (see _VOLUME_BUDGET_STATS's docstring
        for why: a zero-mean shift cannot move a team-level total).

        Each player's share within their stat's group is their own RAW
        simulated draw's share of the group's raw total on that path — so
        players who project for more volume still get more of the budget,
        the budget just makes the group sum honest. A path where the whole
        group's raw draws floor at ~0 (extremely unlikely group, small
        n_simulations) is left unscaled rather than dividing by ~0.
        """
        by_team: dict[str, list[str]] = {}
        for pid, team in team_by_pid.items():
            by_team.setdefault(str(team), []).append(pid)

        for team, budget_path in team_yards_paths.items():
            pids = by_team.get(team)
            if not pids:
                continue
            pass_budget = budget_path * _PASS_YARDS_SHARE
            rush_budget = budget_path * (1.0 - _PASS_YARDS_SHARE)
            # passing_yards is allocated FIRST and receiving_yards is then
            # pinned to whatever the passing group actually realized, rather
            # than to the raw budget. Every passing yard is by definition a
            # receiving yard, so the two group totals must agree per path.
            # Handing both groups the same *budget* only preserves that
            # identity while both groups can absorb it: the zero-rate gate and
            # _MAX_BUDGET_SCALE mean a team whose passers are unavailable
            # legitimately falls short on passing, while its much larger
            # receiver group still soaks up the full budget. That split the
            # league's season totals by ~11% (110.8k passing vs 122.8k
            # receiving) — invisible per-player, but every team's passing and
            # receiving surfaces disagreed.
            realized_pass: Optional[np.ndarray] = None
            for stat, budget in (
                ("passing_yards", pass_budget),
                ("receiving_yards", pass_budget),
                ("rushing_yards", rush_budget),
            ):
                if stat == "receiving_yards" and realized_pass is not None:
                    budget = realized_pass
                group_pids = [pid for pid in pids if (pid, stat) in results]
                if not group_pids:
                    continue
                raw = np.vstack([np.maximum(results[(pid, stat)], 0.0) for pid in group_pids])
                group_sum = raw.sum(axis=0)
                # On paths where availability zeroed EVERY player who owns this
                # stat, fall back to the pre-availability draws for the split.
                # A team's yardage budget belongs to the team playing the game;
                # availability decides WHO produces it, not WHETHER it happens.
                # Because each player's Bernoulli(p_active) is drawn
                # independently, 10.1% of simulated team-weeks league-wide had
                # NO quarterback available at all (MIA: 41%), and that team's
                # entire passing budget silently evaporated on those paths.
                # That is a pure downward bias on every passing and receiving
                # total -- it is why no QB reached 4,000 passing yards and no
                # receiver reached 1,400, both of which happen every real NFL
                # season. The fallback keeps the group sum honest AND lets an
                # available backup absorb the starter's share, which is what
                # actually happens when a starter sits.
                if raw_unmasked is not None:
                    dead = group_sum <= 1e-6
                    if dead.any():
                        fallback = np.vstack([
                            np.maximum(raw_unmasked.get((pid, stat), results[(pid, stat)]), 0.0)
                            for pid in group_pids
                        ])
                        raw = np.where(dead, fallback, raw)
                        group_sum = raw.sum(axis=0)
                has_signal = group_sum > 1e-6
                scale = np.where(has_signal, budget / np.where(has_signal, group_sum, 1.0), 0.0)
                # Defence in depth against the failure this budget can amplify:
                # when the players who actually own a stat are all inactive on
                # a path, the group's surviving raw signal is whatever marginal
                # contributor is left, and an uncapped budget/group_sum would
                # multiply that sliver up to a full team's yardage. Falling
                # short of the budget on such a path is the honest outcome —
                # if nobody who throws is available, the team does not throw
                # for 230 yards — so the scale is capped rather than the
                # group-sum identity being preserved at any cost.
                scale = np.minimum(scale, _MAX_BUDGET_SCALE)
                for i, pid in enumerate(group_pids):
                    results[(pid, stat)] = raw[i] * scale
                if stat == "passing_yards":
                    realized_pass = group_sum * scale

    # ── Team wins / playoffs ─────────────────────────────────────────────────

    def _accumulate_week_wins(
        self,
        resolved: pd.DataFrame,
        team_score_paths: dict[str, np.ndarray],
        team_win_accum: dict[str, np.ndarray],
        n_simulations: int,
    ) -> None:
        """
        Increment each path's win count from team_score_paths (see
        _draw_team_score_paths) — the SAME per-path scores used to couple
        player stats to their team's outcome. Ties award 0.5 wins each.
        """
        for game_id, game_rows in resolved.groupby("game_id"):
            home_row = game_rows[game_rows["is_home"] == 1]
            away_row = game_rows[game_rows["is_home"] == 0]
            if home_row.empty or away_row.empty:
                continue
            home_team = str(home_row.iloc[0]["team"])
            away_team = str(away_row.iloc[0]["team"])
            home_paths = team_score_paths.get(home_team)
            away_paths = team_score_paths.get(away_team)
            if home_paths is None or away_paths is None:
                continue
            team_win_accum.setdefault(home_team, np.zeros(n_simulations))
            team_win_accum.setdefault(away_team, np.zeros(n_simulations))

            home_win = home_paths > away_paths
            away_win = away_paths > home_paths
            tie = ~(home_win | away_win)
            team_win_accum[home_team] += home_win.astype(float) + 0.5 * tie.astype(float)
            team_win_accum[away_team] += away_win.astype(float) + 0.5 * tie.astype(float)

    @staticmethod
    def _playoff_probs_from_wins(
        team_win_accum: dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """Fraction of paths where each team finishes top-7 in its conference."""
        if not team_win_accum:
            return {}
        teams = list(team_win_accum.keys())
        n_sims = len(next(iter(team_win_accum.values())))
        wins_mat = np.column_stack([team_win_accum[t] for t in teams])  # (n_sims, n_teams)

        made = {t: 0 for t in teams}
        for s in range(n_sims):
            row = wins_mat[s]
            # Tiny jitter breaks exact ties deterministically per path
            jitter = rng.uniform(0, 1e-6, size=len(teams))
            scored = row + jitter
            for conf in (_AFC, _NFC):
                idxs = [i for i, t in enumerate(teams) if t in conf]
                if not idxs:
                    continue
                conf_scores = [(scored[i], teams[i]) for i in idxs]
                conf_scores.sort(reverse=True)
                for _, t in conf_scores[:_PLAYOFF_BERTHS]:
                    made[t] += 1

        return {t: made[t] / n_sims for t in teams}

    # ── Elo Update ───────────────────────────────────────────────────────────

    def _update_elo_from_week(self, resolved: pd.DataFrame, week: int) -> None:
        """
        Update this simulation's carried-forward Elo (_ensure_sim_elo) from
        the SAME Phase 4 points-model predictions used to resolve wins in
        _accumulate_week_wins, so Elo movement and win outcomes agree with
        each other rather than being derived from two different notions of
        "how good is this team."
        """
        from ml.team_elo import GameResult

        results = []
        for game_id, game_rows in resolved.groupby("game_id"):
            home_row = game_rows[game_rows["is_home"] == 1]
            away_row = game_rows[game_rows["is_home"] == 0]
            if home_row.empty or away_row.empty:
                continue
            results.append(GameResult(
                season=self.season,
                week=week,
                home_team=str(home_row.iloc[0]["team"]),
                away_team=str(away_row.iloc[0]["team"]),
                home_score=int(round(float(home_row.iloc[0]["points_mean"]))),
                away_score=int(round(float(away_row.iloc[0]["points_mean"]))),
                game_id=str(game_id),
            ))
        if results:
            self._sim_elo.update_week(results)


# ── Convenience Functions ─────────────────────────────────────────────────────

def run_rest_of_season(
    season: int,
    current_week: int,
    players_df: pd.DataFrame,
    prior_game_rows: dict[str, list[dict]],
    schedule_df: Optional[pd.DataFrame] = None,
    injury_projections: Optional[dict[int, dict[str, str]]] = None,
    n_simulations: int = 500,
    stats: Optional[list[str]] = None,
    rng_seed: Optional[int] = None,
    database_url: Optional[str] = None,
) -> SeasonSimulation:
    """
    Convenience function: simulate the rest of the season starting from current_week + 1.

    Args:
        season:             NFL season.
        current_week:       The last completed week. Simulation starts from current_week + 1.
        players_df:         Active roster DataFrame.
        prior_game_rows:    Kalman history through current_week.
        schedule_df:        Optional team schedule [week, home_team, away_team]. Non-empty
                             rows for a week gate real Phase 4 game resolution for that
                             week (see SeasonSimulator.run's docstring).
        injury_projections: Optional {week: {player_id: status}} for projected IR.
        n_simulations:      Monte Carlo season paths.
        stats:              Stats to simulate. Default: 4 core stats.
        rng_seed:           Optional seed for reproducibility.
        database_url:       DB URL for the Phase 4 points model + Elo; defaults to
                             pipeline.db_defaults.DEFAULT_HOST_DATABASE_URL.

    Returns:
        SeasonSimulation with rest-of-season totals and distributions.

    Example:
        results = run_rest_of_season(
            season=2025, current_week=8,
            players_df=nfl_rosters_df,
            prior_game_rows=feature_matrix_dict,
            n_simulations=500,
        )
        print(results.top_players("receiving_yards", n=5))
    """
    sim = SeasonSimulator(
        season=season,
        start_week=current_week + 1,
        end_week=18,
        n_simulations=n_simulations,
        stats=stats or _DEFAULT_STATS,
        use_copula=True,
        use_cpp=False,
        database_url=database_url,
    )
    return sim.run(
        players_df=players_df,
        prior_game_rows=prior_game_rows,
        schedule_df=schedule_df,
        injury_projections=injury_projections,
        rng_seed=rng_seed,
    )


def load_real_schedule(
    database_url: str, season: int, weeks: list[int]
) -> pd.DataFrame:
    """
    Build a [week, home_team, away_team] schedule_df for run()'s gating
    parameter, straight from the `games` table via the same
    build_team_game_forward_frame reader SeasonSimulator itself uses for
    per-week game resolution — one schedule source of truth, so the teams
    named here always match the teams _resolve_week_games predicts for
    (no separate query that could drift to a different abbreviation
    convention, e.g. "LA" vs "LAR").
    """
    from pipeline.team_game_features import build_team_game_forward_frame

    rows = []
    for week in weeks:
        frame = build_team_game_forward_frame(database_url, season, week)
        if frame.empty:
            continue
        home = frame[frame["is_home"] == 1][["game_id", "team", "opponent"]]
        for _, r in home.iterrows():
            rows.append({"week": week, "home_team": r["team"], "away_team": r["opponent"]})
    return pd.DataFrame(rows, columns=["week", "home_team", "away_team"])
