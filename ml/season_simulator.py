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
  1. Pull current Kalman estimates for all rostered players.
  2. Apply Volume Redistribution (injury report for that projected week).
  3. Draw joint samples via Copula Layer (correlated within-team/game).
  4. Run Monte Carlo projection to produce week simulation results.
  5. Feed simulated game stats BACK into the Kalman filter as if observed.
  6. Update Team Elo ratings from simulated game outcomes.
  7. Advance to next week.

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
# Fantasy→NFL score scale used when scoring simulated games from player PPR.
_FANTASY_TO_NFL_SCALE = 4.0


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
        self._sim_elo       = None  # deep-copied Elo for simulation isolation

    # ── Main Entry Point ─────────────────────────────────────────────────────

    def run(
        self,
        players_df: pd.DataFrame,
        prior_game_rows: dict[str, list[dict]],
        schedule_df: Optional[pd.DataFrame] = None,
        injury_projections: Optional[dict[int, dict[str, str]]] = None,
        rng_seed: Optional[int] = None,
    ) -> SeasonSimulation:
        """
        Simulate the rest of the season across multiple Monte Carlo paths.

        Args:
            players_df:          DataFrame with [player_id, position, team, ...].
                                 All rostered players eligible for projection.
            prior_game_rows:     {player_id: [dict of game rows]} — Kalman history
                                 as of start_week-1. From normalize.py / DB load.
            schedule_df:         Optional DataFrame [week, home_team, away_team]
                                 for team matchup context. If None, matchup context
                                 is omitted (projections still work).
            injury_projections:  Optional {week: {player_id: injury_status}}.
                                 Projected injuries per week (e.g. from ESPN IR list
                                 or user-defined scenario).
            rng_seed:            Optional seed for reproducibility.

        Returns:
            SeasonSimulation with season totals, team wins, and week-by-week data.
        """
        rng = np.random.default_rng(rng_seed)
        injury_projections = injury_projections or {}

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

        # ── Main simulation loop: iterate per WEEK ──────────────────────────
        for week in weeks:
            week_injury_report = injury_projections.get(week, {})
            week_schedule = (
                schedule_df[schedule_df["week"] == week]
                if schedule_df is not None and "week" in schedule_df.columns
                else pd.DataFrame()
            )

            # Draw weekly samples: shape (n_simulations, n_players, n_stats)
            week_paths = self._simulate_week(
                kalman_df=initial_kalman_df,
                week=week,
                injury_report=week_injury_report,
                n_simulations=self.n_simulations,
                rng=rng,
            )
            # week_paths: dict[(player_id, stat)] → np.ndarray(n_simulations,)

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

            # Update simulated Elo + accumulate per-path team wins
            if not week_schedule.empty:
                # Build team map for this week
                team_map = dict(zip(initial_kalman_df["player_id"].astype(str), initial_kalman_df["team"]))
                self._update_elo_from_week(week_schedule, week_paths, week, team_map)
                self._accumulate_week_wins(
                    week_schedule, week_paths, team_map, team_win_accum, self.n_simulations
                )

            # Update Kalman priors with simulated week medians (for next week's prior).
            # Using median instead of mean reduces sensitivity to outlier simulations.
            # Note: ideally each simulation path should maintain its own Kalman state,
            # but that requires O(n_sims * n_players * n_stats) memory. Using shared
            # medians is a practical approximation that preserves the central tendency
            # while acknowledging that inter-path Kalman variance is approximated.
            initial_kalman_df = self._advance_kalman(
                kalman_df=initial_kalman_df,
                week_means={(k[0], k[1]): float(np.median(v)) for k, v in week_paths.items()},
            )

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

    def _simulate_week(
        self,
        kalman_df: pd.DataFrame,
        week: int,
        injury_report: dict[str, str],
        n_simulations: int,
        rng: np.random.Generator,
    ) -> dict[tuple[str, str], np.ndarray]:
        """
        Simulate one week's stats for all players across n_simulations paths.

        Uses:
          - Kalman estimates as the mean of a Normal distribution prior.
          - Volume Redistribution for injured player shares.
          - Copula for within-team/game correlated draws (if use_copula=True).

        Returns:
            {(player_id, stat): np.ndarray(n_simulations,)} — per-sim weekly totals.
        """
        results: dict[tuple[str, str], np.ndarray] = {}

        for stat in self.stats:
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
                    results.update(team_results)
            else:
                # Independent draws per player
                for _, row in kalman_with_injury.iterrows():
                    pid = str(row["player_id"])
                    est      = float(row.get(f"kalman_est_{stat}") or 0.0)
                    variance = float(row.get(f"kalman_variance_{stat}") or max(est * 0.5, 5.0) ** 2)
                    std      = max(np.sqrt(variance), 0.1)

                    # Sample from truncated normal (non-negative stats)
                    samples  = rng.normal(est, std, size=n_simulations)
                    samples  = np.maximum(samples, 0.0)
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
            pid      = str(row["player_id"])
            est      = float(row.get(f"kalman_est_{stat}") or 0.0)
            variance = float(row.get(f"kalman_variance_{stat}") or max(est * 0.5, 5.0) ** 2)
            std      = max(np.sqrt(variance), 0.1)
            # Generate reference marginal (we use more draws here for quantile accuracy)
            marginal_samples[pid] = np.maximum(
                rng.normal(est, std, size=_WEEK_SAMPLES * 5), 0.0
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
            pid      = str(row["player_id"])
            est      = float(row.get(f"kalman_est_{stat}") or 0.0)
            variance = float(row.get(f"kalman_variance_{stat}") or max(est * 0.5, 5.0) ** 2)
            std      = max(np.sqrt(variance), 0.1)
            results[(pid, stat)] = np.maximum(rng.normal(est, std, n_simulations), 0.0)
        return results

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
            kfe = KalmanFeatureEngineer()
            augmented_rows = []
            for _, row in players_df.iterrows():
                pid = str(row["player_id"])
                pos = str(row.get("position", ""))
                prior = prior_game_rows.get(pid, [])
                kalman_feats = kfe.compute_kalman_form(prior, position=pos or None)
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

    # ── Team wins / playoffs ─────────────────────────────────────────────────

    @staticmethod
    def _team_fantasy_points(
        week_paths: dict[tuple[str, str], np.ndarray],
        team_map: dict[str, str],
        team: str,
        n_simulations: int,
    ) -> np.ndarray:
        """Sum simulated fantasy_ppr for a team across simulation paths."""
        pts = np.zeros(n_simulations, dtype=float)
        for pid, tm in team_map.items():
            if tm != team:
                continue
            arr = week_paths.get((pid, "fantasy_ppr"))
            if arr is None:
                continue
            pts = pts + np.asarray(arr, dtype=float)
        return pts / _FANTASY_TO_NFL_SCALE

    def _accumulate_week_wins(
        self,
        week_schedule: pd.DataFrame,
        week_paths: dict[tuple[str, str], np.ndarray],
        team_map: dict[str, str],
        team_win_accum: dict[str, np.ndarray],
        n_simulations: int,
    ) -> None:
        """
        Increment per-path win counts from simulated week outcomes.

        Uses fantasy_ppr team totals (scaled) as the game score proxy so each
        Monte Carlo path has an independent win record. Ties award 0.5 wins.
        """
        for _, game_row in week_schedule.iterrows():
            home_team = str(game_row.get("home_team", "") or "")
            away_team = str(game_row.get("away_team", "") or "")
            if not home_team or not away_team:
                continue
            if home_team not in team_win_accum or away_team not in team_win_accum:
                # Ensure keys exist even if schedule had unexpected teams
                team_win_accum.setdefault(home_team, np.zeros(n_simulations))
                team_win_accum.setdefault(away_team, np.zeros(n_simulations))

            home_pts = self._team_fantasy_points(
                week_paths, team_map, home_team, n_simulations
            )
            away_pts = self._team_fantasy_points(
                week_paths, team_map, away_team, n_simulations
            )
            # If neither side has fantasy_ppr samples, skip (avoid all-tie inflation)
            if float(home_pts.sum()) == 0.0 and float(away_pts.sum()) == 0.0:
                continue

            home_win = home_pts > away_pts
            away_win = away_pts > home_pts
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

    def _update_elo_from_week(
        self,
        week_schedule: pd.DataFrame,
        week_paths: dict[tuple[str, str], np.ndarray],
        week: int,
        team_map: dict[str, str],
    ) -> None:
        """
        Update Team Elo ratings from simulated game scores.
        Uses median fantasy_ppr totals mapped by team as a proxy for team quality movement.
        """
        try:
            from ml.team_elo import get_elo_system, GameResult
            if self._sim_elo is None:
                import copy
                self._sim_elo = copy.deepcopy(get_elo_system())
            elo = self._sim_elo
        except Exception:
            return

        for _, game_row in week_schedule.iterrows():
            try:
                home_team = str(game_row.get("home_team", ""))
                away_team = str(game_row.get("away_team", ""))
                if not home_team or not away_team:
                    continue

                # Use median simulated fantasy_ppr as proxy for team points
                # Scale rough fantasy points (~80-120) down to NFL points (~15-30) by dividing by 4
                home_pts = sum(
                    np.median(week_paths.get((pid, "fantasy_ppr"), np.zeros(1)))
                    for pid, tm in team_map.items() if tm == home_team
                ) / 4.0

                away_pts = sum(
                    np.median(week_paths.get((pid, "fantasy_ppr"), np.zeros(1)))
                    for pid, tm in team_map.items() if tm == away_team
                ) / 4.0

                # Fallback if both 0
                if home_pts == 0 and away_pts == 0:
                    home_pts, away_pts = 21, 21

                result = GameResult(
                    season=self.season,
                    week=week,
                    home_team=home_team,
                    away_team=away_team,
                    home_score=int(home_pts),
                    away_score=int(away_pts),
                )
                elo.update_week([result])

            except Exception:
                continue


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
) -> SeasonSimulation:
    """
    Convenience function: simulate the rest of the season starting from current_week + 1.

    Args:
        season:             NFL season.
        current_week:       The last completed week. Simulation starts from current_week + 1.
        players_df:         Active roster DataFrame.
        prior_game_rows:    Kalman history through current_week.
        schedule_df:        Optional team schedule [week, home_team, away_team].
        injury_projections: Optional {week: {player_id: status}} for projected IR.
        n_simulations:      Monte Carlo season paths.
        stats:              Stats to simulate. Default: 4 core stats.
        rng_seed:           Optional seed for reproducibility.

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
    )
    return sim.run(
        players_df=players_df,
        prior_game_rows=prior_game_rows,
        schedule_df=schedule_df,
        injury_projections=injury_projections,
        rng_seed=rng_seed,
    )
