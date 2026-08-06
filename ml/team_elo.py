"""
ml/team_elo.py

Dynamic Team Elo Ratings for the Gridiron Oracle pipeline.

OVERVIEW
--------
Elo is a self-correcting power rating system originally designed for chess.
Here it tracks each NFL team's offensive and defensive strength separately,
updating weekly after each game.

Two variants are maintained per team:
    • Offensive Elo  — quality of the team's scoring offense
    • Defensive Elo  — quality of the team's scoring defense (higher = better D)

WHY TWO SEPARATE ELOs?
-----------------------
A team can be a strong rusher but a weak passer (or vice versa). When we project
WR/TE fantasy, what matters is the OPPONENT's defensive Elo, not the team's
overall Elo. Separate offensive/defensive ratings let us surface:
    - opp_def_elo          → opponent's defensive strength (key for WR projections)
    - team_off_elo         → team's offensive momentum (drives volume)
    - home_elo_advantage   → differential for game script prediction

MATH
----
Standard Elo update after game:

    Expected score: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    Actual score:   S_A = 1 if A wins, 0.5 if draw, 0 if loss

    New rating: R_A_new = R_A + K × (S_A - E_A)

NFL-specific modifications:
    - K = 20 (moderate update speed — NFL has 17 games vs 80 in NBA)
    - Margin of victory multiplier:
        MOV_mult = ln(abs(point_differential) + 1) × (2.2 / (winner_Elo_diff × 0.001 + 2.2))
      (from 538's implementation — prevents Elo inflation from blowouts vs weak opponents)
    - Season-to-season regression:
        R_new_season = R_prev_season × REGRESSION_FACTOR + INITIAL_ELO × (1 - REGRESSION_FACTOR)
      where REGRESSION_FACTOR = 0.75 (teams regress 25% toward mean each season)
    - Home field adjustment: +65 Elo points for home team in expected score calculation

USAGE
-----
    from ml.team_elo import TeamEloSystem, EloSnapshot

    elo = TeamEloSystem()
    elo.fit(game_results_df)       # historical game results from DB
    elo.update_week(week_results)  # update after each week's games

    snap = elo.get_snapshot("MIN", season=2025, week=10)
    # snap.off_elo → 1587.3
    # snap.def_elo → 1542.1
    # snap.off_elo_rank → 8    (rank among 32 teams this week)

    feature_df = elo.enrich_features(feature_matrix_df, season=2025, week=10)
    # Adds columns: team_off_elo, team_def_elo, opp_off_elo, opp_def_elo,
    #               home_elo_diff, elo_implied_win_prob

INTEGRATION
-----------
    FeatureRow (pipeline/feature_engineer.py) — add Bucket 10:
        team_off_elo, team_def_elo, opp_off_elo, opp_def_elo,
        home_elo_diff, elo_implied_win_prob
    These require a populated EloSystem before feature engineering runs.
    TeamEloSystem.enrich_features() is the recommended integration point.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

INITIAL_ELO: float = 1500.0      # All teams start equal
K_FACTOR: float = 20.0           # Elo update speed (NFL: fewer games = smaller K vs chess)


@dataclass(frozen=True)
class EloConfig:
    """Configuration for Team Elo system. Used by tests and consumers."""
    initial_elo: float = INITIAL_ELO


HOME_ADVANTAGE: float = 65.0     # Elo home-field bonus in expected score calculation
REGRESSION_FACTOR: float = 0.75  # Season-to-season regression toward mean (0.75 = 25% decay)

# MOV multiplier cap to prevent runaway Elo from extreme blowouts (e.g. 50-pt wins)
MOV_MULT_CAP: float = 3.0

# Minimum and maximum Elo values (hard clip — prevents weird states after data gaps)
ELO_FLOOR: float = 1000.0
ELO_CEILING: float = 2200.0

# Seasons available in the nflreadpy game data
NFL_SEASONS_AVAILABLE: list[int] = list(range(2000, 2027))  # through CURRENT_SEASON; incomplete years may be empty


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class EloSnapshot:
    """
    Team Elo ratings at a specific (season, week) checkpoint.
    Represents the state AFTER processing all games up to and including
    week-1 of the given week (i.e., the prior heading into this week).
    """
    team: str
    season: int
    week: int
    off_elo: float       # Offensive Elo (quality of scoring attack)
    def_elo: float       # Defensive Elo (quality of preventing points; higher = better D)
    off_elo_rank: int    # 1 = best offense in league this week
    def_elo_rank: int    # 1 = best defense in league this week
    games_played: int    # cumulative games through this point (for credibility weighting)


@dataclass
class GameResult:
    """
    A single NFL game result needed to compute Elo updates.
    Both teams must be present (home/away convention).
    """
    season: int
    week: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    game_id: Optional[str] = None

    @property
    def home_won(self) -> bool:
        return self.home_score > self.away_score

    @property
    def away_won(self) -> bool:
        return self.away_score > self.home_score

    @property
    def point_differential(self) -> int:
        """Absolute score difference (always positive)."""
        return abs(self.home_score - self.away_score)

    @property
    def score_diff(self) -> int:
        return self.point_differential

    @property
    def is_tie(self) -> bool:
        return self.home_score == self.away_score


# ── Elo Math ──────────────────────────────────────────────────────────────────

def _expected_score(rating_a: float, rating_b: float, a_is_home: bool = False) -> float:
    """
    Probability that team A beats team B.

    Args:
        rating_a: Elo rating of team A (combined off+def).
        rating_b: Elo rating of team B.
        a_is_home: If True, adds HOME_ADVANTAGE to team A's rating.

    Returns:
        Float in (0, 1) — expected score for team A.
    """
    rating_a_adj = rating_a + (HOME_ADVANTAGE if a_is_home else 0.0)
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a_adj) / 400.0))


def _mov_multiplier(point_diff: int, winner_elo_diff: float) -> float:
    """
    Margin of Victory multiplier (from 538's NFL Elo model).
    Reduces Elo gained from blowing out weak opponents.

    Args:
        point_diff:     Absolute score difference.
        winner_elo_diff: Elo difference = winner_elo - loser_elo (positive = winner was favored).

    Returns:
        Multiplier > 0; capped at MOV_MULT_CAP.
    """
    ln_factor = math.log(point_diff + 1.0)
    # autocorrelation correction: teams that already have high Elo get smaller MOV bonus
    autocorr = 2.2 / (winner_elo_diff * 0.001 + 2.2)
    return min(ln_factor * autocorr, MOV_MULT_CAP)


def _clip_elo(rating: float) -> float:
    return max(ELO_FLOOR, min(ELO_CEILING, rating))


def _regress_to_mean(rating: float) -> float:
    """Apply season-to-season mean reversion."""
    return INITIAL_ELO + REGRESSION_FACTOR * (rating - INITIAL_ELO)


# ── TeamEloSystem ─────────────────────────────────────────────────────────────

class TeamEloSystem:
    """
    Manages separate offensive and defensive Elo ratings for all 32 NFL teams.

    State:
        _off_elo[team]  — current offensive rating
        _def_elo[team]  — current defensive rating
        _history[team]  — list of EloSnapshot per (season, week), in order

    Conceptual Model for Off/Def Separation:
        When team A (offense) beats team B (defense):
          - A's off_elo goes up   (their offense outperformed expectation)
          - B's def_elo goes down (their defense underperformed expectation)
          - A's def_elo unchanged (it was the offense that won)
          - B's off_elo unchanged (the loser's offense gets credit separately)

        In practice:
          - We compute a single "game quality" Elo update based on combined
            ratings (off + def average) — this drives the win probability.
          - We then split the update between the winner's offense and the
            loser's defense (and vice versa for the other team).

    Note: Strictly the offense vs. defense attribution is approximate. A more
    rigorous approach would model each play individually and attribute yards
    to individual matchups. That's the GNN model (Tier 4). This is a
    good-enough team-level signal for weekly projection confidence levels.
    """

    def __init__(self) -> None:
        """Initialize with all 32 teams at INITIAL_ELO = 1500."""
        self._off_elo: dict[str, float] = {}
        self._def_elo: dict[str, float] = {}
        self._games_played: dict[str, int] = {}
        self._history: dict[str, list[EloSnapshot]] = {}
        self._fitted_through: Optional[tuple[int, int]] = None  # (season, week)

        # Bootstrap all 32 NFL teams
        for team in _ALL_NFL_TEAMS:
            self._off_elo[team] = INITIAL_ELO
            self._def_elo[team] = INITIAL_ELO
            self._games_played[team] = 0
            self._history[team] = []

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(self, game_results_df: pd.DataFrame) -> "TeamEloSystem":
        """
        Fit Elo ratings on historical game results.

        Args:
            game_results_df: DataFrame with columns:
                [season, week, home_team, away_team, home_score, away_score]
                One row per game. Sort order doesn't matter — will be sorted
                by (season, week) internally.

        Returns:
            self (for chaining)
        """
        required = {"season", "week", "home_team", "away_team", "home_score", "away_score"}
        missing = required - set(game_results_df.columns)
        if missing:
            raise ValueError(f"TeamEloSystem.fit(): missing columns {missing}. Have: {list(game_results_df.columns)}")

        df = game_results_df.dropna(subset=list(required)).copy()
        df["season"] = df["season"].astype(int)
        df["week"]   = df["week"].astype(int)
        df = df.sort_values(["season", "week"]).reset_index(drop=True)

        current_season: Optional[int] = None

        for _, row in df.iterrows():
            season = int(row["season"])
            week   = int(row["week"])

            # Season transition: regress ratings toward mean
            if current_season is not None and season != current_season:
                self.apply_season_regression()

            current_season = season

            result = GameResult(
                season=season,
                week=week,
                home_team=str(row["home_team"]),
                away_team=str(row["away_team"]),
                home_score=int(row["home_score"]),
                away_score=int(row["away_score"]),
                game_id=str(row.get("game_id", "")),
            )
            self._process_game(result)

        if current_season is not None and not df.empty:
            last_row = df.iloc[-1]
            self._fitted_through = (int(last_row["season"]), int(last_row["week"]))

        n_games = len(df)
        logger.info(
            "TeamEloSystem fitted on %d games (%d seasons). "
            "Final top 5 offenses: %s",
            n_games,
            len(df["season"].unique()),
            self._top_n_teams("off", n=5),
        )
        return self

    def update_week(self, week_results: list[GameResult]) -> None:
        """
        Update Elo ratings with a new week's results.
        Call this after each NFL week completes.

        Args:
            week_results: List of GameResult for all games that week.
        """
        for result in week_results:
            self._process_game(result)
        if week_results:
            last = week_results[-1]
            self._fitted_through = (last.season, last.week)
            logger.info(
                "TeamEloSystem updated through Season %d Week %d (%d games).",
                last.season, last.week, len(week_results),
            )

    # ── Core Update ──────────────────────────────────────────────────────────

    def _process_game(self, result: GameResult) -> None:
        """Apply Elo update for a single game."""
        h = normalize_team_abbr(result.home_team)
        a = normalize_team_abbr(result.away_team)

        # Combined Elo for win probability (average off+def)
        h_combined = (self._off_elo.get(h, INITIAL_ELO) + self._def_elo.get(h, INITIAL_ELO)) / 2
        a_combined = (self._off_elo.get(a, INITIAL_ELO) + self._def_elo.get(a, INITIAL_ELO)) / 2

        # Expected scores
        e_home = _expected_score(h_combined, a_combined, a_is_home=True)
        e_away = 1.0 - e_home

        # Actual scores
        if result.home_won:
            s_home, s_away = 1.0, 0.0
        elif result.away_won:
            s_home, s_away = 0.0, 1.0
        else:
            s_home = s_away = 0.5

        # MOV multiplier
        if not result.is_tie:
            winner_elo = h_combined if result.home_won else a_combined
            loser_elo  = a_combined if result.home_won else h_combined
            mov = _mov_multiplier(result.point_differential, winner_elo - loser_elo)
        else:
            mov = 1.0

        # Raw Elo delta
        delta_home = K_FACTOR * mov * (s_home - e_home)
        delta_away = K_FACTOR * mov * (s_away - e_away)

        # Split delta between offense and defense:
        #   Winner's OFFENSE goes up, loser's DEFENSE goes down.
        #   Winner's DEFENSE and loser's OFFENSE get mild/neutral update.
        if s_home > s_away:  # home won
            self._off_elo[h] = _clip_elo(self._off_elo.get(h, INITIAL_ELO) + delta_home)
            self._def_elo[a] = _clip_elo(self._def_elo.get(a, INITIAL_ELO) + delta_away)
            # Mild updates for winner's D and loser's O
            self._def_elo[h] = _clip_elo(self._def_elo.get(h, INITIAL_ELO) + delta_home * 0.3)
            self._off_elo[a] = _clip_elo(self._off_elo.get(a, INITIAL_ELO) + delta_away * 0.3)
        elif s_away > s_home:  # away won
            self._off_elo[a] = _clip_elo(self._off_elo.get(a, INITIAL_ELO) + delta_away)
            self._def_elo[h] = _clip_elo(self._def_elo.get(h, INITIAL_ELO) + delta_home)
            self._def_elo[a] = _clip_elo(self._def_elo.get(a, INITIAL_ELO) + delta_away * 0.3)
            self._off_elo[h] = _clip_elo(self._off_elo.get(h, INITIAL_ELO) + delta_home * 0.3)
        else:  # tie — split equally between off and def
            for team, delta in [(h, delta_home), (a, delta_away)]:
                self._off_elo[team] = _clip_elo(self._off_elo.get(team, INITIAL_ELO) + delta * 0.5)
                self._def_elo[team] = _clip_elo(self._def_elo.get(team, INITIAL_ELO) + delta * 0.5)

        # Update game counts
        for team in [h, a]:
            self._games_played[team] = self._games_played.get(team, 0) + 1

        # Store snapshots for lookback
        self._snapshot(h, result.season, result.week)
        self._snapshot(a, result.season, result.week)

    def apply_season_regression(self, season: int = None) -> None:
        """Regress all ratings toward the mean at season boundary."""
        for team in list(self._off_elo.keys()):
            self._off_elo[team] = _regress_to_mean(self._off_elo[team])
            self._def_elo[team] = _regress_to_mean(self._def_elo[team])

    def _snapshot(self, team: str, season: int, week: int) -> None:
        """Record current Elo snapshot for a team after a game."""
        snap = EloSnapshot(
            team=team,
            season=season,
            week=week,
            off_elo=self._off_elo.get(team, INITIAL_ELO),
            def_elo=self._def_elo.get(team, INITIAL_ELO),
            off_elo_rank=0,  # filled lazily via rankings()
            def_elo_rank=0,
            games_played=self._games_played.get(team, 0),
        )
        if team not in self._history:
            self._history[team] = []
        self._history[team].append(snap)

    # ── Query API ─────────────────────────────────────────────────────────────

    def get_current_ratings(self, team: str) -> tuple[float, float]:
        """
        Return (off_elo, def_elo) for a team as of the most recent update.

        Args:
            team: Team abbreviation (e.g. "MIN", "GB").

        Returns:
            (off_elo, def_elo) tuple. Returns (INITIAL_ELO, INITIAL_ELO)
            for unknown teams.
        """
        return (
            self._off_elo.get(team, INITIAL_ELO),
            self._def_elo.get(team, INITIAL_ELO),
        )

    def get_snapshot(self, team: str, season: int, week: int) -> Optional[EloSnapshot]:
        """
        Return the Elo snapshot for a team heading INTO the specified (season, week).
        "Heading into" = after processing all games from previous weeks.

        Args:
            team:   Team abbreviation.
            season: NFL season.
            week:   NFL week (the game we're about to project).

        Returns:
            EloSnapshot or None if no data exists for this team/season/week.
        """
        history = self._history.get(team, [])
        # Find the last snapshot BEFORE this week
        prior = [s for s in history if (s.season, s.week) < (season, week)]
        if not prior:
            # No priors — return initial state
            return EloSnapshot(
                team=team, season=season, week=week,
                off_elo=INITIAL_ELO, def_elo=INITIAL_ELO,
                off_elo_rank=16, def_elo_rank=16,
                games_played=0,
            )
        return prior[-1]

    def win_probability(
        self,
        team_a: str,
        team_b: str,
        team_a_is_home: bool = False,
    ) -> float:
        """
        Implied win probability for team_a vs team_b.

        Args:
            team_a: Team abbreviation.
            team_b: Team abbreviation.
            team_a_is_home: If True, applies home field advantage to team_a.

        Returns:
            Float in (0, 1) — probability that team_a wins.
        """
        a_off, a_def = self.get_current_ratings(team_a)
        b_off, b_def = self.get_current_ratings(team_b)
        a_combined = (a_off + a_def) / 2
        b_combined = (b_off + b_def) / 2
        return _expected_score(a_combined, b_combined, a_is_home=team_a_is_home)

    def rankings(self, season: Optional[int] = None, week: Optional[int] = None) -> pd.DataFrame:
        """
        Return current Elo rankings for all 32 teams.

        Args:
            season: If provided, returns rankings at that (season, week) snapshot.
            week:   Must be provided if season is provided.

        Returns:
            DataFrame with columns [team, off_elo, def_elo, off_rank, def_rank,
            combined_elo, combined_rank] sorted by combined_rank ascending.
        """
        rows = []
        for team in _ALL_NFL_TEAMS:
            if season is not None and week is not None:
                snap = self.get_snapshot(team, season, week)
                off  = snap.off_elo if snap else INITIAL_ELO
                def_ = snap.def_elo if snap else INITIAL_ELO
            else:
                off, def_ = self.get_current_ratings(team)

            rows.append({"team": team, "off_elo": off, "def_elo": def_,
                         "combined_elo": (off + def_) / 2})

        df = pd.DataFrame(rows)
        df["off_rank"]      = df["off_elo"].rank(ascending=False).astype(int)
        df["def_rank"]      = df["def_elo"].rank(ascending=False).astype(int)
        df["combined_rank"] = df["combined_elo"].rank(ascending=False).astype(int)
        return df.sort_values("combined_rank").reset_index(drop=True)

    # ── Feature Engineering Integration ──────────────────────────────────────

    def enrich_features(
        self,
        df: pd.DataFrame,
        season: int,
        week: int,
    ) -> pd.DataFrame:
        """
        Add Elo-based features to a feature matrix DataFrame.

        Expected columns in df: [team, opponent] or [team, opp_team].
        Adds the following columns (all float):
            - team_off_elo       : Team's offensive Elo heading into this week
            - team_def_elo       : Team's defensive Elo heading into this week
            - opp_off_elo        : Opponent's offensive Elo
            - opp_def_elo        : Opponent's defensive Elo
            - elo_matchup_diff   : team_off_elo - opp_def_elo
                                   (+ve = offense advantage, key for WR projections)
            - elo_implied_win_prob: Elo-implied win probability for team (from team's perspective)

        Returns:
            DataFrame with new columns appended. Original rows unchanged.
        """
        out = df.copy()
        rankings_snap = self.rankings(season=season, week=week)
        {row["team"]: row for _, row in rankings_snap.iterrows()}

        # Detect opponent column name
        opp_col = "opp_team" if "opp_team" in df.columns else (
            "opponent" if "opponent" in df.columns else None
        )

        def _get_off(team: str) -> float:
            snap = self.get_snapshot(team, season, week)
            return snap.off_elo if snap else INITIAL_ELO

        def _get_def(team: str) -> float:
            snap = self.get_snapshot(team, season, week)
            return snap.def_elo if snap else INITIAL_ELO

        out["team_off_elo"] = out["team"].map(_get_off) if "team" in df.columns else INITIAL_ELO
        out["team_def_elo"] = out["team"].map(_get_def) if "team" in df.columns else INITIAL_ELO

        if opp_col:
            out["opp_off_elo"] = out[opp_col].map(_get_off)
            out["opp_def_elo"] = out[opp_col].map(_get_def)
            out["elo_matchup_diff"]    = out["team_off_elo"] - out["opp_def_elo"]
            out["elo_implied_win_prob"] = out.apply(
                lambda r: self.win_probability(
                    r["team"], r[opp_col],
                    team_a_is_home=bool(r.get("is_home", 0)),
                ),
                axis=1,
            )
        else:
            out["opp_off_elo"]          = INITIAL_ELO
            out["opp_def_elo"]          = INITIAL_ELO
            out["elo_matchup_diff"]     = 0.0
            out["elo_implied_win_prob"] = 0.5

        return out

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """
        Save current Elo state to a JSON file for persistence between runs.

        Format: {team: {off_elo: float, def_elo: float, games_played: int}}
        """
        import json
        state = {
            team: {
                "off_elo":      self._off_elo.get(team, INITIAL_ELO),
                "def_elo":      self._def_elo.get(team, INITIAL_ELO),
                "games_played": self._games_played.get(team, 0),
            }
            for team in _ALL_NFL_TEAMS
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2))
        logger.info("TeamEloSystem saved to %s", path)

    def load(self, path: Path) -> "TeamEloSystem":
        """
        Load Elo state from a previously saved JSON file.

        Returns:
            self (for chaining)
        """
        import json
        path = Path(path)
        if not path.exists():
            logger.warning("TeamEloSystem.load(): file not found at %s, using defaults.", path)
            return self

        state = json.loads(path.read_text())
        for team, data in state.items():
            self._off_elo[team]      = float(data.get("off_elo", INITIAL_ELO))
            self._def_elo[team]      = float(data.get("def_elo", INITIAL_ELO))
            self._games_played[team] = int(data.get("games_played", 0))

        logger.info("TeamEloSystem loaded from %s", path)
        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _top_n_teams(self, kind: str, n: int = 5) -> list[str]:
        elo_dict = self._off_elo if kind == "off" else self._def_elo
        return sorted(elo_dict, key=lambda t: elo_dict[t], reverse=True)[:n]

    def __repr__(self) -> str:
        top3_off = self._top_n_teams("off", 3)
        top3_def = self._top_n_teams("def", 3)
        through = self._fitted_through or ("?", "?")
        return (
            f"TeamEloSystem(through=S{through[0]}W{through[1]}, "
            f"top_off={top3_off}, top_def={top3_def})"
        )


# ── DB Loader Helper ─────────────────────────────────────────────────────────

def load_elo_from_db(db_url: str, seasons: Optional[list[int]] = None) -> TeamEloSystem:
    """
    Convenience factory: build a fully-fitted TeamEloSystem from game scores
    stored in the Gridiron Oracle `game_logs` or `schedule` table.

    Args:
        db_url:   PostgreSQL DSN (e.g. from os.environ["DATABASE_URL"]).
        seasons:  Optional list of seasons to load. Defaults to all available.

    Returns:
        Fitted TeamEloSystem.
    """
    import psycopg2
    import psycopg2.extras

    seasons = seasons or NFL_SEASONS_AVAILABLE

    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT season, week, home_team, away_team,
                       home_score, away_score
                FROM   games
                WHERE  season = ANY(%s)
                  AND  home_score IS NOT NULL
                ORDER  BY season, week
            """, (list(seasons),))
            rows = cur.fetchall()
    except Exception as e:
        logger.error("load_elo_from_db: could not load game results: %s", e)
        rows = []
    finally:
        conn.close()

    if not rows:
        logger.warning("load_elo_from_db: no game results found for seasons %s", seasons)
        return TeamEloSystem()

    df = pd.DataFrame(rows)
    logger.info("load_elo_from_db: loaded %d game results for %d seasons.", len(df), len(seasons))

    elo = TeamEloSystem()
    return elo.fit(df)


# ── Singleton ─────────────────────────────────────────────────────────────────

_GLOBAL_ELO: Optional[TeamEloSystem] = None


def get_elo_system(db_url: Optional[str] = None) -> TeamEloSystem:
    """
    Module-level singleton TeamEloSystem.
    Fits from DB on first call if db_url is provided.

    Usage in feature_engineer.py and train.py:
        from ml.team_elo import get_elo_system
        elo = get_elo_system(db_url=os.environ.get("DATABASE_URL"))
        enriched_df = elo.enrich_features(feature_df, season=2025, week=10)
    """
    global _GLOBAL_ELO
    if _GLOBAL_ELO is None:
        if db_url:
            _GLOBAL_ELO = load_elo_from_db(db_url)
        else:
            _GLOBAL_ELO = TeamEloSystem()
    return _GLOBAL_ELO


def reset_elo_system() -> None:
    """Reset the global singleton (useful in tests)."""
    global _GLOBAL_ELO
    _GLOBAL_ELO = None


# ── NFL Team Registry ─────────────────────────────────────────────────────────
# All 32 current franchises + common historical aliases used in nflreadpy.

_ALL_NFL_TEAMS: list[str] = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB",  "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV",  "MIA", "MIN", "NE",  "NO",  "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF",  "TB",  "TEN", "WAS",
]

# Franchise relocation mapping: old abbreviation → current abbreviation.
# Ensures Elo continuity: Raiders' OAK rating carries to LV, etc.
FRANCHISE_RELOCATIONS: dict[str, str] = {
    "OAK": "LV",   # Oakland → Las Vegas (2020)
    "STL": "LAR",   # St. Louis → Los Angeles Rams (2016)
    "SD":  "LAC",   # San Diego → Los Angeles Chargers (2017)
}


def normalize_team_abbr(abbr: str) -> str:
    """Map historical franchise abbreviations to their current form."""
    return FRANCHISE_RELOCATIONS.get(abbr, abbr)
