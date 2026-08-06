"""
ml/copula_layer.py

Copula Layer — Joint Distribution Modeling for Same Game Parlay (SGP) Pricing.

PROBLEM
-------
XGBoost, LightGBM, and TFT predict each player's stats INDEPENDENTLY.
The Bayesian layer draws independent posteriors for Jefferson (326 yd dist)
and Diggs (93 yd dist).

If you multiply these independent CDFs to price a parlay:
    P(Jefferson > 100 yds AND Diggs > 6 rec) = P1 × P2 = 0.55 × 0.60 = 0.33

But reality: Jefferson and Diggs are on the SAME TEAM. They share the same
QB and the same game. If the Chiefs blitz and Mahomes is under pressure, BOTH
go down together, not independently. The actual joint probability may be 0.22
(correlated downside) or 0.41 (if the gameplan features both of them).

Treating them as independent overstates parlay probabilities (bad for SGPs)
and produces physically impossible correlated scenarios in the season simulator.

SOLUTION — GAUSSIAN COPULA
---------------------------
The Gaussian copula models the joint CDF of n players' stats by:
    1. Convert each player's marginal posterior samples to uniform scores:
       U_i = Φ⁻¹(rank(S_i) / n_samples)   (inverse normal CDF transform)
    2. Fit a correlation matrix Σ on the multivariate U_i scores.
    3. For new Monte Carlo draws, sample (Z_1, ..., Z_n) ~ N(0, Σ).
    4. Convert back to marginal space: S_i_new = F_i⁻¹(Φ(Z_i))
       where F_i is the empirical marginal CDF from the posterior.

The Σ matrix captures the co-movement structure between players while
preserving each player's individual marginal distribution exactly.

CORRELATION SOURCES
-------------------
Three correlation sources are estimated and blended:

1. Historical residual correlation (best but requires OOF data):
   After training, compute OOF residuals for each player on each game.
   The within-game residual correlation matrix captures true co-movement.
   REQUIRES: ml/oof/(stat)_oof_preds.csv files from training.

2. Teammate structural correlation (available now):
   Same team + same position (WR-WR) → typically medium positive correlation.
   Same team + different position (WR-TE) → low-medium positive.
   Opponent-opponent → slight negative (zero-sum game).
   QB and his WR1 → high positive (passing game correlation).

3. Historical co-performance lookup (from feature_matrix):
   Average correlation of game-level performances for known player pairs.
   Pre-computed at pipeline init for the top 200 player pairs.

SGP PRICING
-----------
    from ml.copula_layer import CopulaLayer, SGPLeg

    copula = CopulaLayer()
    copula.fit(oof_df)    # or use structural correlations only

    legs = [
        SGPLeg(player_id="p1", stat="receiving_yards",  threshold=100.5, over=True),
        SGPLeg(player_id="p2", stat="receiving_yards",  threshold=6.5,   over=False),
        SGPLeg(player_id="p3", stat="rushing_yards",    threshold=75.5,  over=True),
    ]

    parlay_prob = copula.price_sgp(legs, posterior_samples_by_player)
    # Returns probability accounting for between-player correlation.
    # Independent assumption would be: prod(P_i for each leg)

USAGE WITH SEASON SIMULATOR
----------------------------
    joint_samples = copula.draw_joint_samples(
        player_ids=["p1", "p2", "p3"],
        stat="receiving_yards",
        n_samples=2000,
        marginal_samples=posterior_samples_dict,
    )
    # joint_samples shape: (2000, 3) — rows are correlated draws
    # Use these instead of independent draws to avoid game script impossibilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Regularization: shrink toward structural prior when OOF sample is small.
# At MIN_GAMES = 10+, we fully trust OOF correlation. Below: linear blend.
MIN_GAMES_FOR_FULL_OOF_TRUST: int = 20

# Numerical epsilon for CDF inversion safety
_EPSILON: float = 1e-6

# Maximum number of players in a single joint draw (memory limit)
MAX_JOINT_PLAYERS: int = 50


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class SGPLeg:
    """
    A single leg in a Same Game Parlay.

    Args:
        player_id:  Player ID string (as in feature_matrix).
        stat:       Stat type: "receiving_yards", "rushing_yards",
                    "passing_yards", "fantasy_ppr", "receptions", etc.
        threshold:  The bookmaker's line (e.g. 94.5 for receiving yards).
        over:       True = bet the OVER (stat > threshold).
                    False = bet the UNDER (stat ≤ threshold).
        name:       Optional player name for display.
    """
    player_id: str
    stat: str
    threshold: float
    over: bool = True
    name: Optional[str] = None

    @property
    def label(self) -> str:
        direction = "OVER" if self.over else "UNDER"
        return f"{self.name or self.player_id} {direction} {self.threshold} {self.stat}"


@dataclass
class SGPResult:
    """Result of pricing a Same Game Parlay."""
    legs: list[SGPLeg]
    parlay_probability: float       # P(all legs hit) with copula correlation
    independent_probability: float  # P(all legs hit) if independent (naive)
    correlation_adjustment: float   # = parlay_prob - independent_prob
    n_samples_used: int
    leg_probabilities: list[float]  # marginal P for each individual leg

    @property
    def implied_odds(self) -> float:
        """Convert probability to American odds."""
        p = self.parlay_probability
        if p <= 0 or p >= 1:
            return float("nan")
        if p >= 0.5:
            return round(-100 * p / (1 - p))
        return round(100 * (1 - p) / p)

    def __str__(self) -> str:
        leg_lines = "\n  ".join(
            f"{leg.label}: {p:.1%}" for leg, p in zip(self.legs, self.leg_probabilities)
        )
        return (
            f"SGP ({len(self.legs)} legs)\n"
            f"  {leg_lines}\n"
            f"Parlay P (copula):     {self.parlay_probability:.3%}\n"
            f"Parlay P (independent):{self.independent_probability:.3%}\n"
            f"Correlation adjustment:{self.correlation_adjustment:+.3%}\n"
            f"Implied American odds: {self.implied_odds:+d}"
        )


# ── Core Math ─────────────────────────────────────────────────────────────────

def _to_uniform(samples: np.ndarray) -> np.ndarray:
    """
    Convert a 1D sample array to ~Uniform(0,1) via empirical CDF rank transform.

    Args:
        samples: 1D array of stat values.

    Returns:
        1D array of uniform scores ∈ (ε, 1-ε).
    """
    n = len(samples)
    ranks = scipy_stats.rankdata(samples)    # average rank ties
    uniform = ranks / (n + 1)               # (0, 1) exclusive
    return np.clip(uniform, _EPSILON, 1 - _EPSILON)


def _to_gaussian(uniform_samples: np.ndarray) -> np.ndarray:
    """
    Convert Uniform(0,1) scores to standard normal via Φ⁻¹.
    """
    return scipy_stats.norm.ppf(uniform_samples)


def _estimate_correlation_matrix(
    gaussian_scores_matrix: np.ndarray,
    regularization: float = 0.05,
) -> np.ndarray:
    """
    Estimate the Gaussian copula correlation matrix from normal scores.

    Applies Ledoit-Wolf shrinkage regularization to prevent ill-conditioned
    matrices from small sample sizes/many players.

    Args:
        gaussian_scores_matrix: Shape (n_samples, n_players). Already in
                                 standard normal space (from _to_gaussian).
        regularization:          Fraction to regularize toward identity.
                                 0 = no regularization, 0.1 = 10% identity.

    Returns:
        Positive semi-definite correlation matrix (n_players, n_players).
    """
    n_players = gaussian_scores_matrix.shape[1]
    if n_players == 1:
        return np.array([[1.0]])

    # Sample correlation matrix
    corr_raw = np.corrcoef(gaussian_scores_matrix, rowvar=False)

    # Handle NaN/Inf from degenerate samples
    corr_raw = np.nan_to_num(corr_raw, nan=0.0, posinf=1.0, neginf=-1.0)
    np.fill_diagonal(corr_raw, 1.0)  # ensure exact diagonal=1

    # Tikhonov regularization toward identity
    identity = np.eye(n_players)
    corr_reg = (1 - regularization) * corr_raw + regularization * identity

    # Nearest positive-definite if needed (can happen with many players)
    min_eig = np.linalg.eigvalsh(corr_reg).min()
    if min_eig < 0:
        corr_reg += (-min_eig + 1e-8) * identity

    return corr_reg


def _draw_correlated_uniform(
    correlation_matrix: np.ndarray,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw n_samples correlated Uniform(0,1) samples from the Gaussian copula.

    Args:
        correlation_matrix: (n_players, n_players) PSD correlation matrix.
        n_samples:          Number of Monte Carlo draws.
        rng:                NumPy Generator for reproducibility.

    Returns:
        Array shape (n_samples, n_players) of Uniform(0,1) samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_players = correlation_matrix.shape[0]
    mean = np.zeros(n_players)

    # Draw from multivariate normal with given covariance (= correlation since std=1)
    try:
        z_samples = rng.multivariate_normal(mean, correlation_matrix, size=n_samples)
    except np.linalg.LinAlgError:
        # Fallback: independent samples if matrix is still singular
        logger.warning("_draw_correlated_uniform: singular correlation matrix; falling back to independent.")
        z_samples = rng.standard_normal((n_samples, n_players))

    # Convert normal → uniform via Φ (standard normal CDF)
    return scipy_stats.norm.cdf(z_samples)


def _quantile_transform(
    uniform_samples: np.ndarray,
    reference_samples: np.ndarray,
) -> np.ndarray:
    """
    Map Uniform(0,1) samples back to the marginal distribution defined by
    `reference_samples` via the empirical quantile function (F⁻¹).

    Args:
        uniform_samples:   1D array of U(0,1) values.
        reference_samples: 1D array of historical/posterior samples defining
                           the marginal distribution.

    Returns:
        1D array of stat values drawn from the marginal distribution while
        preserving their correlation structure from the copula.
    """
    quantiles = np.clip(uniform_samples * 100, 0.001, 99.999)
    return np.percentile(reference_samples, quantiles)


# ── CopulaLayer ───────────────────────────────────────────────────────────────

class CopulaLayer:
    """
    Gaussian copula for pricing parlays and generating correlated player stat
    draws for the Season Simulator.

    State:
        _oof_corr[stat]:        {frozenset(player_ids): ρ} — empirical OOF
                                 pair correlations per stat.
        _player_metadata:       {player_id: {team, position}} — for structural prior.
        _correlation_matrix:    Cached full correlation matrix per (stat, player_ids_tuple).
    """

    def __init__(self, auto_load: bool = False) -> None:
        self._historical_residuals: dict[tuple[str, str], pd.Series] = {}
        self._oof_corr_cache: dict[frozenset[tuple[str, str]], float] = {}
        self._player_metadata: dict[str, dict] = {}
        self._fitted = False
        
        if auto_load:
            self._auto_load_oof()

    def _auto_load_oof(self) -> None:
        """Automatically find and load OOF predictions to fit the copula."""
        import glob
        import os
        oof_files = glob.glob(os.path.join("ml", "oof", "*_combined.csv"))
        if not oof_files:
            logger.warning("Copula auto_load: no *_combined.csv found in ml/oof/")
            return
            
        dfs = []
        for f in oof_files:
            try:
                df = pd.read_csv(f)
                if {"player_id", "game_id", "actual", "predicted"}.issubset(df.columns):
                    # Infer stat from filename if not present
                    if "stat" not in df.columns:
                        base = os.path.basename(f)
                        # e.g., xgb_receiving_yards_combined.csv -> receiving_yards
                        parts = base.replace("_combined.csv", "").split("_")
                        if len(parts) >= 3:
                            stat = "_".join(parts[1:])
                            df["stat"] = stat
                    dfs.append(df)
            except Exception as e:
                logger.warning("Copula auto_load failed to read %s: %s", f, e)
                
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            self.fit(combined)
            logger.info("CopulaLayer auto-loaded %d OOF prediction files.", len(dfs))
        else:
            logger.warning("Copula auto_load: no valid OOF data found.")

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(
        self,
        oof_df: pd.DataFrame,
        player_meta_df: Optional[pd.DataFrame] = None,
    ) -> "CopulaLayer":
        """
        Fit copula correlation from OOF prediction residuals.

        Args:
            oof_df:         OOF predictions DataFrame. Required columns:
                            [player_id, game_id, stat, actual, predicted]
                            Can be loaded from ml/oof/combined_oof.csv or
                            merged from individual xgb/lgbm OOF files.
            player_meta_df: Optional DataFrame [player_id, team, position]
                            for structural correlation prior blending.

        Returns:
            self (for chaining)
        """
        required = {"player_id", "game_id", "stat", "actual", "predicted"}
        missing = required - set(oof_df.columns)
        if missing:
            logger.warning(
                "CopulaLayer.fit(): missing OOF columns %s. "
                "Falling back to structural-only correlations.", missing,
            )
            if player_meta_df is not None:
                self._load_player_metadata(player_meta_df)
            self._fitted = True
            return self

        n_total_series = 0
        oof_df = oof_df.copy()
        if "residual" not in oof_df.columns:
            oof_df["residual"] = oof_df["actual"] - oof_df["predicted"]
            
        for (pid, stat), group in oof_df.groupby(["player_id", "stat"]):
            sr = group.set_index("game_id")["residual"]
            # Save all series, we'll check length during get_correlation
            self._historical_residuals[(str(pid), str(stat))] = sr
            n_total_series += 1

        if player_meta_df is not None:
            self._load_player_metadata(player_meta_df)

        self._fitted = True
        logger.info(
            "CopulaLayer fitted: %d (player, stat) residual series stored.",
            n_total_series,
        )
        return self

    def _load_player_metadata(self, df: pd.DataFrame) -> None:
        """Index player metadata {player_id: {team, position}} for structural prior."""
        for _, row in df.iterrows():
            self._player_metadata[str(row["player_id"])] = {
                "team":     str(row.get("team", "")),
                "position": str(row.get("position", "")),
            }

    # ── Correlation Estimation ────────────────────────────────────────────────

    def get_correlation(
        self,
        node_a: tuple[str, str],
        node_b: tuple[str, str],
    ) -> float:
        """
        Get the estimated correlation ρ between two nodes (player_id, stat).

        Blends OOF empirical correlation with structural prior using
        credibility weighting: more common games = more trust in OOF.

        Args:
            node_a, node_b: (player_id, stat) tuples.

        Returns:
            Estimated Pearson correlation ∈ [-1, 1].
        """
        pid_a, stat_a = node_a
        pid_b, stat_b = node_b
        
        if node_a == node_b:
            return 1.0
            
        key = frozenset([node_a, node_b])
        if key in self._oof_corr_cache:
            return self._oof_corr_cache[key]

        # Try OOF correlation
        oof_rho: Optional[float] = None
        n_common = 0
        
        sr_a = self._historical_residuals.get(node_a)
        sr_b = self._historical_residuals.get(node_b)
        
        if sr_a is not None and sr_b is not None:
            common_games = sr_a.index.intersection(sr_b.index)
            n_common = len(common_games)
            if n_common >= 5:
                # Fast numpy correlation on aligned values
                r_a = sr_a.loc[common_games].values
                r_b = sr_b.loc[common_games].values
                rho, _ = scipy_stats.pearsonr(r_a, r_b)
                if not np.isnan(rho):
                    oof_rho = float(rho)

        # Structural prior
        struct_rho = self._structural_correlation(pid_a, pid_b, stat_a, stat_b)

        if oof_rho is None:
            final_rho = struct_rho
        else:
            # Credibility blend: weight toward OOF when we have enough data
            w_oof = min(n_common / MIN_GAMES_FOR_FULL_OOF_TRUST, 1.0)
            final_rho = w_oof * oof_rho + (1 - w_oof) * struct_rho
            
        self._oof_corr_cache[key] = final_rho
        return final_rho

    def _structural_correlation(
        self, pid_a: str, pid_b: str, stat_a: str, stat_b: str
    ) -> float:
        """
        Estimate correlation from player metadata (team, position) and stat types.
        """
        if pid_a == pid_b:
            # Same player, different stat => generally medium-high positive correlation
            return 0.40

        meta_a = self._player_metadata.get(pid_a, {})
        meta_b = self._player_metadata.get(pid_b, {})

        team_a    = meta_a.get("team", "?")
        team_b    = meta_b.get("team", "?")
        pos_a     = meta_a.get("position", "?")
        pos_b     = meta_b.get("position", "?")

        same_team = team_a == team_b and team_a != "?"

        if same_team:
            if pos_a == "QB" or pos_b == "QB":
                # Is it QB Passing vs WR Receiving? Very high correlation.
                if (stat_a == "passing_yards" and stat_b == "receiving_yards") or \
                   (stat_b == "passing_yards" and stat_a == "receiving_yards"):
                    return 0.65
                return 0.40      # QB and skill teammate (general)
            if pos_a == pos_b:
                return 0.25      # Same team, same non-QB position
            return 0.15          # Same team, different position

        # Opponents (same game, different team)
        if team_a != "?" and team_b != "?":
            if pos_a == pos_b and pos_a in {"WR", "TE"}:
                return -0.08     # Opposing WRs compete via game script
            return 0.05          # Unrelated players in same game

        return 0.05              # Unknown relationship

    def build_correlation_matrix(
        self,
        nodes: list[tuple[str, str]],
    ) -> np.ndarray:
        """
        Build the full (n, n) correlation matrix for a group of nodes (player_id, stat).

        Args:
            nodes: Ordered list of (player_id, stat) tuples.

        Returns:
            (n, n) positive-definite correlation matrix.
        """
        n = len(nodes)
        corr = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                rho = self.get_correlation(nodes[i], nodes[j])
                corr[i, j] = rho
                corr[j, i] = rho

        # Ensure PSD
        min_eig = np.linalg.eigvalsh(corr).min()
        if min_eig < 0:
            corr += (-min_eig + 1e-8) * np.eye(n)

        return corr

    # ── Joint Sampling ────────────────────────────────────────────────────────

    def draw_joint_samples(
        self,
        nodes: list[tuple[str, str]],
        marginal_samples: dict[tuple[str, str], np.ndarray],
        corr_matrix: Optional[np.ndarray] = None,
        n_samples: int = 2_000,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Generate correlated joint stat samples for a group of nodes.

        Args:
            nodes:             Ordered list of (player_id, stat) tuples.
            marginal_samples:  {(player_id, stat): 1D posterior samples array}.
                               These define each node's marginal distribution.
            corr_matrix:       Optional precomputed correlation matrix.
            n_samples:         Number of joint Monte Carlo draws.
            rng:               NumPy Generator for reproducibility.

        Returns:
            np.ndarray shape (n_samples, n_nodes) — correlated stat draws.
        """
        if rng is None:
            rng = np.random.default_rng()

        n_nodes = len(nodes)
        if n_nodes == 0:
            return np.zeros((n_samples, 0))
        if n_nodes == 1:
            ref = marginal_samples.get(nodes[0], np.array([0.0]))
            u = rng.uniform(0, 1, n_samples)
            return _quantile_transform(u, ref).reshape(-1, 1)

        # Build local correlation matrix if not supplied
        if corr_matrix is None:
            corr_matrix = self.build_correlation_matrix(nodes)

        # Draw correlated uniform scores from the copula
        u_correlated = _draw_correlated_uniform(corr_matrix, n_samples, rng=rng)

        # Map each column back to the marginal distribution
        result = np.zeros((n_samples, n_nodes))
        for col, node in enumerate(nodes):
            ref = marginal_samples.get(node)
            if ref is None or len(ref) == 0:
                result[:, col] = 0.0
            else:
                result[:, col] = _quantile_transform(u_correlated[:, col], ref)

        return result

    # ── SGP Pricing ───────────────────────────────────────────────────────────

    def price_sgp(
        self,
        legs: list[SGPLeg],
        posterior_samples_by_player: dict[str, dict[str, np.ndarray]],
        n_samples: int = 10_000,
        rng: Optional[np.random.Generator] = None,
    ) -> SGPResult:
        """
        Price a Same Game Parlay using the copula for joint probability estimation.

        Args:
            legs:                        List of SGPLeg objects defining the parlay.
            posterior_samples_by_player: Nested dict:
                                         {player_id: {stat: np.ndarray of samples}}
                                         from BayesianProjection.posterior_samples().
            n_samples:                   Monte Carlo draws for probability estimation.
            rng:                         NumPy Generator.

        Returns:
            SGPResult with true joint probability and comparison to independent baseline.
        """
        if rng is None:
            rng = np.random.default_rng()

        if not legs:
            return SGPResult(
                legs=[], parlay_probability=1.0, independent_probability=1.0,
                correlation_adjustment=0.0, n_samples_used=0, leg_probabilities=[],
            )

        # ── Marginal probabilities ────────────────────────────────────────────
        leg_probs: list[float] = []
        for leg in legs:
            samples_dict = posterior_samples_by_player.get(leg.player_id, {})
            samples = samples_dict.get(leg.stat, np.array([0.0]))
            if leg.over:
                p_marginal = float(np.mean(samples > leg.threshold))
            else:
                p_marginal = float(np.mean(samples <= leg.threshold))
            leg_probs.append(p_marginal)

        independent_p = float(np.prod(leg_probs))

        # ── Global Joint Sampling for all legs ────────────────────────────────
        # Extract unique (player, stat) nodes to avoid redundant sampling
        seen = set()
        nodes = []
        for leg in legs:
            key = (leg.player_id, leg.stat)
            if key not in seen:
                seen.add(key)
                nodes.append(key)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        marginals = {
            node: posterior_samples_by_player.get(node[0], {}).get(node[1], np.array([0.0]))
            for node in nodes
        }
        
        # Draw fully joint correlated samples across all stats and players
        joint_draws = self.draw_joint_samples(
            nodes=nodes,
            marginal_samples=marginals,
            corr_matrix=None,  # Builds it automatically
            n_samples=n_samples,
            rng=rng,
        )
        
        # Evaluate each leg against the joint draws
        joint_hit_matrix = np.ones((n_samples, len(legs)), dtype=bool)
        for col, leg in enumerate(legs):
            node = (leg.player_id, leg.stat)
            idx = node_to_idx[node]
            col_draws = joint_draws[:, idx]
            
            if leg.over:
                joint_hit_matrix[:, col] = col_draws > leg.threshold
            else:
                joint_hit_matrix[:, col] = col_draws <= leg.threshold

        # All legs must hit
        parlay_hits = joint_hit_matrix.all(axis=1)
        parlay_p    = float(parlay_hits.mean())

        return SGPResult(
            legs=legs,
            parlay_probability=parlay_p,
            independent_probability=independent_p,
            correlation_adjustment=parlay_p - independent_p,
            n_samples_used=n_samples,
            leg_probabilities=leg_probs,
        )

    def __repr__(self) -> str:
        n_pairs   = len(self._oof_corr_cache)
        n_resid   = len(self._historical_residuals)
        n_players = len(self._player_metadata)
        return (
            f"CopulaLayer(fitted={self._fitted}, residual_series={n_resid}, "
            f"cached_pairs={n_pairs}, metadata_players={n_players})"
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_GLOBAL_COPULA: Optional[CopulaLayer] = None


def get_copula(
    oof_df: Optional[pd.DataFrame] = None,
    player_meta_df: Optional[pd.DataFrame] = None,
) -> CopulaLayer:
    """
    Return the module-level singleton CopulaLayer.
    Fits on oof_df if provided on first call.
    """
    global _GLOBAL_COPULA
    if _GLOBAL_COPULA is None:
        _GLOBAL_COPULA = CopulaLayer()
        if oof_df is not None:
            _GLOBAL_COPULA.fit(oof_df, player_meta_df=player_meta_df)
    return _GLOBAL_COPULA


def reset_copula() -> None:
    """Reset singleton (useful in tests)."""
    global _GLOBAL_COPULA
    _GLOBAL_COPULA = None
