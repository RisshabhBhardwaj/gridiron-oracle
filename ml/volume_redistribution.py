"""
ml/volume_redistribution.py

Volume Redistribution Engine — Hierarchical Zero-Sum Model for Injury Cascades.

PROBLEM
-------
The ensemble predicts Justin Jefferson and Jordan Addison independently. If
Jefferson gets injured, the base model doesn't know where his 30% target share
goes. Sample independent posteriors from two PyMC models and you'll generate
physically impossible game scripts (e.g., QB throws 35 passes, WR1 gets 18
targets and WR2 gets 16 targets = 34 targets from 35 passes, impossible).

SOLUTION — HIERARCHICAL ZERO-SUM MODEL
---------------------------------------
Layer 1: TeamVolumePredictor
    Predict team total offensive pass volume V (pass attempts per game) as a
    function of: spread_line, total_line, opponent rushing yards allowed,
    team pass-heavy tendency (seas_avg_attempts). This is the constraint budget.

Layer 2: DirichletShareDistributor
    Model each player's target share as a Dirichlet distribution where the
    concentration parameters α_i are proportional to their historical target
    share estimates (from Kalman filter).

    When player_i is injured/OUT, set α_i → 0 (drop from distribution).
    The Dirichlet constraint forces the remaining shares to sum to 1.0,
    automatically redistributing the injured player's α weight.

MATHEMATICS
-----------
Let players p_1, ..., p_n have Kalman target share estimates ts_i ∈ [0, 1].

Concentration parameters:
    α_i = max(ts_i × ALPHA_SCALE, ALPHA_MIN)    if player_i is active
    α_i = 0                                       if player_i is injured/OUT

Sample share vector:
    (s_1, ..., s_n) ~ Dirichlet(α_1, ..., α_n)   [sums to 1 by construction]

Expected targets for player_i:
    E[targets_i] = s_i × V_team

where V_team is the team's predicted pass volume (Layer 1).

USAGE
-----
    from ml.volume_redistribution import VolumeRedistributor

    vr = VolumeRedistributor()
    vr.fit(historical_df)    # optional: learn position-level pass volumes

    # Projection week with Jefferson injured:
    result = vr.redistribute(
        team="MIN",
        players=[
            {"player_id": "00-0038877", "name": "Justin Jefferson",  "position": "WR",
             "kalman_est_target_share": 0.30, "injury_status": "OUT"},
            {"player_id": "00-0039162", "name": "Jordan Addison",    "position": "WR",
             "kalman_est_target_share": 0.15, "injury_status": None},
            {"player_id": "00-0035640", "name": "T.J. Hockenson",    "position": "TE",
             "kalman_est_target_share": 0.13, "injury_status": None},
        ],
        game_context={"spread_line": -3.0, "total_line": 48.5, "is_home": 1},
        n_samples=2_000,
    )
    # result["jordan_addison"]["expected_targets"]  → ~9.2 (was 6.5 pre-injury)
    # result["jordan_addison"]["target_share_mean"]  → ~0.26 (was 0.15)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Dirichlet concentration scaling factor. Higher = tighter around the prior mean.
# At ALPHA_SCALE=50: std of each share ≈ sqrt(ts*(1-ts)/51) ≈ small variance.
# At ALPHA_SCALE=10: more diffuse — wider redistribution on injury.
ALPHA_SCALE: float = 20.0

# Minimum concentration to keep a non-injured player in the distribution.
ALPHA_MIN:   float = 0.5

# Position-average pass attempts per game — fallback when no historical data.
# Source: 2019-2024 nflreadpy pass attempt averages per team-game.
_DEFAULT_TEAM_PASS_VOLUME: dict[str, float] = {
    "QB": 35.0,  # team pass attempts per game (QB = team level)
}
_DEFAULT_TEAM_RUSH_VOLUME: dict[str, float] = {
    "RB": 24.0,  # team rush attempts per game
}

# Game script adjustment coefficients for Layer 1 volume prediction.
# spread_line convention (nflreadpy): negative = home team favored, positive = home team underdog.
# Underdogs trail game scripts → throw more; favorites run to protect lead → throw less.
# effective_spread = spread_line for home team, -spread_line for away team.
#   effective_spread > 0 → team is underdog → more passes (positive adjustment)
#   effective_spread < 0 → team is favored  → fewer passes (negative adjustment)
_SPREAD_PASS_COEFFICIENT:  float =  0.35  # each point of underdog spread → +0.35 pass attempts
_TOTAL_PASS_COEFFICIENT:   float =  0.25  # each point of game total → more passing

# Injury status values that remove a player from the Dirichlet distribution.
INACTIVE_STATUSES = frozenset({"out", "dnp", "ir", "injured reserve", "o", "0"})

# Positions eligible for target/carry share allocation. Measured against
# realized game_logs.target_share (season 2024): this group carries 99.89% of
# a team's target mass on average (median leakage to non-skill positions is
# exactly 0, 75th percentile 0 — trick plays / eligible-lineman targets are
# the only source, and they're rare). Including OL/DL/DB rows in the
# Dirichlet group would give each of them a nonzero ALPHA_MIN floor
# concentration despite a real share of ~0, diluting every skill player's
# allocation with noise from players who structurally cannot be targeted.
SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE", "FB"})

# Doubtful/questionable: keep in distribution but reduce concentration.
_DOUBTFUL_MULTIPLIER:      float = 0.30   # 30% of normal concentration
_QUESTIONABLE_MULTIPLIER:  float = 0.65   # 65% of normal concentration
_LIMITED_MULTIPLIER:       float = 0.85   # 85% of normal concentration


# ── Utility ───────────────────────────────────────────────────────────────────

def _injury_multiplier(injury_status: Optional[str]) -> float:
    """
    Convert an injury_status string into a Dirichlet concentration multiplier.

    Returns 0.0 for inactive players (removed from distribution).
    Returns 1.0 for healthy/None (full concentration).
    """
    if injury_status is None:
        return 1.0
    s = str(injury_status).lower().strip()
    if s == "":
        return 1.0
    if s in INACTIVE_STATUSES:
        return 0.0
    if "doubtful" in s:
        return _DOUBTFUL_MULTIPLIER
    if "questionable" in s:
        return _QUESTIONABLE_MULTIPLIER
    if "limited" in s:
        return _LIMITED_MULTIPLIER
    return 1.0


# ── Layer 1: Team Volume Model ─────────────────────────────────────────────────

class TeamVolumePredictor:
    """
    Layer 1: Predict team pass/rush attempt volume per game.

    Uses a simple linear model fit on historical game_logs:
        pass_volume = β₀ + β₁×spread_line + β₂×total_line + β₃×team_tendency

    where team_tendency is the team's historical pass-heavy rate (avg attempts/gm).

    Falls back to position-average constants when no historical data is available.
    """

    def __init__(self) -> None:
        self._team_pass_avg: dict[str, float] = {}   # team → historical pass vol
        self._team_rush_avg: dict[str, float] = {}   # team → historical rush vol
        self._global_pass_avg: float = 35.0
        self._global_rush_avg: float = 24.0
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "TeamVolumePredictor":
        """
        Fit team-level pass and rush attempt averages from historical game_logs.

        Args:
            df: DataFrame with columns [team, attempts, carries, season].
                Typically the full feature_matrix or game_logs loaded via
                ml.utils.load_feature_matrix().
        """
        required = {"team", "attempts", "carries"}
        if not required.issubset(df.columns):
            logger.warning(
                "TeamVolumePredictor.fit(): missing columns %s. Using defaults.",
                required - set(df.columns),
            )
            return self

        # Aggregate per (team, season, game) to get team-level attempts per game.
        # Use seas_avg_attempts if available (already aggregated), else raw carries.
        team_groups = df.groupby("team")

        for team, grp in team_groups:
            pass_vals = grp["attempts"].dropna()
            rush_vals  = grp["carries"].dropna()
            # Only compute from rows where player IS a QB / RB (else inflate)
            # Use average at team-game level — group by (team, game_id) first
            if "game_id" in grp.columns:
                gb = grp.groupby("game_id")
                team_pass = gb["attempts"].sum().mean()
                team_rush  = gb["carries"].sum().mean()
            else:
                team_pass = float(pass_vals.mean()) if len(pass_vals) else 35.0
                team_rush  = float(rush_vals.mean()) if len(rush_vals) else 24.0

            self._team_pass_avg[str(team)] = team_pass
            self._team_rush_avg[str(team)] = team_rush

        if self._team_pass_avg:
            self._global_pass_avg = float(np.mean(list(self._team_pass_avg.values())))
            self._global_rush_avg = float(np.mean(list(self._team_rush_avg.values())))

        self._fitted = True
        logger.info(
            "TeamVolumePredictor fitted: %d teams. "
            "Global avg pass=%.1f rush=%.1f per game.",
            len(self._team_pass_avg),
            self._global_pass_avg,
            self._global_rush_avg,
        )
        return self

    def predict_pass_volume(
        self,
        team: str,
        game_context: Optional[dict] = None,
    ) -> float:
        """
        Predict team pass attempts for this game.

        Layer 1 adjustment:
            V = V_team_avg
                + β₁ × spread_line      (negative spread → team favored → fewer passes)
                + β₂ × (total_line - 45) (higher total → more passing)

        Args:
            team:         Team abbreviation ("MIN", "GB", etc.)
            game_context: dict with optional spread_line, total_line, is_home.

        Returns:
            Predicted pass attempt volume (float, ≥ 1.0).
        """
        base = self._team_pass_avg.get(team, self._global_pass_avg)

        if game_context:
            spread = float(game_context.get("spread_line") or 0.0)
            total  = float(game_context.get("total_line") or 45.0)
            is_home = int(game_context.get("is_home") or 0)

            # Away team's spread is the negative of home spread in nflreadpy convention.
            effective_spread = spread if is_home else -spread

            adjustment = (
                _SPREAD_PASS_COEFFICIENT * effective_spread
                + _TOTAL_PASS_COEFFICIENT * (total - 45.0)
            )
            base = base + adjustment

        return max(base, 1.0)

    def predict_rush_volume(self, team: str, game_context: Optional[dict] = None) -> float:
        """Predict team rush attempts for this game."""
        base = self._team_rush_avg.get(team, self._global_rush_avg)

        if game_context:
            spread = float(game_context.get("spread_line") or 0.0)
            is_home = int(game_context.get("is_home") or 0)
            effective_spread = spread if is_home else -spread
            # Bigger favorites run more (clock management)
            adjustment = -_SPREAD_PASS_COEFFICIENT * effective_spread * 0.5
            base = base + adjustment

        return max(base, 1.0)


# ── Layer 2: Dirichlet Share Distributor ──────────────────────────────────────

class DirichletShareDistributor:
    """
    Layer 2: Sample target/carry share distributions via Dirichlet.

    Converts Kalman-estimated target shares into Dirichlet concentrations,
    removes injured players (concentration → 0), and samples n_samples share
    vectors. Each sample represents one plausible game-week share allocation.

    The Dirichlet constraint guarantees sum(shares) = 1.0 in every sample.
    """

    @staticmethod
    def compute_concentrations(
        players: list[dict],
        stat_key: str = "kalman_est_target_share",
    ) -> np.ndarray:
        """
        Build Dirichlet concentration vector αᵢ for each player.

        Args:
            players:  List of player dicts. Required keys:
                        - kalman_est_target_share (or stat_key)
                        - injury_status (str or None)
            stat_key: Which Kalman estimate to use as the prior share.

        Returns:
            np.ndarray shape (n_players,) — concentration parameters.
            Active players have α > 0. Inactive players have α = 0.
        """
        alphas = np.zeros(len(players), dtype=float)
        for i, p in enumerate(players):
            share_est = float(p.get(stat_key) or 0.0)
            inj_mult  = _injury_multiplier(p.get("injury_status"))

            if inj_mult == 0.0:
                alphas[i] = 0.0          # removed from distribution
            else:
                alpha_raw  = max(share_est * ALPHA_SCALE, ALPHA_MIN)
                alphas[i]  = alpha_raw * inj_mult

        # Normalize: if ALL players are injured (shouldn't happen), use uniform.
        if alphas.sum() == 0.0:
            logger.warning("All players have zero concentration — using uniform fallback.")
            alphas[:] = ALPHA_MIN

        return alphas

    @staticmethod
    def sample_shares(
        alphas: np.ndarray,
        n_samples: int = 2_000,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample share vectors from Dirichlet(α).

        Args:
            alphas:    Concentration vector (n_players,). Zeros treated specially.
            n_samples: Number of Monte Carlo samples.
            rng:       NumPy random Generator (for reproducibility in tests).

        Returns:
            np.ndarray shape (n_samples, n_players) — each row sums to 1.0.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Dirichlet is undefined for αᵢ=0. We handle zero-concentration players
        # by sampling only the active subset, then re-inserting zeros.
        active_mask = alphas > 0.0
        n_active    = active_mask.sum()

        if n_active == 0:
            return np.zeros((n_samples, len(alphas)))

        active_alphas = alphas[active_mask]
        # np.random.Generator.dirichlet: shape = (n_samples, n_active)
        active_samples = rng.dirichlet(active_alphas, size=n_samples)

        # Re-insert inactive columns as zero
        full_samples = np.zeros((n_samples, len(alphas)))
        active_indices = np.where(active_mask)[0]
        for col_out, col_in in enumerate(active_indices):
            full_samples[:, col_in] = active_samples[:, col_out]

        return full_samples


# ── Main API ──────────────────────────────────────────────────────────────────

class VolumeRedistributor:
    """
    Combines TeamVolumePredictor (Layer 1) and DirichletShareDistributor (Layer 2)
    to produce injury-aware player-level volume projections.

    The key guarantee: projected targets(WR1) + projected targets(WR2) + ...
    ≤ team_pass_volume.  On WR1 injury, the Dirichlet redistribution ensures
    remaining targets are reallocated mathematically (not just scaled uniformly).

    Example
    -------
        vr = VolumeRedistributor()
        vr.fit(game_logs_df)

        projections = vr.redistribute(
            team="MIN",
            players=[
                {"player_id": "xx", "name": "Jefferson",  "position": "WR",
                 "kalman_est_target_share": 0.30, "injury_status": "OUT"},
                {"player_id": "yy", "name": "Addison",    "position": "WR",
                 "kalman_est_target_share": 0.15, "injury_status": None},
            ],
            game_context={"spread_line": -3.0, "total_line": 48.5, "is_home": 1},
        )
        # projections["yy"]["expected_targets"] → ~9+ (redistributed from Jefferson)
    """

    def __init__(self) -> None:
        self.volume_model = TeamVolumePredictor()
        self.share_model  = DirichletShareDistributor()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "VolumeRedistributor":
        """
        Fit team-level volume averages from historical game_logs or feature_matrix.

        Args:
            df: DataFrame with columns [team, attempts, carries] at minimum.
        """
        self.volume_model.fit(df)
        self._fitted = True
        return self

    def redistribute(
        self,
        team: str,
        players: list[dict],
        game_context: Optional[dict] = None,
        n_samples: int = 2_000,
        rng: Optional[np.random.Generator] = None,
        stat: str = "receiving",  # "receiving" or "rushing"
    ) -> dict[str, dict[str, Any]]:
        """
        Compute injury-aware target/carry projections for a group of players.

        Args:
            team:         Team abbreviation.
            players:      List of player dicts. Each must have:
                            - player_id (str)
                            - name (str)
                            - position (str)
                            - kalman_est_target_share (float, for stat="receiving")
                            - kalman_est_carries (float, for stat="rushing")
                            - injury_status (str or None)
            game_context: Optional dict with spread_line, total_line, is_home.
            n_samples:    Monte Carlo draws from the Dirichlet.
            rng:          NumPy Generator for reproducibility.
            stat:         "receiving" → target share / pass volume.
                          "rushing"   → carry share / rush volume.

        Returns:
            dict mapping player_id → {
                "name":               str,
                "injury_status":      str or None,
                "is_active":          bool,
                "target_share_mean":  float,
                "target_share_p10":   float,
                "target_share_p50":   float,
                "target_share_p90":   float,
                "expected_targets":   float,   (or expected_carries)
                "targets_p10":        float,
                "targets_p50":        float,
                "targets_p90":        float,
            }
        """
        if stat == "rushing":
            vol_label = "expected_carries"
            vol_p10   = "carries_p10"
            vol_p50   = "carries_p50"
            vol_p90   = "carries_p90"
            share_label = "carry_share_mean"
            share_p10   = "carry_share_p10"
            share_p50   = "carry_share_p50"
            share_p90   = "carry_share_p90"
            # For rushing share: total carries / team rush volume
            team_volume = self.volume_model.predict_rush_volume(team, game_context)
            # Normalise by team rush volume denominator to get share
            # (kalman_est_carries is in raw carries, not fraction)
            # Convert raw carries to approximate share
            _v = team_volume if team_volume > 0 else 24.0
            for p in players:
                if "kalman_est_target_share" not in p:
                    raw_carries = float(p.get("kalman_est_carries") or 0.0)
                    p["_share_for_dirichlet"] = raw_carries / _v
                else:
                    p["_share_for_dirichlet"] = float(p.get("kalman_est_carries") or 0.0) / _v
            stat_key_for_dir = "_share_for_dirichlet"
        else:
            vol_label   = "expected_targets"
            vol_p10     = "targets_p10"
            vol_p50     = "targets_p50"
            vol_p90     = "targets_p90"
            share_label = "target_share_mean"
            share_p10   = "target_share_p10"
            share_p50   = "target_share_p50"
            share_p90   = "target_share_p90"
            team_volume = self.volume_model.predict_pass_volume(team, game_context)
            stat_key_for_dir = "kalman_est_target_share"

        if not players:
            logger.warning("VolumeRedistributor.redistribute(): empty players list.")
            return {}

        # Count active players BEFORE sampling so we can log clearly.
        n_injured = sum(
            1 for p in players
            if _injury_multiplier(p.get("injury_status")) == 0.0
        )
        if n_injured:
            injured_names = [
                p.get("name", p.get("player_id"))
                for p in players
                if _injury_multiplier(p.get("injury_status")) == 0.0
            ]
            logger.info(
                "VolumeRedistributor: %d/%d players inactive (%s). "
                "Redistributing %.1f %s volume among %d remaining players.",
                n_injured, len(players),
                ", ".join(str(n) for n in injured_names),
                team_volume, "targets" if stat == "receiving" else "carries",
                len(players) - n_injured,
            )

        # Layer 2: sample share distributions
        alphas  = DirichletShareDistributor.compute_concentrations(players, stat_key_for_dir)
        samples = DirichletShareDistributor.sample_shares(alphas, n_samples=n_samples, rng=rng)
        # samples shape: (n_samples, n_players) — each row sums to 1.0

        # Multiply shares by team volume to get per-player volume distributions
        volume_samples = samples * team_volume  # broadcast: (n_samples, n_players)

        out: dict[str, dict[str, Any]] = {}
        for i, p in enumerate(players):
            pid = str(p.get("player_id", f"player_{i}"))
            inj  = _injury_multiplier(p.get("injury_status"))

            share_col   = samples[:, i]        # shape (n_samples,)
            volume_col  = volume_samples[:, i] # shape (n_samples,)

            out[pid] = {
                "name":          p.get("name", pid),
                "position":      p.get("position"),
                "injury_status": p.get("injury_status"),
                "is_active":     inj > 0.0,
                share_label:     float(share_col.mean()),
                share_p10:       float(np.percentile(share_col, 10)),
                share_p50:       float(np.percentile(share_col, 50)),
                share_p90:       float(np.percentile(share_col, 90)),
                vol_label:       float(volume_col.mean()),
                vol_p10:         float(np.percentile(volume_col, 10)),
                vol_p50:         float(np.percentile(volume_col, 50)),
                vol_p90:         float(np.percentile(volume_col, 90)),
                "team_volume":   team_volume,
            }

        return out

    def redistribute_team_projections(
        self,
        projections_df: pd.DataFrame,
        injury_report: dict[str, str],
        game_context_by_team: Optional[dict[str, dict]] = None,
        n_samples: int = 2_000,
    ) -> pd.DataFrame:
        """
        Apply volume redistribution to a DataFrame of weekly projections.

        This is the main integration point with ml/train.py. Call this after
        the stacking ensemble produces raw projections, before the Bayesian layer.

        This is the L3 allocation share model (Phase 6), not just an injury
        patch: it runs unconditionally, every week, for every team —
        `injury_report={}` is a legitimate call meaning "everyone healthy,"
        not a signal to skip. The Dirichlet constraint is what makes every
        team-game's shares sum to exactly 1.0 by construction; skipping this
        step (the old behavior when injury_report was empty) left raw,
        independently-estimated Kalman shares in place, which do not sum to
        anything in particular.

        Args:
            projections_df:  DataFrame with columns:
                               [player_id, name, team, position,
                                kalman_est_target_share, kalman_est_carries,
                                projected_targets, projected_receiving_yards, ...]
                              Rows outside SKILL_POSITIONS pass through
                              unmodified — they never entered the Dirichlet
                              group and keep whatever projection they arrived
                              with.
            injury_report:   {player_id: injury_status} dict from EspnAdapter.
                               Any player_id not in this dict is assumed healthy.
            game_context_by_team: {team: {spread_line, total_line, is_home}} per team.
            n_samples:       Dirichlet samples per team.

        Returns:
            DataFrame with same structure but with redistributed projected_targets
            and projected_carries values. Original projections preserved in
            _raw_projected_targets and _raw_projected_carries columns.
        """
        if projections_df.empty:
            return projections_df

        df = projections_df.copy()

        # Preserve originals
        for col in ("projected_targets", "projected_carries"):
            if col in df.columns:
                df[f"_raw_{col}"] = df[col]

        # Add injury status from report
        # Note: fillna(value=pd.NA) is required in pandas ≥2.0 (fillna(None) raises ValueError)
        df["injury_status"] = df["player_id"].map(injury_report).where(
            df["player_id"].isin(injury_report.keys()),
            other=None,
        )

        game_context_by_team = game_context_by_team or {}
        updated_rows: list[dict] = []

        for team, grp in df.groupby("team"):
            game_ctx = game_context_by_team.get(str(team))
            skill_grp = grp[grp["position"].astype(str).str.upper().isin(SKILL_POSITIONS)]

            # ── Receiving redistribution ──────────────────────────────────────
            recv_players = [
                {
                    "player_id":               row["player_id"],
                    "name":                    row.get("name", row["player_id"]),
                    "position":                row.get("position"),
                    "kalman_est_target_share": row.get("kalman_est_target_share"),
                    "injury_status":           row.get("injury_status"),
                }
                for _, row in skill_grp.iterrows()
            ]
            recv_result = self.redistribute(
                team=str(team),
                players=recv_players,
                game_context=game_ctx,
                n_samples=n_samples,
                stat="receiving",
            )

            # ── Rushing redistribution ────────────────────────────────────────
            rush_players = [
                {
                    "player_id":       row["player_id"],
                    "name":            row.get("name", row["player_id"]),
                    "position":        row.get("position"),
                    "kalman_est_carries": row.get("kalman_est_carries"),
                    "injury_status":   row.get("injury_status"),
                }
                for _, row in skill_grp.iterrows()
            ]
            rush_result = self.redistribute(
                team=str(team),
                players=rush_players,
                game_context=game_ctx,
                n_samples=n_samples,
                stat="rushing",
            )

            for _, row in grp.iterrows():
                pid = str(row["player_id"])
                row_dict = dict(row)

                if pid in recv_result:
                    row_dict["projected_targets"]      = recv_result[pid]["expected_targets"]
                    row_dict["target_share_mean"]      = recv_result[pid]["target_share_mean"]
                    row_dict["target_share_p10"]       = recv_result[pid]["target_share_p10"]
                    row_dict["target_share_p90"]       = recv_result[pid]["target_share_p90"]

                if pid in rush_result:
                    row_dict["projected_carries"]      = rush_result[pid]["expected_carries"]
                    row_dict["carry_share_mean"]       = rush_result[pid].get("carry_share_mean")

                updated_rows.append(row_dict)

        return pd.DataFrame(updated_rows)


# ── Convenience accessor ───────────────────────────────────────────────────────

_GLOBAL_VR: Optional[VolumeRedistributor] = None


def get_redistributor(df: Optional[pd.DataFrame] = None) -> VolumeRedistributor:
    """
    Return a module-level singleton VolumeRedistributor.
    Optionally fit it on the provided DataFrame on first call.

    This is the preferred way to access the redistributor from train.py and
    backtest.py without re-fitting on every call.
    """
    global _GLOBAL_VR
    if _GLOBAL_VR is None:
        _GLOBAL_VR = VolumeRedistributor()
        if df is not None:
            _GLOBAL_VR.fit(df)
    return _GLOBAL_VR
