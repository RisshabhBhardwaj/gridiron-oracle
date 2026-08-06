"""
ml/shap_service.py

SHAP feature attribution service.

Computes SHAP values for XGBoost / LightGBM model predictions and
maps raw feature column names to plain-English labels for the UI.

CLAUDE.md §3 (API Rules) requires:
  "SHAP factor labels served by the backend must be plain English strings,
   not raw feature names. A FEATURE_LABELS dict must exist in shap_service.py."

ARCHITECTURE
------------
  1. SHAPService.explain(player_features, stat, position)
       → list[FactorAttribution] sorted by |shap_value| descending

  2. If a real model artifact is available (XGB or LGB), uses the real SHAP
     TreeExplainer. Falls back to synthetic signed-weight approximation when
     no model is loaded (early development / dry-run mode).

  3. FEATURE_LABELS maps every feature column to a human-readable label.
     Used by both explain() and the /explain API endpoint.

FEATURE_LABELS convention:
  Keys: exact column name in the feature_matrix table.
  Values: one sentence for UI (starts with capital, no trailing period).
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from backend.app.core.runtime_mode import ArtifactRequiredError, fallbacks_allowed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FEATURE_LABELS — plain English for every feature column
# ---------------------------------------------------------------------------

FEATURE_LABELS: dict[str, str] = {
    # ── Bucket 1: Kalman-Filtered Player Form (35%) ───────────────────────
    # Keys match kalman_est_* / kalman_variance_* column names in feature_matrix.
    "kalman_est_receiving_yards":        "Kalman estimated receiving yards (filtered ability)",
    "kalman_est_receiving_tds":          "Kalman estimated receiving touchdowns (filtered ability)",
    "kalman_est_targets":                "Kalman estimated targets (filtered ability)",
    "kalman_est_receptions":             "Kalman estimated receptions (filtered ability)",
    "kalman_est_target_share":           "Kalman estimated target share (filtered ability)",
    "kalman_est_red_zone_target_share":  "Kalman estimated red zone target share (filtered ability)",
    "kalman_est_air_yards_share":        "Kalman estimated air yards share (filtered ability)",
    "kalman_est_fantasy_ppr":            "Kalman estimated PPR fantasy points (filtered ability)",
    "kalman_est_carries":                "Kalman estimated rush attempts (filtered ability)",
    "kalman_est_rushing_yards":          "Kalman estimated rushing yards (filtered ability)",
    "kalman_est_rushing_tds":            "Kalman estimated rushing touchdowns (filtered ability)",
    "kalman_est_pass_attempts":          "Kalman estimated pass attempts (filtered ability)",
    "kalman_est_completions":            "Kalman estimated completions (filtered ability)",
    "kalman_est_interceptions":          "Kalman estimated interceptions (filtered ability)",
    "kalman_est_fumbles":                "Kalman estimated fumbles (filtered ability)",
    "kalman_est_passing_yards":          "Kalman estimated passing yards (filtered ability)",
    "kalman_est_passing_tds":            "Kalman estimated passing touchdowns (filtered ability)",
    "kalman_variance_receiving_yards":   "Kalman uncertainty in receiving yards estimate",
    "kalman_variance_receiving_tds":     "Kalman uncertainty in receiving touchdowns estimate",
    "kalman_variance_targets":           "Kalman uncertainty in targets estimate",
    "kalman_variance_receptions":        "Kalman uncertainty in receptions estimate",
    "kalman_variance_target_share":      "Kalman uncertainty in target share estimate",
    "kalman_variance_red_zone_target_share": "Kalman uncertainty in red zone target share estimate",
    "kalman_variance_air_yards_share":   "Kalman uncertainty in air yards share estimate",
    "kalman_variance_fantasy_ppr":       "Kalman uncertainty in PPR fantasy points estimate",
    "kalman_variance_carries":           "Kalman uncertainty in rush attempts estimate",
    "kalman_variance_rushing_yards":     "Kalman uncertainty in rushing yards estimate",
    "kalman_variance_rushing_tds":       "Kalman uncertainty in rushing touchdowns estimate",
    "kalman_variance_pass_attempts":     "Kalman uncertainty in pass attempts estimate",
    "kalman_variance_completions":       "Kalman uncertainty in completions estimate",
    "kalman_variance_interceptions":     "Kalman uncertainty in interceptions estimate",
    "kalman_variance_fumbles":           "Kalman uncertainty in fumbles estimate",
    "kalman_variance_passing_yards":     "Kalman uncertainty in passing yards estimate",
    "kalman_variance_passing_tds":       "Kalman uncertainty in passing touchdowns estimate",

    # ── Bucket 2: Season-to-Date Baseline (20%) ──────────────────────────
    # Keys match seas_* column names in feature_matrix (_FM_COLS).
    "seas_games_played":                 "Games played this season",
    "seas_avg_receiving_yards":          "Season average receiving yards per game",
    "seas_avg_targets":                  "Season average targets per game",
    "seas_avg_receptions":               "Season average receptions per game",
    "seas_avg_target_share":             "Season average target share",
    "seas_avg_fantasy_ppr":              "Season average PPR fantasy points per game",
    "seas_yards_per_target":             "Season yards per target",
    "seas_yards_per_reception":          "Season yards per reception (catch efficiency)",
    "seas_avg_carries":                  "Season average rush attempts per game",
    "seas_avg_rushing_yards":            "Season average rushing yards per game",
    "seas_yards_per_carry":              "Season yards per carry attempt",
    "seas_avg_attempts":                 "Season average pass attempts per game",
    "seas_avg_passing_yards":            "Season average passing yards per game",
    "seas_completion_pct":               "Season completion percentage",
    "seas_avg_passing_cpoe":             "Season average completion % over expectation (QB efficiency)",
    "seas_avg_passing_epa":              "Season average passing EPA per game (QB efficiency)",
    "seas_avg_receiving_epa":            "Season average receiving EPA per target (receiver efficiency)",
    "seas_avg_rushing_epa":              "Season average rushing EPA per carry (RB efficiency)",
    "seas_avg_racr":                    "Season average reception air conversion ratio",
    "seas_avg_wopr":                    "Season average weighted opportunity (target + air share)",
    "seas_avg_receiving_yac":            "Season average yards after catch per reception",

    # ── Bucket 3: Direct Matchup (20%) ───────────────────────────────────
    # Keys match opp_* column names in feature_matrix (_FM_COLS).
    "opp_avg_receiving_yards_allowed":   "Opponent average receiving yards allowed per game",
    "opp_avg_targets_allowed":           "Opponent average targets allowed per game",
    "opp_avg_tds_allowed":               "Opponent average touchdowns allowed per game",
    "opp_avg_fantasy_ppr_allowed":       "Opponent average PPR fantasy points allowed per game",
    "opp_avg_rushing_yards_allowed":     "Opponent average rushing yards allowed per game",

    # ── Bucket 4: Weather & Venue (10%) ──────────────────────────────────
    # Keys match temp_f, wind_mph, is_dome, surface_turf, temp_bucket, wind_bucket in _FM_COLS.
    "temp_f":                            "Game-day temperature (degrees Fahrenheit)",
    "wind_mph":                          "Game-day wind speed (miles per hour)",
    "is_dome":                           "Game played in a dome (weather-neutral)",
    "surface_turf":                      "Artificial turf playing surface",
    "temp_bucket":                       "Temperature bucket (cold / cool / mild / warm)",
    "wind_bucket":                       "Wind bucket (calm / breezy / windy)",

    # ── Bucket 5: Team Context (5%) ──────────────────────────────────────
    # Keys match game_total_line, spread_line, is_home in _FM_COLS.
    "game_total_line":                   "Vegas game over/under total",
    "spread_line":                       "Vegas point spread (positive = home favored)",
    "is_home":                           "Player's team is the home team",

    # ── Bucket 6: Rest & Schedule (5%) ───────────────────────────────────
    # Keys match days_rest, is_short_week, is_bye_prior in _FM_COLS.
    "days_rest":                         "Days of rest since last game",
    "is_short_week":                     "Short week flag (fewer than 6 days rest)",
    "is_bye_prior":                      "Coming off a bye week (extra rest)",

    # ── Bucket 7: Rule & Meta Shifts (5%) ────────────────────────────────
    # Key matches rule_coeff in _FM_COLS.
    "rule_coeff":                        "Season-level rule emphasis coefficient",

    # ── Injury & availability ─────────────────────────────────────────────
    "injury_status_encoded":             "Injury report status (0=Out, 4=Full participant)",
    "games_missed_streak":               "Consecutive games missed due to injury",

    # ── Snap participation ────────────────────────────────────────────────
    "snap_pct_off":                      "Offensive snap participation rate (0-1 scale)",

    # ── Identifiers and game context ─────────────────────────────────────
    "player_id":                         "Player unique identifier (gsis_id)",
    "game_id":                           "Game unique identifier",
    "season":                            "NFL season year",
    "week":                              "NFL week number",
    "position":                          "Player position (WR/RB/TE/QB)",
    "team":                              "Player's current team",
    "opponent_team":                     "Opponent team this week",

    # ── Bucket 3 Extended: new matchup columns ────────────────────────────────
    "opp_avg_carries_allowed":           "Opponent average rush attempts allowed per game",
    "opp_avg_completions_allowed":       "Opponent average completions allowed per game",
    "opp_avg_pass_attempts_allowed":     "Opponent average pass attempts allowed per game",

    # ── Bucket 4 Extended: precipitation ─────────────────────────────────────
    "precipitation_bucket":              "Precipitation level (0=none, 1=rain, 2=snow)",

    # ── Bucket 11: Play-by-Play derived features ──────────────────────────────
    "epa_per_play":                      "Expected points added per offensive play",
    "adot":                              "Average depth of target (yards past line of scrimmage)",
    "ol_pressure_rate":                  "Offensive line pressure rate allowed (fraction of pass plays)",
    "yac_per_reception":                 "Yards after contact per reception",
    # Depth chart + Next Gen Stats
    "depth_chart_rank":                  "Official depth chart rank (1=WR1, 2=WR2)",
    "avg_separation":                    "Average separation from defender at target (Next Gen)",
    "avg_cushion":                       "Pre-snap cushion distance from defender (Next Gen)",
}


def label(feature: str) -> str:
    """
    Return the human-readable label for a feature column.

    Falls back to formatted raw name if not in FEATURE_LABELS.
    """
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    return feature.replace("_", " ").title()


# ---------------------------------------------------------------------------
# FactorAttribution
# ---------------------------------------------------------------------------

@dataclass
class FactorAttribution:
    """One SHAP factor attribution for the /explain response."""
    feature: str    # raw column name
    label: str      # human-readable label (from FEATURE_LABELS)
    impact: float   # SHAP value (positive = pushes projection up)
    value: float    # actual feature value for this player


@dataclass
class SHAPExplanation:
    """SHAP attributions plus provenance for the caller/UI."""
    attributions: list[FactorAttribution]
    source: str


# ---------------------------------------------------------------------------
# SHAPService
# ---------------------------------------------------------------------------

class SHAPService:
    """
    Computes SHAP feature attributions for a player projection.

    Attempts to load a real XGBoost or LightGBM model from MLflow.
    Falls back to a signed-weight synthetic approximation when no
    model artifact is available (early dev / dry-run mode).
    """

    def __init__(self, mlflow_tracking_uri: str = "") -> None:
        self._mlflow_uri = mlflow_tracking_uri
        self._model_cache: dict[str, object] = {}
        self._mlflow_available: Optional[bool] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        features: dict,
        stat: str,
        position: str,
        top_n: int = 5,
    ) -> SHAPExplanation:
        """
        Return the top_n SHAP attributions for a single player projection.

        Args:
            features:  Dict of feature_name → value for one player.
            stat:      Target stat (e.g. "receiving_yards").
            position:  Player position ("WR"/"RB"/"TE"/"QB").
            top_n:     Number of factors to return (sorted by |impact|).

        Returns:
            SHAPExplanation containing sorted attributions and provenance.
        """
        model = self._load_model(stat, position)

        if model is not None:
            return self._real_shap(model, features, stat, position, top_n)
        if not fallbacks_allowed():
            raise ArtifactRequiredError(
                "SHAP model artifact unavailable while PRODUCT_MODE=artifact_backed"
            )
        return SHAPExplanation(
            attributions=self._synthetic_shap(features, stat, position, top_n),
            source="synthetic_fallback",
        )

    # ------------------------------------------------------------------
    # Real SHAP (when model artifact is available)
    # ------------------------------------------------------------------

    def _load_model(self, stat: str, position: str) -> Optional[object]:
        """
        Try to load an XGBoost model from MLflow.
        Returns None if unavailable (graceful degradation).
        """
        cache_key = f"{stat}_{position}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        if not self._mlflow_uri:
            self._model_cache[cache_key] = None
            return None

        if not self._mlflow_reachable():
            self._model_cache[cache_key] = None
            return None

        try:
            import mlflow
            mlflow.set_tracking_uri(self._mlflow_uri)
            client = mlflow.tracking.MlflowClient()

            registered_models = client.search_registered_models(
                filter_string=f"name LIKE 'xgb_{stat}%'"
            )
            if not registered_models:
                self._model_cache[cache_key] = None
                return None

            model_name = registered_models[0].name
            latest = client.get_latest_versions(model_name, stages=["Production", "None"])
            if not latest:
                self._model_cache[cache_key] = None
                return None

            run_id = latest[0].run_id
            model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
            self._model_cache[cache_key] = model
            logger.info("Loaded XGB model for %s/%s from MLflow run %s", stat, position, run_id)
            return model

        except Exception as exc:
            logger.debug("MLflow model load failed (%s) — using synthetic SHAP", exc)
            self._model_cache[cache_key] = None
            return None

    def _mlflow_reachable(self) -> bool:
        """Fast preflight so missing local MLflow does not stall request paths."""
        if self._mlflow_available is not None:
            return self._mlflow_available

        parsed = urlparse(self._mlflow_uri)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            self._mlflow_available = True
            return True

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((parsed.hostname, port), timeout=0.25):
                self._mlflow_available = True
        except OSError:
            logger.debug(
                "MLflow server %s is unreachable — using synthetic SHAP fallback",
                self._mlflow_uri,
            )
            self._mlflow_available = False
        return self._mlflow_available

    def _real_shap(
        self,
        model,
        features: dict,
        stat: str,
        position: str,
        top_n: int,
    ) -> SHAPExplanation:
        """Compute real SHAP values using shap.TreeExplainer."""
        try:
            import shap

            feat_cols = sorted(k for k in features if isinstance(features[k], (int, float)))
            feat_vals = np.array([[features[c] for c in feat_cols]], dtype=float)
            feat_df = pd.DataFrame(feat_vals, columns=feat_cols)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(feat_df)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            attributions = [
                FactorAttribution(
                    feature=col,
                    label=label(col),
                    impact=round(float(shap_values[0, i]), 3),
                    value=float(features[col]),
                )
                for i, col in enumerate(feat_cols)
            ]
            attributions.sort(key=lambda a: abs(a.impact), reverse=True)
            return SHAPExplanation(
                attributions=attributions[:top_n],
                source="model_artifact",
            )

        except Exception as exc:
            logger.warning("Real SHAP failed (%s) — falling back to synthetic", exc)
            return SHAPExplanation(
                attributions=self._synthetic_shap(features, stat=stat, position=position, top_n=top_n),
                source="synthetic_fallback",
            )

    # ------------------------------------------------------------------
    # Synthetic SHAP (graceful fallback when no model artifact available)
    # ------------------------------------------------------------------

    def _synthetic_shap(
        self,
        features: dict,
        stat: str,
        position: str,
        top_n: int,
    ) -> list[FactorAttribution]:
        """
        Signed-weight approximation when no real model is available.

        Uses known feature importance priors (bucket weights from §5.1)
        scaled by feature values. Returns approximate SHAP-style attributions
        for UI display, clearly marked as estimates.
        """
        # Bucket weights from CLAUDE.md §5.1
        # Prefix → bucket weight (bucket names match _FM_COLS column prefixes).
        _BUCKET_WEIGHTS: dict[str, float] = {
            "kalman_est_":  0.35,
            "kalman_var":   0.05,   # variance columns (sub-bucket of Kalman)
            "seas_":        0.20,
            "opp_":         0.20,
            "temp_":        0.03,   # Bucket 4 (weather)
            "wind_":        0.03,
            "is_dome":      0.02,
            "surface":      0.02,
            "game_total":   0.02,   # Bucket 5 (context)
            "spread_":      0.02,
            "is_home":      0.01,
            "days_rest":    0.02,   # Bucket 6 (rest)
            "is_short":     0.02,
            "is_bye":       0.01,
            "rule_":        0.05,   # Bucket 7
        }
        # Stat-specific: Kalman estimate for the target stat should dominate
        primary_kalman = f"kalman_est_{stat}"

        rng = np.random.default_rng(abs(hash(f"{stat}_{position}")) % 2**31)

        attributions: list[FactorAttribution] = []
        for feat, val in features.items():
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if np.isnan(fval):
                continue

            # Feature importance weight
            weight = 0.01
            for prefix, w in _BUCKET_WEIGHTS.items():
                if feat.startswith(prefix):
                    weight = w
                    break

            # Primary target stat Kalman estimate: strongest positive factor
            if feat == primary_kalman:
                impact = weight * max(fval / 100.0, 0.01) * rng.uniform(0.5, 1.5)
            else:
                sign = rng.choice([-1.0, 1.0], p=[0.35, 0.65])
                impact = sign * weight * abs(fval) * rng.uniform(0.01, 0.15)

            attributions.append(FactorAttribution(
                feature=feat,
                label=f"Estimated: {label(feat)}",
                impact=round(float(impact), 3),
                value=round(fval, 3),
            ))

        attributions.sort(key=lambda a: abs(a.impact), reverse=True)
        return attributions[:top_n]
