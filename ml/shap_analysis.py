"""
ml/shap_analysis.py

SHAP Analysis Module — Model Interpretability for Gridiron Oracle.

PURPOSE
-------
After training XGBoost, LightGBM, and TFT models, we need to understand:
  1. Which features drive each prediction? (Global feature importance)
  2. For specific player-week predictions: why did the model predict X yards?
  3. Are there biases by position, team, or week?

This module wraps `shap.TreeExplainer` for XGB/LGB models and provides:
  - Global SHAP summary reports
  - Per-player prediction explanations
  - Feature importance CSV exports
  - SHAP beeswarm and bar plots

USAGE
-----
    from ml.shap_analysis import SHAPAnalyzer

    analyzer = SHAPAnalyzer(model_path="ml/models/xgb_receiving_yards.ubj",
                            model_type="xgb")
    report = analyzer.run(feature_df=feature_matrix_df, stat="receiving_yards")
    report.save("ml/shap_reports/xgb_receiving_yards/")

    # For a specific player:
    explanation = analyzer.explain_prediction(
        feature_row=feature_matrix_df.iloc[42],
        stat="receiving_yards"
    )
    print(explanation.to_string())
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.utils import FEATURE_COLS

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_TYPE_XGB  = "xgb"
MODEL_TYPE_LGB  = "lgb"
MODEL_TYPE_AUTO = "auto"

_SUPPORTED_MODEL_TYPES = {MODEL_TYPE_XGB, MODEL_TYPE_LGB}


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class SHAPReport:
    """
    Container for SHAP analysis results.

    Attributes:
        importance_df:  DataFrame[feature, mean_abs_shap, rank] sorted by importance
        shap_values:    np.ndarray (n_samples, n_features) of SHAP values
        feature_names:  list of feature column names matching shap_values
        model_type:     "xgb" or "lgb"
        stat:           target stat name (e.g. "receiving_yards")
    """
    importance_df:  pd.DataFrame
    shap_values:    np.ndarray
    feature_names:  list[str]
    model_type:     str
    stat:           str
    n_rows:         int

    def top_features(self, n: int = 20) -> pd.DataFrame:
        """Return the top N most important features."""
        return self.importance_df.head(n)

    def save(self, out_dir: str) -> None:
        """
        Save the SHAP report to out_dir/:
          - shap_importance.csv
          - shap_summary_beeswarm.png   (requires matplotlib + shap)
          - shap_summary_bar.png        (requires matplotlib + shap)
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        csv_path = Path(out_dir) / "shap_importance.csv"
        self.importance_df.to_csv(csv_path, index=False)
        logger.info("SHAP importance saved: %s", csv_path)

        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")  # headless save
            import matplotlib.pyplot as plt

            # Beeswarm plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values, features=None,
                feature_names=self.feature_names,
                plot_type="dot", show=False,
                max_display=min(25, len(self.feature_names)),
            )
            beeswarm_path = Path(out_dir) / "shap_summary_beeswarm.png"
            plt.tight_layout()
            plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Beeswarm plot saved: %s", beeswarm_path)

            # Bar plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                self.shap_values, features=None,
                feature_names=self.feature_names,
                plot_type="bar", show=False,
                max_display=min(20, len(self.feature_names)),
            )
            bar_path = Path(out_dir) / "shap_summary_bar.png"
            plt.tight_layout()
            plt.savefig(bar_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Bar plot saved: %s", bar_path)

        except ImportError:
            logger.warning("shap or matplotlib not installed — skipping plots.")

    def __repr__(self) -> str:
        return (
            f"SHAPReport(stat={self.stat}, model={self.model_type}, "
            f"rows={self.n_rows}, features={len(self.feature_names)})"
        )


@dataclass
class PredictionExplanation:
    """
    SHAP explanation for a single player-week prediction.

    Provides base value + feature contributions sorted by absolute impact.
    """
    player_id:   str
    stat:        str
    prediction:  float
    base_value:  float
    contributions: pd.DataFrame  # [feature, shap_value, feature_value]

    def to_string(self, top_n: int = 10) -> str:
        """Human-readable summary of the top N contributing features."""
        top = self.contributions.head(top_n)
        lines = [
            f"Prediction for {self.player_id} | {self.stat}: {self.prediction:.1f}",
            f"Base value (mean): {self.base_value:.1f}",
            "── Top contributing features ──",
        ]
        for _, row in top.iterrows():
            direction = "▲" if row["shap_value"] > 0 else "▼"
            lines.append(
                f"  {direction} {row['feature']:30s}  "
                f"SHAP={row['shap_value']:+.2f}  val={row['feature_value']:.3g}"
            )
        return "\n".join(lines)


# ── SHAPAnalyzer ───────────────────────────────────────────────────────────────

class SHAPAnalyzer:
    """
    Wraps shap.TreeExplainer for XGB and LightGBM tree models.

    Supports:
        - Global SHAP importance reports
        - Single-prediction explanations
        - MLflow model path loading

    Args:
        model_path: Path to saved model file (.ubj for XGB, .txt for LGB)
        model_type: "xgb", "lgb", or "auto" (detected from extension)
        feature_cols: Feature columns to use. Defaults to FEATURE_COLS.
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = MODEL_TYPE_AUTO,
        feature_cols: Optional[list[str]] = None,
    ) -> None:
        self.model_path   = model_path
        self.model_type   = self._detect_type(model_path, model_type)
        self.feature_cols = feature_cols or FEATURE_COLS
        self._model       = None
        self._explainer   = None
        self._loaded      = False

    def _detect_type(self, path: str, hint: str) -> str:
        if hint != MODEL_TYPE_AUTO:
            return hint
        ext = Path(path).suffix.lower()
        if ext in (".ubj", ".model", ".json"):
            return MODEL_TYPE_XGB
        elif ext in (".txt", ".bin", ".lgb"):
            return MODEL_TYPE_LGB
        logger.warning("Cannot auto-detect model type from extension '%s'; assuming xgb.", ext)
        return MODEL_TYPE_XGB

    def load(self) -> "SHAPAnalyzer":
        """Load the model and build the SHAP TreeExplainer."""
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "shap is required for SHAPAnalyzer. Install: pip install shap"
            ) from exc

        if self.model_type == MODEL_TYPE_XGB:
            try:
                import xgboost as xgb
                self._model = xgb.Booster()
                self._model.load_model(self.model_path)
                logger.info("Loaded XGBoost model from %s", self.model_path)
            except Exception as exc:
                raise RuntimeError(f"Failed to load XGBoost model: {exc}") from exc

        elif self.model_type == MODEL_TYPE_LGB:
            try:
                import lightgbm as lgb
                self._model = lgb.Booster(model_file=self.model_path)
                logger.info("Loaded LightGBM model from %s", self.model_path)
            except Exception as exc:
                raise RuntimeError(f"Failed to load LightGBM model: {exc}") from exc

        self._explainer = shap.TreeExplainer(self._model)
        self._loaded = True
        return self

    def _prepare_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Select and order feature columns; fill missing with 0."""
        present = [c for c in self.feature_cols if c in feature_df.columns]
        missing = [c for c in self.feature_cols if c not in feature_df.columns]
        if missing:
            logger.debug("SHAP: %d feature columns absent in df (will use 0): %s",
                         len(missing), missing[:5])
        X = feature_df[present].copy()
        for c in missing:
            X[c] = 0.0
        return X[self.feature_cols]  # ensure canonical order

    def run(
        self,
        feature_df: pd.DataFrame,
        stat: str = "receiving_yards",
        max_rows: Optional[int] = 5000,
    ) -> SHAPReport:
        """
        Compute global SHAP values for the full feature matrix.

        Args:
            feature_df:  Feature matrix DataFrame.
            stat:        Target stat string (for report metadata only).
            max_rows:    Cap rows for performance (random sample if exceeded).

        Returns:
            SHAPReport with importance_df and raw shap_values.
        """
        if not self._loaded:
            self.load()

        X = self._prepare_features(feature_df)
        if max_rows and len(X) > max_rows:
            X = X.sample(n=max_rows, random_state=42)
            logger.info("SHAP: sampled %d rows for performance.", max_rows)

        logger.info("Computing SHAP values (%d rows × %d features)…", len(X), len(self.feature_cols))
        shap_values = self._explainer.shap_values(X.values)

        if isinstance(shap_values, list):
            # Multi-output tree (e.g. early LGB versions): take first output
            shap_values = shap_values[0]

        mean_abs = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature":        self.feature_cols,
            "mean_abs_shap":  mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        importance_df["rank"] = importance_df.index + 1

        logger.info(
            "SHAP complete — top feature: %s (%.4f)",
            importance_df.iloc[0]["feature"],
            importance_df.iloc[0]["mean_abs_shap"],
        )
        return SHAPReport(
            importance_df=importance_df,
            shap_values=shap_values,
            feature_names=self.feature_cols,
            model_type=self.model_type,
            stat=stat,
            n_rows=len(X),
        )

    def explain_prediction(
        self,
        feature_row: pd.Series,
        stat: str = "receiving_yards",
    ) -> PredictionExplanation:
        """
        Explain a single player-week prediction.

        Args:
            feature_row: Single DataFrame row (pd.Series) with player features.
            stat:        Target stat name (metadata only).

        Returns:
            PredictionExplanation with per-feature SHAP contributions.
        """
        if not self._loaded:
            self.load()

        row_df = feature_row.to_frame().T
        X = self._prepare_features(row_df)

        shap_values = self._explainer.shap_values(X.values)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        base_value = float(self._explainer.expected_value)
        if hasattr(base_value, "__iter__"):
            base_value = float(list(base_value)[0])

        contributions = pd.DataFrame({
            "feature":       self.feature_cols,
            "shap_value":    shap_values[0],
            "feature_value": X.values[0],
        })
        contributions["abs_shap"] = contributions["shap_value"].abs()
        contributions = contributions.sort_values("abs_shap", ascending=False).drop(columns="abs_shap")

        prediction = base_value + shap_values[0].sum()
        player_id  = str(feature_row.get("player_id", "unknown"))

        return PredictionExplanation(
            player_id=player_id,
            stat=stat,
            prediction=prediction,
            base_value=base_value,
            contributions=contributions,
        )

    @staticmethod
    def run_from_mlflow(
        run_id: str,
        stat: str,
        feature_df: pd.DataFrame,
        tracking_uri: Optional[str] = None,
        out_dir: Optional[str] = None,
    ) -> SHAPReport:
        """
        Load a model from MLflow by run_id and compute SHAP values.

        Args:
            run_id:       MLflow run ID.
            stat:         Target stat (used to find model artifact: '{stat}_xgb').
            feature_df:   Feature matrix DataFrame.
            tracking_uri: MLflow tracking URI (defaults to MLFLOW_TRACKING_URI env).
            out_dir:      Directory to save the SHAP report. Optional.

        Returns:
            SHAPReport
        """
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError("mlflow is required for run_from_mlflow.") from exc

        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient()

        # Download the model artifact
        artifact_name = f"{stat}_xgb"
        try:
            local_dir = client.download_artifacts(run_id, artifact_name, "/tmp")
            model_path = Path(local_dir) / "model.ubj"
            if not model_path.exists():
                model_path = next(Path(local_dir).glob("*.ubj"), None)
                if not model_path:
                    raise FileNotFoundError(f"No .ubj file in {local_dir}")
        except Exception as exc:
            raise RuntimeError(f"Failed to download MLflow artifact: {exc}") from exc

        analyzer = SHAPAnalyzer(str(model_path), model_type=MODEL_TYPE_XGB)
        report = analyzer.run(feature_df, stat=stat)

        if out_dir:
            report.save(out_dir)

        return report


# ── Convenience function ───────────────────────────────────────────────────────

def generate_shap_report(
    model_path: str,
    feature_df: pd.DataFrame,
    stat: str = "receiving_yards",
    model_type: str = MODEL_TYPE_AUTO,
    out_dir: Optional[str] = None,
) -> SHAPReport:
    """
    One-line SHAP analysis entry point.

    Args:
        model_path:  Path to trained model file.
        feature_df:  Feature matrix DataFrame.
        stat:        Target stat name.
        model_type:  "xgb", "lgb", or "auto".
        out_dir:     Save report to this directory if specified.

    Returns:
        SHAPReport
    """
    analyzer = SHAPAnalyzer(model_path, model_type=model_type)
    report = analyzer.run(feature_df, stat=stat)
    if out_dir:
        report.save(out_dir)
    return report
