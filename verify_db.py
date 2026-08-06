"""
verify_db.py

Sanity checks for the feature_matrix database table.

Run:
    python verify_db.py

Checks:
  1. Model loading: XGB, LGBM, TFT can each load the feature matrix from DB.
  2. NULL rate audit: warn on feature columns with > 80% NULL values.
     High NULL rates cause silent training degradation (models learn from noise).
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# NULL rate threshold above which we warn
_NULL_RATE_WARN_PCT: float = 80.0


def verify() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL not set!")
        return

    import ml.xgb_model as xgb
    import ml.lgbm_model as lgbm
    import ml.tft_model as tft

    print("Verifying XGB ...")
    xgb_df = xgb.load_feature_matrix(db_url, seasons=[2024])
    print(f"XGB loaded {len(xgb_df)} rows")

    print("Verifying LGBM ...")
    lgbm_df = lgbm.load_feature_matrix(db_url, seasons=[2024])
    print(f"LGBM loaded {len(lgbm_df)} rows")

    print("Verifying TFT ...")
    if hasattr(tft, "load_feature_matrix"):
        tft_df = tft.load_feature_matrix(db_url, seasons=[2024])
        print(f"TFT loaded {len(tft_df)} rows")
    else:
        print("TFT does not have a standalone load_feature_matrix method.")

    # ------------------------------------------------------------------
    # Issue 21 fix: NULL rate audit
    # Columns with > _NULL_RATE_WARN_PCT% NULLs silently degrade training
    # because imputed zeros / means can dominate gradient updates.
    # ------------------------------------------------------------------
    print(f"\n── NULL rate audit (threshold: {_NULL_RATE_WARN_PCT}%) ──")
    for df_name, df in [("xgb", xgb_df), ("lgbm", lgbm_df)]:
        if df is None or df.empty:
            continue
        high_null_cols = []
        for col in df.columns:
            null_rate = df[col].isna().mean() * 100.0
            if null_rate > _NULL_RATE_WARN_PCT:
                high_null_cols.append((col, round(null_rate, 1)))

        if high_null_cols:
            logger.warning(
                "[%s] %d columns exceed %.0f%% NULL rate — these may corrupt training:\n%s",
                df_name,
                len(high_null_cols),
                _NULL_RATE_WARN_PCT,
                "\n".join(f"  {col}: {pct}%" for col, pct in sorted(high_null_cols)),
            )
        else:
            logger.info("[%s] All %d columns within NULL rate threshold.", df_name, len(df.columns))

    print("\nVerification complete.")


if __name__ == "__main__":
    verify()
