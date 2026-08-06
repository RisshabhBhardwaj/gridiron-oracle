"""
pipeline/injury_pipeline.py

Multi-season IR history pipeline.

Loads nflreadpy.load_injuries() for seasons 2009-2025 (the longest available
history), computes:
  - games_missed_streak (consecutive weeks absent)
  - time_to_return (weeks until next active game — survival model training target)
  - injury_type (knee, hamstring, etc.)
  - event_type ('injury' or 'return')

Writes to `injury_history` table. This feeds:
  - ml/survival_model.py (CoxPH) for player availability probabilities
  - feature_matrix.games_missed_streak (retrospective absence counter)

Run AFTER pipeline.orchestrator:
    python -m pipeline.injury_pipeline

Data source: nflreadpy.load_injuries() — official NFL weekly injury report,
same data used by nflfastR analytics community. Free, no scraping required.
Coverage: 2009-present (17 seasons, ~200k rows).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("injury_pipeline")

# Load from 2009 for maximum survival model training data
INJURY_SEASONS = list(range(2009, 2027))

# Statuses that count as "out" / missing a game
OUT_STATUSES = {"Out", "IR", "Injured Reserve", "PUP-F", "COVID-19"}
# Statuses that suggest limited availability
RISK_STATUSES = {"Doubtful", "Questionable"}


# ── DB schema ──────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS injury_history (
    player_id            VARCHAR NOT NULL,
    full_name            VARCHAR,
    season               INTEGER NOT NULL,
    week                 INTEGER NOT NULL,
    team                 VARCHAR,
    position             VARCHAR,
    report_status        VARCHAR,   -- "Out", "Doubtful", "Questionable", "Limited", "Full", "DNP"
    practice_status      VARCHAR,   -- Mid-week practice designation
    injury_type          VARCHAR,   -- "Knee", "Hamstring", "Ankle", etc.
    is_out               BOOLEAN,   -- TRUE when player missed the game
    games_missed_streak  INTEGER,   -- consecutive games missed heading into this week
    time_to_return       INTEGER,   -- weeks until next active game (NULL if still IR/season end)
    event_type           VARCHAR,   -- 'active', 'injury_start', 'injury_ongoing', 'return'
    PRIMARY KEY (player_id, season, week)
)
"""


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(_DDL)
    conn.commit()
    logger.info("injury_history table ensured.")


# ── Ingestion ──────────────────────────────────────────────────────────────────

def _load_injuries() -> pd.DataFrame:
    import nflreadpy
    import warnings
    warnings.filterwarnings("ignore")
    logger.info("Loading injury reports for seasons %d–%d...", INJURY_SEASONS[0], INJURY_SEASONS[-1])
    
    dfs = []
    for s in INJURY_SEASONS:
        try:
            df_season = nflreadpy.load_injuries([s])
            if hasattr(df_season, "to_pandas"):
                df_season = df_season.to_pandas()
            if not df_season.empty:
                dfs.append(df_season)
        except Exception as e:
            logger.warning("Failed to load injury stats for season %d: %s", s, e)
            
    if not dfs:
        logger.error("No injury data could be loaded for any season.")
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d injury report rows.", len(df))
    return df


def _process_injuries(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform raw injury report data into the injury_history schema.

    Key columns expected from nflreadpy.load_injuries():
        gsis_id (player_id), full_name, season, week, team, position,
        report_status, practice_status, primary_injury

    Returns cleaned DataFrame with all injury_history columns computed.
    """
    # ── Normalize column names ─────────────────────────────────────────────
    col_map = {
        "gsis_id":        "player_id",
        "full_name":      "full_name",
        "season":         "season",
        "week":           "week",
        "team":           "team",
        "position":       "position",
        "report_status":  "report_status",
        "practice_status":"practice_status",
        "primary_injury": "injury_type",
    }
    available = {k: v for k, v in col_map.items() if k in raw.columns}
    df = raw.rename(columns=available)[list(available.values())].copy()

    # Fill missing columns
    for col in col_map.values():
        if col not in df.columns:
            df[col] = None

    # ── Classify active / out per row ──────────────────────────────────────
    df["is_out"] = df["report_status"].isin(OUT_STATUSES)

    # ── Sort for streak calculation ────────────────────────────────────────
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # ── Compute games_missed_streak (consecutive weeks out heading into this week) ──
    streaks = []
    for player_id, grp in df.groupby("player_id"):
        streak = 0
        for _, row in grp.iterrows():
            streaks.append(streak)
            if row["is_out"]:
                streak += 1
            else:
                streak = 0
    df["games_missed_streak"] = streaks

    # ── Compute time_to_return (weeks until next non-out game) ────────────
    # This is the Cox PH survival model training target.
    # time_to_return = None means player is still on IR at season end (right-censored).
    time_to_return = []
    for player_id, grp in df.groupby("player_id"):
        grp.index.tolist()
        is_out_arr = grp["is_out"].values
        n = len(grp)
        for i in range(n):
            if not is_out_arr[i]:
                time_to_return.append(None)  # active — not an event start
            else:
                # Find the next non-out game
                returned = None
                for j in range(i + 1, n):
                    if not is_out_arr[j]:
                        # Count weeks between i and j (accounts for byes/off-weeks)
                        returned = int(grp.iloc[j]["week"]) - int(grp.iloc[i]["week"])
                        if returned <= 0:
                            returned = j - i  # fallback: count row distance
                        break
                time_to_return.append(returned)
    df["time_to_return"] = time_to_return

    # ── Event type classification ──────────────────────────────────────────
    def _classify_event(row):
        if not row["is_out"]:
            if row["games_missed_streak"] > 0:
                return "return"
            return "active"
        if row["games_missed_streak"] == 0:
            return "injury_start"
        return "injury_ongoing"

    df["event_type"] = df.apply(_classify_event, axis=1)

    # Keep only rows where player is identified
    df = df[df["player_id"].notna()].copy()
    df["season"] = df["season"].astype(int)
    df["week"]   = df["week"].astype(int)

    # Deduplicate primary key before upserting into the database
    df = df.drop_duplicates(subset=["player_id", "season", "week"], keep="last")

    return df


def _write_to_db(conn, df: pd.DataFrame) -> int:
    cur = conn.cursor()
    upsert_sql = """
        INSERT INTO injury_history (
            player_id, full_name, season, week, team, position,
            report_status, practice_status, injury_type,
            is_out, games_missed_streak, time_to_return, event_type
        )
        VALUES %s
        ON CONFLICT (player_id, season, week) DO UPDATE SET
            full_name           = EXCLUDED.full_name,
            team                = EXCLUDED.team,
            position            = EXCLUDED.position,
            report_status       = EXCLUDED.report_status,
            practice_status     = EXCLUDED.practice_status,
            injury_type         = EXCLUDED.injury_type,
            is_out              = EXCLUDED.is_out,
            games_missed_streak = EXCLUDED.games_missed_streak,
            time_to_return      = EXCLUDED.time_to_return,
            event_type          = EXCLUDED.event_type
    """
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r["player_id"]),
            str(r["full_name"]) if r["full_name"] else None,
            int(r["season"]),
            int(r["week"]),
            str(r["team"]) if r["team"] else None,
            str(r["position"]) if r["position"] else None,
            str(r["report_status"]) if r["report_status"] else None,
            str(r["practice_status"]) if r["practice_status"] else None,
            str(r["injury_type"]) if r["injury_type"] else None,
            bool(r["is_out"]),
            int(r["games_missed_streak"]),
            int(r["time_to_return"]) if r["time_to_return"] is not None and not (isinstance(r["time_to_return"], float) and np.isnan(r["time_to_return"])) else None,
            str(r["event_type"]),
        ))

    psycopg2.extras.execute_values(cur, upsert_sql, rows, page_size=2000)
    conn.commit()
    return len(rows)


def _update_feature_matrix_streaks(conn, df: pd.DataFrame) -> None:
    """
    Update feature_matrix.games_missed_streak for each (player_id, season, week)
    using the computed injury history. This is one of the most predictive features
    for snap share and target volume after returning from injury.
    """
    cur = conn.cursor()
    cur.execute("ALTER TABLE feature_matrix ADD COLUMN IF NOT EXISTS games_missed_streak INTEGER DEFAULT 0")
    conn.commit()

    update_sql = """
        UPDATE feature_matrix SET games_missed_streak = %(streak)s
        WHERE player_id = %(player_id)s AND season = %(season)s AND week = %(week)s
    """
    # Only update rows where player was returning (streak > 0)
    streak_rows = df[df["games_missed_streak"] > 0][["player_id", "season", "week", "games_missed_streak"]]
    batch = [
        {"player_id": str(r["player_id"]), "season": int(r["season"]),
         "week": int(r["week"]), "streak": int(r["games_missed_streak"])}
        for _, r in streak_rows.iterrows()
    ]
    if batch:
        psycopg2.extras.execute_batch(cur, update_sql, batch, page_size=5000)
        conn.commit()
        logger.info("✓ Updated games_missed_streak for %d feature_matrix rows.", len(batch))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("  Injury Pipeline — Seasons 2009–2025")
    logger.info("=" * 60)

    conn = _get_conn()
    try:
        _ensure_schema(conn)
        raw = _load_injuries()
        df = _process_injuries(raw)
        n = _write_to_db(conn, df)
        logger.info("✓ injury_history: %d rows written.", n)

        _update_feature_matrix_streaks(conn, df)

        # Summary stats
        n_injury_starts = (df["event_type"] == "injury_start").sum()
        n_returns = (df["event_type"] == "return").sum()
        n_censored = df["is_out"] & df["time_to_return"].isna()
        logger.info("  Injury events: %d starts, %d returns, %d censored (still IR at season end)",
                    n_injury_starts, n_returns, int(n_censored.sum()))

    finally:
        conn.close()

    logger.info("=" * 60)
    logger.info("  Injury pipeline complete. injury_history ready for survival_model.py.")
    logger.info("=" * 60)


def _get_conn():
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        raise RuntimeError("DATABASE_URL not set.")
    return psycopg2.connect(dsn.replace("postgresql+asyncpg://", "postgresql://"))


if __name__ == "__main__":
    main()
