"""
pipeline/enrich_elo_embeddings.py

Post-ETL enrichment script:
  1. Fits TeamEloSystem on all historical game results (from game_logs table)
  2. Bulk-updates feature_matrix with the 6 Elo columns for every row
  3. Fits PlayerProfileEmbedder on all players (from game_logs physical attrs)
  4. Bulk-updates feature_matrix with player_emb_0..31 embedding columns

Run AFTER pipeline.orchestrator has populated game_logs + feature_matrix:
    python -m pipeline.enrich_elo_embeddings

Environment:
    DATABASE_URL — postgresql://... (required)

Idempotent: safe to re-run; updates are upserted (UPDATE ... SET).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("enrich_elo_embeddings")

# ── DB helpers ─────────────────────────────────────────────────────────────────

def _get_conn():
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        raise RuntimeError("DATABASE_URL not set.")
    dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")
    return psycopg2.connect(dsn)


# ── Step 1: Elo Enrichment ─────────────────────────────────────────────────────

def enrich_elo(conn) -> None:
    """
    Fit TeamEloSystem on all game_logs results, then UPDATE feature_matrix
    with the 6 Elo columns for every (player, game) row.

    Strategy:
        - Pull one row per game (distinct game_id + team combos) from game_logs.
        - Sort ascending by season, week to preserve temporal order (no leakage).
        - For each game: call elo.update(home_team, away_team, home_score, away_score,
          week). This is the standard ELO update used by 538-style models.
        - After fitting all games, call elo.enrich_features(df) on the full
          feature_matrix to compute team_off_elo, opp_def_elo, etc. for each row.
        - Bulk UPDATE feature_matrix in batches of 5,000.
    """
    logger.info("Step 1: Building Elo ratings from game_logs...")

    from ml.team_elo import get_elo_system

    elo = get_elo_system()

    # ── Load game-level results ────────────────────────────────────────────────
    # game_logs.team (not recent_team); scores and home/away from games table.
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT ON (gl.game_id, gl.team)
            gl.game_id,
            gl.season,
            gl.week,
            gl.team,
            gl.opponent_team,
            COALESCE(CASE WHEN gl.team = g.home_team THEN g.home_score ELSE g.away_score END, 0)::float AS team_score,
            COALESCE(CASE WHEN gl.team = g.home_team THEN g.away_score ELSE g.home_score END, 0)::float AS opp_score,
            CASE WHEN gl.team = g.home_team THEN 'home' ELSE 'away' END AS home_away
        FROM game_logs gl
        JOIN games g ON gl.game_id = g.id
        WHERE gl.game_id IS NOT NULL
          AND gl.team IS NOT NULL
          AND gl.opponent_team IS NOT NULL
          AND gl.season IS NOT NULL
          AND gl.week IS NOT NULL
        ORDER BY gl.game_id, gl.team, gl.season, gl.week
    """)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    games_df = pd.DataFrame(rows, columns=cols)

    if games_df.empty:
        logger.warning("No game_logs rows found — skipping Elo enrichment.")
        return

    # ── Reconstruct game-level scores (one row per game) ───────────────────────
    # Each game appears twice (once per team). We need home_score + away_score.
    games_df["is_home"] = (games_df["home_away"] == "home").astype(int)
    home_games = games_df[games_df["is_home"] == 1].set_index("game_id")
    away_games = games_df[games_df["is_home"] == 0].set_index("game_id")

    # Merge to get both sides of each game
    paired = home_games.join(away_games, lsuffix="_home", rsuffix="_away", how="inner")
    fit_df = pd.DataFrame({
        "season": paired["season_home"].astype(int),
        "week": paired["week_home"].astype(int),
        "home_team": paired["team_home"].astype(str),
        "away_team": paired["team_away"].astype(str),
        "home_score": paired["team_score_home"].astype(int),
        "away_score": paired["team_score_away"].astype(int),
    })
    fit_df = fit_df.sort_values(["season", "week"]).reset_index(drop=True)
    elo.fit(fit_df)
    fitted_games = len(fit_df)
    logger.info("Elo fitted on %d games. Saving ratings...", fitted_games)
    elo_path = Path("ml/elo_ratings.pkl")
    elo.save(str(elo_path))
    logger.info("Elo ratings saved → %s", elo_path)

    # ── Load full feature_matrix for enrichment ────────────────────────────────
    logger.info("Loading feature_matrix for Elo enrichment...")
    cur.execute("""
        SELECT player_id, game_id, season, week,
               team, opponent_team
        FROM feature_matrix
        WHERE team IS NOT NULL
    """)
    fm_rows = cur.fetchall()
    fm_cols = [d[0] for d in cur.description]
    fm_df = pd.DataFrame(fm_rows, columns=fm_cols)
    logger.info("  feature_matrix rows: %d", len(fm_df))

    if fm_df.empty:
        logger.warning("feature_matrix is empty — skipping Elo UPDATE.")
        return

    # enrich_features expects opp_team or opponent; we have opponent_team
    fm_df = fm_df.copy()
    if "opp_team" not in fm_df.columns and "opponent_team" in fm_df.columns:
        fm_df["opp_team"] = fm_df["opponent_team"]

    # ── Enrich: compute Elo columns per (season, week) ─────────────────────────
    # enrich_features(season, week) uses Elo snapshot heading into that week.
    enriched_parts: list[pd.DataFrame] = []
    for (seas, wk), grp in fm_df.groupby(["season", "week"]):
        part = elo.enrich_features(grp, season=int(seas), week=int(wk))
        enriched_parts.append(part)
    enriched = pd.concat(enriched_parts, ignore_index=True) if enriched_parts else fm_df

    elo_cols = [
        "team_off_elo", "team_def_elo",
        "opp_off_elo", "opp_def_elo",
        "elo_matchup_diff", "elo_implied_win_prob",
    ]

    # Ensure columns are present after enrichment (fallback to 0 if missing)
    for col in elo_cols:
        if col not in enriched.columns:
            enriched[col] = 0.0

    # ── Bulk UPDATE in batches of 5,000 ───────────────────────────────────────
    logger.info("Bulk-updating feature_matrix Elo columns (%d rows)...", len(enriched))
    batch_size = 5000
    update_sql = """
        UPDATE feature_matrix SET
            team_off_elo         = %(team_off_elo)s,
            team_def_elo         = %(team_def_elo)s,
            opp_off_elo          = %(opp_off_elo)s,
            opp_def_elo          = %(opp_def_elo)s,
            elo_matchup_diff     = %(elo_matchup_diff)s,
            elo_implied_win_prob = %(elo_implied_win_prob)s
        WHERE player_id = %(player_id)s
          AND game_id   = %(game_id)s
    """

    n_updated = 0
    rows_data = []
    for _, row in enriched.iterrows():
        rows_data.append({
            "player_id":           str(row["player_id"]),
            "game_id":             str(row["game_id"]),
            "team_off_elo":        _safe_float(row.get("team_off_elo")),
            "team_def_elo":        _safe_float(row.get("team_def_elo")),
            "opp_off_elo":         _safe_float(row.get("opp_off_elo")),
            "opp_def_elo":         _safe_float(row.get("opp_def_elo")),
            "elo_matchup_diff":    _safe_float(row.get("elo_matchup_diff")),
            "elo_implied_win_prob":_safe_float(row.get("elo_implied_win_prob")),
        })
        if len(rows_data) >= batch_size:
            psycopg2.extras.execute_batch(cur, update_sql, rows_data)
            conn.commit()
            n_updated += len(rows_data)
            logger.info("  Elo: updated %d / %d rows...", n_updated, len(enriched))
            rows_data = []

    if rows_data:
        psycopg2.extras.execute_batch(cur, update_sql, rows_data)
        conn.commit()
        n_updated += len(rows_data)

    logger.info("✓ Elo enrichment complete: %d feature_matrix rows updated.", n_updated)


# ── Step 2: Player Embedding Enrichment ───────────────────────────────────────

def enrich_embeddings(conn) -> None:
    """
    Fit PlayerProfileEmbedder on all players then UPDATE feature_matrix
    with player_emb_0..player_emb_31 (32-dim PCA physical profile embedding).

    This gives every XGB/LGB/TFT model a dense representation of the player's
    physical identity, which is critical for projecting rookies and low-sample
    players where historical stats are sparse.
    """
    raise RuntimeError(
        "Player embeddings are disabled by the as-of feature contract: the legacy "
        "embedder uses CURRENT_DATE/current players snapshots and broadcasts them "
        "over history. Implement dated causal embeddings before enabling this path."
    )

    try:
        from ml.player_embeddings import EmbeddingStore, EMBED_DIM
    except ImportError as e:
        logger.warning("player_embeddings not available: %s — skipping.", e)
        return

    cur = conn.cursor()

    # ── Load player physical data from players table ───────────────────────────
    # height is stored as "6-0" (ft-in); parse to inches. draft_round not in players → NULL.
    cur.execute("""
        SELECT
            p.id AS player_id,
            p.position,
            (SPLIT_PART(p.height, '-', 1)::int * 12 + SPLIT_PART(NULLIF(TRIM(SPLIT_PART(p.height, '-', 2)), ''), '-', 1)::int)::float AS height,
            p.weight,
            NULL::float AS draft_round,
            CASE WHEN p.birth_date IS NOT NULL
                 THEN EXTRACT(YEAR FROM CURRENT_DATE)::int - EXTRACT(YEAR FROM p.birth_date)::int
                 ELSE NULL END AS age,
            COALESCE(p.years_exp, 0)::float AS years_experience
        FROM players p
        WHERE p.id IS NOT NULL
          AND p.position IS NOT NULL
          AND p.height IS NOT NULL
          AND p.height ~ '^[0-9]+-[0-9]+$'
    """)
    rows = cur.fetchall()
    player_df = pd.DataFrame(rows, columns=[d[0] for d in cur.description])
    logger.info("  players for embedding: %d", len(player_df))

    if player_df.empty:
        logger.warning("No player data for embeddings — skipping.")
        return

    # ── Fit embedder ───────────────────────────────────────────────────────────
    store = EmbeddingStore()
    store.build(player_df)

    embed_path = Path("ml/player_embeddings_store.pkl")
    store.save(str(embed_path))
    logger.info("EmbeddingStore saved → %s", embed_path)

    # ── Load player_ids from feature_matrix ────────────────────────────────────
    cur.execute("SELECT DISTINCT player_id FROM feature_matrix WHERE player_id IS NOT NULL")
    fm_player_ids = [row[0] for row in cur.fetchall()]
    logger.info("  feature_matrix unique players: %d", len(fm_player_ids))

    # ── Build embedding dict ───────────────────────────────────────────────────
    emb_dict = {}  # player_id → np.ndarray(EMBED_DIM,)
    for pid in fm_player_ids:
        emb = store.get(str(pid))
        if emb is not None:
            emb_dict[pid] = emb

    logger.info("  embeddings computed for %d / %d players", len(emb_dict), len(fm_player_ids))

    # ── Bulk UPDATE ────────────────────────────────────────────────────────────
    if not emb_dict:
        logger.warning("No embeddings to write — skipping UPDATE.")
        return

    emb_cols_set = ", ".join([f"player_emb_{i} = %(emb_{i})s" for i in range(EMBED_DIM)])
    update_sql = f"""
        UPDATE feature_matrix SET {emb_cols_set}
        WHERE player_id = %(player_id)s
    """

    batch_size = 2000
    n_updated = 0
    rows_data = []
    for pid, emb in emb_dict.items():
        row_data = {"player_id": str(pid)}
        for i in range(EMBED_DIM):
            row_data[f"emb_{i}"] = float(emb[i]) if i < len(emb) else 0.0
        rows_data.append(row_data)
        if len(rows_data) >= batch_size:
            psycopg2.extras.execute_batch(cur, update_sql, rows_data)
            conn.commit()
            n_updated += len(rows_data)
            logger.info("  Embeddings: updated %d players...", n_updated)
            rows_data = []

    if rows_data:
        psycopg2.extras.execute_batch(cur, update_sql, rows_data)
        conn.commit()
        n_updated += len(rows_data)

    logger.info("✓ Embedding enrichment complete: %d players updated.", n_updated)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else f  # NaN → None
    except (TypeError, ValueError):
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("  Post-ETL Enrichment: Elo Ratings + Player Embeddings")
    logger.info("=" * 60)

    conn = _get_conn()
    try:
        enrich_elo(conn)
        enrich_embeddings(conn)
    finally:
        conn.close()

    logger.info("=" * 60)
    logger.info("  Enrichment complete. feature_matrix is ready for training.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
