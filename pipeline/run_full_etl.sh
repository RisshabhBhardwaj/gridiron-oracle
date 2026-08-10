#!/bin/bash
# pipeline/run_full_etl.sh
#
# Full ETL pipeline: start Docker services, ingest all seasons, verify row counts.
#
# Steps:
#   1. Start Docker (PostgreSQL)
#   2. ETL orchestrator (game_logs + feature_matrix for 2019-2025)
#   3. Elo ratings + player embeddings (Bucket 10 + embedding cols in feature_matrix)
#   4. PBP pipeline (Bucket 11: EPA, ADOT, YAC, pressure, GNN matchup edges)
#   5. Injury pipeline (2009-2025 IR history — survival model training data)
#   6. Row count verification
#
# RUNTIME ESTIMATE: 60-120 min depending on nflreadpy cache state.
# Run from project root:  bash pipeline/run_full_etl.sh

set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export DATABASE_URL=${DATABASE_URL:-postgresql://oracle:oracle@localhost:15439/oracle}
export ODDS_API_KEY=${ODDS_API_KEY:-}
export OPENWEATHER_API_KEY=${OPENWEATHER_API_KEY:-}
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"

# Artifact-backed runs are evidence-producing runs.  Do not continue with stale
# or NULL feature groups after an enrichment failure; graceful fallback remains
# available for local development without external credentials.
require_artifact_source() {
  local label="$1"
  if [ "${PRODUCT_MODE:-graceful_fallback}" = "artifact_backed" ]; then
    echo "ERROR: $label failed in artifact_backed mode; refusing partial feature matrix." >&2
    exit 1
  fi
}

# Use .venv_311 — same requirement as train_all_models.sh.
VENV_DIR="$(cd "$(dirname "$0")/.." && pwd)/.venv_311"
if [ ! -f "$VENV_DIR/bin/python3.11" ]; then
  echo "ERROR: .venv_311 not found. Run: python3.11 -m venv .venv_311 && pip install -r requirements.txt"
  exit 1
fi
PYTHON="$VENV_DIR/bin/python3.11"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 1 — Starting Docker services (PostgreSQL only)"
echo "════════════════════════════════════════════════════════════════"
# Start only db — ETL needs PostgreSQL. Skip mlflow (port 5000) so you can run
# MLflow manually in another tab without conflict.
docker compose -f infra/docker-compose.yml up -d db
echo "Waiting 15s for PostgreSQL to be ready..."
sleep 15

# Verify DB is up before proceeding.
echo "Verifying PostgreSQL connection..."
docker compose -f infra/docker-compose.yml exec -T db \
  psql -U oracle -d oracle -c "SELECT 'DB ready' AS status;" \
  || { echo "ERROR: PostgreSQL is not ready. Check: docker compose -f infra/docker-compose.yml logs db"; exit 1; }

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 1a — Alembic schema migrations"
echo "════════════════════════════════════════════════════════════════"
$PYTHON scripts/migrate.py upgrade

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 1b — Clear dead_letter + reset nextgen/combine staging for remap"
echo "════════════════════════════════════════════════════════════════"
docker compose -f infra/docker-compose.yml exec -T db \
  psql -U oracle -d oracle -c "TRUNCATE TABLE dead_letter;" || true
docker compose -f infra/docker-compose.yml exec -T db \
  psql -U oracle -d oracle -c "UPDATE staging_nflreadpy SET processed = FALSE WHERE source_type IN ('nextgen_stats', 'combine');" \
  || true

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 2 — Running full ETL pipeline (all 7 seasons)"
echo "════════════════════════════════════════════════════════════════"
$PYTHON -m pipeline.orchestrator \
  --seasons 2019 2020 2021 2022 2023 2024 2025 2026

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 3 — Elo ratings + player embeddings enrichment"
echo "  Populates: team_off_elo, opp_def_elo, elo_matchup_diff,"
echo "             elo_implied_win_prob, player_emb_0..31"
echo "════════════════════════════════════════════════════════════════"
$PYTHON -m pipeline.enrich_elo_embeddings \
  || { require_artifact_source "Elo/embedding enrichment"; echo "  WARNING: enrichment step had errors — check logs. Continuing."; }

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 4 — PBP Pipeline (EPA, ADOT, YAC, OL pressure, GNN matchup edges, ftn_player_game)"
echo "  Populates: pbp_features, ftn_player_game (FTN+PBP join), Bucket 11 cols in feature_matrix"
echo "  Source: nflreadpy.load_pbp() + load_ftn_charting() 2019-2025"
echo "════════════════════════════════════════════════════════════════"
$PYTHON -m pipeline.pbp_pipeline \
  || { require_artifact_source "PBP pipeline"; echo "  WARNING: PBP pipeline had errors — check logs. Bucket 11 features will be NULL. Continuing."; }

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 5 — Injury Pipeline (2009-2025 IR history for survival model)"
echo "  Populates: injury_history table + feature_matrix.games_missed_streak"
echo "  Source: nflreadpy.load_injuries() — official NFL weekly injury reports"
echo "════════════════════════════════════════════════════════════════"
$PYTHON -m pipeline.injury_pipeline \
  || echo "  WARNING: Injury pipeline had errors — check logs. Continuing."

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 5b — Pregame weather forecast + odds enrichment (optional; needs API keys)"
echo "  Populates: weather_forecasts (timestamped OpenWeather snapshots)"
echo "             prop_odds (The Odds API)"
echo "════════════════════════════════════════════════════════════════"
$PYTHON -m scraper.adapters.weather_adapter --db-url "$DATABASE_URL" --capture-forecasts \
  || { require_artifact_source "Weather forecast capture"; echo "  (Weather skipped — set OPENWEATHER_API_KEY for forecast capture)"; }
$PYTHON -m scraper.adapters.odds_adapter --db-url "$DATABASE_URL" \
  || { require_artifact_source "Odds enrichment"; echo "  (Odds skipped — set ODDS_API_KEY for prop lines)"; }

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 6 — Row count verification"
echo "════════════════════════════════════════════════════════════════"
docker compose -f infra/docker-compose.yml exec -T db \
  psql -U oracle -d oracle -c "
    SELECT 'players'        AS table_name, COUNT(*) AS rows FROM players
    UNION ALL
    SELECT 'game_logs',     COUNT(*) FROM game_logs
    UNION ALL
    SELECT 'feature_matrix', COUNT(*) FROM feature_matrix
    UNION ALL
    SELECT 'dead_letter',   COUNT(*) FROM dead_letter
    UNION ALL
    SELECT 'pbp_features',  COUNT(*) FROM pbp_features
    UNION ALL
    SELECT 'pbp_matchups',  COUNT(*) FROM pbp_matchups
    UNION ALL
    SELECT 'injury_history', COUNT(*) FROM injury_history
    UNION ALL
    SELECT 'depth_charts', COUNT(*) FROM depth_charts
    UNION ALL
    SELECT 'nextgen_stats', COUNT(*) FROM nextgen_stats
    UNION ALL
    SELECT 'team_game_stats', COUNT(*) FROM team_game_stats
    UNION ALL
    SELECT 'ftn_play', COUNT(*) FROM ftn_play
    UNION ALL
    SELECT 'ftn_player_game', COUNT(*) FROM ftn_player_game
    UNION ALL
    SELECT 'participation_player_game', COUNT(*) FROM participation_player_game
    UNION ALL
    SELECT 'combine', COUNT(*) FROM combine
    ORDER BY table_name;
  " || true

docker compose -f infra/docker-compose.yml exec -T db \
  psql -U oracle -d oracle -c "SELECT 'projections' AS table_name, COUNT(*) AS rows FROM projections;" 2>/dev/null || true

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ETL COMPLETE"
echo "  Expected minimums (live DB from prior runs):"
echo "    game_logs:      ~426,000 rows"
echo "    feature_matrix: ~129,000 rows"
echo "    projections:    ~161,000 rows"
echo "════════════════════════════════════════════════════════════════"
