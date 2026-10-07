Type: Architecture
Status: Accepted
Last verified against code: 2026-10-07 (module existence and constants checked; model quality not re-measured)

# Architecture

## Data flow

1. **Ingest** (`scraper/adapters/`, `pipeline/`): `nflreadpy` is the primary source. Rows are validated (Pydantic); failures go to a `dead_letter` table. Normalization order matters because of foreign keys: schedules, rosters, player stats, snap counts. `game_id` is derived from the schedule where the source omits it. Sources: [DATA_SOURCES.md](DATA_SOURCES.md).
2. **Features** (`pipeline/feature_engineer.py`, `pipeline/features/`): feature matrix in PostgreSQL. Player form uses a scalar Kalman filter per player and stat (`ml/kalman_tracker.py`, default Q=1.0), which is causal by construction.
3. **Models** (`ml/`): XGBoost, LightGBM, CatBoost and a Temporal Fusion Transformer produce out-of-fold predictions. A Ridge meta-learner is fit per (stat, position) (`ml/stacking_ensemble.py`). Some served cells are a plain LightGBM passthrough, not a stack; `ml/served_learner.py` and `releases/candidates/*/CONSTRAINED_STACK_SELECTION.json` say which.
4. **Uncertainty**: weekly intervals use residual quantiles (`ml/conformal.py`). A PyMC Bayesian layer and Monte Carlo (`ml/bayesian_model.py`, `ml/monte_carlo.py`) exist in code; whether they are on the served path was not verified.
5. **Other simulators**: season (`ml/season_simulator.py`, C++ `engine/src/season_simulator.cpp`), team-game and drive models (`ml/team_game_model.py`, `ml/drive_engine.py`, `engine/src/drive_mcmc.cpp`), draft board and mock draft (`ml/draft_projection.py`, `ml/vor.py`, `ml/mock_draft.py`). Draft ranks use VOR for an 8-team PPR league.
6. **Serving** (`backend/app/`): FastAPI. Routes: `/predict`, `/projections/*`, `/explain`, `/scenario`, `/backtest`, `/settings`, `/alerts`, `/season/current`, `/draft/board`, `/mock-draft/*`, `/team-games/*`, `/drive/simulate`. Optional `X-API-Key`. `PRODUCT_MODE=artifact_backed` (default) serves only pinned artifacts; `graceful_fallback` allows labelled fallbacks.
7. **Frontend** (`frontend/src/`): React 18, TypeScript, Vite, Tailwind 3, TanStack Query, Recharts. Routed pages: Home, Projections (+ game detail), Season review, Draft, Player, Backtest, Settings.

## C++ engine (`engine/`)

Lock-free single-producer single-consumer ring buffer, fractional Kelly sizing (0.25x, 5% cap per bet), exposure limits (5% per player, 20% per game, 20% total), WebSocket consumer, Unix-socket bridge to Python. 72 Catch2 tests pass locally. Connection to a live odds feed was not verified. Measurements: [the benchmark log](plans/2026-09-08-spsc-latency-benchmarks.md).

## Release gating

`scripts/reprojection_gate.py` compares a candidate to a frozen prior release; `scripts/freeze_baseline.py` writes `releases/current_baseline.json`; `ml/artifact_manifest.py` fails closed on a missing entry or digest mismatch. Note: the stored gate reports compare against a prior LightGBM reference release, not against naive or trailing-3 baselines.

## Known gaps

- The 2026 weekly forecast surface depends on data that does not exist before games are played; the UI shows a hard-coded (`frontend/src/components/shared/NoForecastBanner.tsx`) "2026 in-season projections are not yet available" banner (see USERS.md).
- Settings weights (`user_config.json`) are stored and validated by `/settings`, but no consumer was found in projection code.
- A drive simulator exists (`/drive/simulate`); its accuracy was not evaluated in this audit.
- Injury data: the nflverse feed stopped after 2024 (per earlier docs; not re-checked).
