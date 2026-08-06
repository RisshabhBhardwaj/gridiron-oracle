# Gridiron Oracle — Architecture

_Last verified against the code: 2026-08-01._

## Layer 0 — Ingestion

`scraper/adapters/` holds one adapter per data source. `nflreadpy` is primary; there are no paid APIs.

Every external row passes through a **Pydantic validator before touching any non-staging table**. Rows that fail validation go to the `dead_letter` table. Nothing is silently dropped — a pipeline that quietly discards 3% of its input produces a model whose errors nobody can explain.

`pipeline/normalize.py` handles two known source defects:

- **`game_id` is absent** from `nflreadpy` player stats for 2019, 2020, 2021, and 2024. It is derived via a schedule lookup (`_build_game_id_lookup()`).
- **Foreign-key processing order is load-bearing:** schedules → rosters → player_stats → snap_counts. Breaking that order raises `IntegrityError`.

`pipeline/feature_engineer.py` produces eight feature buckets, including the Kalman form estimate and an injury bucket.

## Layer 1 — Latent state (Kalman filter)

A player's true ability is a latent state observed through noisy weekly outcomes. A rolling average is a lagging, crude estimator of that state; a scalar Kalman filter per (player, stat) is the correct one — and, critically, it is **causal by construction**, so it cannot leak future information.

Configuration: `Q = 1.0`, `R = empirical variance`, `x0 = position-average prior`. NumPy only, no dependency on the ML stack.

**`snap_pct_off` is stored as a `[0.0, 1.0]` fraction, not a percentage.** The minimum threshold is `MIN_SNAP_PCT = 0.34` (`ml/utils.py`). This was raised from 0.10 because bench and mop-up appearances inject noise that corrupts the Kalman estimate — that noise was the cause of a QB projection collapsing to roughly 60 yards.

## Layer 2 — Base learners

Four learners, each producing out-of-fold predictions with a position column:

| Learner | Why it earns its place |
|---|---|
| **XGBoost** | Strong tabular baseline; Optuna-tuned, walk-forward CV |
| **LightGBM** | Leaf-wise growth gives a different bias profile and different gain importances |
| **CatBoost** | Symmetric trees, native NaN handling, a genuinely different error surface |
| **Temporal Fusion Transformer** | Handles static and time-varying covariates natively — the only learner here that models sequence structure directly |

The TFT is not optional. It is the component that makes this a sequence-modeling project rather than four flavors of gradient boosting.

TFT note: `team_id` is a **time-varying known** covariate, not a static one. Players change teams between seasons.

## Layer 3 — Stacking

Ridge meta-learner over the out-of-fold predictions, fit **per (stat, position) pair**.

This is not a stylistic choice. Fitting one Ridge across positions produces biased intercepts — combining QB, WR, RB, and TE for passing yards drove the intercept to −9.054 — because the positions have entirely different outcome distributions for the same stat. Each pair gets its own model, persisted as `ridge_{stat}_{pos}_coefs.json`, with `intercept_` saved alongside the coefficients and applied at inference.

Alpha selection uses `ElasticNetCV` with **`TimeSeriesSplit`**. It previously used `cv=5`, which is random k-fold, which leaks the future into the meta-learner.

`VALID_POSITION_STATS` restricts each position to its meaningful stats (QB → passing/rushing, WR → receiving/rushing, and so on) so the backtest cannot be inflated by nonsense pairs.

## Layer 4 — Uncertainty and simulation

**Bayesian layer** (`ml/bayesian_model.py`): PyMC NUTS, 2 chains, 500 tune, with a Kalman-informed prior. A point projection is not enough to size a position — the distribution is what matters.

**Monte Carlo** (`ml/monte_carlo.py`): 10,000 draws producing projection, floor, ceiling, boom/bust probabilities, and fantasy PPR.

Calibration note: backtest sigma carries a ×3.0 correction. `kalman_variance` tracks *mean* uncertainty (≈31 at steady state, so std ≈ 5.6 yards), not per-game outcome variance (WR std ≈ 35 yards). Fast mode requires the correction; without it, intervals are absurdly tight.

## Validation

**Walk-forward only. Never random k-fold on time-series data.** Expanding-window cross-validation across seven seasons (2019–2025).

Every training run logs to MLflow before it counts as complete: `mae`, `rmse`, `feature_importances`, `model_artifact`, `training_data_hash`, `timestamp`. **No model is promoted unless its validation MAE beats the incumbent in MLflow.**

The `/backtest` endpoint reports MAE, RMSE, Brier score, simulated P&L, Sharpe, and max drawdown. Brier score matters because a projection system that is accurate on average but badly calibrated will still lose money.

See the model-status section of the [README](../README.md) for what the evaluation currently shows. It is not yet a positive result on yardage.

## Serving

**FastAPI backend** (`backend/app/`): every endpoint returns a typed Pydantic response model, and every `/predict` response carries a `data_freshness` timestamp. SHAP attributions come back as plain-English factor labels (`FEATURE_LABELS` in `shap_service.py`).

**React frontend** (`frontend/src/`): TypeScript strict mode, no `any`. All API calls go through typed hooks in `hooks/`. What-If sliders debounce 300 ms before calling `/scenario`. The Backtest Explorer page is required — it is the surface that shows whether the model has edge, and removing it would make the product unfalsifiable.

## C++ execution engine

| Component | Design |
|---|---|
| `ring_buffer.hpp` | Lock-free SPSC. `alignas(64)` on `head_` and `tail_` to avoid false sharing. Memory-ordering comments required at every atomic load and store: acquire on load, release on store, relaxed on self-reads. |
| `kelly_sizer.hpp` | f\* = (b·p − q)/b, applied at 0.25× fractional Kelly with a hard 5% cap. Quarter Kelly retains ≈94% of log growth at ≈50% of full-Kelly volatility. |
| `exposure_manager.hpp` | Per-player 5%, per-game 20%, total 20%. |
| `signal_processor.cpp` | Ring buffer → Kelly → exposure manager. |
| `game_event.hpp` | Fixed-size POD, ≤256 bytes. |
| `ws_consumer.cpp` | Boost.Beast synchronous WebSocket with exponential backoff, 1s → 30s. |
| `ipc_bridge.cpp` | Unix domain socket, nlohmann/json. |

**Zero heap allocation in the hot path** — no `new`, no `std::string`. SPSC rather than MPMC because there is exactly one producer (the WebSocket consumer) and one consumer (the signal processor); a general-purpose queue would cost throughput for concurrency that does not exist.

Catch2 benchmarks are required for the ring buffer, targeting under 200 ns/op.

## Decisions log

| Decision | Rationale |
|---|---|
| `nflreadpy` over `nfl_data_py` | `nfl_data_py` was archived in September 2025 |
| Stacking over simple averaging | A meta-learner learns *when* to trust each base model |
| Walk-forward CV only | Random k-fold leaks the future into the past |
| TFT as a base learner | Native handling of static plus temporal covariates |
| CatBoost as the fourth learner | Symmetric trees, native NaN handling, different error surface |
| Fractional Kelly at 0.25× | ≈94% of log growth at ≈50% of the volatility |
| SPSC rather than MPMC | One producer, one consumer — MPMC would cost throughput for nothing |
| Ridge per (stat, position) | Cross-position fitting produced a −9.054 intercept on passing yards |
| OOF files carry a position column | Required for position-specific Ridge filtering |
| `ElasticNetCV` with `TimeSeriesSplit` | The previous `cv=5` was random k-fold in the meta-learner |
| Ridge intercept persisted to JSON | Loaded and applied at inference; omitting it silently biased predictions |
| `_FM_COLS` derived dynamically | `[f.name for f in dc_fields(FeatureRow)]` eliminates schema drift from a hardcoded list |
| Kalman replaces `form_*` features | Correct estimator for a latent state, and causal by construction |
| `MIN_SNAP_PCT = 0.34` | Bench and mop-up noise corrupted Kalman estimates at 0.10 |
| `game_id` derivation for 2019–2021 and 2024 | `nflreadpy` omits it for those seasons |
| Injury data gap accepted | The nflverse injury feed died after 2024; the ESPN adapter provides practice participation. Do not block training on it. |
| `AlertService` thread safety | `threading.Lock` for singleton and history operations; publish captures subscribers under the lock and iterates outside it |

## Planned extensions

Ordered by dependency, not by appeal. Nothing here starts before the base system shows positive expected value in backtest.

- **Correlation and joint distributions** — Gaussian copula for same-game-parlay pricing, Dirichlet volume redistribution on injury, scheme-interaction features (aDOT × coverage shell), expanded defensive tendencies.
- **Season-long simulation** — autoregressive multi-horizon rollout feeding week *N* predictions back into the Kalman filter, Cox proportional-hazards injury discount (blocked on the injury data gap), player embeddings, dynamic team Elo per simulation path.
- **Drive and play level** — drive-level Markov chain (requires a play-by-play pipeline), GNN matchup topology (blocked on formation data), SIMD Monte Carlo for sub-millisecond live re-simulation.
- **Deep RL hedging** — an SAC/PPO agent replacing static exposure limits. **Only after** backtest confirms positive expected value on the base system.
