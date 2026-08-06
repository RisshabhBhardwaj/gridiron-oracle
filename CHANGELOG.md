# Changelog

All notable changes to Gridiron Oracle are documented here.

## [Unreleased]

### Added
- `ml/inference_client.py` — extracted model-loading/serving logic from `train.py` into a dedicated `InferenceClient` class. `PipelineRunner` now delegates to it via thin wrappers, keeping test patchability intact (M1).
- `pipeline/features/` subpackage — `FeatureRow` dataclass and bucket compute functions extracted from `feature_engineer.py` into `feature_row.py` and `buckets.py`; `__init__.py` re-exports all symbols for backward compatibility (M2).
- `make mlflow-backup` target — timestamped local backup of the `mlruns/` artifact store.
- `.coveragerc` — coverage source/omit config so `pytest --cov` skips tests, migrations, OOF CSVs, and scripts.
- `CHANGELOG.md` — this file.
- `pyproject.toml` — Ruff configuration that excludes generated artifacts and mutation-test output while linting active Python code.
- `make diagnose-backtest` and `scripts/diagnose_backtest_edge.py` — quick stat/position report for the current backtest edge.

### Changed
- `backend/app/main.py`: scheduler startup failure now logged at `ERROR` (was `WARNING`).
- `pipeline/normalize.py`: added assertion in `_build_game_id_lookup` — raises immediately if the lookup is empty, catching scheduling-order bugs before silent game_id derivation failures accumulate.
- `ml/kalman_tracker.py`: `POSITION_PRIORS` comment cites `scripts/compute_position_priors.py` as the derivation source.
- `engine/src/drive_mcmc.cpp`: Phase 5 stub banner added (mirrors `engine/include/simd_mc.hpp`).
- CI/frontend tooling now uses npm consistently (`npm ci`, `npm run typecheck`, `npm test -- --run`) because `package-lock.json` is the committed lockfile.
- Non-integration test commands now exclude `network` and `slow` markers by default.
- XGB/LGBM test ONNX exports and TFT test OOF output write under temporary output directories instead of tracked artifacts.
- `freeze-baseline` refuses to write a blocked runtime baseline unless `ALLOW_BLOCKED_BASELINE=1` is explicitly set.

## [2025-04-30] — Security + Quality hardening (C1–H5, M3–M5)

### Fixed
- C1: SQL injection surface removed from raw query paths.
- C2: JWT secret validation at startup; reject empty/default secrets.
- H1–H5: SHAP fallback hardened; Kalman variance corrected ×3.0; backtest cross-position contamination fixed via `VALID_POSITION_STATS`; Ridge stacking now per `(stat, position)`; OOF files include `position` column.
- M3: `AlertService` thread safety (threading.Lock on singleton + history ops).
- M4: TFT `team_id` moved from STATIC to TIME_VARYING_KNOWN.
- M5: `ElasticNetCV` replaced random k-fold with `TimeSeriesSplit`; Ridge intercept persisted to JSON.

## [2025-04-15] — C++ Engine

### Added
- `engine/include/ring_buffer.hpp` — SPSC lock-free ring buffer, `alignas(64)`, acquire/release documented.
- `engine/include/kelly_sizer.hpp` — Kelly criterion at 0.25× fractional, hard cap 5%.
- `engine/include/exposure_manager.hpp` — per-player 5%, per-game 20%, total 20% limits.
- `engine/src/signal_processor.cpp`, `ipc_bridge.cpp`, `ws_consumer.cpp`.
- 49 Catch2 unit tests + 10 ws_integration_tests.

## [2025-03-01] — ML Stack Complete

### Added
- XGBoost, LightGBM, CatBoost base learners with walk-forward CV + Optuna + MLflow.
- TFT base learner (Temporal Fusion Transformer) with static + time-varying covariates.
- Ridge stacking meta-learner per `(stat, position)` over OOF predictions.
- Bayesian uncertainty layer (PyMC NUTS, 2 chains).
- Monte Carlo simulation (10k draws) — projection/floor/ceiling, boom/bust, fantasy PPR.
- Walk-forward backtesting (7 seasons 2019–2025) with Brier, Sharpe, max drawdown.
- SHAP service with 80 plain-English feature labels.
- FastAPI backend (14 routes, typed Pydantic, `data_freshness` on `/predict`).
- React 18 + TypeScript strict frontend — Dashboard, PlayerDetail, BacktestExplorer, Settings.
