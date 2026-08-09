# Changelog

All notable changes to Gridiron Oracle are documented here.

## [Unreleased]

### Fixed — adversarial audit 2026-08-09, hygiene stream (C-07, C-23, C-24, C-25, C-28, C-29, C-30, C-31, C-32)

- **C-07 / C-30 — public-push gate.** `.githooks/pre-push` set `range="$local_sha"` for a new remote ref and ran a one-commit `git diff-tree`, so a blocked path committed in an ancestor and deleted before the tip pushed cleanly (reproduced: exit 0). On existing refs it diffed endpoints, hiding add-then-delete inside the range. It now enumerates every object introduced by the push via `git rev-list --objects --root <local> --not --remotes=<remote>`, uniformly for new and existing refs. Added content-secret scanning (a built-in regex set applied across every introduced commit, plus `gitleaks` when installed and **required** under `CI=true`) and a 5 MB object-size limit. `make hooks` and `scripts/bootstrap_local.sh` now set `core.hooksPath=.githooks` instead of copying into `.git/hooks`, so clones are covered and the tracked hook is the one that runs. A `push-gate` job in `.github/workflows/ci.yml` runs the same hook over everything reachable from `HEAD` — mark it as a required check; the local hook is convenience only, and `CONTRIBUTING.md` now says so. Both bypasses are locked by `backend/tests/test_push_gate.py`.
- **`.gitignore` — unanchored `lib/`.** The packaging block's `lib/` matched every directory named `lib` at any depth, silently excluding `e2e/lib/http_assert.sh` — a file the smoke test sources, which would therefore have been missing from every fresh clone. Anchored to `/lib/` and `/lib64/` (the only `lib` directory in the tree is `e2e/lib`; `.venv_311/` is ignored on its own line). Found while fixing C-32; same class as the mid-line-comment ignore bug the audit credited as fully fixed.
- **C-23 — FantasyPros importer alias collision.** `_COLUMN_ALIASES` mapped both `rank` and `avg` onto `adp`, so a standard FantasyPros export silently imported Rank as ADP; the `required - set(columns)` guard could not detect it because the rename produced two columns named `adp`. Aliases now resolve in explicit priority order with at most one source column per canonical name. `rank` is preserved as `fp_rank` and is no longer an ADP alias, so a Rank-only export raises instead of importing a rank as a draft position. Covered by `backend/tests/test_fantasypros_adp_importer.py` against a real export header shape.
- **C-24 — coaching seed data.** Deleted `data/coaching/coaching_2026.csv` and the `DEFAULT_ROWS` fallback that regenerated it. The seed listed a former wide receiver as a defensive coordinator, had two coordinators each holding the same role on two teams, and said `verify` in every row; `team_coaching` has zero consumers, so it was inert wrong data. Added `validate_coaching()` — team codes among the 32 franchises exactly once, no placeholder text, no coordinator on two teams, HC/coordinator dual roles declared via `dual_role`, pass-rate prior in (0, 1) — plus required `source` / `verified_by` / `verified_on` provenance columns enforced at load. Template and policy in `data/coaching/`.
- **C-25 — "PFR stub gone" was half true.** The retired scraper adapter does raise on construction with zero callers, but `pipeline/pbp_pipeline.py::_load_pfr_drop_rates` still calls `nflreadpy.load_pfr_advstats` and `drop_rate` is sourced from it. Documented the live PFR-derived path accurately at the call site, in the retired adapter's docstring, in `docs/ARCHITECTURE.md`, and in a new `docs/DATA_SOURCES.md`. A drift guard fails if the call exists without the documentation.
- **C-28 — season caps were advisory.** `_parse_seasons` dropped out-of-range seasons with a warning, so `--seasons 2019-2026` quietly trained on 2019–2025 and reported success; it now raises. `cap_seasons` no longer clamps the *lower* bound — `TRAIN_SEASON_START` is the default start of the walk-forward range, not a floor, and clamping rewrote `2018-2024` as `2019-2024`. Added `assert_seasons_within_cap`, and called `assert_not_fitting_incomplete_season` at all four trainer CLI entrypoints (`xgb_model`, `lgbm_model`, `catboost_model`, `tft_cli`), where it had never been called in production code. `test_xgb_model.py::TestParseSeasons::test_range_notation` is green.
- **C-29 — evidence lived under ignored paths.** Added `releases/artifacts/` (un-ignored) with a written policy and a `MANIFEST.json` recording path, SHA-256, size, producing command, and status for all 25 tracked evidence artifacts under `reports/` and `ml/experiments/`. `scripts/verify_evidence_manifest.py` (`make verify-evidence`, plus a CI step) reports content drift, missing files, and unlisted evidence sitting in ignored directories. Every entry is `provisional` — the audit invalidated the evidence itself, so nothing is frozen yet. Nothing under `ml/oof/` was moved.
- **C-31 — metric mislabelling.** `reports/eval_causal_volume_summary.json` reported mean Poisson deviance under the key `pooled_mae`: the metric is not MAE (count targets route to `poisson_deviance`) and the value is not pooled (it is the unweighted mean of per-season scores). Both summaries are now generated by `scripts/summarize_causal_evals.py`, which carries the metric name through from the per-cell CSVs and emits `metric`, `mean_season_score`, and `n_weighted_score`. Every other report key was audited against `target_metric_family`; the rest were correct. All `beat_both` counts reproduce exactly.
- **C-32 — e2e smoke test could not validate `/predict`.** It sent obsolete `player_id`/`position` parameters instead of the required `player`, and asserted via `grep -qE '\[|\]'` — which passes on an empty array and on most error payloads. Requests now use the current contract against a cell the materializer actually covered (`receiving_yards` is not one), and assertions check status code plus response shape: required fields, the requested stat present in the stat projection, ordered percentiles, non-null `model_version`. The helper lives in `e2e/lib/http_assert.sh` and is tested against a live server in `backend/tests/test_http_assert.py`, including cases proving it fails when it should.

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
