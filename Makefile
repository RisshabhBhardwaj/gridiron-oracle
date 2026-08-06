COMPOSE := docker compose -f infra/docker-compose.yml
NPROC   := $(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
PYTHON  ?= .venv_311/bin/python
MUTMUT_BIN := .venv_311/bin/mutmut
MUTMUT_JOBS ?= 2
export ODDS_API_KEY ?=
export OPENWEATHER_API_KEY ?=

# ── Environment ───────────────────────────────────────────────────────────────
.PHONY: setup
setup:
	$(COMPOSE) build

.PHONY: hooks
hooks:
	@mkdir -p .git/hooks
	@cp .githooks/pre-push .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo "Installed public-push gate into .git/hooks/pre-push"

.PHONY: migrate
migrate:
	$(PYTHON) scripts/migrate.py upgrade

.PHONY: migrate-stamp
migrate-stamp:
	$(PYTHON) scripts/migrate.py stamp

.PHONY: bootstrap-local
bootstrap-local:
	bash scripts/bootstrap_local.sh
	$(MAKE) hooks

.PHONY: verify-local
verify-local:
	bash scripts/verify_local.sh

.PHONY: verify-readiness
verify-readiness:
	$(PYTHON) scripts/verify_release_readiness.py

.PHONY: verify-artifact-prerequisites
verify-artifact-prerequisites:
	$(PYTHON) scripts/verify_artifact_prerequisites.py

.PHONY: yardage-diagnostic
yardage-diagnostic:
	@test -n "$(INPUT)" || (echo "Usage: make yardage-diagnostic INPUT=path.csv TARGET=passing_yards POSITION=QB"; exit 2)
	$(PYTHON) scripts/yardage_diagnostic_report.py --input $(INPUT) --target $(or $(TARGET),passing_yards) --position $(or $(POSITION),QB) --json-out reports/yardage_diagnostic.json

.PHONY: freeze-baseline
freeze-baseline:
	$(PYTHON) scripts/freeze_baseline.py

.PHONY: reprojection-gate
reprojection-gate:
	$(PYTHON) scripts/reprojection_gate.py --holdout-season $(or $(HOLDOUT),2024) --position $(or $(POSITION),WR) --target $(or $(TARGET),fantasy_ppr)

.PHONY: up
up:
	$(COMPOSE) up -d

.PHONY: down
down:
	$(COMPOSE) down -v

.PHONY: logs
logs:
	$(COMPOSE) logs -f

.PHONY: tracing-logs
tracing-logs:
	$(COMPOSE) logs -f backend otel-collector jaeger

# ── Data ──────────────────────────────────────────────────────────────────────
.PHONY: ingest
ingest:
	$(COMPOSE) run --rm backend python -m pipeline.orchestrator

.PHONY: clear-dead-letter
clear-dead-letter:
	$(COMPOSE) exec -T db psql -U oracle -d oracle -c "TRUNCATE TABLE dead_letter;"

# Reset nextgen/combine staging so normalize re-runs with new mappings.
.PHONY: reset-staging-for-remap
reset-staging-for-remap:
	@echo "Resetting processed=FALSE for nextgen_stats and combine…"
	$(COMPOSE) exec -T db psql -U oracle -d oracle -c \
		"UPDATE staging_nflreadpy SET processed = FALSE WHERE source_type IN ('nextgen_stats', 'combine');"

# ── ML ────────────────────────────────────────────────────────────────────────
.PHONY: retrain
retrain:
	$(COMPOSE) run --rm backend python -m ml.train

.PHONY: backtest
backtest:
	$(COMPOSE) run --rm backend python -m ml.backtest

.PHONY: diagnose-backtest
diagnose-backtest:
	$(PYTHON) scripts/diagnose_backtest_edge.py

.PHONY: verify
verify:
	$(COMPOSE) run --rm backend python -m ml.verify_pipeline_setup

.PHONY: verify-smoke
verify-smoke:
	$(COMPOSE) run --rm backend python -m ml.verify_pipeline_setup --smoke-test

# ── C++ Engine ────────────────────────────────────────────────────────────────
.PHONY: engine
engine:
	cmake -B engine/build engine/ \
	      -DCMAKE_BUILD_TYPE=Release \
	      -DCMAKE_CXX_STANDARD=17 \
	      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	cmake --build engine/build -j$(NPROC)

.PHONY: bench
bench:
	$(MAKE) engine
	./engine/build/ring_buffer_test --benchmark-samples 100

# ── MLflow ────────────────────────────────────────────────────────────────────
.PHONY: mlflow-backup
mlflow-backup:
	@BACKUP_DIR="mlruns_backup_$$(date +%Y%m%d_%H%M%S)"; \
	cp -r mlruns "$$BACKUP_DIR" && \
	echo "MLflow artifacts backed up to $$BACKUP_DIR"

# ── Testing ───────────────────────────────────────────────────────────────────
.PHONY: test
test:
	# Default repo test pass: skip DB-backed integration tests.
	$(COMPOSE) run --rm backend pytest backend/tests/ -v -m "not integration"
	$(MAKE) engine
	ctest --test-dir engine/build --output-on-failure

.PHONY: test-integration
test-integration:
	$(COMPOSE) run --rm backend pytest backend/tests/test_integration.py -v -m integration

.PHONY: mutmut
mutmut:
	test -x $(MUTMUT_BIN)
	$(MUTMUT_BIN) run --max-children $(MUTMUT_JOBS)

.PHONY: mutmut-results
mutmut-results:
	test -x $(MUTMUT_BIN)
	$(MUTMUT_BIN) results

.PHONY: mutmut-browse
mutmut-browse:
	test -x $(MUTMUT_BIN)
	$(MUTMUT_BIN) browse

.PHONY: mutmut-time-estimates
mutmut-time-estimates:
	test -x $(MUTMUT_BIN)
	$(MUTMUT_BIN) print-time-estimates

.PHONY: ci
ci:
	# Skip integration/network/slow tests and enforce 80% coverage gate.
	$(COMPOSE) run --rm backend pytest backend/tests/ -v -m "not integration and not network and not slow" \
		--cov=ml --cov=pipeline --cov=backend \
		--cov-report=term-missing --cov-fail-under=80
	$(MAKE) engine
	ctest --test-dir engine/build --output-on-failure
	$(MAKE) typecheck
	$(COMPOSE) run --rm backend ruff check .
	$(MAKE) complexity

.PHONY: complexity
complexity:
	# Hard gate: fail CI on rank F functions (CC ≥ 26). These should never exist.
	# Use `make complexity-report` to see all C/D/E violations for refactor tracking.
	@output=$$($(COMPOSE) run --rm backend radon cc -n F backend/app ml pipeline); \
	if [ -n "$$output" ]; then \
		echo "$$output"; \
		echo "Complexity gate FAILED: rank-F functions (CC ≥ 26) detected. Refactor before merging."; \
		exit 1; \
	fi

.PHONY: complexity-report
complexity-report:
	# Report all functions with CC ≥ 11 (rank C+). Informational — does not fail.
	# Known D/E violations in xgb_model.train, lgbm_model.train, stacking_ensemble.stack
	# are inherent to walk-forward ML training loops. Track and reduce over time.
	$(COMPOSE) run --rm backend radon cc -n C -s backend/app ml pipeline || true

.PHONY: typecheck
typecheck:
	$(COMPOSE) run --rm frontend npm run typecheck

.PHONY: smoke
smoke:
	# End-to-end smoke test. Requires `make up` first (stack must be running).
	# Does NOT run in `make ci` automatically — call explicitly after `make up`.
	bash e2e/smoke_test.sh
