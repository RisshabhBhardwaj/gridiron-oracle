# Gridiron Oracle

NFL player projections (weekly, season, fantasy draft) built from a Python data and ML pipeline, served by a FastAPI backend and a React frontend. A C++ engine (ring buffer, Kelly sizing, exposure limits, Monte Carlo) exists in `engine/`. Long-term goal: projections that are measurably better than simple baselines and public consensus, with every claim backed by a reproducible artifact. Betting (fractional Kelly sizing with exposure limits via the C++ engine) is in scope.

## Commands (verified 2026-10-07)

- Backend tests: `.venv_311/bin/python -m pytest backend/tests -q -m "not integration and not slow and not network"`. Use the CI env: `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OTEL_TRACES_EXPORTER=none`.
- Lint: `.venv_311/bin/ruff check .`
- Frontend: `cd frontend && npm run typecheck && npm run test && npm run build`
- Engine: `cmake -B engine/build engine && cmake --build engine/build && ctest --test-dir engine/build`
- Evidence manifest: `python scripts/verify_evidence_manifest.py`
- Dev servers: `.claude/launch.json` (api on 8000, web on 5173). Full stack: `make up` (docker).
- Not run in the audit (long or need a database): full ETL, model training, `make ci`, integration tests.

## Invariants not obvious from code

- Evaluation must be causal: predictions and baselines use only information available before the game. Walk-forward validation only, never random k-fold. Training and OOF stop at `LAST_COMPLETE_SEASON` (`ml/season_constants.py`).
- A leak was found and fixed in August 2026 (target-game snap data). Do not reintroduce features known only after kickoff. Contemporaneous expected-fantasy-points is forbidden as a model input.
- Serving is artifact-backed: `releases/current_baseline.json` pins model artifacts by SHA-256 and fails closed on mismatch. Do not edit pinned artifacts by hand; re-freeze with `make freeze-baseline`.
- Evidence files under ignored paths (`reports/`, `ml/oof/`, `ml/experiments/`) must be listed in `releases/artifacts/MANIFEST.json`. All entries are currently "provisional".
- `docs/DATA_SOURCES.md` must keep documenting the `load_pfr_advstats` call (a test enforces it).
- Public-push gate: `.githooks/pre-push` (enable with `make hooks`) blocks secrets and large objects; CI job `push-gate` is the authoritative check.
- Cite a performance number only if a repo file reproduces it, or label it "owner-verified; reproduction artifact not committed" (the 74/75 baseline-beating count is in that state; see README).
- Code comments citing "CLAUDE.md §5" refer to a file that no longer exists; ignore them.

## Principle

Every decision is made for the project's best long-term outcome, including scale and correctness, even when slower or harder. Never choose the easiest or quickest option for its own sake.

## Working principles

- **Trust the code.** Docs and earlier "done" claims can be stale. Verify by reading and running the code, and never report work complete without running the quality gate above.
- **Know the users.** Principles: `docs/UX_PRINCIPLES.md`. Who uses this and what each person needs is in `docs/USERS.md`. Show each person the least that lets them do everything they need, with a clear next step, in plain words anyone can understand.
- **Do not reinvent UI.** Use the primitives, component library, and tokens chosen in `docs/DESIGN_SYSTEM.md`. The experience should feel seamless and thoughtful.
- **Keep docs honest.** Short, plain, dated, and clear about what is built versus planned. Durable docs live in `docs/`; plans and research are labeled with date and status.
- **Keep context moving between tools.** Read `docs/status/CURRENT.md` at the start of a session. At the end of a session or milestone, update it (state, in progress, next steps, to-dos, open questions) and add a line to `docs/status/HISTORY.md`. Record decisions in `docs/decisions/` so no other tool overrules them without a new decision. Doc layout: `_internal-docs/_platform/DOCS_STANDARD.md`.
- **Research before building.** For any substantial capability (auth, UI components, data, infrastructure, runtime), first check whether a maintained open-source solution or dataset already exists and choose what is best long-term.
