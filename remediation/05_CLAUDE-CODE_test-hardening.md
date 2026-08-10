# 05 · CLAUDE CODE · Test hardening

| | |
|---|---|
| **Order** | Wave 5 — after `04` |
| **Concurrency** | ⛔ **SOLO** |
| **Depends on** | `03A`, `03B`, `03C`, `04` all merged |
| **Blocks** | `06` |
| **Findings** | C-19 (complete — `01A` covered 5 locks; this finishes the set) |

### Setup

```bash
cd ~/Projects/Active/gridiron_oracle
git checkout main && git pull
git checkout -b fix/05-tests
```

Run in the main tree. Needs Postgres, MLflow, `.venv_311`, and a scratch DB for the schema tests.

---

Read `CONSOLIDATED_ADVERSARIAL_AUDIT_2026-08-09.md` row C-19.

Three tests are red on HEAD and the suite locks none of the invariants that actually matter.
`backend/tests/test_draft_api.py` is 10 lines of name normalization; `test_adp_eval.py` is Spearman and name
helpers. Across three independent reviews, the claims most at risk were the least tested.

## Part 1 — Fix what is red

Three genuine failures at the audit baseline (reviews reported 882/911 pass with ~3 real failures; a
targeted run showed 133/1). Fix:

- `test_kalman_tracker.py::TestSchemaSync` — should already be green from `03B`; confirm.
- `test_xgb_model.py::TestParseSeasons::test_range_notation` — should already be green from `01C`; confirm.
- `test_stacking_ensemble.py` four-learner assertions — should already be retired by `01A`; confirm and
  finish any that remain.
- Anything still failing after waves 1–4. **Report environmental failures separately from genuine ones** and
  say which is which — earlier reviews disagreed on counts purely because they conflated the two.

## Part 2 — Add the remaining locks

`01A` added five: learner policy, artifact pinning, manifest integrity, discovery hygiene, no-purge. Add:

1. **As-of feature contract** — no feature column equals any same-game postgame field. This is the C-01
   lock; it is the single most important test in the repo.
2. **Draft no-actuals / as-of** — target-season OOF unreachable from the draft path; rank↔games-played
   correlation bounded.
3. **Materializer completeness** — every cell declared in the manifest reaches the DB.
4. **Artifact-backed fail-closed** — no Kalman fallback under `PRODUCT_MODE=artifact_backed`, including for
   a coef file containing `NaN`, a missing intercept, or an unknown key.
5. **Gate can fail** — a deliberately regressed candidate is rejected; missing evidence fails closed.
6. **Push-gate ancestor scan** — the `.env`-in-ancestor reproduction exits nonzero.
7. **Collapse recurrence** — per-season prediction variance floor. This currently exists only as a shell
   heredoc in `scripts/retrain_fantasy_ppr_collapsed.sh`, not as a test.
8. **Two-team Phase-4 fixture** — should already exist from `02`; confirm it covers
   `pipeline/features/buckets.py` aggregation.
9. **Migration round-trip** — `upgrade → downgrade → upgrade` on a scratch DB, as CI.
10. **DraftBoard frontend tests** — should already exist from `03C`; confirm.

## Part 3 — Prove each lock is a lock

**Every new test must be shown to fail against the pre-fix commit.** A test that passes both before and
after locks nothing.

```bash
git worktree add ../go_locktest audit-baseline-2026-08-09
# copy each new test in, run it, record the failure
```

Record in `backend/tests/README.md`: for each lock, which finding it guards and which commit it was verified
red against. Delete or rewrite any test you cannot make fail against the baseline — and say which those were,
because that is a finding in itself.

## Done criteria

- Full suite green; environmental vs. genuine failures reported separately.
- All 10 locks present and each demonstrated red against `audit-baseline-2026-08-09`.
- `backend/tests/README.md` maps every lock to its finding and its verification commit.
- CI runs the migration round-trip and the push-gate reproduction.
