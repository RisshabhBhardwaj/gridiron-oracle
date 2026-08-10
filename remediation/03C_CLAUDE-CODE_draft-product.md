# 03C · CLAUDE CODE · Draft product

| | |
|---|---|
| **Order** | Wave 3, stream C |
| **Concurrency** | ⇄ **CONCURRENT** with `03A` and `03B` |
| **Depends on** | `02` merged |
| **Blocks** | `05` |
| **Findings** | C-02, C-03, C-14, C-15, C-22 |
| **Shared resources** | **Read-only on Postgres** — `03A` owns writes. If you need materialized rows, wait for `03A` to land. |

### Setup

```bash
cd ~/Projects/Active/gridiron_oracle
git worktree add ../go_03C_draft -b fix/03c-draft main
cd ../go_03C_draft
```

Interpreter: `/Users/risshabh/Projects/Active/gridiron_oracle/.venv_311/bin/python`.
Frontend: `npm ci` inside the worktree (node_modules do not carry across worktrees).
**You own `backend/app/api/draft.py` this wave** — `03B` has been told to stay out of it, including
the C-15 query fix which now belongs to you.

---

Read `CONSOLIDATED_ADVERSARIAL_AUDIT_2026-08-09.md` rows C-02, C-03, C-14, C-15, C-22. This is the
largest behavioural change in the programme: **the draft output is currently not a prediction.**

## C-03 — Replace the rank basis (do this first; everything else depends on it)

`backend/app/api/draft.py:114-135` and `ml/adp_eval.py:186-211` sum target-season weekly OOF rows,
through week 22 (postseason included). Measured: Spearman(season_sum, games_played) = **0.926**;
games-played rank alone explains **86%** of season-sum rank variance. The headline draft output is
dominated by realized availability, which is unknowable at draft time. Adam Thielen 2025 is ranked from
eight actual appearances.

- Build an as-of-draft-date projection: **per-game mean × a games-played prior derived from pre-draft
  information only.** Fixed eligible-player universe, not "whoever appeared."
- **Ban target-season OOF as a draft source.** The endpoint must refuse it, not prefer it.
- Carry explicit `as_of` and `source` metadata on every response.
- Add a test asserting rank↔games-played correlation stays below a threshold. **It must fail against
  `audit-baseline-2026-08-09`.**

## C-02 — Make 2026 actually work

`projections` spans 2021–2025; `fantasy_adp` 2019–2025; zero 2026 rows; no preseason projection path.
`frontend/src/pages/DraftBoard.tsx:14-25` hardwires the current season with no selector, so
`/draft/board?season=2026` returns 404 — the default page, for the only season a drafter needs.

- Ingest 2026 ADP via the Sleeper public API or the FantasyPros manual CSV path. Both are already
  implemented and free-source-clean. **Do not add a scraper.**
  - If `01C` fixed the FP importer alias collision (C-23), rebase onto it first — otherwise a standard FP
    export imports Rank as ADP.
- Generate genuine preseason (week-0) projections using the C-03 machinery.
- Add a season selector to the frontend; remove the hardwired current-season assumption; fix the source
  dropdown that currently offers options that always error.
- **Add DraftBoard tests — there are currently none.** The frontend build passing is not coverage.

## C-15 — Fix the dead DB fallback

`backend/app/api/draft.py:138-166` and `ml/adp_eval.py:105-144` probe for projection columns `mean` /
`projected_mean`; the schema, ORM, service and materializer all use `projection`
(`pipeline/schema_ddl.py:487-512`, `backend/app/models/production.py:425-463`). Verified:
`_fetch_db_projections_ppr(conn, 2024)` returns **zero rows** against a populated 48,058-row DB. The
documented stack→DB→404 chain can never reach its middle link. Even fixed, it uses `AVG` where the stack
path uses `SUM` — different scales, so switching `model_source` silently changes ranks.

- Query `projection`; put both sources on one scale.
- Validate one row per (player, game, stat).
- Add a test exercising DB-only draft behaviour with OOF discovery out of scope.

## C-22 — Join on IDs, not names

`fantasy_adp.player_id` exists but every local value is NULL (`scripts/import_historical_adp.py:63-81`);
`draft.py:214-223` and `ml/adp_eval.py:159-183` join on normalized names. Name-only joins drop 7–10 real
players per season (Hollywood Brown, Josh Palmer verified) and the board shows `model_rank=—` for them.
The `fantasy_adp` PK on `player_name` will collide on duplicate names.

- Resolve to `player_id` at import with an audited match table. **Report unmatched rows rather than
  silently dropping them.**
- Re-key the primary key.

## C-14 — Stop citing a metric with no discriminating power

Reported ADP Spearman reproduces (0.459/0.485/0.438/0.455 for 2022–25) — but ranking by **season actuals**,
which the model may not use, scores 0.458/0.474/0.436/0.479. The metric measures ADP quality, not model skill.

- Report the Spearman **gap** against an actuals-oracle and a prev-season baseline, or drop it as a model
  claim entirely.
- `README.md:11` headlines `--from-actuals` (ρ=0.514) as "Beat ADP?" — this instructs users to cite a
  hindsight number. Fix it.
- `reports/adp_spearman_stack_ranks.json` is hand-assembled and omits QB-2025 (ρ=0.055, p=0.78). Regenerate
  from code with p-values and no omissions.

## Done criteria

- `/draft/board?season=2026` returns a real board built from preseason projections.
- No target-season OOF reachable from the draft path.
- rank↔games-played test passes, and fails against the audit-baseline tag.
- ADP joins on `player_id`; unmatched rows reported, not dropped.
- DB-only draft path returns rows under test.
- Frontend has a season selector and DraftBoard tests.
- README no longer advertises a hindsight metric.
