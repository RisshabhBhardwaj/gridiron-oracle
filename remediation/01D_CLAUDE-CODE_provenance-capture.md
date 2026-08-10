# 01D · CLAUDE CODE · Provenance capture for UNPROVEN features

| | |
|---|---|
| **Order** | Wave 1, stream D — **inserted before 02** |
| **Concurrency** | ⛔ **SOLO** — but short. Blocks `02`. |
| **Depends on** | wave 1 merged (`6c17c43` + the containment follow-up commit) |
| **Blocks** | `02` — its output determines 02's final feature list |
| **Findings** | resolves the `UNPROVEN` rows in 01B's as-of audit |
| **Runtime** | one session. ETL work, no model training. |

### Setup

```bash
cd ~/Projects/Active/gridiron_oracle
git checkout main && git pull
git checkout -b fix/01d-provenance
```

Needs Postgres `:15439` and network access for nflreadpy. No MLflow, no training.

---

Session `01B` audited every default feature for as-of safety. It produced three verdicts: `LEAK`
(reads the target game — session `02` removes these), `LATENT LEAK` (inert today, would leak on a data
refresh — `02` fixes these too), and **`UNPROVEN`** — features where nobody recorded *when* the source
value was captured, so pregame availability cannot be demonstrated.

Session `02`'s standing instruction is "treat UNPROVEN as LEAK." Applied literally that removes roughly
ten features that are probably legitimate. **Your job is to convert every `UNPROVEN` into a proven
`SAFE` or a documented `UNOBTAINABLE`, so `02` starts from a decided list rather than a blanket rule.**

Read `remediation/02_01B_PASTE_READY_FINDINGS.md` for the full table. The `UNPROVEN` set is:
`temp_f`, `wind_mph`, `temp_bucket`, `wind_bucket`, `wind_x_qb`, `wind_x_wr`, `precip_x_pass`,
`height`, `weight`, `depth_chart_rank`, `injury_status_encoded`.

## GUARDRAILS

1. **Do not rebuild `feature_matrix`, do not retrain, do not restack.** That is `02`. You are
   establishing what is provable, not acting on it.
2. Do not remove any feature from `FEATURE_COLS`. Recommend; `02` executes.
3. Do not touch `ml/oof/`. Session `01A` just finished cleaning it and every serving artifact is
   SHA-pinned.
4. Schema changes are fine and expected, but they must go through Alembic as new revisions — do not add
   columns via `ensure_schema` or runtime DDL (finding C-12).

## Part 1 — Establish provenance where it exists

For each `UNPROVEN` column, determine what the source actually is and whether a pregame-dated value can
be recovered. Expected outcomes, to be confirmed rather than assumed:

- **`height`, `weight`** — `pipeline/feature_engineer.py:178-194` reads current `players.*`; DB shows
  129,103 feature values exactly equal the current roster value, with no historical effective dates.
  nflreadpy exposes per-season rosters. Recover as-of-season values, store them with the season they
  belong to, and join by `(player_id, season)`.
- **`depth_chart_rank`** — `feature_engineer.py:793-803` joins `depth_charts` on the same week. Depth
  charts are week-granular and published before the game; the gap is a recorded publication timestamp,
  not the data. Establish and store one.
- **`temp_f`, `wind_mph`** and everything derived from them (`temp_bucket`, `wind_bucket`, `wind_x_qb`,
  `wind_x_wr`, `precip_x_pass`) — `games.temp` / `games.wind` most likely originate from nflreadpy
  schedules, which record **observed game conditions, not pre-kickoff forecasts**. Verify this. If
  confirmed, these are `UNOBTAINABLE` retroactively — a 2021 forecast cannot be manufactured, and free
  forecast archives do not reach back. Say so plainly rather than inventing a proxy.
- **`injury_status_encoded`** — 0/129,128 non-null because `FeatureEngineer.run()` never passes
  `injury_df`. Confirm, then mark **moot**: there is nothing to prove about a column that is never
  populated. Note for `02` that wiring it up later requires dated ESPN practice reports.

## Part 2 — Record provenance in the schema

Where a value is recoverable, the fix is not just to fetch it — it is to make the as-of relationship
**checkable**.

- Add capture/effective-date columns to the relevant source tables via a new Alembic revision (e.g.
  `players_history.effective_season`, `depth_charts.published_at`, `games.weather_captured_at`).
- Backfill where the data supports it. Leave NULL where it does not, and treat NULL as illegal at
  feature-build time rather than silently passing.
- Every feature built from these sources must be able to assert `source_captured_at < kickoff` (or
  `effective_season <= season`) at build time. That assertion is what `02` will wire into the as-of
  contract — give it something to assert against.

## Part 3 — Forward capture, even where retroactive fails

For anything you mark `UNOBTAINABLE`: **start capturing it correctly now.** Weather is the likely case —
add a pre-kickoff forecast capture step to the pipeline (scheduled ahead of each slate, stored with a
timestamp) so 2026 onward has real provenance even though 2021–2025 never will.

This is the difference between "we deleted weather" and "weather is unavailable historically and will be
available prospectively." Document it in `docs/DATA_SOURCES.md`, which `01C` created.

## Part 4 — Deliverable

`remediation/01D_PROVENANCE_DECISIONS.md`, one row per `UNPROVEN` column:

| Column | Source | Retroactive verdict | Evidence | Forward capture | 02 disposition |
|---|---|---|---|---|---|

`Retroactive verdict` ∈ `PROVEN SAFE` (as-of value recovered and dated) · `UNOBTAINABLE` (no pregame
value exists, with the reason) · `MOOT` (never populated).

`02 disposition` must be one of: **keep** (with the join it must now use), **remove** (with a one-line
justification), or **remove-and-recapture** (removed historically, captured going forward).

Then **edit `remediation/02_CLAUDE-CODE_feature-contract-retrain.md` directly**: replace the blanket
"treat UNPROVEN as LEAK" instruction with your decided list, so `02` acts on evidence rather than a
default. Note in that edit which features `02` should now expect to *keep*, because the current wording
would have it delete them.

## Done criteria

- Every `UNPROVEN` column resolved to `PROVEN SAFE` / `UNOBTAINABLE` / `MOOT` with evidence.
- Recoverable as-of values stored and dated, via Alembic revisions only.
- A build-time assertion exists that `02` can wire into the as-of contract.
- Forward capture in place for anything unobtainable retroactively.
- `02`'s prompt edited to carry the decided list.
- No feature removed, no feature matrix rebuilt, no model trained.

**Report the split explicitly** — how many of the eleven are keepable, how many are gone, and how much of
`02`'s scope that changes. If most turn out unobtainable, say so directly; that is a legitimate result and
better than a proxy that quietly reintroduces the same class of defect.
