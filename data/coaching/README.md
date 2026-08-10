# Coaching / scheme seed data

## No seed CSV ships with this repository

`coaching_2026.csv` was removed. It was provably wrong: it listed a former wide
receiver as New England's defensive coordinator, and had Kliff Kingsbury and Jim
Schwartz each coordinating two different teams. Every row's own `notes` field
said `seed — verify`, and the adapter performed no semantic validation, so all
32 rows would have upserted cleanly (audit finding C-24).

`team_coaching` has **zero consumers** — no feature, model, API route, or report
reads it — so shipping the file bought nothing and asserted 32 unverified facts.
Deleting it was cheaper and more honest than guessing at corrections.

If you want the table populated, supply a CSV you have actually verified.

## Producing a CSV

```bash
cp data/coaching/coaching_TEMPLATE.csv data/coaching/coaching_2026.csv
# fill it in, then:
python -m scraper.adapters.coaching_adapter --season 2026 --validate-only
python -m scraper.adapters.coaching_adapter --season 2026
```

### Columns

| Column | Required | Meaning |
|---|---|---|
| `season` | yes | Must match the `--season` argument |
| `team` | yes | One of the 32 current franchise codes, each exactly once |
| `head_coach` | yes | Full name |
| `offensive_coordinator` | yes | Full name |
| `defensive_coordinator` | yes | Full name |
| `dual_role` | when applicable | `true` when the HC also holds a coordinator title |
| `scheme_pass_rate_prior` | no | Probability in (0, 1) |
| `source` | yes | Where the staff list came from (team site, press release, URL) |
| `verified_by` | yes | Who checked it |
| `verified_on` | yes | ISO date, not in the future |
| `notes` | no | Free text |

## Validation

`scraper.adapters.coaching_adapter.validate_coaching` runs on every load and
raises `CoachingValidationError` listing **every** problem it finds:

- team codes among the 32 current franchises, each present exactly once
- no placeholder text (`verify`, `TBD`, `TBA`, `unknown`, `n/a`, `?`, `todo`,
  blank) in any name or provenance field
- no coordinator listed for two teams in the same role
- an HC who is also OC or DC must set `dual_role=true` — a real arrangement
  (Brian Schottenheimer, Liam Coen, Arthur Smith all held both titles at points)
  but one that must be declared rather than inferred
- `scheme_pass_rate_prior` strictly inside (0, 1)
- `source` / `verified_by` present, `verified_on` a real past date

What it deliberately does **not** check: whether a named person actually holds
the job. There is no coach registry in this repo to resolve names against, and a
check that cannot be performed should not be claimed. That is what `source`,
`verified_by`, and `verified_on` are for — provenance recorded per row, so a
stale or unsourced seed is visible instead of implied.

Coverage lives in `backend/tests/test_coaching_adapter.py`, including a fixture
that reproduces the exact defect shape of the deleted seed.

## Open handoff — Alembic must add the provenance columns

`coaching_adapter.CREATE_TEAM_COACHING` now declares `source`, `verified_by`,
and `verified_on`. `alembic/versions/20260806_0003_draft_and_coaching.py`
creates `team_coaching` **without** them, and `CREATE TABLE IF NOT EXISTS` is a
no-op against an existing table — so on any Alembic-migrated database
`upsert_coaching` would fail with `column "source" does not exist`.

This cannot fire today: no seed ships, the table has zero consumers, and the
adapter is not invoked by any pipeline. It is left for the schema work (audit
C-12), which owns that migration — **add the three columns in a new Alembic
revision before anything writes to `team_coaching`.**
