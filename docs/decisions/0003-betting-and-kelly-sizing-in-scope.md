# 0003: Betting and Kelly sizing stay in scope

Type: Decision
Status: Accepted
Last verified against code: 2026-10-07 (engine exists and its 72 tests pass; live odds connection not verified)
Date: 2026-10-07 (owner decision)

## Context
The repo has a C++ engine (fractional Kelly 0.25x, 5% per-bet cap, exposure limits 5% player / 20% game / 20% total) and a frontend that shows edge against a book line. It was unclear whether betting was still a goal.

## Choice
The owner confirmed on 2026-10-07 that betting and Kelly sizing are in scope, alongside fantasy and draft projections.

## Consequences
Projections must be calibrated, not only well ranked. The UI has no bet slip and shows "Book Line 0" when no line exists; the engine's link to a live odds feed is unverified. Both are open work (see `docs/status/CURRENT.md`).
