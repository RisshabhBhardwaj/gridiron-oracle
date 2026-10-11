# 0005: Retire the SP0-SP4 rebuild plan; keep the built pieces

Type: Decision
Status: Accepted (owner asked for the call, 2026-10-10)
Last verified against code: 2026-10-10
Date: recorded 2026-10-10

## Context
The archived spec `2026-08-19-prediction-surfaces-rebuild-design.md` (in `_internal-docs`, archive-2026-10-07) set out SP0 (enablers), SP1 (scoreboard), SP2 (playing-time and role model), SP3 (season-long) and SP4 (draft). Its status line says "awaiting review". The code still carries the SP labels: `ml/playing_time.py` ("SP2"), `backend/app/services/projection.py`, `ml/draft_projection.py`, and `pipeline/features/feature_row.py`.

## Options
1. Continue the SP0-SP4 program as written.
2. Retire the program and keep the pieces that are built and served. Any further rebuild needs its own decision.
3. Delete the SP2 model and its consumers.

## Choice
Option 2.
- SP0: the fantasy-player ID map is built (`scraper/adapters/ff_playerids.py`).
- SP2: the playing-time and role model is built and served. It feeds the season and rest-of-season path (`projection.py`) and the draft's expected games (`ml/draft_projection.py`). It has tests (`backend/tests/test_playing_time.py`), including a cold-start QB check.
- SP1, SP3, SP4 are not active. Nothing in the repo records the Marcel comparison that the spec asked to run before SP2. `reports/marcel_vs_stack.json` covers fantasy points only.
- The SP labels stay in code for now. Renaming them is a separate change.

## Why
Season, rest-of-season and draft outputs already depend on the SP2 model. Rebuilding or deleting it would change served numbers without evidence. A new rebuild must beat Marcel and the naive baselines in the causal evaluation (decision 0001) before it is served.
