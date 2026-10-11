# 0006: Forward rows take game context from the target game

Type: Decision
Status: Accepted for the code; served numbers change only after a materialize or refresh run with production data
Last verified against code: 2026-10-10
Date: recorded 2026-10-10

## Context
A pre-kickoff row (`build_forward_week`, `pipeline/feature_engineer.py`) copied every non-identity column from the player's latest prior-season row. That copied the lines, rest, dome and surface of a different game into the target game. The row is causal, but it describes the wrong game. Also, `ml/served_learner.py` disclosed the stale candidate set `causal_20260810`, not the pinned `causal_20260821_fixed`.

## Options
1. Keep copying every column from the prior row.
2. Take the lines, rest, venue, and the lines-derived script columns from the target game. Keep copying player and team history from the prior row. Leave opponent, weather and injury columns as they were.
3. Also null the opponent, weather and injury columns for forward rows. This was proposed on `agent/gor-ml-02-forward-rows-served-learner` (945faa5) and is not adopted. Its reasoning is still open.

## Choice
Option 2, plus a fallback for missing lines:
- A missing target total or spread takes the training median: total 45.0, spread 2.5 (`_FWD_LINE_FALLBACK`).
- The cached training matrices (1,960 games, 2019-2025) have no missing lines. LightGBM maps an unseen NaN to 0, which would read as a 0-point total, so a missing forward line must not stay null.
- The served learner discloses `causal_20260821_fixed`. `tests/test_served_learner.py` reads the pin from `releases/current_baseline.json`.

## Why
Lines, rest and venue are schedule facts for the target game, so the target game is the only correct source. Option 3 needs its own decision because it changes what the model sees for opponent-based features.

## Caveats
- The fallback values are a choice. Owner to confirm before the VPS materialize run.
- Tests run without a database. The forward-row tests are fake-connection unit tests. A real materialize on production data is still needed to measure the change.
