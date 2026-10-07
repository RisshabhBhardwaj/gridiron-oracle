# 0001: Causal, walk-forward evaluation only

Type: Decision
Status: Accepted
Last verified against code: 2026-10-07
Date: recorded 2026-10-07 (the rule predates the record; the August 2026 leak fix enforced it)

## Context
Random k-fold on time series leaks the future. In August 2026 an audit found a leak (target-game snap data in features and in the eval cohort filter) that made results look far better than they were. One report records the headline falling from 74/75 to 15/75 after the fix (`reports/POST_LEAK_FIX_COMPARISON.md`).

## Options
1. Random or k-fold CV. Rejected: leaks.
2. Walk-forward folds with out-of-fold predictions, and baselines built only from prior information.

## Choice
Option 2. Predictions and baselines (previous season, trailing 3 games, Marcel) use only information available before the game. Training and OOF stop at `LAST_COMPLETE_SEASON` (`ml/season_constants.py`). Contemporaneous expected-fantasy-points is not allowed as a model input. Code: `ml/eval_causal.py`, `ml/baselines.py`, `ml/cohort_guards.py`.

## Why
Edge claimed on leaked data is worse than no claim. Skipping this check once already produced a false headline.
