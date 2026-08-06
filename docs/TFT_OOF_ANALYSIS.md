# TFT OOF Duplicate Analysis — Deep Dive

**Date:** 2026-03-06  
**Purpose:** Understand why TFT OOF has multiple rows per (player_id, game_id), whether TFT training was affected, and whether retraining is required.

---

## 1. Empirical Findings

### Row counts
| Source | Total rows | Unique (player_id, game_id) |
|--------|------------|-----------------------------|
| TFT rushing_tds | 178,524 | 99,465 |
| XGB rushing_tds (fold 0) | 4,076 | 4,076 |

### TFT duplicate distribution
- **80,429 games** have exactly 1 row (unique)
- **19,036 games** have 2+ rows (duplicates)
- **Max 13 rows** per game
- **Mean 5.15 rows** when duplicates exist

### Example: Tom Brady (00-0019596), 2020 Super Bowl (2020_21_KC_TB)
12 TFT predictions for the same game, all very similar:
```
y_pred: -0.000488, -0.000489, -0.000488, -0.000490, -0.000491, -0.000491,
        -0.000493, -0.000495, -0.000499, -0.000500, -0.000502, -0.000507
```
Range: 0.00002 (negligible variance).

### XGB vs TFT
- XGB: **one row per (player_id, game_id)** — correct for stacking
- TFT: **multiple rows per (player_id, game_id)** — causes merge duplication
- Overlap: 3,551 games appear in both (XGB is position-filtered; TFT is all-position)

---

## 2. Root Cause: `allow_missing_timesteps=True`

### How TFT builds validation samples
1. **TimeSeriesDataSet** uses `group_ids=[player_id]` and `time_idx` (global week index)
2. Each validation sample = one (encoder window, decoder target) pair
3. Decoder predicts exactly 1 step (`max_prediction_length=1`)

### Effect of `allow_missing_timesteps=True`
- Players have **gaps** in their timeline: bye weeks, injuries, DNPs
- With gaps, **multiple encoder windows** can predict the **same** (player_id, time_idx)
- Example: player has games at time_idx 5, 6, 10 (gaps at 7, 8, 9)
  - Window A: encoder [3,4,5,6] → predict 10
  - Window B: encoder [4,5,6]   → predict 10
  - Window C: encoder [5,6]     → predict 10
  - Window D: encoder [6]       → predict 10
- Each window has different encoder context → slightly different prediction
- **Result:** Multiple OOF rows per (player_id, game_id)

### Why this is intentional, not a bug
- `allow_missing_timesteps` is required for NFL data (bye weeks, injuries)
- pytorch_forecasting is designed to create these overlapping windows
- The docstring in `tft_model.py` states: *"allow_missing_timesteps: players have bye weeks and injury absences"*

---

## 3. Impact on TFT Training

### During training
- Each validation sample contributes to the loss
- Games after gaps appear in **multiple** validation samples
- Those games receive **more gradient updates** than games with no gaps
- ~19% of games (19,036 / 99,465) are affected

### Severity
- **Low.** Duplicate predictions are very similar (e.g. -0.000488 to -0.000507)
- The model is not learning contradictory targets; it's seeing the same target from different encoder lengths
- Can be viewed as mild data augmentation (multiple perspectives on the same game)

### Verdict: **No retraining required**
- The effect is small and likely benign
- Retraining would take days with minimal expected improvement
- The current TFT models are valid for use

---

## 4. Impact on Stacking

### Problem
- Ridge meta-learner expects **one row per (player_id, game_id)**
- Merge of XGB + LGBM + TFT produces duplicate rows when TFT has multiple rows per game
- Duplicates over-represent those games in the meta-learner

### Current fix (stacking_ensemble.py)
- `drop_duplicates(subset=["player_id", "game_id"], keep="first")`
- Keeps one row per game; discards the rest
- **Arbitrary:** "first" depends on merge order; we lose TFT information

### Better fix: **aggregate before merge**
- For OOF files with duplicates: `groupby(player_id, game_id).agg(mean(y_pred))`
- Use mean of duplicate predictions instead of keep="first"
- Preserves y_true, season, week, fold_idx (same across duplicates)
- **No retraining** — pure stacking-side change

---

## 5. Recommendations

### 1. Do NOT retrain TFT
- Root cause is library behavior, not a bug
- Training impact is minimal
- Your 4–5 days of training are valid

### 2. Improve stacking deduplication (optional)
- Change from `keep="first"` to **mean aggregation** when loading OOF files with duplicates
- More principled use of TFT's multiple predictions
- Implement in `load_and_align_oofs()` in `stacking_ensemble.py`

### 3. Re-run stacking only
- Your XGB, LGBM, TFT models are fine
- Re-run the stacking step (~10 min) with the current or improved dedup logic
- No base model retraining needed

---

## 6. Summary Table

| Question | Answer |
|----------|--------|
| Why does TFT have duplicates? | `allow_missing_timesteps=True` + player gaps → multiple encoder windows predict same game |
| Is TFT training wrong? | No. Slight over-weighting of post-gap games; effect is small |
| Do we need to retrain TFT? | **No** |
| Do we need to retrain XGB/LGBM? | **No** |
| What to do? | Re-run stacking step only. Optionally improve to mean aggregation. |
