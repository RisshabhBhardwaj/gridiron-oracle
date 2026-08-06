# Contributing to Gridiron Oracle

> Reference for adding new capabilities to the platform.
> Start with [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`PIPELINE_RUNBOOK.md`](PIPELINE_RUNBOOK.md).

---

## Table of Contents

1. [Adding a New Stat](#adding-a-new-stat)
2. [Adding a New Base Learner](#adding-a-new-base-learner)
3. [Running Training](#running-training)
4. [Running the Backtest](#running-the-backtest)
5. [Running Tests](#running-tests)

---

## Adding a New Stat

A "stat" is a numeric target column (e.g. `passing_yards`, `receptions`). Adding one touches five places:

### 1. Feature matrix column

In `pipeline/feature_engineer.py`, add the raw stat to the `FeatureRow` dataclass (Bucket 1 or whichever bucket owns it). The `_FM_COLS` list is derived dynamically from `dc_fields(FeatureRow)`, so no secondary list needs updating.

### 2. Position filter

In `ml/backtest.py`, add the stat to `VALID_POSITION_STATS` for every position that can produce it:

```python
VALID_POSITION_STATS = {
    "QB":  {"passing_yards", "rushing_yards", ...},
    "WR":  {"receiving_yards", "receptions", ...},
    # add your stat here for each relevant position
}
```

### 3. SHAP label

In `ml/shap_service.py`, add a plain-English label to `FEATURE_LABELS`:

```python
FEATURE_LABELS = {
    ...
    "my_new_stat": "My New Stat (plain English)",
}
```

### 4. OOF column

Each base learner (`xgb_model.py`, `lgbm_model.py`, `catboost_model.py`, `tft_model.py`) writes OOF predictions for each stat. The stat name in the OOF CSV header must exactly match the column name in `feature_matrix`. No code change is needed unless you are adding a new learner (see below).

### 5. Ridge stacking coefficient file

After training, `ml/stacking_ensemble.py` writes `ml/oof/ridge_{stat}_{pos}_coefs.json` for every `(stat, position)` pair. These files are generated automatically during training — no manual step needed.

---

## Adding a New Base Learner

The stack currently has four base learners: XGBoost, LightGBM, CatBoost, TFT. To add a fifth:

### 1. Create the model file

Mirror the structure of `ml/xgb_model.py`:

```
ml/my_model.py
  class MyModel:
      def train(self, X_train, y_train, X_val, y_val) -> None
      def predict(self, X) -> np.ndarray
      def save(self, path: Path) -> None
      def load(self, path: Path) -> None
```

Required behaviour:
- Walk-forward CV only — no random k-fold.
- Log `mae`, `rmse`, `feature_importances`, `model_artifact`, `training_data_hash`, `timestamp` to MLflow before returning.
- Emit an OOF CSV to `ml/oof/my_model_{stat}_{pos}_oof.csv` with columns `[actual, predicted, position]`.

### 2. Register in train.py

In `ml/train.py`, add the model to `PipelineRunner._base_learners`:

```python
from ml.my_model import MyModel

self._base_learners = [
    ("xgb",  XGBModel),
    ("lgb",  LGBModel),
    ("cb",   CatBoostModel),
    ("tft",  TFTModel),
    ("my",   MyModel),   # ← add here
]
```

### 3. Register the OOF path in stacking_ensemble.py

In `ml/stacking_ensemble.py`, add the OOF glob pattern to `load_and_align_oofs()`:

```python
OOF_PATTERNS = [
    "xgb_*_oof.csv",
    "lgb_*_oof.csv",
    "cb_*_oof.csv",
    "tft_*_oof.csv",
    "my_*_oof.csv",   # ← add here
]
```

### 4. Verify promotion gate

A new learner is only promoted to the stack when its validation MAE beats the incumbent in MLflow. Run `make backtest` after training to confirm.

---

## Running Training

### Prerequisites

```bash
make up        # start Docker services (Postgres, MLflow)
make ingest    # full ETL (required on first run or after data update)
```

### Quick single-learner run

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# XGBoost only (fastest, ~5 min on a sample)
python -m ml.xgb_model --stat passing_yards --position QB
```

### Full training pipeline

```bash
bash ml/train_all_models.sh          # XGB + LGB + CB + TFT + stack + pipeline (~30-40h full)
bash ml/train_all_models.sh --resume # Resume after crash (skips completed checkpoints)
```

The script writes MLflow runs to `http://localhost:5001`. Open that URL to compare run metrics before and after any change.

### Weekly retrain (manual)

```bash
make retrain
```

### Required env vars (Apple Silicon)

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

Add to `~/.zshrc` permanently. Without them, XGBoost + PyTorch + scikit-learn deadlock in the OpenMP thread barrier.

---

## Running the Backtest

The backtest validates the full stack against seven seasons of held-out data (2019–2025) using expanding-window walk-forward CV — no random k-fold, no future data leakage.

### Run

```bash
make backtest
# or directly:
python -m ml.backtest --seasons 2019 2020 2021 2022 2023 2024 2025
```

### What it measures

| Metric | Description |
|--------|-------------|
| MAE | Mean absolute error per (position, stat) |
| RMSE | Root mean squared error |
| Brier score | Calibration of over/under probabilities |
| Simulated P&L | Bet edge × implied probability, season-length |
| Sharpe ratio | P&L / std-dev of weekly returns |
| Max drawdown | Worst peak-to-trough loss in the simulation |

Results are written to `ml/backtest_results/` and logged to MLflow.

### Calibration check

Brier score < 0.25 and calibration curves within 5% of the diagonal line are required before promoting a model to production. Verify in the MLflow UI (`/backtest` endpoint also serves these metrics).

---

## Running Tests

```bash
# Unit tests only (no DB required, fast)
pytest backend/tests/ -m "not integration"

# All tests including integration (requires make up)
pytest backend/tests/

# With coverage report
pytest backend/tests/ --cov=ml --cov=pipeline --cov=backend --cov-report=term-missing

# CI gate (70% coverage required, Catch2 C++ tests, typecheck, lint)
make ci
```

### Hypothesis property tests

Kalman filter edge cases are covered by Hypothesis in `backend/tests/test_kalman_tracker.py`. These run automatically in the unit test suite. To run them in isolation:

```bash
pytest backend/tests/test_kalman_tracker.py::TestKalmanProperty -v
```

### C++ engine tests

```bash
make bench                           # Catch2 ring buffer + Kelly benchmarks
cd engine/build && ./ws_integration_tests   # WebSocket integration tests
```
