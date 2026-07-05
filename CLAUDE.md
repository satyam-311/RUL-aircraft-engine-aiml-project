# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management

This project uses `uv`. Never use `pip` or `uv pip` directly — all dependency changes go through `pyproject.toml`.

```powershell
uv sync --all-groups          # Install all deps including dev
uv add <package>              # Add a runtime dep (updates pyproject.toml + uv.lock)
uv add --dev <package>        # Add a dev dep
uv run python <script>        # Run inside the managed venv
```

`requires-python = ">=3.12,<3.13"` is intentionally narrow — do not widen it. The `numba>=0.60` pin is a deliberate constraint to prevent the resolver from backtracking to `llvmlite<0.40`, which cannot build on Python 3.12.

## Common Commands

```powershell
# Lint
uv run ruff check src/ tests/

# Type-check
uv run mypy src/

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_foo.py -v

# Smoke-test data ingestion
uv run python -m rul_prediction.components.data_ingestion

# Smoke-test preprocessor (saves to data/processed/ and artifacts/)
uv run python -m rul_prediction.components.preprocessor

# Run the Streamlit dashboard
uv run streamlit run app/dashboard.py
# Or double-click run_dashboard.bat from Explorer

# Execute EDA notebook headlessly
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 notebooks/01_eda_fd001.ipynb
```

## Component Architecture Rule

All pipeline step classes go as individual files inside `src/rul_prediction/components/` — one class per file, named after the step. `src/rul_prediction/pipeline/` holds orchestration scripts that chain components. Never create a separate top-level folder per step.

## Architecture

### src layout
The package lives in `src/rul_prediction/` and is installed as an editable package by `uv sync`. Always import as `from rul_prediction.X import Y`.

### Constants vs Config
Two separate configuration layers — do not conflate them:

- **`src/rul_prediction/constants/constants.py`** — immutable dataset schema: `ALL_COLUMNS`, `SUBSETS`, `RANDOM_SEED`, and `NON_INFORMATIVE_SENSORS` (7 sensors empirically confirmed flat in FD001: sensor_1, 5, 6, 10, 16, 18, 19). Must not be made configurable.
- **`configs/config.yaml`** — runtime knobs: `active_subset`, `window_size` (30), `rul_cap` (125), `model.type` (currently `"lstm"`), DL hyperparameters. Loaded via `src/rul_prediction/config/configuration.py`.

### Logging
Use `get_logger(__name__)` from `rul_prediction.logger.logger` at module level. Writes to `logs/rul_prediction.log` with a 5 MB rotating handler (3 backups). Handlers are deduplicated on repeated calls. Avoid Unicode arrow characters (`→`) in log strings — Windows console (cp1252) cannot encode them.

### Exception handling
Wrap all `except` blocks with `raise RULException(e, sys) from e`. `RULException` (in `rul_prediction.exception.exception`) extracts originating filename and line number from the traceback.

### Data layout
Raw CMAPSS files live under `data/raw/<subset>/` (no header, space-delimited, sometimes trailing NaN columns). `parse_cmapss_bytes(buf, columns)` in `data_ingestion.py` is the single shared parser — used by both `DataIngestion._read_raw_file()` and the dashboard upload handler. Training RUL is computed as `max_cycle_for_engine - current_cycle`; test RUL comes from `RUL_<subset>.txt` and must **never be capped** (ground truth for evaluation).

Currently only FD001 data is present. FD002–FD004 require downloading raw files to `data/raw/FD002/` etc. and retraining.

### Preprocessor outputs
```
data/processed/
    train_X_flat.parquet   train_X_seq.npy   train_y.npy   train_engine_ids.npy
    val_X_flat.parquet     val_X_seq.npy     val_y.npy     val_engine_ids.npy
    test_X_flat.parquet    test_X_seq.npy                  test_engine_ids.npy

artifacts/
    preprocessor.pkl       # fitted MinMaxScaler + feature list — required by inference
```

- `X_seq` shape: `(n_windows, 30, 17)` — for LSTM/GRU/BiLSTM
- `X_flat` shape: `(n_windows, 510)` — for XGBoost/LightGBM; parquet preserves column names
- Train/val split is **by engine ID** (80/20), never by row, to prevent data leakage
- `y` for train/val is RUL capped at 125; test set has no `y` (use `test_rul_df` from `DataIngestion` at eval time)

### Model training and saved models

Two trainer classes exist side by side:

- **`ModelTrainer`** (`model_trainer.py`) — scikit-learn / XGBoost / LightGBM / CatBoost. Saves as `.pkl` via joblib. Runs Optuna HPO. Uses `X_flat`.
- **`SequenceTrainer`** (`sequence_trainer.py`) — PyTorch LSTM/GRU/BiLSTM. Saves weights as `.pt` (state dict) plus a companion `_config.json` for architecture reconstruction. Uses `X_seq`. Never joblib-pickles the full class.

Trained models land in `saved_models/` with a timestamp stem (e.g. `lstm_20260705_012557.pt`). `model.type` in `config.yaml` controls which model `InferenceEngine` loads.

**Production model: LSTM** — chosen because its low-RUL bias is −0.43 (conservative/safe). BiLSTM and XGBoost both have positive bias (overestimate RUL near failure). RMSE difference vs BiLSTM is noise-level (15.04 vs 15.02). Full comparison in `reports/model_comparison.csv`.

### Asymmetric loss
`asymmetric_rul_loss` in `sequence_trainer.py` applies penalty weight `alpha=2.0` when `(y_pred > y_true) AND (y_true < rul_threshold=30)`. This directly targets optimistic bias in the danger zone. Do not replace with standard MSE.

### Inference pipeline
`InferencePipeline` (`pipeline/inference_pipeline.py`) is the end-to-end entry point:
```
engine_df (raw CMAPSS rows)
  -> Preprocessor.transform()   -> X_seq (n_windows, 30, 17)
  -> X_seq[-1:]                 -> (1, 30, 17)  last window = current state
  -> InferenceEngine.predict()  -> PredictionResult
```
`InferencePipeline.predict_trajectory(engine_df)` calls `predict()` for every window and returns a list of `PredictionResult` — used for the RUL trend chart in the dashboard.

Both the preprocessor and model are lazy-loaded and cached, safe to instantiate once at server startup.

**Pickle compatibility**: `preprocessor.pkl` was saved with `Preprocessor` recorded as its originating class. Any script that loads it must have `from rul_prediction.components.preprocessor import Preprocessor  # noqa: F401` at module level so pickle can resolve the class, even if the symbol is unused. Same applies to `ModelTrainer` when loading baseline `.pkl` files.

### Dashboard (`app/dashboard.py`)
Four-tab Streamlit app: Dataset Overview, Upload & Predict, Model Comparison, SHAP Explainability. All source strings are ASCII-only — do not introduce Unicode/emoji (Windows cp1252 console encoding causes corruption). Theme is set via `.streamlit/config.toml` (dark navy, sky-blue accent). Run via `run_dashboard.bat` or the `uv run streamlit` command above.

### SHAP explainability
`ModelExplainer` (`components/explainer.py`) generates global (bar + beeswarm) and local (waterfall) SHAP figures for both XGBoost and the configured DL model. Output lands in `reports/figures/`. Key finding: both models rank `sensor_11` (HPC static pressure) as the top predictor, consistent with EDA Pearson correlations.

### Phase roadmap
**Complete:** 1 (infrastructure), 2 (data ingestion + EDA), 3 (preprocessing), 4 (baseline ML), 5 (deep learning), 6 (SHAP explainability), 8 (inference pipeline), 9 (Streamlit dashboard).

**Upcoming:** Phase 10 — FastAPI inference endpoint (`uv add fastapi uvicorn pydantic`).
