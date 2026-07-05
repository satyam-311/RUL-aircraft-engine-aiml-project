# Aircraft Engine RUL Prediction

Predicting the **Remaining Useful Life (RUL)** of turbofan aircraft engines using multivariate
time-series sensor data from the NASA CMAPSS benchmark dataset.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rul-aircraft-engine-aiml-project-jc326sqju3ue7tvwzoor6y.streamlit.app/)

---

## Live Demo

| Service | URL |
|---------|-----|
| Streamlit Dashboard | [https://rul-aircraft-engine-aiml-project-jc326sqju3ue7tvwzoor6y.streamlit.app/](https://rul-aircraft-engine-aiml-project-jc326sqju3ue7tvwzoor6y.streamlit.app/) |
| FastAPI Endpoint (Render) | Deploy your own — see [Render Deployment](#render-deployment-fastapi) below |
| GitHub Repository | [https://github.com/satyam-311/RUL-aircraft-engine-aiml-project](https://github.com/satyam-311/RUL-aircraft-engine-aiml-project) |

---

## What It Does

| Tab | Description |
|-----|-------------|
| **Dataset Overview** | Sensor distributions, RUL histograms, degradation trend charts for FD001 |
| **Upload & Predict** | Upload a CMAPSS-format `.txt` file, get per-engine RUL predictions + safety flags |
| **Model Comparison** | RMSE, MAE, bias metrics across LSTM / GRU / BiLSTM / XGBoost |
| **SHAP Explainability** | Global feature importance and local per-engine waterfall explanations |

---

## Production Model

**LSTM** — trained on NASA CMAPSS FD001 with an asymmetric loss function that applies
a 2× penalty for optimistic predictions when true RUL < 30 cycles (the danger zone).

| Metric | LSTM | BiLSTM | XGBoost |
|--------|------|--------|---------|
| RMSE | 15.04 | 15.02 | 16.31 |
| Low-RUL bias | **-0.43** | +1.42 | +2.54 |

LSTM chosen because its bias is **conservative** (slightly under-predicts near failure),
which is the safe direction for a maintenance scheduling system.

---

## Tech Stack

- **Data:** NASA CMAPSS FD001 (100 engines, 21 sensors, ~20 000 training cycles)
- **Models:** PyTorch (LSTM / GRU / BiLSTM), XGBoost, LightGBM, scikit-learn
- **Explainability:** SHAP (TreeExplainer + DeepExplainer)
- **Dashboard:** Streamlit + Plotly
- **API:** FastAPI + Uvicorn (Phase 10)
- **Package management:** `uv` (see `pyproject.toml` + `uv.lock`)

---

## Local Setup

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (`pip install uv` or `winget install astral-sh.uv`)

### Install

```powershell
git clone https://github.com/satyam-311/RUL-aircraft-engine-aiml-project.git
cd RUL-aircraft-engine-aiml-project
uv sync --all-groups
```

### Run the dashboard

```powershell
uv run streamlit run app/dashboard.py
# OR double-click run_dashboard.bat
```

### Run the FastAPI inference endpoint

```powershell
uv run uvicorn app.api:app --port 8000 --reload
# OR double-click run_api.bat
# Swagger UI: http://127.0.0.1:8000/docs
```

### Reproduce training (optional)

The trained models are already committed. If you want to retrain from scratch:

```powershell
# 1. Re-run preprocessing
uv run python -m rul_prediction.components.preprocessor

# 2. Train baseline ML (XGBoost / LightGBM)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline_models.ipynb

# 3. Train deep learning (LSTM / GRU / BiLSTM)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_deep_learning_fd001.ipynb
```

---

## Project Structure

```
.
├── app/
│   ├── dashboard.py        # Streamlit dashboard (4 tabs)
│   └── api.py              # FastAPI inference endpoint
├── configs/
│   └── config.yaml         # Runtime knobs (model type, window size, RUL cap)
├── data/
│   ├── raw/FD001/          # NASA CMAPSS raw files (train/test/RUL)
│   └── processed/          # Preprocessed numpy arrays (gitignored, regeneratable)
├── notebooks/
│   ├── 01_eda_fd001.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_deep_learning_fd001.ipynb
├── reports/
│   ├── model_comparison.csv
│   ├── dl_comparison.csv
│   └── figures/            # SHAP plots, residual plots, sensor charts
├── saved_models/           # Trained model weights (.pt) + companion configs
├── artifacts/
│   └── preprocessor.pkl    # Fitted MinMaxScaler + feature list
├── src/rul_prediction/
│   ├── components/         # DataIngestion, Preprocessor, ModelTrainer,
│   │                       # SequenceTrainer, InferenceEngine, Explainer
│   ├── pipeline/           # InferencePipeline (end-to-end orchestration)
│   ├── config/             # YAML loader
│   ├── constants/          # Fixed column names and sensor lists
│   ├── logger/             # Rotating file logger
│   └── exception/          # RULException with traceback extraction
├── tests/
├── pyproject.toml          # Package manifest + dependency groups
├── uv.lock                 # Locked dependency graph (reproducible installs)
└── requirements.txt        # pip-compatible deps (used by Streamlit Cloud)
```

---

## Streamlit Cloud Deployment

The app is deployed at the URL above. To deploy your own fork:

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account
3. Set **Main file path** to `app/dashboard.py`
4. Set **Python version** to `3.12`
5. Click **Deploy** — Streamlit Cloud will install from `requirements.txt` automatically

---

## Render Deployment (FastAPI)

The FastAPI inference endpoint can be deployed for free on [Render](https://render.com).
A `render.yaml` is already included in the repo — Render auto-detects it.

1. Go to [render.com](https://render.com) and sign in with GitHub
2. Click **New + → Web Service → Connect a repository** → select this repo
3. Render reads `render.yaml` and pre-fills everything automatically
4. Click **Deploy** — first build takes ~5 minutes (PyTorch install)
5. Your API will be live at `https://rul-prediction-api.onrender.com`
6. Swagger UI: `https://rul-prediction-api.onrender.com/docs`

> **Note:** The free tier spins down after 15 minutes of inactivity.
> The first request after idle takes ~30 seconds to wake up.

---

## API Quick Reference

```bash
# Health check
curl http://localhost:8000/health

# Predict RUL for one engine (JSON readings: 24 values per cycle)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"engine_id": 1, "readings": [[0.0, 0.0, 100.0, ...]]}'

# Predict all engines from a CMAPSS file
curl -X POST http://localhost:8000/predict/file \
  -F "file=@data/raw/FD001/test_FD001.txt"
```

Full interactive docs: `http://localhost:8000/docs`

---

## Dataset

**NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) FD001**

- 100 training engines, 100 test engines
- 21 sensors + 3 operating settings per cycle
- Fault mode: HPC degradation (single operating condition)
- RUL capped at 125 cycles (standard CMAPSS convention)

Download: [NASA CMAPSS on Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps?resource=download)

---

## Completed Phases

- [x] Phase 1: Project infrastructure (logging, exceptions, config, src layout)
- [x] Phase 2: Data ingestion and EDA
- [x] Phase 3: Preprocessing pipeline (sliding window, MinMax scaling, train/val split by engine)
- [x] Phase 4: Baseline ML models (XGBoost, LightGBM, Random Forest with Optuna HPO)
- [x] Phase 5: Deep learning (LSTM, GRU, BiLSTM with asymmetric loss)
- [x] Phase 6: SHAP explainability (global bar/beeswarm + local waterfall)
- [x] Phase 8: Inference pipeline (preprocessor + model in one call)
- [x] Phase 9: Streamlit dashboard
- [x] Phase 10: FastAPI inference endpoint

---

## Author

**Satyam Mishra** — [satyam3112003@gmail.com](mailto:satyam3112003@gmail.com)
