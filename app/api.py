"""
Phase 10 -- FastAPI inference endpoint for the RUL prediction system.

Endpoints
---------
GET  /health                  Server and model status
POST /predict                 Single engine from JSON sensor readings
POST /predict/batch           Multiple engines from JSON
POST /predict/trajectory      Per-window RUL trend for one engine
POST /predict/file            CMAPSS file upload -> all engines predicted

Run
---
    uv run uvicorn app.api:app --reload
    uv run uvicorn app.api:app --host 0.0.0.0 --port 8000

The InferencePipeline is instantiated once at startup (lifespan) and cached
in app.state. Both the preprocessor and model are lazy-loaded on the first
request, then reused. Thread-safe: no mutable state changes after warm-up.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rul_prediction.components.data_ingestion import parse_cmapss_bytes
from rul_prediction.components.preprocessor import Preprocessor  # noqa: F401  # pickle compat
from rul_prediction.config.configuration import load_config
from rul_prediction.constants.constants import (
    ALL_COLUMNS,
    OPERATIONAL_SETTING_COLUMNS,
    SENSOR_COLUMNS,
)
from rul_prediction.exception.exception import RULException
from rul_prediction.logger.logger import get_logger
from rul_prediction.pipeline.inference_pipeline import InferencePipeline

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature columns accepted in JSON payloads (unit_number + time_in_cycles
# are synthetic -- we add them automatically from the row index)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = OPERATIONAL_SETTING_COLUMNS + SENSOR_COLUMNS  # 24 cols


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Single-engine prediction request."""

    engine_id: int = Field(default=1, ge=1, description="Integer engine identifier")
    readings: list[list[float]] = Field(
        ...,
        min_length=1,
        description=(
            "Sensor readings, one row per cycle. Each row must have exactly 24 values: "
            "[op_setting_1, op_setting_2, op_setting_3, sensor_1 ... sensor_21]"
        ),
    )


class BatchPredictRequest(BaseModel):
    """Multi-engine prediction request."""

    engines: list[PredictRequest] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    engine_id: int
    predicted_rul: float
    safety_flag: bool
    model_used: str
    timestamp: str


class TrajectoryResponse(BaseModel):
    engine_id: int
    model_used: str
    window_predictions: list[float]
    safety_flags: list[bool]
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_type: str
    rul_threshold: float
    timestamp: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _readings_to_df(readings: list[list[float]], engine_id: int) -> pd.DataFrame:
    """Convert a list of reading rows to a labelled CMAPSS-style DataFrame."""
    n_features = len(FEATURE_COLUMNS)
    for i, row in enumerate(readings):
        if len(row) != n_features:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Row {i} has {len(row)} values; expected {n_features} "
                    f"({OPERATIONAL_SETTING_COLUMNS} + {SENSOR_COLUMNS})"
                ),
            )
    df = pd.DataFrame(readings, columns=FEATURE_COLUMNS)
    df.insert(0, "time_in_cycles", range(1, len(df) + 1))
    df.insert(0, "unit_number", engine_id)
    return df


# ---------------------------------------------------------------------------
# App lifespan: load config + create pipeline singleton
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    app.state.pipeline = InferencePipeline(cfg)
    app.state.config = cfg
    logger.info("InferencePipeline initialised at startup")
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="RUL Prediction API",
    description=(
        "Remaining Useful Life prediction for turbofan aircraft engines. "
        "Production model: LSTM trained on NASA CMAPSS FD001 with asymmetric loss."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health() -> HealthResponse:
    """Return server status and active model configuration."""
    cfg: dict[str, Any] = app.state.config
    return HealthResponse(
        status="ok",
        model_type=cfg["model"]["type"],
        rul_threshold=cfg["model"]["rul_threshold"],
        timestamp=_now_iso(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictRequest) -> PredictionResponse:
    """Predict RUL for a single engine from JSON sensor readings.

    Supply one row of readings per cycle (in chronological order).
    The model uses the last 30-cycle sliding window; shorter histories
    are automatically zero-padded.
    """
    try:
        engine_df = _readings_to_df(request.readings, request.engine_id)
        pipeline: InferencePipeline = app.state.pipeline
        result = pipeline.predict(engine_df, engine_id=request.engine_id)
        return PredictionResponse(
            engine_id=result.engine_id,
            predicted_rul=result.predicted_rul,
            safety_flag=result.safety_flag,
            model_used=result.model_used,
            timestamp=_now_iso(),
        )
    except HTTPException:
        raise
    except RULException as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=list[PredictionResponse], tags=["Prediction"])
def predict_batch(request: BatchPredictRequest) -> list[PredictionResponse]:
    """Predict RUL for multiple engines in a single call."""
    try:
        pipeline: InferencePipeline = app.state.pipeline
        responses: list[PredictionResponse] = []
        ts = _now_iso()
        for eng in request.engines:
            engine_df = _readings_to_df(eng.readings, eng.engine_id)
            result = pipeline.predict(engine_df, engine_id=eng.engine_id)
            responses.append(
                PredictionResponse(
                    engine_id=result.engine_id,
                    predicted_rul=result.predicted_rul,
                    safety_flag=result.safety_flag,
                    model_used=result.model_used,
                    timestamp=ts,
                )
            )
        return responses
    except HTTPException:
        raise
    except RULException as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/trajectory", response_model=TrajectoryResponse, tags=["Prediction"])
def predict_trajectory(request: PredictRequest) -> TrajectoryResponse:
    """Return per-window RUL predictions across the engine's full history.

    Returns one prediction per sliding window (window_size=30 cycles).
    Useful for plotting how the model's estimate evolved over time.
    """
    try:
        engine_df = _readings_to_df(request.readings, request.engine_id)
        pipeline: InferencePipeline = app.state.pipeline
        results = pipeline.predict_trajectory(engine_df, engine_id=request.engine_id)
        return TrajectoryResponse(
            engine_id=request.engine_id,
            model_used=results[0].model_used if results else "",
            window_predictions=[r.predicted_rul for r in results],
            safety_flags=[r.safety_flag for r in results],
            timestamp=_now_iso(),
        )
    except HTTPException:
        raise
    except RULException as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/file", response_model=list[PredictionResponse], tags=["Prediction"])
async def predict_file(file: UploadFile) -> list[PredictionResponse]:
    """Upload a CMAPSS-format text file and predict RUL for every engine in it.

    The file must be headerless and space-delimited (standard CMAPSS format).
    Multiple engines (identified by the first column, unit_number) are all
    predicted and returned in unit_number order.
    """
    try:
        raw_bytes = await file.read()
        df = parse_cmapss_bytes(raw_bytes, ALL_COLUMNS)

        if "unit_number" not in df.columns:
            raise HTTPException(
                status_code=422,
                detail="Uploaded file could not be parsed as a CMAPSS dataset.",
            )

        pipeline: InferencePipeline = app.state.pipeline
        engine_dfs: dict[int, pd.DataFrame] = {
            int(eid): grp.reset_index(drop=True)
            for eid, grp in df.groupby("unit_number", sort=True)
        }
        results = pipeline.predict_batch(engine_dfs)
        ts = _now_iso()
        return [
            PredictionResponse(
                engine_id=r.engine_id,
                predicted_rul=r.predicted_rul,
                safety_flag=r.safety_flag,
                model_used=r.model_used,
                timestamp=ts,
            )
            for r in results
        ]
    except HTTPException:
        raise
    except RULException as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
