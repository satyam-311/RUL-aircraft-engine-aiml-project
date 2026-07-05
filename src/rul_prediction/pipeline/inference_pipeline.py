"""
InferencePipeline: end-to-end orchestration from raw CMAPSS sensor data to RUL prediction.

Loads the fitted preprocessor from artifacts/, applies the same scaling and windowing
used during training, then delegates to InferenceEngine for the model forward pass.
Both the preprocessor and the model are lazy-loaded and cached, so this class is safe
to instantiate once at FastAPI startup and reuse across requests (Phase 10).

Data flow:
    engine_df (raw CMAPSS rows, >= 1 row)
      -> preprocessor.transform()    -> PreprocessedData.X_seq (n_windows, 30, 17)
      -> X_seq[-1:]                  -> (1, 30, 17)  last window = current health state
      -> InferenceEngine.predict()   -> PredictionResult
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rul_prediction.components.inference_engine import InferenceEngine, PredictionResult
from rul_prediction.components.preprocessor import Preprocessor  # noqa: F401  # pickle compat
from rul_prediction.config.configuration import load_config
from rul_prediction.exception.exception import RULException
from rul_prediction.logger.logger import get_logger

logger = get_logger(__name__)


class InferencePipeline:
    """End-to-end inference: raw sensor DataFrame -> PredictionResult.

    Usage
    -----
    pipeline = InferencePipeline(load_config())
    result   = pipeline.predict(engine_df, engine_id=42)
    results  = pipeline.predict_batch({42: df_42, 17: df_17})
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.artifacts_dir = Path(config["paths"]["artifacts_dir"])
        self.window_size: int = config["dataset"]["window_size"]
        self._preprocessor = None   # lazy-loaded on first predict() call
        self._engine = InferenceEngine(config)

    # ------------------------------------------------------------------

    def _load_preprocessor(self):
        """Load the fitted Preprocessor from artifacts/preprocessor.pkl (cached)."""
        try:
            pkl_path = self.artifacts_dir / "preprocessor.pkl"
            if not pkl_path.exists():
                raise FileNotFoundError(
                    f"Preprocessor not found at {pkl_path}. "
                    "Run: uv run python -m rul_prediction.components.preprocessor"
                )
            self._preprocessor = Preprocessor.load(pkl_path)
            logger.info(f"Preprocessor loaded <- {pkl_path}")
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e

    def predict(
        self,
        engine_df: pd.DataFrame,
        engine_id: int = -1,
    ) -> PredictionResult:
        """Predict RUL for a single engine from its raw sensor readings.

        Args:
            engine_df: Raw CMAPSS-format DataFrame for one engine.
                       Must contain the sensor and op_setting columns.
                       Does NOT need a 'unit_number' column — engine_id is used.
                       Needs at least 1 row; short sequences are left-padded to 30.
            engine_id: Integer engine identifier (used in the result and logs).

        Returns:
            PredictionResult with predicted_rul, safety_flag, model_used.
        """
        try:
            if self._preprocessor is None:
                self._load_preprocessor()

            # Attach a unit_number column so preprocessor can group by engine
            df = engine_df.copy()
            if "unit_number" not in df.columns:
                df["unit_number"] = engine_id if engine_id != -1 else 1
            if "time_in_cycles" not in df.columns:
                df["time_in_cycles"] = range(1, len(df) + 1)

            data = self._preprocessor.transform(df, cap_rul=False)
            # Last window = most recent 30 cycles = current health state
            X_window: np.ndarray = data.X_seq[-1:]   # (1, 30, 17)
            return self._engine.predict(X_window, engine_id=engine_id)
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e

    def predict_trajectory(
        self,
        engine_df: pd.DataFrame,
        engine_id: int = -1,
    ) -> list[PredictionResult]:
        """Predict RUL for every sliding window in the engine's history.

        Returns one PredictionResult per window in chronological order.
        The first result corresponds to the window ending at cycle window_size;
        the last is the same prediction as predict(). Used by the dashboard
        RUL trend plot.

        For an engine with n cycles, returns n - window_size + 1 results
        (or 1 result if n < window_size, using zero-padded window).
        """
        try:
            if self._preprocessor is None:
                self._load_preprocessor()

            df = engine_df.copy()
            if "unit_number" not in df.columns:
                df["unit_number"] = engine_id if engine_id != -1 else 1
            if "time_in_cycles" not in df.columns:
                df["time_in_cycles"] = range(1, len(df) + 1)

            data = self._preprocessor.transform(df, cap_rul=False)

            # Ensure model is loaded before batch predict
            if self._engine._trainer is None:
                self._engine._load_model()

            y_preds = self._engine._trainer.predict(data.X_seq)  # (n_windows,)

            stem = self._engine._model_stem
            threshold = self._engine.rul_threshold
            return [
                PredictionResult(
                    engine_id=engine_id,
                    predicted_rul=float(y),
                    safety_flag=float(y) < threshold,
                    model_used=stem,
                )
                for y in y_preds
            ]
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e

    def predict_batch(
        self,
        engine_dfs: dict[int, pd.DataFrame],
    ) -> list[PredictionResult]:
        """Predict RUL for multiple engines.

        Args:
            engine_dfs: Mapping of engine_id -> raw sensor DataFrame.

        Returns:
            List of PredictionResult, one per engine, in input order.
        """
        try:
            results: list[PredictionResult] = []
            for eid, df in engine_dfs.items():
                results.append(self.predict(df, engine_id=eid))
            return results
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e


# ---------------------------------------------------------------------------
# Smoke test — run with: uv run python -m rul_prediction.pipeline.inference_pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from rul_prediction.components.data_ingestion import DataIngestion

    cfg = load_config()
    ingestion = DataIngestion(
        raw_data_dir=cfg["paths"]["raw_data_dir"],
        subset=cfg["dataset"]["active_subset"],
    )
    _, test_df, test_rul_df = ingestion.load_all()

    # Build per-engine DataFrames from the test set
    engine_dfs: dict[int, pd.DataFrame] = {
        int(eid): grp.reset_index(drop=True)
        for eid, grp in test_df.groupby("unit_number", sort=True)
    }
    # test_rul_df has one column "RUL" indexed by row; engine_id = row_index + 1
    true_rul: dict[int, float] = {
        int(idx) + 1: float(row["RUL"])
        for idx, row in test_rul_df.iterrows()
    }

    pipeline = InferencePipeline(cfg)
    results = pipeline.predict_batch(engine_dfs)

    print(f"\nInference complete: {len(results)} engines\n")
    print(f"{'Engine':>8}  {'Pred RUL':>10}  {'True RUL':>10}  {'Error':>8}  {'Flag':>5}")
    print("-" * 52)
    for r in results[:5]:
        true = true_rul.get(r.engine_id, float("nan"))
        err = r.predicted_rul - true
        flag = "SAFE" if not r.safety_flag else "FLAG"
        print(f"{r.engine_id:>8}  {r.predicted_rul:>10.1f}  {true:>10.1f}  "
              f"{err:>+8.1f}  {flag:>5}")

    flagged = [r for r in results if r.safety_flag]
    print(f"\nSafety flags raised: {len(flagged)}/{len(results)} engines "
          f"(predicted RUL < {cfg['model']['rul_threshold']})")
    if flagged:
        sample_eids = [r.engine_id for r in flagged[:5]]
        print(f"First flagged engines: {sample_eids}")
