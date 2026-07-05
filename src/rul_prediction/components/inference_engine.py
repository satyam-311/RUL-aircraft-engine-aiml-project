"""
InferenceEngine: loads the production DL model and runs single-window predictions.

Single responsibility: accept a preprocessed window (1, 30, 17) and return a
PredictionResult. Does not own the preprocessor — InferencePipeline handles that.
Model is lazy-loaded on first call and cached for all subsequent calls (safe for
the FastAPI startup pattern used in Phase 10).
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rul_prediction.config.configuration import load_config
from rul_prediction.exception.exception import RULException
from rul_prediction.logger.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """Structured output from a single-engine RUL prediction."""
    engine_id: int
    predicted_rul: float
    safety_flag: bool    # True when predicted_rul < rul_threshold
    model_used: str      # stem of the .pt file used, e.g. "bilstm_20260705_013021"


class InferenceEngine:
    """Wraps the production sequence model for inference.

    Usage
    -----
    engine = InferenceEngine(config)
    result = engine.predict(X_window, engine_id=42)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.saved_models_dir = Path(config["paths"]["saved_models_dir"])
        self.model_type: str = config["model"]["type"]           # e.g. "lstm"
        self.rul_threshold: float = config["model"]["rul_threshold"]  # 30.0
        self._trainer = None   # lazy-loaded on first predict() call
        self._model_stem: str = ""

    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Locate the latest .pt file for self.model_type and load it."""
        try:
            from rul_prediction.components.sequence_trainer import SequenceTrainer  # noqa: PLC0415

            candidates = sorted(
                p for p in self.saved_models_dir.glob("*.pt")
                if p.stem.startswith(self.model_type)
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No saved .pt model found for type '{self.model_type}' "
                    f"in {self.saved_models_dir}"
                )
            model_path = candidates[-1]
            self._trainer = SequenceTrainer.load(model_path)
            self._model_stem = model_path.stem
            logger.info(f"Loaded model: {model_path}")
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e

    def predict(self, X_window: np.ndarray, engine_id: int = -1) -> PredictionResult:
        """Run inference on a single preprocessed window.

        Args:
            X_window:  Shape (1, 30, 17) — last 30-cycle window, already scaled.
            engine_id: Integer engine identifier; -1 if unknown.

        Returns:
            PredictionResult with predicted_rul, safety_flag, model_used.
        """
        try:
            if self._trainer is None:
                self._load_model()

            y_pred = float(self._trainer.predict(X_window)[0])
            safety_flag = y_pred < self.rul_threshold

            if safety_flag:
                logger.warning(
                    f"Engine {engine_id}: predicted RUL={y_pred:.1f} is below "
                    f"safety threshold {self.rul_threshold} -- flag raised"
                )
            else:
                logger.info(f"Engine {engine_id}: predicted RUL={y_pred:.1f}")

            return PredictionResult(
                engine_id=engine_id,
                predicted_rul=y_pred,
                safety_flag=safety_flag,
                model_used=self._model_stem,
            )
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e


if __name__ == "__main__":
    _config = load_config()
    _engine = InferenceEngine(_config)
    # Quick sanity: random window to verify load + forward pass
    _dummy = np.random.default_rng(42).random((1, 30, 17)).astype(np.float32)
    _result = _engine.predict(_dummy, engine_id=0)
    print(f"Sanity check -- predicted_rul={_result.predicted_rul:.2f}, "
          f"safety_flag={_result.safety_flag}, model={_result.model_used}")
