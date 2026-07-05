"""
ModelTrainer component for training, evaluating, and persisting baseline ML models.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from rul_prediction.config.configuration import load_config
from rul_prediction.exception.exception import RULException
from rul_prediction.logger.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Train, evaluate, and persist a configurable sklearn-compatible regressor."""

    SUPPORTED_MODELS: frozenset[str] = frozenset(
        {"linear_regression", "random_forest", "xgboost", "lightgbm"}
    )

    def __init__(self, config: dict[str, Any]) -> None:
        try:
            self.model_type: str = config["model"]["type"]
            if self.model_type not in self.SUPPORTED_MODELS:
                raise ValueError(
                    f"Unsupported model type '{self.model_type}'. "
                    f"Choose from: {sorted(self.SUPPORTED_MODELS)}"
                )
            self.hyperparameters: dict[str, Any] = dict(
                config["model"].get("hyperparameters", {})
            )
            self.saved_models_dir = Path(config["paths"]["saved_models_dir"])
            self.seed: int = config.get("seed", 42)
            self.model_: Any = None
            self.is_fitted_: bool = False
        except Exception as e:
            raise RULException(e, sys) from e

    def _build_model(self) -> Any:
        params = dict(self.hyperparameters)
        if self.model_type == "linear_regression":
            return LinearRegression(**params)
        if self.model_type == "random_forest":
            params.setdefault("random_state", self.seed)
            params.setdefault("n_jobs", -1)
            return RandomForestRegressor(**params)
        if self.model_type == "xgboost":
            params.setdefault("device", "cpu")
            params.setdefault("tree_method", "hist")
            return XGBRegressor(**params)
        if self.model_type == "lightgbm":
            params.setdefault("n_jobs", -1)
            params.setdefault("verbosity", -1)
            return LGBMRegressor(**params)
        raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> ModelTrainer:
        try:
            logger.info(
                f"Training {self.model_type} on {X_train.shape[0]} windows "
                f"({X_train.shape[1]} features)"
            )
            self.model_ = self._build_model()
            _t0 = time.perf_counter()
            self.model_.fit(X_train, y_train)
            self.training_time_s_: float = time.perf_counter() - _t0
            self.is_fitted_ = True
            logger.info(
                f"{self.model_type} training complete in {self.training_time_s_:.1f}s"
            )
            return self
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e

    def predict(self, X: np.ndarray) -> np.ndarray:
        try:
            if not self.is_fitted_:
                raise RuntimeError("ModelTrainer is not fitted. Call fit() first.")
            return self.model_.predict(X)
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        try:
            y_pred = self.predict(X)
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae = float(mean_absolute_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))
            logger.info(f"Evaluation -- RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}")
            return {"rmse": rmse, "mae": mae, "r2": r2}
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e

    def save(self, path: str | Path | None = None) -> Path:
        try:
            if not self.is_fitted_:
                raise RuntimeError("Cannot save an unfitted model. Call fit() first.")
            if path is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = self.saved_models_dir / f"{self.model_type}_{ts}.pkl"
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, path)
            logger.info(f"Model saved -> {path}")
            return path
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e

    @classmethod
    def load(cls, path: str | Path) -> ModelTrainer:
        try:
            path = Path(path)
            logger.info(f"Loading model from {path}")
            trainer: ModelTrainer = joblib.load(path)
            logger.info(f"Loaded {trainer.model_type} model (fitted={trainer.is_fitted_})")
            return trainer
        except Exception as e:
            raise RULException(e, sys) from e


if __name__ == "__main__":
    import pandas as pd

    _config = load_config()
    _processed = Path(_config["paths"]["processed_data_dir"])

    _X_train = pd.read_parquet(_processed / "train_X_flat.parquet").to_numpy(dtype=np.float32)
    _y_train = np.load(_processed / "train_y.npy")
    _X_val = pd.read_parquet(_processed / "val_X_flat.parquet").to_numpy(dtype=np.float32)
    _y_val = np.load(_processed / "val_y.npy")

    _trainer = ModelTrainer(_config).fit(_X_train, _y_train)
    _metrics = _trainer.evaluate(_X_val, _y_val)
    logger.info(f"Val metrics: {_metrics}")
    _saved = _trainer.save()
    logger.info(f"Smoke test complete. Model at {_saved}")
