"""
Phase 3 preprocessing pipeline for NASA CMAPSS data.

WHY: Raw DataFrames from DataIngestion need three transformations before any
model can consume them: (1) drop non-informative sensors identified in EDA,
(2) scale features to [0,1] without fitting on test data (leakage prevention),
(3) reshape into fixed-length windows — flat for tree models, 3-D for sequence
models. Centralising this in a sklearn-style class means Phase 4, Phase 5, and
Phase 8 inference all apply identical transformations.

HOW: MinMaxScaler is preferred over StandardScaler because CMAPSS sensors have
stable physical bounds (controlled test-bed, no outliers) and LSTM/GRU
activations (sigmoid, tanh) work best with bounded [0,1] inputs. The scaler is
fitted ONLY on training-engine data, then applied unchanged to val and test.

WHERE: src/rul_prediction/components/preprocessor.py
    Called by: training scripts (Phase 4/5), inference pipeline (Phase 8)
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from rul_prediction.constants.constants import NON_INFORMATIVE_SENSORS
from rul_prediction.exception.exception import RULException
from rul_prediction.logger.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessedData:
    """Container for a single split's preprocessed arrays."""
    X_flat:     np.ndarray            # (n_windows, window_size * n_features)
    X_seq:      np.ndarray            # (n_windows, window_size, n_features)
    y:          Optional[np.ndarray]  # (n_windows,) — None for test set
    engine_ids: np.ndarray            # (n_windows,) — source engine for each window


class Preprocessor:
    """
    Sklearn-style preprocessing for CMAPSS data.

    Typical usage
    -------------
    preprocessor = Preprocessor(cfg)
    train_eng_df, val_eng_df = preprocessor.split_engines(train_df)
    train_data = preprocessor.fit_transform(train_eng_df)   # fits scaler here
    val_data   = preprocessor.transform(val_eng_df)
    test_data  = preprocessor.transform(test_df, cap_rul=False)
    preprocessor.save("artifacts/preprocessor.pkl")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.rul_cap:     int   = config["dataset"]["rul_cap"]
        self.window_size: int   = config["dataset"]["window_size"]
        self.seed:        int   = config["seed"]
        self.val_size:    float = config["training"]["test_size"]

        self.scaler_:       Optional[MinMaxScaler] = None
        self.feature_cols_: Optional[List[str]]    = None
        self.drop_cols_:    List[str]               = list(NON_INFORMATIVE_SENSORS)
        self.is_fitted_:    bool                    = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_engines(
        self, train_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split full training DataFrame by engine ID, not by row.

        Splitting by row would leak the same engine's degradation history
        across train and validation, producing optimistic val metrics.
        """
        try:
            engine_ids = train_df["unit_number"].unique()
            train_engines, val_engines = train_test_split(
                engine_ids,
                test_size=self.val_size,
                random_state=self.seed,
                shuffle=True,
            )
            train_eng_df = train_df[train_df["unit_number"].isin(train_engines)].copy()
            val_eng_df   = train_df[train_df["unit_number"].isin(val_engines)].copy()
            logger.info(
                f"Engine split: {len(train_engines)} train / {len(val_engines)} val "
                f"(val_size={self.val_size}, seed={self.seed})"
            )
            return train_eng_df, val_eng_df
        except Exception as e:
            raise RULException(e, sys) from e

    def fit(self, train_eng_df: pd.DataFrame) -> "Preprocessor":
        """Fit MinMaxScaler on training-engine features only.

        Must be called before transform(). Returns self for chaining.
        """
        try:
            self.feature_cols_ = self._get_feature_cols(train_eng_df)
            self.scaler_ = MinMaxScaler()
            self.scaler_.fit(train_eng_df[self.feature_cols_].values)
            self.is_fitted_ = True
            logger.info(
                f"Fitted MinMaxScaler on {len(self.feature_cols_)} features "
                f"({len(self.drop_cols_)} sensors dropped)"
            )
            return self
        except Exception as e:
            raise RULException(e, sys) from e

    def transform(
        self,
        df: pd.DataFrame,
        cap_rul: bool = True,
    ) -> PreprocessedData:
        """Apply fitted scaler and generate sliding windows.

        Args:
            df:      DataFrame from DataIngestion (with or without RUL column).
            cap_rul: If True, clip RUL labels at self.rul_cap (use for train/val).
                     If False, keep raw RUL values (use for test evaluation).

        Returns:
            PreprocessedData with X_flat, X_seq, y (None if no RUL column), engine_ids.
        """
        try:
            if not self.is_fitted_:
                raise RuntimeError("Call fit() before transform().")
            return self._make_windows(df, cap_rul=cap_rul)
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e

    def fit_transform(self, train_eng_df: pd.DataFrame) -> PreprocessedData:
        """Fit on training data and immediately transform it."""
        self.fit(train_eng_df)
        return self.transform(train_eng_df, cap_rul=True)

    def save(self, path: "str | Path") -> None:
        """Persist the entire fitted Preprocessor via joblib."""
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            # When run via `python -m`, the class's __module__ is '__main__' rather
            # than the canonical 'rul_prediction.components.preprocessor', which
            # causes pickle to record __main__.Preprocessor. Any process that later
            # loads the pkl (e.g. uvicorn) cannot resolve __main__.Preprocessor and
            # raises AttributeError. Fix: temporarily register the running class
            # object under its canonical path so pickle sees both the module name
            # AND the module attribute agree before writing.
            import importlib
            _canon = "rul_prediction.components.preprocessor"
            _mod = importlib.import_module(_canon)
            _orig_cls_in_mod = getattr(_mod, "Preprocessor", None)
            _orig_module_attr = type(self).__module__
            _mod.Preprocessor = type(self)
            type(self).__module__ = _canon
            try:
                joblib.dump(self, path)
            finally:
                type(self).__module__ = _orig_module_attr
                if _orig_cls_in_mod is not None:
                    _mod.Preprocessor = _orig_cls_in_mod
                else:
                    delattr(_mod, "Preprocessor")
            logger.info(f"Preprocessor saved -> {path}")
        except Exception as e:
            raise RULException(e, sys) from e

    @classmethod
    def load(cls, path: "str | Path") -> "Preprocessor":
        """Load a previously saved Preprocessor."""
        try:
            preprocessor = joblib.load(path)
            logger.info(f"Preprocessor loaded <- {path}")
            return preprocessor
        except Exception as e:
            raise RULException(e, sys) from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Return feature columns: sensor_* + op_setting_* minus non-informative."""
        all_feat = [
            c for c in df.columns
            if c.startswith("sensor_") or c.startswith("op_setting_")
        ]
        return [c for c in all_feat if c not in self.drop_cols_]

    def _make_windows(self, df: pd.DataFrame, cap_rul: bool) -> PreprocessedData:
        """Generate sliding windows of length window_size per engine.

        For engines with fewer than window_size cycles, a single window is
        produced by left-padding with zeros (standard CMAPSS convention).
        """
        df = df.sort_values(["unit_number", "time_in_cycles"])
        has_rul = "RUL" in df.columns

        X_flat_list:    list = []
        X_seq_list:     list = []
        y_list:         list = []
        engine_id_list: list = []

        for engine_id, engine_df in df.groupby("unit_number", sort=True):
            features = self.scaler_.transform(engine_df[self.feature_cols_].values)
            rul_vals = engine_df["RUL"].values if has_rul else None
            n = len(features)

            if n >= self.window_size:
                for end in range(self.window_size - 1, n):
                    start  = end - self.window_size + 1
                    window = features[start : end + 1]       # (W, F)
                    X_seq_list.append(window)
                    X_flat_list.append(window.reshape(-1))
                    engine_id_list.append(engine_id)
                    if rul_vals is not None:
                        label = float(rul_vals[end])
                        if cap_rul:
                            label = min(label, self.rul_cap)
                        y_list.append(label)
            else:
                # Left-pad short engines with zeros — one window per engine
                pad    = np.zeros((self.window_size - n, len(self.feature_cols_)))
                window = np.vstack([pad, features])          # (W, F)
                X_seq_list.append(window)
                X_flat_list.append(window.reshape(-1))
                engine_id_list.append(engine_id)
                if rul_vals is not None:
                    label = float(rul_vals[-1])
                    if cap_rul:
                        label = min(label, self.rul_cap)
                    y_list.append(label)

        result = PreprocessedData(
            X_flat     = np.array(X_flat_list,    dtype=np.float32),
            X_seq      = np.array(X_seq_list,     dtype=np.float32),
            y          = np.array(y_list,         dtype=np.float32) if y_list else None,
            engine_ids = np.array(engine_id_list, dtype=np.int32),
        )
        cap_info = f", RUL capped at {self.rul_cap}" if cap_rul and has_rul else ""
        logger.info(
            f"Windows: X_seq={result.X_seq.shape}, X_flat={result.X_flat.shape}"
            + (f", y={result.y.shape}" if result.y is not None else ", y=None")
            + cap_info
        )
        return result


# ---------------------------------------------------------------------------
# Smoke test — run with: uv run python -m rul_prediction.components.preprocessor
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from rul_prediction.components.data_ingestion import DataIngestion
    from rul_prediction.config.configuration import load_config

    cfg = load_config()
    processed_dir = Path(cfg["paths"]["processed_data_dir"])
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # --- Load raw data ---
    ingestion = DataIngestion(
        raw_data_dir=cfg["paths"]["raw_data_dir"],
        subset=cfg["dataset"]["active_subset"],
    )
    train_df, test_df, _ = ingestion.load_all()

    # --- Split, fit, transform ---
    preprocessor = Preprocessor(cfg)
    train_eng_df, val_eng_df = preprocessor.split_engines(train_df)

    train_data = preprocessor.fit_transform(train_eng_df)
    val_data   = preprocessor.transform(val_eng_df,  cap_rul=True)
    test_data  = preprocessor.transform(test_df,     cap_rul=False)

    # --- Save arrays ---
    for split, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        np.save(processed_dir / f"{split}_X_seq.npy",      data.X_seq)
        np.save(processed_dir / f"{split}_engine_ids.npy", data.engine_ids)

        flat_cols = [
            f"{col}_t{t}"
            for t in range(preprocessor.window_size)
            for col in preprocessor.feature_cols_
        ]
        pd.DataFrame(data.X_flat, columns=flat_cols).to_parquet(
            processed_dir / f"{split}_X_flat.parquet", index=False
        )
        if data.y is not None:
            np.save(processed_dir / f"{split}_y.npy", data.y)

    # --- Save preprocessor ---
    preprocessor.save(artifacts_dir / "preprocessor.pkl")

    # --- Summary ---
    print("\n=== Preprocessing complete ===")
    print(f"Features ({len(preprocessor.feature_cols_)}): {preprocessor.feature_cols_}")
    print(f"Dropped  ({len(preprocessor.drop_cols_)}): {preprocessor.drop_cols_}")
    for split, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        y_info = f"y={data.y.shape}" if data.y is not None else "y=None"
        print(f"{split:5s}  X_seq={data.X_seq.shape}  X_flat={data.X_flat.shape}  {y_info}")
