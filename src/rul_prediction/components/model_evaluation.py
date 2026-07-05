"""
ModelEvaluator component for post-hoc evaluation of saved models.

Loads any .pkl from saved_models/, measures accuracy (RMSE/MAE/R²),
efficiency (inference latency, model size, training time), and safety
(low-RUL error analysis). Produces a multi-model comparison table and
diagnostic plots.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from rul_prediction.components.data_ingestion import DataIngestion
# ModelTrainer must be importable at module level so that joblib can unpickle
# saved models whose class was recorded as __main__.ModelTrainer (produced when
# model_trainer.py is run directly via -m). Add future trainer classes here too.
from rul_prediction.components.model_trainer import ModelTrainer  # noqa: F401
from rul_prediction.config.configuration import load_config
from rul_prediction.exception.exception import RULException
from rul_prediction.logger.logger import get_logger

logger = get_logger(__name__)

_LOW_RUL_THRESHOLD: int = 30
_LOW_RUL_FLAG_RATIO: float = 1.5
_DEFAULT_BUCKET_EDGES: list[int] = [0, 30, 60, 90, 125, 9999]


class ModelEvaluator:
    """Post-hoc evaluator that loads saved .pkl models and measures accuracy and efficiency."""

    def __init__(self, config: dict[str, Any]) -> None:
        try:
            self.saved_models_dir = Path(config["paths"]["saved_models_dir"])
            self.processed_dir = Path(config["paths"]["processed_data_dir"])
            self.reports_dir = Path("reports")
            self.figures_dir = self.reports_dir / "figures"
            self.subset: str = config["dataset"]["active_subset"]
            self.figures_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RULException(e, sys) from e

    def _load_trainer(self, model_path: Path) -> Any:
        """Load a saved trainer from .pkl (baseline) or .pt (DL) file."""
        if model_path.suffix == ".pkl":
            return joblib.load(model_path)
        elif model_path.suffix == ".pt":
            from rul_prediction.components.sequence_trainer import SequenceTrainer  # lazy — avoids hard torch dep when only evaluating baseline models
            return SequenceTrainer.load(model_path)
        else:
            raise ValueError(f"Unsupported model file extension: {model_path.suffix}")

    def load_test_data(
        self, data_format: str = "flat"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and align test features with uncapped ground-truth RUL.

        Args:
            data_format: "flat" for tree models (n x 510);
                         "seq" for sequence models (n x 30 x 17) — Phase 5.

        Returns:
            (X_test_eval, y_test, engine_ids_eval) with one row per test engine
            (last-window alignment). y_test is uncapped from RUL_FD001.txt.
        """
        try:
            engine_ids_test = np.load(self.processed_dir / "test_engine_ids.npy")

            if data_format == "flat":
                X_test: np.ndarray = pd.read_parquet(
                    self.processed_dir / "test_X_flat.parquet"
                ).to_numpy(dtype=np.float32)
            elif data_format == "seq":
                X_test = np.load(self.processed_dir / "test_X_seq.npy")
            else:
                raise ValueError(
                    f"Unknown data_format '{data_format}'. Use 'flat' or 'seq'."
                )

            y_test = DataIngestion(subset=self.subset).load_test_rul()["RUL"].values

            # Keep only the last window per engine — aligns with RUL_FD001.txt ordering
            unique_engines = np.unique(engine_ids_test)
            last_idx = [
                int(np.where(engine_ids_test == eid)[0][-1]) for eid in unique_engines
            ]
            X_test_eval = X_test[last_idx]
            engine_ids_eval = engine_ids_test[last_idx]

            if len(X_test_eval) != len(y_test):
                raise RuntimeError(
                    f"Alignment mismatch: {len(X_test_eval)} windows vs "
                    f"{len(y_test)} ground-truth RUL values"
                )

            logger.info(
                f"Test data loaded: {len(y_test)} engines, "
                f"y_test range [{y_test.min()}, {y_test.max()}]"
            )
            return X_test_eval, y_test, engine_ids_eval
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e

    def evaluate_model(
        self,
        model_path: str | Path,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate one saved model file. Returns a flat metrics dict."""
        try:
            model_path = Path(model_path)
            trainer = self._load_trainer(model_path)
            label: str = getattr(trainer, "model_type", model_path.stem)

            # Inference latency
            _t0 = time.perf_counter()
            y_pred: np.ndarray = trainer.predict(X_test)
            inference_s = time.perf_counter() - _t0

            # Accuracy metrics (vs uncapped y_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            # Low-RUL safety metrics
            low_mask = y_test < _LOW_RUL_THRESHOLD
            n_low = int(low_mask.sum())
            if n_low > 0:
                low_rmse = float(
                    np.sqrt(mean_squared_error(y_test[low_mask], y_pred[low_mask]))
                )
                low_bias = float((y_pred[low_mask] - y_test[low_mask]).mean())
            else:
                low_rmse = float("nan")
                low_bias = float("nan")

            low_flag = not np.isnan(low_rmse) and low_rmse > _LOW_RUL_FLAG_RATIO * rmse

            logger.info(
                f"{label}: RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}  "
                f"low_RUL_RMSE={low_rmse:.3f}  flag={low_flag}  "
                f"latency={inference_s * 1000:.1f}ms"
            )

            return {
                "model": label,
                "test_rmse": round(rmse, 4),
                "test_mae": round(mae, 4),
                "test_r2": round(r2, 4),
                "low_rul_rmse": round(low_rmse, 4) if not np.isnan(low_rmse) else None,
                "low_rul_bias": round(low_bias, 4) if not np.isnan(low_bias) else None,
                "low_rul_flag": low_flag,
                "inference_ms_total": round(inference_s * 1000, 3),
                "inference_us_per_sample": round((inference_s / len(y_test)) * 1e6, 3),
                "training_time_s": getattr(trainer, "training_time_s_", None),
                "model_size_kb": round(model_path.stat().st_size / 1024, 2),
                "_model_path": str(model_path),
            }
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e

    def evaluate_all(
        self,
        model_paths: list[str | Path] | None = None,
    ) -> pd.DataFrame:
        """Evaluate multiple saved models; save comparison to reports/model_comparison.csv.

        Auto-discovers *.pkl (baseline) and *.pt (DL) files. Baseline models
        receive flat (n x 510) features; DL models receive sequence (n x 30 x 17).

        Args:
            model_paths: explicit list of paths, or None to auto-discover both
                         *.pkl and *.pt files in saved_models_dir.

        Returns:
            DataFrame sorted by test_rmse ascending. Contains an internal
            '_model_path' column (dropped before CSV) for downstream use.
        """
        try:
            if model_paths is None:
                pkl_paths = sorted(self.saved_models_dir.glob("*.pkl"))
                pt_paths = sorted(self.saved_models_dir.glob("*.pt"))
                model_paths = pkl_paths + pt_paths
                if not model_paths:
                    raise FileNotFoundError(
                        f"No .pkl or .pt files found in {self.saved_models_dir}"
                    )

            logger.info(f"Evaluating {len(model_paths)} model(s)")
            X_test_flat, y_test, _ = self.load_test_data(data_format="flat")
            X_test_seq, _, _ = self.load_test_data(data_format="seq")

            rows = []
            for p in model_paths:
                X_test = X_test_seq if Path(p).suffix == ".pt" else X_test_flat
                rows.append(self.evaluate_model(p, X_test, y_test))

            df = pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)

            out_path = self.reports_dir / "model_comparison.csv"
            df.drop(columns=["_model_path"], errors="ignore").to_csv(out_path, index=False)
            logger.info(f"Comparison table saved -> {out_path}")
            return df
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e

    def plot_residuals(
        self,
        model_path: str | Path,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Path:
        """Scatter + residual plots coloured by RUL zone. Returns figures_dir."""
        try:
            model_path = Path(model_path)
            trainer = self._load_trainer(model_path)
            label = getattr(trainer, "model_type", model_path.stem)
            y_pred = trainer.predict(X_test)
            residuals = y_pred - y_test

            zone_colors = [
                "red" if r < 30 else ("orange" if r < 80 else "steelblue")
                for r in y_test
            ]
            legend_elements = [
                Patch(facecolor="red",       label="Low RUL < 30 (safety-critical)"),
                Patch(facecolor="orange",    label="Mid RUL 30-80"),
                Patch(facecolor="steelblue", label="High RUL > 80"),
                plt.Line2D([0], [0], color="black", linestyle="--", label="Perfect prediction"),
            ]

            # Scatter: predicted vs true
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(y_test, y_pred, c=zone_colors, alpha=0.7, s=40, edgecolors="none")
            _max = max(float(y_test.max()), float(y_pred.max())) + 10
            ax.plot([0, _max], [0, _max], "k--", lw=1.5)
            ax.set_xlabel("True RUL (cycles)")
            ax.set_ylabel("Predicted RUL (cycles)")
            ax.set_title(f"{label} - Predicted vs True RUL")
            ax.legend(handles=legend_elements, fontsize=9)
            plt.tight_layout()
            scatter_path = self.figures_dir / f"residuals_scatter_{label}.png"
            plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
            plt.close()

            # Residuals vs true RUL
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(y_test, residuals, c=zone_colors, alpha=0.7, s=40, edgecolors="none")
            ax.axhline(0, color="black", linestyle="--", lw=1.5)
            ax.set_xlabel("True RUL (cycles)")
            ax.set_ylabel("Residual = Predicted - True (cycles)")
            ax.set_title(f"{label} - Residuals vs True RUL")
            ax.legend(handles=legend_elements[:3], fontsize=9)
            plt.tight_layout()
            residual_path = self.figures_dir / f"residuals_plot_{label}.png"
            plt.savefig(residual_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(
                f"Residual plots saved -> {scatter_path.name}, {residual_path.name}"
            )
            return self.figures_dir
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e

    def plot_error_by_bucket(
        self,
        model_path: str | Path,
        X_test: np.ndarray,
        y_test: np.ndarray,
        bucket_edges: list[int] | None = None,
    ) -> Path:
        """Per-RUL-bucket RMSE/MAE bar chart. Returns the figure path."""
        try:
            if bucket_edges is None:
                bucket_edges = _DEFAULT_BUCKET_EDGES

            model_path = Path(model_path)
            trainer = self._load_trainer(model_path)
            label = getattr(trainer, "model_type", model_path.stem)
            y_pred = trainer.predict(X_test)
            overall_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

            b_labels, b_rmse, b_mae, b_n = [], [], [], []
            for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
                mask = (y_test >= lo) & (y_test < hi)
                n = int(mask.sum())
                hi_str = str(hi) if hi < 9999 else "+"
                b_labels.append(f"{lo}-{hi_str}")
                b_n.append(n)
                if n > 0:
                    b_rmse.append(
                        float(np.sqrt(mean_squared_error(y_test[mask], y_pred[mask])))
                    )
                    b_mae.append(float(mean_absolute_error(y_test[mask], y_pred[mask])))
                else:
                    b_rmse.append(0.0)
                    b_mae.append(0.0)

            bar_colors = [
                "red" if (n > 0 and r > _LOW_RUL_FLAG_RATIO * overall_rmse) else "steelblue"
                for r, n in zip(b_rmse, b_n)
            ]

            x = np.arange(len(b_labels))
            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax1.bar(x, b_rmse, color=bar_colors, alpha=0.8, label="RMSE")
            ax1.set_xlabel("True RUL bucket (cycles)")
            ax1.set_ylabel("RMSE (cycles)")
            ax1.set_xticks(x)
            ax1.set_xticklabels(
                [f"{lbl}\n(n={n})" for lbl, n in zip(b_labels, b_n)], fontsize=9
            )
            ax1.set_title(
                f"{label} - Error by RUL Bucket  "
                f"(red = RMSE > {_LOW_RUL_FLAG_RATIO}x overall)"
            )

            ax2 = ax1.twinx()
            ax2.plot(x, b_mae, "o--", color="darkorange", lw=1.5, label="MAE")
            ax2.set_ylabel("MAE (cycles)")

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, fontsize=9)

            plt.tight_layout()
            out_path = self.figures_dir / f"error_by_bucket_{label}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Error-by-bucket plot saved -> {out_path.name}")
            return out_path
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e


if __name__ == "__main__":
    _config = load_config()
    _evaluator = ModelEvaluator(_config)

    _df = _evaluator.evaluate_all()
    print(_df.drop(columns=["_model_path"], errors="ignore").to_string(index=False))

    _best_path = Path(_df.iloc[0]["_model_path"])
    _fmt = "seq" if _best_path.suffix == ".pt" else "flat"
    _X_test, _y_test, _ = _evaluator.load_test_data(data_format=_fmt)
    _evaluator.plot_residuals(str(_best_path), _X_test, _y_test)
    _evaluator.plot_error_by_bucket(str(_best_path), _X_test, _y_test)
    logger.info("Smoke test complete.")
