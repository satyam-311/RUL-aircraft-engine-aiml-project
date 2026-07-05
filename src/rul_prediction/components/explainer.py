"""
ModelExplainer component for Phase 6 — SHAP explainability.

Produces global importance (bar chart + beeswarm) and local waterfall
explanations for both the XGBoost baseline (.pkl) and the configured DL model (.pt).
Cross-checks SHAP sensor rankings against Phase 2 EDA
Pearson correlates to validate physical plausibility.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import shap
import torch

from rul_prediction.components.data_ingestion import DataIngestion
# ModelTrainer must be importable at module level so pickle can resolve it when
# loading .pkl files whose class was recorded as __main__.ModelTrainer.
from rul_prediction.components.model_trainer import ModelTrainer  # noqa: F401
from rul_prediction.config.configuration import load_config
from rul_prediction.constants.constants import RANDOM_SEED
from rul_prediction.exception.exception import RULException
from rul_prediction.logger.logger import get_logger

logger = get_logger(__name__)

# EDA Phase 2 top-4 Pearson correlates with RUL in FD001 (HPC degradation mode)
_EDA_TOP4 = ["sensor_11", "sensor_4", "sensor_12", "sensor_7"]


class ModelExplainer:
    """SHAP-based explainer for saved .pkl (XGBoost) and .pt (DL model) models."""

    def __init__(self, config: dict[str, Any]) -> None:
        try:
            self.saved_models_dir = Path(config["paths"]["saved_models_dir"])
            self.processed_dir = Path(config["paths"]["processed_data_dir"])
            self.reports_dir = Path("reports")
            self.figures_dir = self.reports_dir / "figures"
            self.figures_dir.mkdir(parents=True, exist_ok=True)
            self.subset: str = config["dataset"]["active_subset"]
            self.dl_model_type: str = config["model"]["type"]   # e.g. "lstm"
            self.n_background: int = 50
            self.low_rul_threshold: float = float(config["model"]["rul_threshold"])
        except Exception as e:
            raise RULException(e, sys) from e

    # ---------- internal helpers ----------

    def _load_trainer(self, model_path: Path) -> Any:
        if model_path.suffix == ".pkl":
            return joblib.load(model_path)
        elif model_path.suffix == ".pt":
            from rul_prediction.components.sequence_trainer import SequenceTrainer
            return SequenceTrainer.load(model_path)
        else:
            raise ValueError(f"Unsupported model extension: {model_path.suffix}")

    def _load_test_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return X_flat (n,510), X_seq (n,30,17), y_test (n,), engine_ids (n,).

        Applies last-window-per-engine alignment so each row = one test engine.
        """
        try:
            engine_ids_test = np.load(self.processed_dir / "test_engine_ids.npy")
            X_flat: np.ndarray = pd.read_parquet(
                self.processed_dir / "test_X_flat.parquet"
            ).to_numpy(dtype=np.float32)
            X_seq: np.ndarray = np.load(self.processed_dir / "test_X_seq.npy")
            y_test = DataIngestion(subset=self.subset).load_test_rul()["RUL"].values

            unique_engines = np.unique(engine_ids_test)
            last_idx = [
                int(np.where(engine_ids_test == eid)[0][-1]) for eid in unique_engines
            ]
            X_flat_eval = X_flat[last_idx]
            X_seq_eval = X_seq[last_idx]
            engine_ids_eval = engine_ids_test[last_idx]

            if len(X_flat_eval) != len(y_test):
                raise RuntimeError(
                    f"Alignment mismatch: {len(X_flat_eval)} windows vs "
                    f"{len(y_test)} ground-truth RUL values"
                )
            logger.info(f"Test data loaded: {len(y_test)} engines, y range [{y_test.min()}, {y_test.max()}]")
            return X_flat_eval, X_seq_eval, y_test, engine_ids_eval
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e

    def _feature_names(self) -> list[str]:
        """Extract 17 sensor-level feature names from test_X_flat.parquet column names."""
        cols = pd.read_parquet(self.processed_dir / "test_X_flat.parquet").columns.tolist()
        seen: dict[str, bool] = {}
        for c in cols:
            base = re.sub(r"_t\d+$", "", c)
            if base not in seen:
                seen[base] = True
        return list(seen.keys())

    def _get_background(self, n: int = 50) -> torch.Tensor:
        """Random subset of training windows for GradientExplainer background."""
        X_train = np.load(self.processed_dir / "train_X_seq.npy")
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(X_train), size=min(n, len(X_train)), replace=False)
        return torch.from_numpy(X_train[idx]).float()

    @staticmethod
    def _collapse_xgb(values: np.ndarray, n_sensors: int, reduce: str = "sum") -> np.ndarray:
        """Collapse (n, 510) → (n, 17) by grouping all 30 timestep values per sensor.

        The flat parquet layout is TIME-FIRST: columns are ordered as all n_sensors features
        at t0, then all n_sensors features at t1, ..., then at t29.  Sensor j therefore
        occupies columns j, j+n_sensors, j+2*n_sensors, ..., j+29*n_sensors (stride = n_sensors).

        reduce="sum"  for SHAP values (total attribution across the window)
        reduce="mean" for raw feature values (representative per-sensor value)
        """
        result = np.zeros((values.shape[0], n_sensors), dtype=np.float32)
        for j in range(n_sensors):
            block = values[:, j::n_sensors]   # shape (n, 30) — all timesteps for sensor j
            result[:, j] = block.sum(axis=1) if reduce == "sum" else block.mean(axis=1)
        return result

    @staticmethod
    def _collapse_bilstm(values: np.ndarray, reduce: str = "sum") -> np.ndarray:
        """Collapse (n, 30, 17) → (n, 17) by reducing the timestep axis."""
        return values.sum(axis=1) if reduce == "sum" else values.mean(axis=1)

    # ---------- SHAP computation ----------

    def compute_xgb_shap(
        self,
        model_path: Path,
        X_test_flat: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """TreeExplainer SHAP for XGBoost. Returns (shap_values (n,510), expected_value)."""
        try:
            trainer = self._load_trainer(model_path)
            explainer = shap.TreeExplainer(trainer.model_)
            shap_values = explainer.shap_values(X_test_flat)
            expected = float(
                explainer.expected_value[0]
                if hasattr(explainer.expected_value, "__len__")
                else explainer.expected_value
            )
            logger.info(f"XGB SHAP computed: {shap_values.shape}  expected_value={expected:.3f}")
            return np.array(shap_values), expected
        except Exception as e:
            raise RULException(e, sys) from e

    def compute_bilstm_shap(
        self,
        model_path: Path,
        X_test_seq: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """GradientExplainer SHAP for BiLSTM. Returns (shap_values (n,30,17), base_value).

        Estimated runtime: 2–5 min on CPU for 100 test engines with 50 background samples.
        """
        try:
            from rul_prediction.components.sequence_trainer import SequenceTrainer

            class _Unsqueeze(torch.nn.Module):
                """Wrap a (batch,) model output to (batch, 1) — required by GradientExplainer."""
                def __init__(self, base: torch.nn.Module) -> None:
                    super().__init__()
                    self.base = base

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.base(x).unsqueeze(-1)

            nn_model = SequenceTrainer.load(model_path).model_
            nn_model.eval()
            wrapped = _Unsqueeze(nn_model)

            background = self._get_background(self.n_background)
            logger.info(
                f"BiLSTM SHAP: background={tuple(background.shape)}, "
                f"test={X_test_seq.shape} — this may take 2-5 minutes..."
            )
            explainer = shap.GradientExplainer(wrapped, background)
            X_tensor = torch.from_numpy(X_test_seq).float()
            raw = explainer.shap_values(X_tensor)
            # GradientExplainer returns list[array] (one per output) or a single array.
            # For a wrapped (batch,1) output, SHAP may append the output dim as the last
            # axis giving shape (n, 30, 17, 1) — squeeze it back to (n, 30, 17).
            shap_values = np.array(raw[0] if isinstance(raw, list) else raw)
            if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
                shap_values = shap_values[..., 0]

            with torch.no_grad():
                base_value = float(nn_model(background).mean().item())

            logger.info(f"BiLSTM SHAP computed: {shap_values.shape}  base_value={base_value:.3f}")
            return shap_values, base_value
        except Exception as e:
            raise RULException(e, sys) from e

    # ---------- global explanations ----------

    def plot_global_importance(
        self,
        shap_2d: np.ndarray,
        feature_names: list[str],
        model_label: str,
    ) -> Path:
        """Horizontal bar chart of mean |SHAP| per sensor. EDA top-4 highlighted in red."""
        try:
            mean_abs = np.abs(shap_2d).mean(axis=0)
            order = np.argsort(mean_abs)  # ascending for horizontal bar (bottom = highest)
            names_sorted = [feature_names[i] for i in order]
            vals_sorted = mean_abs[order]
            colors = ["red" if n in _EDA_TOP4 else "steelblue" for n in names_sorted]

            fig, ax = plt.subplots(figsize=(8, 7))
            y_pos = np.arange(len(names_sorted))
            ax.barh(y_pos, vals_sorted, color=colors, alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names_sorted)
            ax.set_xlabel("Mean |SHAP value| (cycles)")
            ax.set_title(f"{model_label} — Global Feature Importance (SHAP)")
            legend_elements = [
                Patch(facecolor="red", label="EDA top-4 Pearson correlate"),
                Patch(facecolor="steelblue", label="Other sensor"),
            ]
            ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
            plt.tight_layout()
            out = self.figures_dir / f"shap_global_{model_label}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Global importance plot saved: {out.name}")
            return out
        except Exception as e:
            raise RULException(e, sys) from e

    def plot_summary_beeswarm(
        self,
        shap_2d: np.ndarray,
        X_summarised: np.ndarray,
        feature_names: list[str],
        model_label: str,
    ) -> Path:
        """SHAP beeswarm summary plot — one dot per sample per feature, coloured by value."""
        try:
            shap.summary_plot(
                shap_2d,
                X_summarised,
                feature_names=feature_names,
                show=False,
                max_display=17,
            )
            plt.title(f"{model_label} — SHAP Beeswarm Summary", pad=12)
            plt.tight_layout()
            out = self.figures_dir / f"shap_beeswarm_{model_label}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close("all")
            logger.info(f"Beeswarm plot saved: {out.name}")
            return out
        except Exception as e:
            raise RULException(e, sys) from e

    # ---------- local explanations ----------

    def _select_local_engines(
        self,
        y_test: np.ndarray,
        engine_ids: np.ndarray,
    ) -> list[int]:
        """Return up to 3 indices: near-failure, high-RUL, median-RUL engine."""
        idx_low = int(np.argmin(y_test))
        idx_high = int(np.argmax(y_test))
        median_val = float(np.median(y_test))
        idx_mid = int(np.argmin(np.abs(y_test - median_val)))
        return list(dict.fromkeys([idx_low, idx_mid, idx_high]))[:3]

    def plot_local_explanation(
        self,
        shap_values: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        engine_ids: np.ndarray,
        feature_names: list[str],
        model_label: str,
        base_value: float,
        is_seq: bool,
    ) -> list[Path]:
        """Waterfall plots for 3 representative engines (near-failure, median, high-RUL)."""
        try:
            saved: list[Path] = []
            for idx in self._select_local_engines(y_test, engine_ids):
                eid = int(engine_ids[idx])
                if is_seq:
                    shap_per_feat = shap_values[idx].sum(axis=0).astype(np.float64)
                    X_repr = X_test[idx].mean(axis=0).astype(np.float64)
                else:
                    n = len(feature_names)
                    shap_per_feat = np.array([
                        shap_values[idx, i * 30 : (i + 1) * 30].sum() for i in range(n)
                    ], dtype=np.float64)
                    X_repr = np.array([
                        X_test[idx, i * 30 : (i + 1) * 30].mean() for i in range(n)
                    ], dtype=np.float64)

                exp = shap.Explanation(
                    values=shap_per_feat,
                    base_values=float(base_value),
                    data=X_repr,
                    feature_names=feature_names,
                )
                shap.waterfall_plot(exp, show=False)
                plt.gcf().suptitle(
                    f"{model_label} | Engine {eid} | "
                    f"True RUL: {y_test[idx]:.0f} | Predicted: {y_pred[idx]:.0f}",
                    fontsize=10,
                    y=1.01,
                )
                plt.tight_layout()
                out = self.figures_dir / f"shap_local_{model_label}_engine{eid}.png"
                plt.savefig(out, dpi=150, bbox_inches="tight")
                plt.close("all")
                logger.info(f"Local explanation saved: {out.name}")
                saved.append(out)
            return saved
        except Exception as e:
            raise RULException(e, sys) from e

    # ---------- cross-check ----------

    def cross_check_vs_eda(
        self,
        shap_2d: np.ndarray,
        feature_names: list[str],
        model_label: str,
    ) -> dict[str, Any]:
        """Check whether EDA top-4 sensors appear in SHAP top-5 ranking."""
        mean_abs = np.abs(shap_2d).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        ranking = [feature_names[i] for i in order]

        matched = [s for s in _EDA_TOP4 if s in ranking[:5]]
        mismatched = [s for s in _EDA_TOP4 if s not in ranking[:5]]

        for sensor in _EDA_TOP4:
            rank = ranking.index(sensor) + 1 if sensor in ranking else -1
            if rank <= 5:
                status = "MATCH (top-5)"
            elif rank <= 10:
                status = "MATCH (top-10)"
            else:
                status = "MISMATCH"
            logger.info(f"[{model_label}] {sensor} ranks #{rank} in SHAP — {status}")

        return {"shap_top5": ranking[:5], "matched": matched, "mismatched": mismatched}

    # ---------- write-up ----------

    def write_findings(
        self,
        xgb_cross: dict[str, Any],
        dl_cross: dict[str, Any],
        xgb_ranking: list[str],
        dl_ranking: list[str],
        dl_label: str = "lstm",
    ) -> Path:
        """Write reports/phase6_findings.md — plain language summary for stakeholders."""
        try:
            def _rank_table(ranking: list[str]) -> str:
                rows = ["| Rank | Sensor | In EDA top-4? |", "|---|---|---|"]
                for i, s in enumerate(ranking[:10], 1):
                    mark = "YES" if s in _EDA_TOP4 else ""
                    rows.append(f"| {i} | {s} | {mark} |")
                return "\n".join(rows)

            all_matched_top5 = set(xgb_cross["matched"]) & set(dl_cross["matched"])
            xgb_top10 = set(xgb_ranking[:10])
            dl_top10  = set(dl_ranking[:10])
            all_in_top10 = all(s in xgb_top10 and s in dl_top10 for s in _EDA_TOP4)
            if len(all_matched_top5) == 4:
                cross_verdict = (
                    "Both models rank all four EDA sensors in their top-5 SHAP features. "
                    "Strong positive signal: the models rely on physically meaningful "
                    "HPC degradation indicators rather than spurious correlations."
                )
            elif all_in_top10:
                in5 = sorted(all_matched_top5)
                out5 = sorted(set(_EDA_TOP4) - all_matched_top5)
                cross_verdict = (
                    f"All four EDA sensors appear in the top-10 for both models. "
                    f"Sensors {in5} are in both top-5; sensors {out5} rank 6th–10th. "
                    "This is a positive result — the slight demotion reflects shared credit "
                    "among correlated HPC sensors, not absence of signal."
                )
            else:
                cross_verdict = (
                    "One or more EDA top-4 sensors rank outside the top-10 for at least one model. "
                    "This warrants investigation: it may reflect a secondary degradation pathway "
                    "or a spurious correlation worth examining before deployment."
                )

            lines = [
                "# Phase 6 — SHAP Explainability Findings",
                "",
                "## Overview",
                "",
                "This report summarises SHAP-based feature attribution for two trained models on the",
                f"NASA CMAPSS FD001 turbofan dataset: the XGBoost baseline (Phase 4) and the {dl_label.upper()}",
                "production model (Phase 5). The goal is to explain *why* each model predicts a given",
                "RUL and whether those explanations align with known HPC degradation physics.",
                "",
                "## XGBoost — Global Results",
                "",
                f"Top-5 sensors by mean |SHAP|: **{', '.join(xgb_cross['shap_top5'])}**",
                "",
                _rank_table(xgb_ranking),
                "",
                "![XGBoost global importance](figures/shap_global_xgboost.png)",
                "![XGBoost beeswarm](figures/shap_beeswarm_xgboost.png)",
                "",
                f"## {dl_label.upper()} — Global Results",
                "",
                f"Top-5 sensors by mean |SHAP|: **{', '.join(dl_cross['shap_top5'])}**",
                "",
                _rank_table(dl_ranking),
                "",
                f"![{dl_label.upper()} global importance](figures/shap_global_{dl_label}.png)",
                f"![{dl_label.upper()} beeswarm](figures/shap_beeswarm_{dl_label}.png)",
                "",
                "## EDA Cross-Check",
                "",
                "Phase 2 EDA identified four sensors as the strongest Pearson correlates with RUL",
                "in FD001 (High Pressure Compressor degradation mode):",
                "",
                "| Sensor | Symbol | Physical Role |",
                "|---|---|---|",
                "| sensor_11 | Ps30 | HPC static pressure — drops as compressor degrades |",
                "| sensor_4  | T50  | LPT outlet temperature — rises as efficiency falls |",
                "| sensor_12 | phi  | Fuel/pressure ratio — increases to compensate |",
                "| sensor_7  | P30  | HPC outlet pressure — falls with deterioration |",
                "",
                cross_verdict,
                "",
                "## Local Explanations",
                "",
                "Waterfall plots for three representative engines are saved in `reports/figures/`.",
                "Each plot shows how individual sensors pushed the predicted RUL above or below the",
                "model's average (base value), for:",
                "- **Near-failure engine** (lowest true RUL — most safety-critical)",
                "- **High-RUL engine** (furthest from failure — normal operation)",
                "- **Median-RUL engine** (representative mid-life prediction)",
                "",
                f"## Limitations of GradientExplainer for {dl_label.upper()}",
                "",
                "- **Approximate**: GradientExplainer uses integrated gradients, satisfying the",
                "  completeness axiom but approximating true Shapley values. Exact SHAP via",
                "  KernelExplainer would require ~hours at 510 effective input dimensions.",
                f"- **Timestep aggregation**: {dl_label.upper()} SHAP values have shape (n, 30, 17). Collapsing",
                "  across timesteps (sum) loses information about *when* in the degradation window",
                "  each sensor matters most — a limitation for time-resolved attribution.",
                "- **Background sensitivity**: Results depend on the 50-sample background set.",
                "  Top-ranked sensors are typically stable across random seeds, but magnitudes may vary.",
                "",
                "## Why Explainability Matters in Aviation",
                "",
                "Predictive maintenance decisions in safety-critical systems cannot rest on black-box",
                "outputs alone. A maintenance engineer grounding an aircraft needs to know *which*",
                "engine parameters are driving a low-RUL alert before acting on it.",
                "",
                "SHAP explanations provide that justification:",
                "- If sensor_11 (Ps30) and sensor_7 (P30) dominate a low-RUL prediction, the engineer",
                "  has a concrete HPC health story: compressor pressure is dropping, consistent with",
                "  expected blade tip clearance growth or fouling.",
                "- If an unexpected sensor leads the explanation, that is a flag to investigate sensor",
                "  calibration or data quality before trusting the RUL estimate.",
                "- Emerging regulatory frameworks for AI in aviation — including the EASA AI Roadmap 2.0",
                "  and the FAA's Safety Risk Management (SRM) process, which is increasingly being applied",
                "  to AI/ML-based aviation systems — increasingly require interpretability as part of",
                "  airworthiness arguments, making SHAP outputs directly relevant to certification.",
            ]
            out = self.reports_dir / "phase6_findings.md"
            out.write_text("\n".join(lines), encoding="utf-8")
            logger.info(f"Findings written: {out}")
            return out
        except Exception as e:
            raise RULException(e, sys) from e

    # ---------- orchestrator ----------

    def explain_all(self) -> None:
        """Run the full Phase 6 pipeline for XGBoost and BiLSTM."""
        try:
            pkl_paths = sorted(self.saved_models_dir.glob("*.pkl"))
            # Select .pt file whose stem starts with the configured model type (e.g. "bilstm")
            pt_paths = sorted(
                p for p in self.saved_models_dir.glob("*.pt")
                if p.stem.startswith(self.dl_model_type)
            )
            if not pkl_paths:
                raise FileNotFoundError(f"No .pkl files in {self.saved_models_dir}")
            if not pt_paths:
                raise FileNotFoundError(
                    f"No .pt files matching '{self.dl_model_type}_*.pt' in {self.saved_models_dir}"
                )
            xgb_path = pkl_paths[-1]
            bilstm_path = pt_paths[-1]
            logger.info(f"XGBoost model: {xgb_path.name}")
            logger.info(f"BiLSTM model:  {bilstm_path.name}")

            X_flat, X_seq, y_test, engine_ids = self._load_test_data()
            feature_names = self._feature_names()
            logger.info(f"Features ({len(feature_names)}): {feature_names}")

            # ---- XGBoost ----
            xgb_trainer = self._load_trainer(xgb_path)
            xgb_pred = xgb_trainer.predict(X_flat)
            xgb_shap, xgb_base = self.compute_xgb_shap(xgb_path, X_flat)
            xgb_2d = self._collapse_xgb(xgb_shap, len(feature_names), reduce="sum")
            X_flat_sensor = self._collapse_xgb(X_flat, len(feature_names), reduce="mean")

            self.plot_global_importance(xgb_2d, feature_names, "xgboost")
            self.plot_summary_beeswarm(xgb_2d, X_flat_sensor, feature_names, "xgboost")
            self.plot_local_explanation(
                xgb_shap, X_flat, y_test, xgb_pred, engine_ids,
                feature_names, "xgboost", xgb_base, is_seq=False,
            )
            xgb_cross = self.cross_check_vs_eda(xgb_2d, feature_names, "xgboost")

            # ---- DL model (label from config: bilstm / lstm / gru) ----
            dl_label = self.dl_model_type
            bilstm_trainer = self._load_trainer(bilstm_path)
            bilstm_pred = bilstm_trainer.predict(X_seq)
            bilstm_shap, bilstm_base = self.compute_bilstm_shap(bilstm_path, X_seq)
            bilstm_2d = self._collapse_bilstm(bilstm_shap, reduce="sum")
            X_seq_sensor = self._collapse_bilstm(X_seq, reduce="mean")

            self.plot_global_importance(bilstm_2d, feature_names, dl_label)
            self.plot_summary_beeswarm(bilstm_2d, X_seq_sensor, feature_names, dl_label)
            self.plot_local_explanation(
                bilstm_shap, X_seq, y_test, bilstm_pred, engine_ids,
                feature_names, dl_label, bilstm_base, is_seq=True,
            )
            bilstm_cross = self.cross_check_vs_eda(bilstm_2d, feature_names, dl_label)

            xgb_mean_abs = np.abs(xgb_2d).mean(axis=0)
            xgb_ranking = [feature_names[i] for i in np.argsort(xgb_mean_abs)[::-1]]
            bilstm_mean_abs = np.abs(bilstm_2d).mean(axis=0)
            bilstm_ranking = [feature_names[i] for i in np.argsort(bilstm_mean_abs)[::-1]]

            self.write_findings(xgb_cross, bilstm_cross, xgb_ranking, bilstm_ranking, dl_label=dl_label)

            logger.info("Phase 6 complete. Output files:")
            for p in sorted(self.figures_dir.glob("shap_*.png")):
                logger.info(f"  figures/{p.name}")
            logger.info(f"  {self.reports_dir / 'phase6_findings.md'}")
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e


if __name__ == "__main__":
    _config = load_config()
    ModelExplainer(_config).explain_all()
