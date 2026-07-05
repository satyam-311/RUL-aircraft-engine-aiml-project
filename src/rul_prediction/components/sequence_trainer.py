"""
SequenceTrainer component for training, evaluating, and persisting LSTM/GRU/BiLSTM models.

Saves model weights as .pt (torch.save state_dict) plus a companion _config.json
for architecture reconstruction — never joblib of the whole class.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

from rul_prediction.config.configuration import load_config
from rul_prediction.exception.exception import RULException
from rul_prediction.logger.logger import get_logger

logger = get_logger(__name__)


def asymmetric_rul_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    rul_threshold: float = 30.0,
    alpha: float = 2.0,
) -> torch.Tensor:
    """MSE loss that penalises over-prediction in the low-RUL danger zone.

    Weight = alpha when (y_pred > y_true) AND (y_true < rul_threshold), else 1.
    This directly targets the optimistic bias observed at low RUL values.
    """
    residual = y_pred - y_true
    weight = torch.where(
        (residual > 0) & (y_true < rul_threshold),
        torch.full_like(residual, alpha),
        torch.ones_like(residual),
    )
    return (weight * residual**2).mean()


class _RULDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class RULSequenceModel(nn.Module):
    """RNN-based RUL regressor. Supports LSTM, GRU, and BiLSTM."""

    SUPPORTED_TYPES: frozenset[str] = frozenset({"lstm", "gru", "bilstm"})

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        model_type: str,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if model_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Choose from {sorted(self.SUPPORTED_TYPES)}"
            )
        # Store arch params so save() can serialise them to JSON
        self.arch: dict[str, Any] = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "model_type": model_type,
            "dropout": dropout,
        }
        bidirectional = model_type == "bilstm"
        rnn_cls = nn.LSTM if model_type in ("lstm", "bilstm") else nn.GRU
        rnn_dropout = dropout if num_layers > 1 else 0.0

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        fc_in = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.rnn(x)            # (batch, seq_len, hidden * dirs)
        last = out[:, -1, :]            # last time step: (batch, hidden * dirs)
        last = self.dropout(last)
        return self.fc(last).squeeze(-1)  # (batch,)


class SequenceTrainer:
    """Train, evaluate, and persist a configurable LSTM/GRU/BiLSTM regressor.

    Unlike ModelTrainer, fit() requires both train and val arrays because
    early stopping and LR scheduling depend on validation loss.
    """

    SUPPORTED_MODELS: frozenset[str] = frozenset({"lstm", "gru", "bilstm"})

    def __init__(self, config: dict[str, Any]) -> None:
        try:
            model_cfg = config["model"]
            train_cfg = config["training"]
            self.model_type: str = model_cfg["type"]
            if self.model_type not in self.SUPPORTED_MODELS:
                raise ValueError(
                    f"Unsupported model type '{self.model_type}'. "
                    f"Choose from: {sorted(self.SUPPORTED_MODELS)}"
                )
            self.hidden_size: int = int(model_cfg["hidden_size"])
            self.num_layers: int = int(model_cfg["num_layers"])
            self.dropout: float = float(model_cfg["dropout"])
            self.patience: int = int(model_cfg["early_stopping_patience"])
            self.rul_threshold: float = float(model_cfg["rul_threshold"])
            self.loss_alpha: float = float(model_cfg["loss_alpha"])
            self.lr: float = float(train_cfg["learning_rate"])
            self.batch_size: int = int(train_cfg["batch_size"])
            self.num_epochs: int = int(train_cfg["num_epochs"])
            self.saved_models_dir = Path(config["paths"]["saved_models_dir"])
            self.seed: int = int(config.get("seed", 42))
            self.model_: RULSequenceModel | None = None
            self.is_fitted_: bool = False
            self.training_time_s_: float | None = None
        except Exception as e:
            raise RULException(e, sys) from e

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "SequenceTrainer":
        """Train with early stopping; restores best weights before returning."""
        try:
            torch.manual_seed(self.seed)
            input_size = X_train.shape[2]
            self.model_ = RULSequenceModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                model_type=self.model_type,
                dropout=self.dropout,
            )

            train_loader = DataLoader(
                _RULDataset(X_train, y_train),
                batch_size=self.batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                _RULDataset(X_val, y_val),
                batch_size=self.batch_size,
                shuffle=False,
            )

            optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5
            )

            best_val_loss = float("inf")
            best_state: dict = {}
            epochs_no_improve = 0

            n_params = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
            logger.info(
                f"Training {self.model_type} | hidden={self.hidden_size} "
                f"layers={self.num_layers} params={n_params:,} | "
                f"max_epochs={self.num_epochs} patience={self.patience}"
            )
            _t0 = time.perf_counter()

            for epoch in range(1, self.num_epochs + 1):
                # --- train ---
                self.model_.train()
                train_losses: list[float] = []
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    y_hat = self.model_(X_batch)
                    loss = asymmetric_rul_loss(
                        y_hat, y_batch, self.rul_threshold, self.loss_alpha
                    )
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                # --- validate ---
                self.model_.eval()
                val_losses: list[float] = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        y_hat = self.model_(X_batch)
                        loss = asymmetric_rul_loss(
                            y_hat, y_batch, self.rul_threshold, self.loss_alpha
                        )
                        val_losses.append(loss.item())

                train_loss = float(np.mean(train_losses))
                val_loss = float(np.mean(val_losses))
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model_.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epoch % 20 == 0 or epochs_no_improve == 0:
                    logger.info(
                        f"Epoch {epoch:03d}/{self.num_epochs} "
                        f"train={train_loss:.4f}  val={val_loss:.4f}  "
                        f"best={best_val_loss:.4f}  no_improve={epochs_no_improve}"
                    )

                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch} (patience={self.patience})")
                    break

            self.model_.load_state_dict(best_state)
            self.training_time_s_ = time.perf_counter() - _t0
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
            if not self.is_fitted_ or self.model_ is None:
                raise RuntimeError("SequenceTrainer is not fitted. Call fit() first.")
            self.model_.eval()
            with torch.no_grad():
                X_t = torch.from_numpy(X).float()
                return self.model_(X_t).numpy()
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
        """Save state dict to .pt and architecture to companion _config.json."""
        try:
            if not self.is_fitted_ or self.model_ is None:
                raise RuntimeError("Cannot save an unfitted model. Call fit() first.")
            if path is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = self.saved_models_dir / f"{self.model_type}_{ts}.pt"
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(self.model_.state_dict(), path)

            arch_data = dict(self.model_.arch)
            arch_data["training_time_s"] = self.training_time_s_
            config_path = path.with_name(path.stem + "_config.json")
            config_path.write_text(json.dumps(arch_data, indent=2))

            logger.info(f"Model saved -> {path}")
            logger.info(f"Arch config saved -> {config_path}")
            return path
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e

    @classmethod
    def load(cls, path: str | Path) -> "SequenceTrainer":
        """Reconstruct from .pt weights + companion _config.json. No full config dict needed."""
        try:
            path = Path(path)
            config_path = path.with_name(path.stem + "_config.json")
            if not config_path.exists():
                raise FileNotFoundError(f"Arch config not found: {config_path}")

            arch = json.loads(config_path.read_text())
            training_time_s = arch.pop("training_time_s", None)

            model = RULSequenceModel(**arch)
            state_dict = torch.load(path, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()

            # Build a SequenceTrainer for inference without requiring the full config
            instance = cls.__new__(cls)
            instance.model_ = model
            instance.is_fitted_ = True
            instance.model_type = arch["model_type"]
            instance.hidden_size = arch["hidden_size"]
            instance.num_layers = arch["num_layers"]
            instance.dropout = arch["dropout"]
            instance.training_time_s_ = training_time_s
            # Sentinel values — unused for inference
            instance.patience = 0
            instance.rul_threshold = 30.0
            instance.loss_alpha = 2.0
            instance.lr = 0.0
            instance.batch_size = 64
            instance.num_epochs = 0
            instance.saved_models_dir = path.parent
            instance.seed = 42

            logger.info(f"Loaded {arch['model_type']} model from {path}")
            return instance
        except RULException:
            raise
        except Exception as e:
            raise RULException(e, sys) from e


if __name__ == "__main__":
    _config = load_config()
    _processed = Path(_config["paths"]["processed_data_dir"])

    _X_train = np.load(_processed / "train_X_seq.npy")
    _y_train = np.load(_processed / "train_y.npy")
    _X_val = np.load(_processed / "val_X_seq.npy")
    _y_val = np.load(_processed / "val_y.npy")

    _trainer = SequenceTrainer(_config).fit(_X_train, _y_train, _X_val, _y_val)
    _val_metrics = _trainer.evaluate(_X_val, _y_val)
    logger.info(f"Val metrics: {_val_metrics}")
    _saved = _trainer.save()
    logger.info(f"Model saved at {_saved}")

    # Round-trip load test
    _loaded = SequenceTrainer.load(_saved)
    _loaded_metrics = _loaded.evaluate(_X_val, _y_val)
    logger.info(f"Loaded model val metrics: {_loaded_metrics}")
    logger.info("Smoke test complete.")
