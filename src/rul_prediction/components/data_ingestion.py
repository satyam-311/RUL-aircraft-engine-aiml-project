"""
Data ingestion component for the NASA CMAPSS dataset.

WHY: The raw CMAPSS .txt files have no header row, are space-delimited
with irregular whitespace (often a trailing space causing an extra phantom
26th column), and require RUL to be computed manually for the training
set. If every script re-implements this parsing, small bugs (like an
off-by-one on column count) get duplicated everywhere. Centralizing it
here means every phase (EDA, preprocessing, training, inference) loads
data identically.

HOW: pandas.read_csv with sep='\\s+' (regex for one-or-more whitespace)
handles irregular spacing. We explicitly assign column names from
constants.py rather than relying on inferred headers (there are none).
RUL for training is computed per-engine as (max_cycle_for_engine - current_cycle).

WHERE: src/rul_prediction/components/data_ingestion.py
    Used by: notebooks/01_eda_fd001.ipynb (Phase 2),
             preprocessing pipeline (Phase 3),
             inference pipeline (Phase 8)
"""

import io
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

from rul_prediction.constants.constants import ALL_COLUMNS
from rul_prediction.exception.exception import RULException
from rul_prediction.logger.logger import get_logger

logger = get_logger(__name__)


def parse_cmapss_bytes(buf: bytes, columns: list) -> pd.DataFrame:
    """Parse a headerless space-delimited CMAPSS file from raw bytes.

    Same logic as DataIngestion._read_raw_file() but accepts bytes so it can
    be called by both DataIngestion and the dashboard upload handler without
    duplicating parsing code.

    Args:
        buf: Raw file bytes (e.g. from Path.read_bytes() or an uploaded file).
        columns: Column names to assign (e.g. ALL_COLUMNS from constants).

    Returns:
        DataFrame with named columns, trailing NaN columns stripped.
    """
    df = pd.read_csv(io.BytesIO(buf), sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    df = df.iloc[:, : len(columns)]
    df.columns = columns[: len(df.columns)]
    return df


class DataIngestion:
    """Loads raw CMAPSS train/test/RUL text files for a given sub-dataset."""

    def __init__(self, raw_data_dir: str = "data/raw", subset: str = "FD001"):
        """
        Args:
            raw_data_dir: Root directory containing per-subset folders,
                e.g. "data/raw/FD001/train_FD001.txt".
            subset: Which CMAPSS sub-dataset to load (FD001-FD004).
        """
        self.subset = subset
        self.subset_dir = Path(raw_data_dir) / subset

    def _read_raw_file(self, filename: str, columns: list) -> pd.DataFrame:
        """Read a single space-delimited CMAPSS file with no header."""
        try:
            file_path = self.subset_dir / filename
            logger.info(f"Reading raw file: {file_path}")
            return parse_cmapss_bytes(file_path.read_bytes(), columns)
        except Exception as e:
            raise RULException(e, sys) from e

    def load_train_data(self) -> pd.DataFrame:
        """Load training data and compute RUL for every row.

        Returns:
            DataFrame with all raw columns plus a computed 'RUL' column.
        """
        try:
            df = self._read_raw_file(f"train_{self.subset}.txt", ALL_COLUMNS)

            # RUL = max cycle observed for that engine - current cycle.
            # This works because in training data every engine runs to
            # failure, so the last recorded cycle IS the failure point.
            max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
            df["RUL"] = max_cycles - df["time_in_cycles"]

            logger.info(
                f"Loaded train_{self.subset}: {df.shape[0]} rows, "
                f"{df['unit_number'].nunique()} engines"
            )
            return df
        except Exception as e:
            raise RULException(e, sys) from e

    def load_test_data(self) -> pd.DataFrame:
        """Load test data (no RUL column - truncated before failure).

        Returns:
            DataFrame with raw columns only (no RUL).
        """
        try:
            df = self._read_raw_file(f"test_{self.subset}.txt", ALL_COLUMNS)
            logger.info(
                f"Loaded test_{self.subset}: {df.shape[0]} rows, "
                f"{df['unit_number'].nunique()} engines"
            )
            return df
        except Exception as e:
            raise RULException(e, sys) from e

    def load_test_rul(self) -> pd.DataFrame:
        """Load the ground-truth RUL values for the test set.

        Returns:
            DataFrame with one column 'RUL', indexed implicitly by engine
            order (row 0 = engine 1, row 1 = engine 2, etc.)
        """
        try:
            file_path = self.subset_dir / f"RUL_{self.subset}.txt"
            logger.info(f"Reading test RUL file: {file_path}")
            df = pd.read_csv(file_path, sep=r"\s+", header=None, names=["RUL"])
            return df
        except Exception as e:
            raise RULException(e, sys) from e

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convenience method to load train, test, and test-RUL together.

        Returns:
            Tuple of (train_df, test_df, test_rul_df).
        """
        train_df = self.load_train_data()
        test_df = self.load_test_data()
        test_rul_df = self.load_test_rul()
        return train_df, test_df, test_rul_df


if __name__ == "__main__":
    # Quick manual smoke test - run with:
    #   uv run python -m rul_prediction.components.data_ingestion
    ingestion = DataIngestion()
    train_df, test_df, test_rul_df = ingestion.load_all()

    print("\nTrain shape:", train_df.shape)
    print(train_df.head())
    print("\nTest shape:", test_df.shape)
    print(test_df.head())
    print("\nTest RUL shape:", test_rul_df.shape)
    print(test_rul_df.head())
