# -*- coding: utf-8 -*-
"""
src/data_processor.py
---------------------
Data loading, validation, and stratified splitting for large-scale
telemetry datasets stored as compressed Parquet files.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class TelemetryDataProcessor:
    """
    Loads a Parquet telemetry dataset and prepares stratified train/test
    splits, ensuring class-imbalance ratios are preserved in both partitions.

    Parameters
    ----------
    filepath : str
        Path to the compressed Parquet dataset.
    target_col : str
        Name of the binary target column (default: 'value_compressed_flag').

    Examples
    --------
    >>> processor = TelemetryDataProcessor("data/synthetic_telecom_retention_1m.parquet")
    >>> X_train, X_test, y_train, y_test = processor.prepare_splits()
    """

    # Columns that should never enter the feature matrix
    _LEAKAGE_COLS = frozenset(["change_rev", "change_mou", "avg3rev", "totrev"])
    _ID_COLS      = frozenset(["customer_id"])

    def __init__(self, filepath: str, target_col: str = "value_compressed_flag"):
        self.filepath   = filepath
        self.target_col = target_col
        self._df: pd.DataFrame | None = None  # lazy-load cache

    # ── Public API ──────────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        """
        Loads the Parquet dataset, applying temporal-leakage guards.
        Result is cached so subsequent calls are free.

        Returns
        -------
        pd.DataFrame
            Cleaned dataset with leakage columns removed.
        """
        if self._df is None:
            raw = pd.read_parquet(self.filepath)
            cols_to_drop = [
                c for c in raw.columns
                if c in self._LEAKAGE_COLS
            ]
            if cols_to_drop:
                print(f"  [DataProcessor] Dropping {len(cols_to_drop)} leakage column(s): {cols_to_drop}")
            self._df = raw.drop(columns=cols_to_drop)
            print(f"  [DataProcessor] Loaded → shape: {self._df.shape}  |  "
                  f"positive rate: {self._df[self.target_col].mean():.3f}")
        return self._df

    def prepare_splits(
        self,
        test_size: float = 0.30,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Executes a stratified train / test split.

        Identifier columns (e.g. customer_id) are automatically excluded
        from the feature matrix to prevent data leakage.

        Parameters
        ----------
        test_size : float
            Fraction of records allocated to the test partition.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        X_train, X_test : pd.DataFrame
        y_train, y_test : pd.Series
        """
        df = self.load_data()

        exclude = self._ID_COLS | {self.target_col}
        feature_cols = [c for c in df.columns if c not in exclude]

        X = df[feature_cols]
        y = df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )

        print(
            f"  [DataProcessor] Train: {X_train.shape}  |  "
            f"Test: {X_test.shape}  |  "
            f"Positive rate — Train: {y_train.mean():.4f}  "
            f"Test: {y_test.mean():.4f}"
        )
        return X_train, X_test, y_train, y_test
