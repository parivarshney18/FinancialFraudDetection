"""
Cross-validation pipeline module for Financial Fraud Detection.
Implements leak-free Stratified 5-Fold Cross-Validation with strict fold isolation.
"""

from typing import Generator, Tuple, List, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ai.src.config import (
    RANDOM_SEED,
    CV_N_SPLITS,
    CV_SHUFFLE,
    TARGET_COL,
)
from ai.src.preprocessing import FraudPreprocessor


def get_cv_split_indices(
    df: pd.DataFrame,
    n_splits: int = CV_N_SPLITS,
    shuffle: bool = CV_SHUFFLE,
    random_state: int = RANDOM_SEED,
    target_col: str = TARGET_COL,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate train and validation index splits using StratifiedKFold.

    Args:
        df: Dataset containing the target column.
        n_splits: Number of cross-validation folds (default 5).
        shuffle: Whether to shuffle before splitting (default True).
        random_state: Random seed for reproducibility (default 42).
        target_col: Name of the target column.

    Returns:
        List of (train_indices, val_indices) numpy arrays.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    y = df[target_col].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = list(skf.split(df, y))
    return splits


def generate_cv_folds(
    df: pd.DataFrame,
    n_splits: int = CV_N_SPLITS,
    shuffle: bool = CV_SHUFFLE,
    random_state: int = RANDOM_SEED,
    target_col: str = TARGET_COL,
    as_df: bool = False,
    scale_numerical: bool = True,
) -> Generator[
    Tuple[int, Union[np.ndarray, pd.DataFrame], np.ndarray, Union[np.ndarray, pd.DataFrame], np.ndarray, FraudPreprocessor],
    None,
    None,
]:
    """Generator yielding leak-free preprocessed training and validation fold partitions.

    For each fold:
    1. Splits the dataset into raw training and validation subsets.
    2. Initializes an independent FraudPreprocessor.
    3. Fits the preprocessor strictly on the training partition.
    4. Transforms the validation partition using the training-fitted parameters.
    5. Yields (fold_idx, X_train, y_train, X_val, y_val, preprocessor).

    Guarantees:
    - Zero data leakage: scalers, encoders, and imputers never observe validation rows.
    - Zero target leakage: target_col is separated and excluded from feature matrices.
    - No synthetic resampling (SMOTE) is applied to validation partitions.
    - Every row appears in a validation fold exactly once.
    - Raw input DataFrame is never modified.

    Args:
        df: Input DataFrame containing raw transactions and target column.
        n_splits: Number of folds (default 5).
        shuffle: Whether to shuffle before splitting (default True).
        random_state: Random seed for reproducibility (default 42).
        target_col: Target column name (default 'Is_Fraud').
        as_df: If True, yields X_train and X_val as pandas DataFrames.
               If False, yields 2D NumPy arrays (TabNet / PyTorch ready).
        scale_numerical: Whether to apply RobustScaler to numerical features.

    Yields:
        fold_idx: 0-indexed fold number (0 to n_splits - 1).
        X_train: Transformed training features.
        y_train: Training binary labels (1D array).
        X_val: Transformed validation features.
        y_val: Validation binary labels (1D array).
        preprocessor: The fitted FraudPreprocessor instance for the fold.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    y = df[target_col].values.astype(np.int64)
    splits = get_cv_split_indices(
        df,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        target_col=target_col,
    )

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_raw = df.iloc[train_idx]
        val_raw = df.iloc[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        # Initialize fresh preprocessor for this fold
        preprocessor = FraudPreprocessor(
            target_col=target_col,
            scale_numerical=scale_numerical,
        )

        # Fit exclusively on training partition
        X_train = preprocessor.fit_transform(train_raw, as_df=as_df)

        # Transform validation partition using training-fitted parameters
        X_val = preprocessor.transform(val_raw, as_df=as_df)

        yield fold_idx, X_train, y_train, X_val, y_val, preprocessor


def validate_cv_integrity(
    df: pd.DataFrame,
    n_splits: int = CV_N_SPLITS,
    shuffle: bool = CV_SHUFFLE,
    random_state: int = RANDOM_SEED,
    target_col: str = TARGET_COL,
) -> Dict[str, Any]:
    """Execute structural verification on cross-validation folds.

    Checks:
    - Exact 100% validation coverage (no duplicates, no omissions).
    - Strict train/validation disjointness in every fold.
    - Class balance preservation across all validation folds.
    - Both classes present in every fold's train and validation sets.
    """
    n_rows = len(df)
    splits = get_cv_split_indices(
        df,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        target_col=target_col,
    )

    val_indices_collected = []
    fold_summaries = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # Disjointness check
        intersection = np.intersect1d(train_idx, val_idx)
        if len(intersection) > 0:
            raise AssertionError(f"Fold {fold_idx} has {len(intersection)} overlapping train/val rows.")

        val_indices_collected.extend(val_idx)

        val_y = df.iloc[val_idx][target_col].values
        train_y = df.iloc[train_idx][target_col].values

        n_val_fraud = int(np.sum(val_y == 1))
        n_val_legit = int(np.sum(val_y == 0))
        val_fraud_rate = float(np.mean(val_y == 1) * 100)

        n_train_fraud = int(np.sum(train_y == 1))
        n_train_legit = int(np.sum(train_y == 0))

        if n_val_fraud == 0 or n_val_legit == 0:
            raise AssertionError(f"Fold {fold_idx} validation set is missing a class.")
        if n_train_fraud == 0 or n_train_legit == 0:
            raise AssertionError(f"Fold {fold_idx} train set is missing a class.")

        fold_summaries.append({
            "fold": fold_idx,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "train_fraud_count": n_train_fraud,
            "val_fraud_count": n_val_fraud,
            "val_fraud_rate_pct": round(val_fraud_rate, 4),
        })

    # Exact coverage check
    unique_val_indices = np.unique(val_indices_collected)
    if len(unique_val_indices) != n_rows:
        raise AssertionError(f"Validation indices cover {len(unique_val_indices)} rows instead of {n_rows}.")
    if len(val_indices_collected) != n_rows:
        raise AssertionError("Some rows appeared in validation folds multiple times.")

    return {
        "n_rows": n_rows,
        "n_splits": n_splits,
        "exact_coverage_verified": True,
        "disjointness_verified": True,
        "both_classes_present_all_folds": True,
        "fold_summaries": fold_summaries,
    }
