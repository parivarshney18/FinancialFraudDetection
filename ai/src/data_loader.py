"""
Data loader and schema validation module for Bank Transaction Fraud Detection.
Loads raw data without modifying the source file and validates dimensions, schema, and types.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from ai.src.config import (
    RAW_DATA_PATH,
    TARGET_COL,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    DROP_COLS,
)


def load_raw_data(filepath: Optional[Path] = None, validate: bool = True) -> pd.DataFrame:
    """Load the raw dataset from disk in read-only mode and optionally validate schema.

    Args:
        filepath: Custom path to dataset. If None, defaults to config.RAW_DATA_PATH.
        validate: Whether to perform runtime integrity assertions.

    Returns:
        pd.DataFrame: The loaded transaction dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If schema or validation assertions fail.
    """
    path = Path(filepath) if filepath else RAW_DATA_PATH

    if not path.exists():
        raise FileNotFoundError(f"Raw transaction dataset not found at {path}")

    # Read dataset
    df = pd.read_csv(path)

    if validate:
        validate_dataset(df)

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate dataset structure, dimensions, target distribution, and integrity.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If any structural constraint is violated.
    """
    expected_rows = 200000
    expected_cols = 24

    if df.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {df.shape[0]}")

    if df.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns, found {df.shape[1]}")

    # Verify target column
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' is missing.")

    unique_targets = set(df[TARGET_COL].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Target column must contain only binary values {{0, 1}}, got {unique_targets}")

    # Verify numerical and categorical columns
    for col in NUMERICAL_COLS:
        if col not in df.columns:
            raise ValueError(f"Numerical feature '{col}' is missing.")

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            raise ValueError(f"Categorical feature '{col}' is missing.")

    for col in DROP_COLS:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' is missing.")

    # Null value verification
    null_count = df.isnull().sum().sum()
    if null_count != 0:
        raise ValueError(f"Dataset contains {null_count} unexpected null values.")
