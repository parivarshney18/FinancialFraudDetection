"""
Data loader and validation module for the IEEE-CIS Fraud Detection dataset.
Supports merging train_transaction.csv and train_identity.csv via TransactionID,
validating schema and primary key integrity, checking class distributions,
and reporting memory utilization.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IEEE_RAW_DIR = PROJECT_ROOT / "ai" / "data" / "ieee_cis" / "raw"
IEEE_PROCESSED_DIR = PROJECT_ROOT / "ai" / "data" / "ieee_cis" / "processed"

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"

REQUIRED_TRANSACTION_COLS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
]


def check_dataset_availability(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Check whether official IEEE-CIS raw files exist on disk.

    Args:
        data_dir: Path to directory containing raw files. Defaults to ai/data/ieee_cis/raw.

    Returns:
        Dict detailing file existence and sizes.
    """
    raw_path = Path(data_dir) if data_dir else IEEE_RAW_DIR
    tx_file = raw_path / TRANSACTION_FILE
    id_file = raw_path / IDENTITY_FILE

    # Also check if compressed versions (.zip, .tar.gz) exist
    tx_zip = raw_path / f"{TRANSACTION_FILE}.zip"
    id_zip = raw_path / f"{IDENTITY_FILE}.zip"

    status = {
        "raw_dir": str(raw_path),
        "raw_dir_exists": raw_path.exists(),
        "train_transaction_exists": tx_file.exists(),
        "train_transaction_size_mb": round(tx_file.stat().st_size / (1024 * 1024), 2) if tx_file.exists() else 0.0,
        "train_identity_exists": id_file.exists(),
        "train_identity_size_mb": round(id_file.stat().st_size / (1024 * 1024), 2) if id_file.exists() else 0.0,
        "train_transaction_zip_exists": tx_zip.exists(),
        "train_identity_zip_exists": id_zip.exists(),
        "is_ready_for_loading": tx_file.exists(),
    }
    return status


def validate_ieee_transaction_df(df: pd.DataFrame) -> None:
    """Validate structure, primary keys, and target column of IEEE-CIS transaction table."""
    if "TransactionID" not in df.columns:
        raise ValueError("Transaction table must contain 'TransactionID' primary key.")

    if df["TransactionID"].duplicated().any():
        n_dups = int(df["TransactionID"].duplicated().sum())
        raise ValueError(f"Transaction table contains {n_dups} duplicate TransactionID values.")

    if "isFraud" not in df.columns:
        raise ValueError("Transaction table must contain 'isFraud' target column.")

    unique_targets = set(df["isFraud"].dropna().unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Target column 'isFraud' must contain only {{0, 1}}, got {unique_targets}")


def validate_ieee_identity_df(df: pd.DataFrame) -> None:
    """Validate primary keys of IEEE-CIS identity table."""
    if "TransactionID" not in df.columns:
        raise ValueError("Identity table must contain 'TransactionID' primary key.")

    if df["TransactionID"].duplicated().any():
        n_dups = int(df["TransactionID"].duplicated().sum())
        raise ValueError(f"Identity table contains {n_dups} duplicate TransactionID values.")


def merge_ieee_tables(
    transaction_df: pd.DataFrame,
    identity_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Perform a left merge of transaction table with identity table using TransactionID.

    Args:
        transaction_df: Main transaction table (590,540 rows in full dataset).
        identity_df: Optional identity table (~144,233 rows in full dataset).

    Returns:
        pd.DataFrame: Merged DataFrame preserving all transaction rows.
    """
    validate_ieee_transaction_df(transaction_df)

    if identity_df is None or identity_df.empty:
        return transaction_df.copy()

    validate_ieee_identity_df(identity_df)

    n_tx_orig = len(transaction_df)
    merged = pd.merge(transaction_df, identity_df, on="TransactionID", how="left")

    if len(merged) != n_tx_orig:
        raise ValueError(f"Merge row count mismatch: Expected {n_tx_orig} rows, got {len(merged)}.")

    return merged


def load_ieee_cis_raw(
    data_dir: Optional[Path] = None,
    nrows: Optional[int] = None,
    load_identity: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """Load and merge raw IEEE-CIS dataset from disk.

    Args:
        data_dir: Path to directory containing raw CSVs. Defaults to ai/data/ieee_cis/raw.
        nrows: Optional row limit for prototyping or debugging.
               NOTE: If nrows is passed and load_identity=True, nrows transactions are
               loaded and left-joined with the identity table on TransactionID, returning
               nrows merged records with 434 total columns.
        load_identity: Whether to load and left-merge identity table.
        validate: Whether to validate schemas and primary keys.

    Returns:
        pd.DataFrame: Merged IEEE-CIS transactions DataFrame.

    Raises:
        FileNotFoundError: If train_transaction.csv is missing.
    """
    raw_path = Path(data_dir) if data_dir else IEEE_RAW_DIR
    tx_file = raw_path / TRANSACTION_FILE
    id_file = raw_path / IDENTITY_FILE

    if not tx_file.exists():
        raise FileNotFoundError(
            f"IEEE-CIS transaction file not found at: {tx_file}.\n"
            f"Please download 'train_transaction.csv' and 'train_identity.csv' from Kaggle "
            f"(https://www.kaggle.com/c/ieee-fraud-detection/data) and place them in: {raw_path}"
        )

    print(f"[IEEE-CIS Loader] Reading {tx_file.name} (nrows={nrows})...")
    tx_df = pd.read_csv(tx_file, nrows=nrows)

    if validate:
        validate_ieee_transaction_df(tx_df)

    if load_identity and id_file.exists():
        print(f"[IEEE-CIS Loader] Reading {id_file.name}...")
        id_df = pd.read_csv(id_file)
        if validate:
            validate_ieee_identity_df(id_df)
        merged_df = merge_ieee_tables(tx_df, id_df)
    else:
        merged_df = tx_df

    return merged_df


def get_dataset_memory_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute detailed memory utilization, null distribution, and type breakdown."""
    mem_bytes = df.memory_usage(deep=True).sum()
    mem_mb = mem_bytes / (1024 * 1024)
    mem_gb = mem_mb / 1024

    dtypes_count = df.dtypes.value_counts().to_dict()
    dtypes_str = {str(k): int(v) for k, v in dtypes_count.items()}

    total_cells = df.shape[0] * df.shape[1]
    null_cells = int(df.isnull().sum().sum())
    null_pct = round((null_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

    target_stats = {}
    if "isFraud" in df.columns:
        vc = df["isFraud"].value_counts().to_dict()
        n_fraud = int(vc.get(1, 0))
        n_legit = int(vc.get(0, 0))
        rate = (n_fraud / len(df)) * 100 if len(df) > 0 else 0.0
        target_stats = {
            "total_transactions": len(df),
            "legitimate_count": n_legit,
            "fraud_count": n_fraud,
            "fraud_prevalence_pct": round(rate, 3),
            "imbalance_ratio": f"{round(n_legit / max(1, n_fraud), 2)}:1",
        }

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "memory_bytes": int(mem_bytes),
        "memory_mb": round(mem_mb, 4),
        "memory_gb": round(mem_gb, 4),
        "total_null_cells": null_cells,
        "overall_null_pct": null_pct,
        "dtype_breakdown": dtypes_str,
        "target_distribution": target_stats,
    }


def get_full_dataset_fraud_prevalence(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Calculate target distribution from full train_transaction.csv without loading all columns into memory.

    Memory-efficient: Reads only the 'isFraud' column (~4.7 MB memory).
    """
    raw_path = Path(data_dir) if data_dir else IEEE_RAW_DIR
    tx_file = raw_path / TRANSACTION_FILE
    if not tx_file.exists():
        raise FileNotFoundError(f"Transaction file not found: {tx_file}")

    target_series = pd.read_csv(tx_file, usecols=["isFraud"])["isFraud"]
    vc = target_series.value_counts().to_dict()
    n_fraud = int(vc.get(1, 0))
    n_legit = int(vc.get(0, 0))
    total = len(target_series)
    rate = (n_fraud / total) * 100 if total > 0 else 0.0
    ratio = round(n_legit / max(1, n_fraud), 2)
    return {
        "total_transactions": total,
        "legitimate_count": n_legit,
        "fraud_count": n_fraud,
        "fraud_prevalence_pct": round(rate, 4),
        "legitimate_pct": round(100 - rate, 4),
        "imbalance_ratio": f"{ratio}:1",
    }


def validate_dataset_integrity(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Validate full-scale IEEE-CIS dataset integrity, key uniqueness, and merge relationship.

    Memory-efficient: loads only schemas, TransactionID columns, and isFraud (~15 MB RAM).
    """
    raw_path = Path(data_dir) if data_dir else IEEE_RAW_DIR
    tx_file = raw_path / TRANSACTION_FILE
    id_file = raw_path / IDENTITY_FILE

    if not tx_file.exists():
        raise FileNotFoundError(f"Transaction file not found: {tx_file}")
    if not id_file.exists():
        raise FileNotFoundError(f"Identity file not found: {id_file}")

    # 1. Header & columns validation
    tx_cols = pd.read_csv(tx_file, nrows=0).columns.tolist()
    id_cols = pd.read_csv(id_file, nrows=0).columns.tolist()

    for col in REQUIRED_TRANSACTION_COLS:
        if col not in tx_cols:
            raise ValueError(f"Missing required transaction column: {col}")

    required_id_cols = ["TransactionID", "DeviceType", "DeviceInfo"]
    for col in required_id_cols:
        if col not in id_cols:
            raise ValueError(f"Missing required identity column: {col}")

    # 2. Key uniqueness and referential integrity
    tx_ids = pd.read_csv(tx_file, usecols=["TransactionID"])["TransactionID"]
    id_ids = pd.read_csv(id_file, usecols=["TransactionID"])["TransactionID"]

    tx_dups = int(tx_ids.duplicated().sum())
    id_dups = int(id_ids.duplicated().sum())
    if tx_dups > 0:
        raise ValueError(f"Transaction table has {tx_dups} duplicate TransactionID values.")
    if id_dups > 0:
        raise ValueError(f"Identity table has {id_dups} duplicate TransactionID values.")

    set_tx = set(tx_ids)
    set_id = set(id_ids)
    id_in_tx = set_id.issubset(set_tx)
    if not id_in_tx:
        n_unmatched = len(set_id - set_tx)
        raise ValueError(f"{n_unmatched} identity TransactionIDs not found in transaction table.")

    overlap_count = len(set_id.intersection(set_tx))
    overlap_pct = round((overlap_count / len(tx_ids)) * 100, 2)

    # 3. Fraud distribution
    fraud_stats = get_full_dataset_fraud_prevalence(data_dir)

    return {
        "train_transaction_rows": len(tx_ids),
        "train_transaction_cols": len(tx_cols),
        "train_identity_rows": len(id_ids),
        "train_identity_cols": len(id_cols),
        "merged_expected_cols": len(tx_cols) + len(id_cols) - 1,
        "transaction_id_duplicates_tx": tx_dups,
        "transaction_id_duplicates_id": id_dups,
        "identity_is_subset_of_tx": id_in_tx,
        "transactions_with_identity": overlap_count,
        "transactions_without_identity": len(tx_ids) - overlap_count,
        "identity_coverage_pct": overlap_pct,
        "target_distribution": fraud_stats,
    }


if __name__ == "__main__":
    avail = check_dataset_availability()
    print("=" * 70)
    print("IEEE-CIS DATASET AVAILABILITY REPORT")
    print("=" * 70)
    for k, v in avail.items():
        print(f"  {k}: {v}")
    if avail["is_ready_for_loading"]:
        print("\n" + "=" * 70)
        print("RUNNING FULL DATASET INTEGRITY VALIDATION")
        print("=" * 70)
        integrity = validate_dataset_integrity()
        for k, v in integrity.items():
            print(f"  {k}: {v}")

