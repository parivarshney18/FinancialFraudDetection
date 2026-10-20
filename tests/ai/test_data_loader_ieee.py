"""
Tests for IEEE-CIS Data Loader and Validation Module.
Verifies schema validation, primary key integrity, left-merge semantics,
dataset availability reporting, memory analysis, and error handling.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from ai.src.data_loader_ieee import (
    check_dataset_availability,
    validate_ieee_transaction_df,
    validate_ieee_identity_df,
    merge_ieee_tables,
    load_ieee_cis_raw,
    get_dataset_memory_report,
    get_full_dataset_fraud_prevalence,
    validate_dataset_integrity,
    IEEE_RAW_DIR,
)


@pytest.fixture
def valid_transaction_df():
    """Create a minimal valid synthetic IEEE-CIS transaction DataFrame."""
    return pd.DataFrame({
        "TransactionID": [2987000, 2987001, 2987002, 2987003, 2987004],
        "isFraud": [0, 0, 1, 0, 0],
        "TransactionDT": [86400, 86401, 86402, 86405, 86410],
        "TransactionAmt": [68.5, 29.0, 59.0, 50.0, 102.5],
        "ProductCD": ["W", "W", "W", "H", "C"],
        "card1": [13926, 2755, 4663, 18132, 4461],
    })


@pytest.fixture
def valid_identity_df():
    """Create a minimal valid synthetic IEEE-CIS identity DataFrame."""
    return pd.DataFrame({
        "TransactionID": [2987001, 2987003, 2987004],
        "DeviceType": ["mobile", "desktop", "desktop"],
        "DeviceInfo": ["Samsung", "MacOS", "Windows"],
    })


def test_check_dataset_availability_keys():
    """Verify check_dataset_availability returns required diagnostic keys."""
    status = check_dataset_availability()
    assert isinstance(status, dict)
    assert "raw_dir" in status
    assert "raw_dir_exists" in status
    assert "train_transaction_exists" in status
    assert "train_transaction_size_mb" in status
    assert "train_identity_exists" in status
    assert "train_identity_size_mb" in status
    assert "is_ready_for_loading" in status
    assert status["raw_dir_exists"] is True


def test_check_dataset_availability_custom_dir(tmp_path):
    """Verify check_dataset_availability handles an empty custom directory properly."""
    status = check_dataset_availability(tmp_path)
    assert status["raw_dir_exists"] is True
    assert status["train_transaction_exists"] is False
    assert status["train_identity_exists"] is False
    assert status["is_ready_for_loading"] is False


def test_validate_ieee_transaction_df_success(valid_transaction_df):
    """Verify valid transaction dataframe passes validation without error."""
    validate_ieee_transaction_df(valid_transaction_df)


def test_validate_ieee_transaction_df_missing_primary_key(valid_transaction_df):
    """Verify ValueError is raised if TransactionID is missing."""
    df_missing_key = valid_transaction_df.drop(columns=["TransactionID"])
    with pytest.raises(ValueError, match="Transaction table must contain 'TransactionID'"):
        validate_ieee_transaction_df(df_missing_key)


def test_validate_ieee_transaction_df_duplicate_key(valid_transaction_df):
    """Verify ValueError is raised if duplicate TransactionID values exist."""
    dup_df = valid_transaction_df.copy()
    dup_df.loc[1, "TransactionID"] = dup_df.loc[0, "TransactionID"]
    with pytest.raises(ValueError, match="duplicate TransactionID"):
        validate_ieee_transaction_df(dup_df)


def test_validate_ieee_transaction_df_missing_target(valid_transaction_df):
    """Verify ValueError is raised if isFraud is missing."""
    df_missing_target = valid_transaction_df.drop(columns=["isFraud"])
    with pytest.raises(ValueError, match="Transaction table must contain 'isFraud'"):
        validate_ieee_transaction_df(df_missing_target)


def test_validate_ieee_transaction_df_invalid_target_values(valid_transaction_df):
    """Verify ValueError is raised if isFraud contains non-binary values."""
    df_invalid_target = valid_transaction_df.copy()
    df_invalid_target.loc[0, "isFraud"] = 2
    with pytest.raises(ValueError, match="must contain only {0, 1}"):
        validate_ieee_transaction_df(df_invalid_target)


def test_validate_ieee_identity_df_success(valid_identity_df):
    """Verify valid identity dataframe passes validation without error."""
    validate_ieee_identity_df(valid_identity_df)


def test_validate_ieee_identity_df_missing_primary_key(valid_identity_df):
    """Verify ValueError is raised if identity table lacks TransactionID."""
    df_no_key = valid_identity_df.drop(columns=["TransactionID"])
    with pytest.raises(ValueError, match="Identity table must contain 'TransactionID'"):
        validate_ieee_identity_df(df_no_key)


def test_validate_ieee_identity_df_duplicate_key(valid_identity_df):
    """Verify ValueError is raised if identity table contains duplicate TransactionID."""
    dup_id = valid_identity_df.copy()
    dup_id.loc[1, "TransactionID"] = dup_id.loc[0, "TransactionID"]
    with pytest.raises(ValueError, match="duplicate TransactionID"):
        validate_ieee_identity_df(dup_id)


def test_merge_ieee_tables_preserves_all_transactions(valid_transaction_df, valid_identity_df):
    """Verify left merge preserves exact transaction count and enriches matching rows."""
    merged = merge_ieee_tables(valid_transaction_df, valid_identity_df)

    assert len(merged) == len(valid_transaction_df)
    assert "DeviceType" in merged.columns
    assert "DeviceInfo" in merged.columns

    # Transactions with identity info should be populated
    row_with_id = merged[merged["TransactionID"] == 2987001].iloc[0]
    assert row_with_id["DeviceType"] == "mobile"
    assert row_with_id["DeviceInfo"] == "Samsung"

    # Transactions without identity info should have NaN
    row_without_id = merged[merged["TransactionID"] == 2987000].iloc[0]
    assert pd.isna(row_without_id["DeviceType"])
    assert pd.isna(row_without_id["DeviceInfo"])


def test_merge_ieee_tables_none_or_empty_identity(valid_transaction_df):
    """Verify merge handles None or empty identity gracefully without changes."""
    merged_none = merge_ieee_tables(valid_transaction_df, None)
    assert len(merged_none) == len(valid_transaction_df)
    assert merged_none.columns.tolist() == valid_transaction_df.columns.tolist()

    merged_empty = merge_ieee_tables(valid_transaction_df, pd.DataFrame())
    assert len(merged_empty) == len(valid_transaction_df)


def test_load_ieee_cis_raw_missing_file_raises_filenotfound(tmp_path):
    """Verify load_ieee_cis_raw raises informative FileNotFoundError when files are absent."""
    with pytest.raises(FileNotFoundError, match="IEEE-CIS transaction file not found"):
        load_ieee_cis_raw(data_dir=tmp_path)


def test_load_ieee_cis_raw_with_mock_files(tmp_path, valid_transaction_df, valid_identity_df):
    """Verify end-to-end load and merge from mock CSV files on disk."""
    tx_file = tmp_path / "train_transaction.csv"
    id_file = tmp_path / "train_identity.csv"

    valid_transaction_df.to_csv(tx_file, index=False)
    valid_identity_df.to_csv(id_file, index=False)

    df_loaded = load_ieee_cis_raw(data_dir=tmp_path, nrows=3, load_identity=True, validate=True)
    assert len(df_loaded) == 3
    assert "DeviceType" in df_loaded.columns
    assert "isFraud" in df_loaded.columns


def test_get_dataset_memory_report(valid_transaction_df):
    """Verify memory report structure, null calculations, and target distribution."""
    # Add a null value to test null calculation
    df = valid_transaction_df.copy()
    df.loc[0, "card1"] = np.nan

    rep = get_dataset_memory_report(df)
    assert rep["rows"] == 5
    assert rep["columns"] == 6
    assert rep["memory_bytes"] > 0
    assert rep["memory_mb"] >= 0
    assert rep["total_null_cells"] == 1
    assert rep["overall_null_pct"] > 0
    assert "target_distribution" in rep
    assert rep["target_distribution"]["total_transactions"] == 5
    assert rep["target_distribution"]["fraud_count"] == 1
    assert rep["target_distribution"]["legitimate_count"] == 4
    assert rep["target_distribution"]["fraud_prevalence_pct"] == 20.0


def test_get_full_dataset_fraud_prevalence_mock(tmp_path, valid_transaction_df):
    """Verify get_full_dataset_fraud_prevalence works accurately on mock file."""
    tx_file = tmp_path / "train_transaction.csv"
    valid_transaction_df.to_csv(tx_file, index=False)

    stats = get_full_dataset_fraud_prevalence(tmp_path)
    assert stats["total_transactions"] == 5
    assert stats["fraud_count"] == 1
    assert stats["legitimate_count"] == 4
    assert stats["fraud_prevalence_pct"] == 20.0
    assert stats["imbalance_ratio"] == "4.0:1"


def test_validate_dataset_integrity_mock(tmp_path, valid_transaction_df, valid_identity_df):
    """Verify validate_dataset_integrity verifies key uniqueness, referential integrity, and columns."""
    tx_file = tmp_path / "train_transaction.csv"
    id_file = tmp_path / "train_identity.csv"

    valid_transaction_df.to_csv(tx_file, index=False)
    valid_identity_df.to_csv(id_file, index=False)

    integrity = validate_dataset_integrity(tmp_path)
    assert integrity["train_transaction_rows"] == 5
    assert integrity["train_transaction_cols"] == 6
    assert integrity["train_identity_rows"] == 3
    assert integrity["train_identity_cols"] == 3
    assert integrity["merged_expected_cols"] == 8
    assert integrity["transaction_id_duplicates_tx"] == 0
    assert integrity["transaction_id_duplicates_id"] == 0
    assert integrity["identity_is_subset_of_tx"] is True
    assert integrity["transactions_with_identity"] == 3
    assert integrity["transactions_without_identity"] == 2
    assert integrity["identity_coverage_pct"] == 60.0
    assert integrity["target_distribution"]["fraud_count"] == 1


def test_validate_dataset_integrity_unmatched_identity_raises(tmp_path, valid_transaction_df, valid_identity_df):
    """Verify ValueError is raised if identity table has TransactionID not in transaction table."""
    tx_file = tmp_path / "train_transaction.csv"
    id_file = tmp_path / "train_identity.csv"

    valid_transaction_df.to_csv(tx_file, index=False)
    bad_id_df = valid_identity_df.copy()
    bad_id_df.loc[0, "TransactionID"] = 9999999
    bad_id_df.to_csv(id_file, index=False)

    with pytest.raises(ValueError, match="identity TransactionIDs not found in transaction table"):
        validate_dataset_integrity(tmp_path)


def test_validate_dataset_integrity_duplicate_keys_raises(tmp_path, valid_transaction_df, valid_identity_df):
    """Verify ValueError is raised if duplicate keys exist during full integrity check."""
    tx_file = tmp_path / "train_transaction.csv"
    id_file = tmp_path / "train_identity.csv"

    dup_tx = valid_transaction_df.copy()
    dup_tx.loc[1, "TransactionID"] = dup_tx.loc[0, "TransactionID"]
    dup_tx.to_csv(tx_file, index=False)
    valid_identity_df.to_csv(id_file, index=False)

    with pytest.raises(ValueError, match="duplicate TransactionID"):
        validate_dataset_integrity(tmp_path)


def test_live_full_dataset_verification():
    """Verify live raw IEEE-CIS dataset integrity, row counts, and fraud prevalence if present on disk."""
    status = check_dataset_availability()
    if not status["is_ready_for_loading"]:
        pytest.skip("Raw IEEE-CIS files not found on disk, skipping live verification.")

    integrity = validate_dataset_integrity()
    assert integrity["train_transaction_rows"] == 590540
    assert integrity["train_transaction_cols"] == 394
    assert integrity["train_identity_rows"] == 144233
    assert integrity["train_identity_cols"] == 41
    assert integrity["merged_expected_cols"] == 434
    assert integrity["transaction_id_duplicates_tx"] == 0
    assert integrity["transaction_id_duplicates_id"] == 0
    assert integrity["identity_is_subset_of_tx"] is True
    assert integrity["transactions_with_identity"] == 144233
    assert integrity["transactions_without_identity"] == 446307
    assert integrity["identity_coverage_pct"] == 24.42

    target_stats = integrity["target_distribution"]
    assert target_stats["total_transactions"] == 590540
    assert target_stats["fraud_count"] == 20663
    assert target_stats["legitimate_count"] == 569877
    assert target_stats["fraud_prevalence_pct"] == 3.4990
    assert target_stats["imbalance_ratio"] == "27.58:1"

