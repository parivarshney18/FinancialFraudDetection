"""
Comprehensive unit and integration tests for Phase 2:
Preprocessing and Stratified Cross-Validation Pipeline.
Verifies fold stratification, class presence, zero-leakage isolation,
raw data immutability, missing/corrupt timestamp resilience, and TabNet compatibility.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ai.src.config import TARGET_COL, RANDOM_SEED, CV_N_SPLITS
from ai.src.data_loader import load_raw_data
from ai.src.preprocessing import FraudPreprocessor, PROCESSED_CATEGORICAL_COLS, PROCESSED_NUMERICAL_COLS
from ai.src.cv_pipeline import (
    get_cv_split_indices,
    generate_cv_folds,
    validate_cv_integrity,
)


@pytest.fixture(scope="module")
def sample_dataset():
    """Load a representative subset (2,000 rows) of the raw dataset for fast, deterministic testing."""
    df = load_raw_data(validate=True)
    return df.iloc[:2000].copy()


def test_cv_splits_coverage_and_disjointness(sample_dataset):
    """Verify that 5 folds partition the dataset completely with zero overlap."""
    n_rows = len(sample_dataset)
    splits = get_cv_split_indices(sample_dataset, n_splits=5, shuffle=True, random_state=42)

    assert len(splits) == 5, "Expected exactly 5 folds."
    all_val_indices = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # 1. Train and validation sets must be disjoint within each fold
        intersection = np.intersect1d(train_idx, val_idx)
        assert len(intersection) == 0, f"Fold {fold_idx} has {len(intersection)} overlapping indices."

        # 2. Sizes must match standard 80/20 train/val split
        assert len(train_idx) + len(val_idx) == n_rows
        all_val_indices.extend(val_idx)

    # 3. Every row must appear in validation exactly once
    unique_val = np.unique(all_val_indices)
    assert len(unique_val) == n_rows, f"Validation set covered {len(unique_val)} rows, expected {n_rows}."
    assert len(all_val_indices) == n_rows, "Some rows appeared in validation multiple times."


def test_stratification_and_class_presence(sample_dataset):
    """Verify both fraud and non-fraud classes appear in all train and validation folds."""
    splits = get_cv_split_indices(sample_dataset, n_splits=5, shuffle=True, random_state=42)
    y = sample_dataset[TARGET_COL].values

    global_fraud_rate = np.mean(y == 1)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_y = y[train_idx]
        val_y = y[val_idx]

        # Both classes must exist
        assert np.sum(train_y == 1) > 0, f"Fold {fold_idx} train set missing fraud class."
        assert np.sum(train_y == 0) > 0, f"Fold {fold_idx} train set missing legitimate class."
        assert np.sum(val_y == 1) > 0, f"Fold {fold_idx} validation set missing fraud class."
        assert np.sum(val_y == 0) > 0, f"Fold {fold_idx} validation set missing legitimate class."

        # Stratified distribution: validation fraud rate close to global rate
        val_fraud_rate = np.mean(val_y == 1)
        assert abs(val_fraud_rate - global_fraud_rate) < 0.02, (
            f"Fold {fold_idx} validation fraud rate ({val_fraud_rate:.4f}) diverged significantly from global ({global_fraud_rate:.4f})."
        )


def test_target_exclusion_from_features(sample_dataset):
    """Verify that Is_Fraud is strictly excluded from preprocessor feature outputs."""
    prep = FraudPreprocessor(target_col=TARGET_COL)
    X = prep.fit_transform(sample_dataset)

    # 1. Target column must not be in feature names
    feature_names = prep.get_feature_names()
    assert TARGET_COL not in feature_names, f"Target '{TARGET_COL}' leaked into feature_names."

    # 2. Feature matrix width must equal expected processed columns (9 categorical + 13 numerical = 22)
    expected_dim = len(PROCESSED_CATEGORICAL_COLS) + len(PROCESSED_NUMERICAL_COLS)
    assert X.shape[1] == expected_dim, f"Expected {expected_dim} features, got {X.shape[1]}."

    # 3. Check generator as well
    for _, X_tr, y_tr, X_va, y_va, p in generate_cv_folds(sample_dataset.iloc[:200], n_splits=2):
        assert TARGET_COL not in p.get_feature_names()
        assert len(y_tr.shape) == 1
        assert len(y_va.shape) == 1
        assert set(np.unique(y_tr)).issubset({0, 1})
        break


def test_raw_data_immutability(sample_dataset):
    """Verify raw input DataFrame is not modified in-place by the preprocessing pipeline."""
    original_copy = sample_dataset.copy(deep=True)
    original_columns = list(sample_dataset.columns)
    original_shape = sample_dataset.shape

    # Run CV pipeline
    list(generate_cv_folds(sample_dataset, n_splits=2))

    # Assert completely unchanged
    assert list(sample_dataset.columns) == original_columns, "Raw dataset columns were mutated."
    assert sample_dataset.shape == original_shape, "Raw dataset shape was mutated."
    pd.testing.assert_frame_equal(sample_dataset, original_copy)


def test_train_validation_isolation_and_no_leakage(sample_dataset):
    """Verify scalers and encoders are fitted solely on training folds."""
    splits = get_cv_split_indices(sample_dataset, n_splits=3, shuffle=True, random_state=42)

    scalers_centers = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_df = sample_dataset.iloc[train_idx]
        val_df = sample_dataset.iloc[val_idx]

        prep = FraudPreprocessor()
        prep.fit(train_df)

        # Scaler center_ must match median of training fold numerical columns
        train_engineered = prep._engineer_features(train_df)
        expected_medians = [
            train_engineered[col].median() for col in prep.numerical_cols
        ]
        np.testing.assert_allclose(prep.scaler_.center_, expected_medians, rtol=1e-5)

        scalers_centers.append(prep.scaler_.center_)

    # Across different folds, scaler parameters must differ because training subsets differ
    assert not np.allclose(scalers_centers[0], scalers_centers[1]), (
        "Scaler parameters for fold 0 and fold 1 are identical, indicating lack of fold independence."
    )


def test_unseen_categorical_value_handling():
    """Verify validation categories absent from training fold map to unknown index safely."""
    train_data = pd.DataFrame({
        "Gender": ["Male", "Female", "Male", "Female"],
        "State": ["Delhi", "Maharashtra", "Delhi", "Karnataka"],
        "City": ["Delhi", "Mumbai", "Delhi", "Bengaluru"],
        "Account_Type": ["Savings", "Checking", "Savings", "Business"],
        "Transaction_Type": ["Debit", "Credit", "Transfer", "Debit"],
        "Merchant_Category": ["Groceries", "Electronics", "Groceries", "Restaurant"],
        "Device_Type": ["Mobile", "Desktop", "POS", "ATM"],
        "Transaction_Device": ["Mobile Device", "Web Browser", "POS Terminal", "ATM"],
        "Transaction_Description": ["Grocery", "Electronics", "Transfer", "Cash"],
        "Age": [25, 45, 32, 50],
        "Transaction_Amount": [1000.0, 5000.0, 200.0, 15000.0],
        "Account_Balance": [25000.0, 10000.0, 8000.0, 50000.0],
        "Transaction_Date": ["01-01-2025", "02-01-2025", "03-01-2025", "04-01-2025"],
        "Transaction_Time": ["10:00:00", "14:30:00", "23:15:00", "02:00:00"],
    })

    val_data = pd.DataFrame({
        "Gender": ["Female", "Non-Binary"],  # 'Non-Binary' is unseen
        "State": ["Goa", "Maharashtra"],     # 'Goa' is unseen
        "City": ["Panaji", "Mumbai"],         # 'Panaji' is unseen
        "Account_Type": ["Savings", "Student"], # 'Student' is unseen
        "Transaction_Type": ["Debit", "Crypto"],# 'Crypto' is unseen
        "Merchant_Category": ["Groceries", "Luxury"], # 'Luxury' is unseen
        "Device_Type": ["Mobile", "Kiosk"],  # 'Kiosk' is unseen
        "Transaction_Device": ["Smart Watch", "ATM"], # 'Smart Watch' is unseen
        "Transaction_Description": ["UnknownNarration", "Cash"], # unseen narration
        "Age": [30, 40],
        "Transaction_Amount": [2000.0, 4000.0],
        "Account_Balance": [15000.0, 20000.0],
        "Transaction_Date": ["05-01-2025", "06-01-2025"],
        "Transaction_Time": ["11:00:00", "15:00:00"],
    })

    prep = FraudPreprocessor()
    X_train = prep.fit_transform(train_data)
    X_val = prep.transform(val_data)

    assert X_train.shape[1] == 22
    assert X_val.shape[1] == 22
    assert not np.isnan(X_val).any(), "Validation output contains NaNs."

    # Check that unknown categories mapped to unknown_category_values_ (which equals K)
    for idx, k in prep.unknown_category_values_.items():
        # Any value in val that was unseen must equal k
        val_col_vals = X_val[:, idx]
        # Max index cannot exceed k
        assert np.all(val_col_vals <= k), f"Categorical index exceeded cardinality bound: {val_col_vals}"
        assert prep.cat_dims[idx] == k + 1, "cat_dims must equal k + 1."


def test_missing_and_invalid_datetime_handling():
    """Verify corrupt, invalid, or null date/time strings are safely imputed without raising exceptions."""
    corrupt_data = pd.DataFrame({
        "Gender": ["Male", None],
        "State": ["Delhi", "Maharashtra"],
        "City": ["Delhi", None],
        "Account_Type": ["Savings", "Checking"],
        "Transaction_Type": ["Debit", "Credit"],
        "Merchant_Category": ["Groceries", "Electronics"],
        "Device_Type": ["Mobile", "Desktop"],
        "Transaction_Device": ["Mobile Device", "Web Browser"],
        "Transaction_Description": ["Grocery", "Electronics"],
        "Age": [None, 45],
        "Transaction_Amount": [None, 5000.0],
        "Account_Balance": [25000.0, None],
        "Transaction_Date": ["99-99-9999", None],  # Completely invalid and missing dates
        "Transaction_Time": ["invalid_time", None], # Completely invalid and missing times
    })

    prep = FraudPreprocessor()
    X = prep.fit_transform(corrupt_data)

    assert X.shape == (2, 22)
    assert not np.isnan(X).any(), "Transformed output from corrupt data contains NaNs."
    assert np.all(np.isfinite(X)), "Transformed output contains infinite values."


def test_reproducibility_with_fixed_seed(sample_dataset):
    """Verify cross-validation splits and transformed matrices are 100% reproducible with fixed seed."""
    df_subset = sample_dataset.iloc[:300]

    # First run
    folds_run1 = list(generate_cv_folds(df_subset, n_splits=3, random_state=42))
    # Second run
    folds_run2 = list(generate_cv_folds(df_subset, n_splits=3, random_state=42))

    for (f1, X_tr1, y_tr1, X_va1, y_va1, _), (f2, X_tr2, y_tr2, X_va2, y_va2, _) in zip(folds_run1, folds_run2):
        assert f1 == f2
        np.testing.assert_array_equal(y_tr1, y_tr2)
        np.testing.assert_array_equal(y_va1, y_va2)
        np.testing.assert_allclose(X_tr1, X_tr2, rtol=1e-6)
        np.testing.assert_allclose(X_va1, X_va2, rtol=1e-6)

    # Different seed should yield different split
    splits_seed_42 = get_cv_split_indices(df_subset, n_splits=3, random_state=42)
    splits_seed_99 = get_cv_split_indices(df_subset, n_splits=3, random_state=99)
    assert not np.array_equal(splits_seed_42[0][1], splits_seed_99[0][1]), (
        "Different seeds yielded identical validation splits."
    )


def test_cv_integrity_utility(sample_dataset):
    """Verify validate_cv_integrity runs cleanly and passes all assertions."""
    audit = validate_cv_integrity(sample_dataset, n_splits=5, shuffle=True, random_state=42)
    assert audit["exact_coverage_verified"] is True
    assert audit["disjointness_verified"] is True
    assert audit["both_classes_present_all_folds"] is True
    assert len(audit["fold_summaries"]) == 5
