"""
Unit tests for Phase 4.5: Class-Imbalance Experiments and SMOTENC Pipeline.
Verifies categorical feature integrity, train/validation isolation, 1:1 balanced resampling,
and metric dictionary contracts.
"""

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE, SMOTENC

from ai.src.cv_pipeline import generate_cv_folds
from ai.src.experiments_imbalance import (
    get_smotenc_resampler,
    train_and_eval_smotenc_fold,
)


@pytest.fixture
def mixed_synthetic_data():
    """Create mixed categorical and numerical imbalanced dataset."""
    np.random.seed(42)
    n_samples = 300
    # 3 categorical columns (values in 0..4) and 3 continuous numerical columns
    cat_cols = np.random.randint(0, 5, size=(n_samples, 3)).astype(np.float64)
    num_cols = np.random.randn(n_samples, 3).astype(np.float64)
    X = np.hstack([cat_cols, num_cols])

    # 5% fraud rate (15 positive cases, 285 negative cases)
    y = np.zeros(n_samples, dtype=np.int64)
    y[:15] = 1
    np.random.shuffle(y)

    cat_idxs = [0, 1, 2]
    return X, y, cat_idxs


def test_smotenc_preserves_categorical_integers(mixed_synthetic_data):
    """Verify SMOTENC strictly preserves integer values in categorical columns while standard SMOTE corrupts them."""
    X, y, cat_idxs = mixed_synthetic_data

    # 1. Standard continuous SMOTE produces illegal fractional categories
    standard_smote = SMOTE(random_state=42)
    X_std, _ = standard_smote.fit_resample(X, y)
    has_fractional_in_std = False
    for c in cat_idxs:
        if not np.all(np.equal(np.mod(X_std[:, c], 1), 0)):
            has_fractional_in_std = True
            break
    assert has_fractional_in_std, "Standard SMOTE was expected to create fractional numbers in categorical columns."

    # 2. SMOTENC strictly keeps all categorical columns as discrete integers
    smotenc = get_smotenc_resampler(cat_idxs=cat_idxs, random_state=42)
    X_res, y_res = smotenc.fit_resample(X, y)

    for c in cat_idxs:
        assert np.all(np.equal(np.mod(X_res[:, c], 1), 0)), f"SMOTENC produced non-integer in categorical feature {c}"


def test_smotenc_balanced_distribution(mixed_synthetic_data):
    """Verify SMOTENC resamples the minority class to exactly match majority count."""
    X, y, cat_idxs = mixed_synthetic_data
    smotenc = get_smotenc_resampler(cat_idxs=cat_idxs, random_state=42)
    X_res, y_res = smotenc.fit_resample(X, y)

    n_neg = int(np.sum(y_res == 0))
    n_pos = int(np.sum(y_res == 1))

    assert n_pos == n_neg, f"Expected 1:1 balance, got {n_pos} positives vs {n_neg} negatives."
    assert len(y_res) == 2 * int(np.sum(y == 0))


def test_smotenc_train_val_isolation_and_no_leakage():
    """Verify that resampling is applied only to training folds and validation fold remains natural and untouched."""
    # Synthetic DataFrame simulating raw bank transactions
    np.random.seed(42)
    n_rows = 500
    df = pd.DataFrame({
        "Account_Type": np.random.choice(["Savings", "Current"], size=n_rows),
        "Gender": np.random.choice(["M", "F"], size=n_rows),
        "Transaction_Type": np.random.choice(["Transfer", "Debit", "Credit"], size=n_rows),
        "Merchant_Category": np.random.choice(["Retail", "Food", "Travel"], size=n_rows),
        "Device_Type": np.random.choice(["Mobile", "Web", "ATM"], size=n_rows),
        "Transaction_Device": np.random.choice(["Android", "iOS", "Windows"], size=n_rows),
        "State": np.random.choice(["CA", "NY", "TX"], size=n_rows),
        "City": np.random.choice(["CityA", "CityB"], size=n_rows),
        "Transaction_Description": np.random.choice(["Purchase", "Bill", "ATM"], size=n_rows),
        "Age": np.random.randint(18, 70, size=n_rows),
        "Transaction_Amount": np.random.uniform(10, 5000, size=n_rows),
        "Account_Balance": np.random.uniform(100, 50000, size=n_rows),
        "Transaction_Date": "2024-01-15",
        "Transaction_Time": "14:30:00",
        "Is_Fraud": [1] * 25 + [0] * (n_rows - 25),  # 5% fraud
    })
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    cv_gen = generate_cv_folds(df, n_splits=3, random_state=42, target_col="Is_Fraud")
    for fold_idx, X_tr, y_tr, X_val, y_val, prep in cv_gen:
        # Pre-resample check
        n_val_orig = len(y_val)
        val_fraud_rate_orig = np.mean(y_val)

        # Apply SMOTENC to training fold
        smotenc = get_smotenc_resampler(cat_idxs=prep.cat_idxs, random_state=42 + fold_idx)
        X_tr_res, y_tr_res = smotenc.fit_resample(X_tr, y_tr)

        # Validation set must remain 100% untouched
        assert len(y_val) == n_val_orig, "Validation set size changed after resampling!"
        assert np.isclose(np.mean(y_val), val_fraud_rate_orig), "Validation fraud rate changed!"
        assert np.sum(y_tr_res == 1) == np.sum(y_tr_res == 0), "Training set must be 1:1 balanced."
        break


def test_smotenc_model_evaluation_contract(mixed_synthetic_data):
    """Verify train_and_eval_smotenc_fold runs cleanly and returns complete metric dictionary."""
    X, y, cat_idxs = mixed_synthetic_data
    smotenc = get_smotenc_resampler(cat_idxs=cat_idxs, random_state=42)
    X_res, y_res = smotenc.fit_resample(X[:200], y[:200])

    class DummyPrep:
        cat_idxs = [0, 1, 2]
        cat_dims = [6, 6, 6]

    metrics, elapsed = train_and_eval_smotenc_fold(
        model_name="logistic_regression",
        X_train_res=X_res,
        y_train_res=y_res,
        X_val=X[200:],
        y_val=y[200:],
        preprocessor=DummyPrep(),
        fold_idx=0,
        verbose=False,
        smoke_test=True,
    )

    required_keys = ["pr_auc", "roc_auc", "recall_at_fpr_1pct", "f1", "macro_f1", "precision", "recall", "confusion_matrix"]
    for k in required_keys:
        assert k in metrics, f"Missing metric {k} in return dictionary."
    assert elapsed > 0.0
