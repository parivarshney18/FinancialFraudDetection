"""
Unit tests for Phase 5: Behavioral Feature Engineering and Leakage Prevention.
Verifies chronological causality, zero target leakage, causal expanding statistics,
and RevisedFraudPreprocessor train/validation isolation.
"""

import numpy as np
import pandas as pd
import pytest

from ai.src.behavioral_features import (
    engineer_causal_behavioral_dataframe,
    RevisedFraudPreprocessor,
    REVISED_CATEGORICAL_COLS,
    REVISED_NUMERICAL_COLS,
)


@pytest.fixture
def sample_raw_dataframe():
    """Create realistic raw transaction sample DataFrame."""
    np.random.seed(42)
    n_rows = 100
    dates = ["01-01-2025", "02-01-2025", "03-01-2025", "04-01-2025"]
    times = ["10:00:00", "11:30:00", "14:15:00", "18:45:00", "22:10:00"]

    df = pd.DataFrame({
        "Customer_ID": [f"cust_{i}" for i in range(n_rows)],
        "Customer_Name": [f"Name {i}" for i in range(n_rows)],
        "Gender": np.random.choice(["Male", "Female"], size=n_rows),
        "Age": np.random.randint(18, 70, size=n_rows),
        "State": np.random.choice(["Maharashtra", "Karnataka", "Delhi"], size=n_rows),
        "City": np.random.choice(["Mumbai", "Bangalore", "Delhi"], size=n_rows),
        "Bank_Branch": np.random.choice(["Mumbai Branch", "Bangalore Branch"], size=n_rows),
        "Account_Type": np.random.choice(["Savings", "Checking", "Business"], size=n_rows),
        "Transaction_ID": [f"tx_{i}" for i in range(n_rows)],
        "Transaction_Date": np.random.choice(dates, size=n_rows),
        "Transaction_Time": np.random.choice(times, size=n_rows),
        "Transaction_Amount": np.random.uniform(50, 10000, size=n_rows).round(2),
        "Merchant_ID": [f"merch_{i}" for i in range(n_rows)],
        "Transaction_Type": np.random.choice(["Transfer", "Debit", "Credit"], size=n_rows),
        "Merchant_Category": np.random.choice(["Restaurant", "Retail", "Groceries"], size=n_rows),
        "Account_Balance": np.random.uniform(500, 50000, size=n_rows).round(2),
        "Transaction_Device": np.random.choice(["POS Terminal", "ATM", "Web Browser"], size=n_rows),
        "Transaction_Location": "Location",
        "Device_Type": np.random.choice(["POS", "ATM", "Mobile", "Desktop"], size=n_rows),
        "Transaction_Currency": "INR",
        "Customer_Contact": "+919XXXXXXXXXX",
        "Transaction_Description": "Purchase",
        "Customer_Email": "user@example.com",
        "Is_Fraud": [1] * 5 + [0] * (n_rows - 5),
    })
    return df


def test_causal_ordering_and_timestamp_parsing(sample_raw_dataframe):
    """Verify that dataframe is strictly sorted chronologically."""
    df_aug = engineer_causal_behavioral_dataframe(sample_raw_dataframe)

    assert "datetime" in df_aug.columns
    # Check monotonically increasing timestamps
    assert df_aug["datetime"].is_monotonic_increasing, "DataFrame is not sorted chronologically!"
    assert len(df_aug) == len(sample_raw_dataframe)


def test_no_target_leakage_in_features(sample_raw_dataframe):
    """Verify that Is_Fraud is never used or included in feature matrix."""
    df_aug = engineer_causal_behavioral_dataframe(sample_raw_dataframe)

    prep = RevisedFraudPreprocessor()
    X = prep.fit_transform(df_aug)

    assert prep.target_col not in prep.feature_names_
    assert "Is_Fraud" not in prep.feature_names_
    assert X.shape[1] == len(prep.feature_names_)


def test_causality_expanding_category_mean(sample_raw_dataframe):
    """Verify that expanding category mean uses strictly preceding transactions."""
    df_aug = engineer_causal_behavioral_dataframe(sample_raw_dataframe)

    # For the very first transaction in each category, prior mean must default to current amount (ratio ~ 1.0)
    for cat, group in df_aug.groupby("Merchant_Category"):
        first_row = group.iloc[0]
        expected_ratio = first_row["Transaction_Amount"] / (first_row["Transaction_Amount"] + 1.0)
        assert np.isclose(first_row["Cat_Amount_Ratio"], expected_ratio, atol=1e-3)


def test_revised_preprocessor_isolation(sample_raw_dataframe):
    """Verify RevisedFraudPreprocessor fits strictly on train and transforms val without leakage."""
    df_aug = engineer_causal_behavioral_dataframe(sample_raw_dataframe)

    train_df = df_aug.iloc[:70].copy()
    val_df = df_aug.iloc[70:].copy()

    prep = RevisedFraudPreprocessor()
    X_train = prep.fit_transform(train_df)
    X_val = prep.transform(val_df)

    assert X_train.shape == (70, 29)
    assert X_val.shape == (30, 29)
    assert np.all(np.isfinite(X_train))
    assert np.all(np.isfinite(X_val))


def test_causal_rolling_density(sample_raw_dataframe):
    """Verify rolling 1-hour density is non-negative and finite."""
    df_aug = engineer_causal_behavioral_dataframe(sample_raw_dataframe)

    assert "Cat_Tx_Density_1h" in df_aug.columns
    assert "City_Tx_Density_1h" in df_aug.columns
    assert np.all(df_aug["Cat_Tx_Density_1h"] >= 0)
    assert np.all(df_aug["City_Tx_Density_1h"] >= 0)
    assert not df_aug["Cat_Tx_Density_1h"].isnull().any()
    assert not df_aug["City_Tx_Density_1h"].isnull().any()
