"""
Tests for IEEE-CIS Preprocessing, Feature Engineering, and Leakage Prevention.
Verifies:
1. Feature schema and target separation.
2. Missing-value handling without fraud assumption.
3. Categorical encoding and TabNet cardinalities.
4. Train/validation isolation (no refitting on validation data).
5. Temporal feature leakage prevention (strictly causal, forward-only).
6. Repeatable transformations (idempotency).
7. Stratified 5-Fold and Chronological holdout evaluation strategies.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from ai.src.preprocessing_ieee import (
    IEEEFraudPreprocessor,
    IEEE_FEATURE_COLS,
    IEEE_CATEGORICAL_COLS,
    IEEE_RAW_NUMERICAL_COLS,
    IEEE_ENGINEERED_NUMERICAL_COLS,
    IEEE_NUMERICAL_COLS,
    TARGET_COL,
    ID_COL,
    TIME_COL,
    get_ieee_stratified_cv_splits,
    get_ieee_chronological_split,
    generate_ieee_folds,
)


@pytest.fixture
def synthetic_ieee_df():
    """Create a minimal synthetic merged IEEE-CIS dataset with candidate columns."""
    n = 200
    np.random.seed(42)

    data = {
        "TransactionID": np.arange(1000, 1000 + n),
        "isFraud": np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
        "TransactionDT": np.sort(np.random.randint(86400, 86400 * 30, size=n)),
        "TransactionAmt": np.random.exponential(scale=100.0, size=n) + 5.0,
        "ProductCD": np.random.choice(["W", "C", "R", "H", "S"], size=n),
        "card1": np.random.choice([1000, 2000, 3000, 4000], size=n),
        "card4": np.random.choice(["visa", "mastercard", "discover"], size=n),
        "card6": np.random.choice(["debit", "credit"], size=n),
        "P_emaildomain": np.random.choice(["gmail.com", "yahoo.com", "anonymous.com", None], size=n),
        "R_emaildomain": np.random.choice(["gmail.com", "hotmail.com", None], size=n),
        "DeviceType": np.random.choice(["mobile", "desktop", None], size=n),
        "DeviceInfo": np.random.choice(["Windows", "iOS Device", "Samsung", None], size=n),
        "addr1": np.random.choice([100.0, 200.0, 300.0, None], size=n),
        "addr2": np.random.choice([87.0, None], size=n),
        "M1": np.random.choice(["T", "F", None], size=n),
        "M4": np.random.choice(["M0", "M1", "M2", None], size=n),
        "M5": np.random.choice(["T", "F", None], size=n),
        "M6": np.random.choice(["T", "F", None], size=n),
    }

    # C-features (0% nulls in reality)
    for c in ["C1", "C2", "C5", "C6", "C11", "C13", "C14"]:
        data[c] = np.random.poisson(lam=3.0, size=n).astype(float)

    # D-features (has nulls)
    for d in ["D1", "D2", "D15"]:
        vals = np.random.exponential(scale=50.0, size=n)
        vals[np.random.choice([True, False], size=n, p=[0.2, 0.8])] = np.nan
        data[d] = vals

    # V-features (has nulls)
    v_cols = [
        "V12", "V13", "V35", "V36", "V75", "V76", "V90", "V91",
        "V126", "V127", "V128", "V257", "V258", "V282", "V283"
    ]
    for v in v_cols:
        vals = np.random.normal(loc=1.0, scale=2.0, size=n)
        vals[np.random.choice([True, False], size=n, p=[0.25, 0.75])] = np.nan
        data[v] = vals

    return pd.DataFrame(data)


def test_feature_schema_and_column_counts():
    """Verify feature counts, taxonomy, and exclusion of identifiers and target."""
    assert len(IEEE_CATEGORICAL_COLS) == 11
    assert len(IEEE_RAW_NUMERICAL_COLS) == 29
    assert len(IEEE_ENGINEERED_NUMERICAL_COLS) == 6
    assert len(IEEE_NUMERICAL_COLS) == 35
    assert len(IEEE_FEATURE_COLS) == 46

    # Verify target and transaction ID are strictly excluded
    assert TARGET_COL not in IEEE_FEATURE_COLS
    assert ID_COL not in IEEE_FEATURE_COLS
    assert TIME_COL not in IEEE_FEATURE_COLS


def test_target_separation_enforced(synthetic_ieee_df):
    """Verify preprocessor refuses to fit or transform if target column is present in feature input."""
    prep = IEEEFraudPreprocessor()

    # Attempting to fit with isFraud in X should raise ValueError
    with pytest.raises(ValueError, match="Target column 'isFraud' detected"):
        prep.fit(synthetic_ieee_df)

    # Fit properly on X without target
    X_train = synthetic_ieee_df.drop(columns=[TARGET_COL])
    prep.fit(X_train)

    # Attempting to transform with isFraud in X should raise ValueError
    with pytest.raises(ValueError, match="Target column 'isFraud' detected"):
        prep.transform(synthetic_ieee_df)


def test_missing_value_handling(synthetic_ieee_df):
    """Verify all missing values are imputed cleanly without introducing NaNs."""
    X = synthetic_ieee_df.drop(columns=[TARGET_COL])

    # Assert raw data has missing values in both categoricals and numericals
    assert X[IEEE_CATEGORICAL_COLS].isna().sum().sum() > 0
    assert X[IEEE_RAW_NUMERICAL_COLS].isna().sum().sum() > 0

    prep = IEEEFraudPreprocessor(as_df=True)
    X_trans = prep.fit_transform(X)

    # Output DataFrame must have 0 missing values
    assert X_trans.isna().sum().sum() == 0
    assert X_trans.shape == (len(X), len(IEEE_FEATURE_COLS))


def test_categorical_encoding_and_tabnet_cardinalities(synthetic_ieee_df):
    """Verify TabNet cat_dims, cat_idxs, and unseen category handling."""
    X = synthetic_ieee_df.drop(columns=[TARGET_COL])
    prep = IEEEFraudPreprocessor(as_df=False)
    X_arr = prep.fit_transform(X)

    assert isinstance(X_arr, np.ndarray)
    assert X_arr.shape == (len(X), 46)

    # Check cat_dims and cat_idxs
    assert len(prep.cat_dims) == 11
    assert prep.cat_idxs == list(range(11))
    assert all(d >= 2 for d in prep.cat_dims)

    # Check that all encoded categorical values in output matrix are valid integers
    for i, dim in enumerate(prep.cat_dims):
        col_vals = X_arr[:, i]
        assert np.all(col_vals >= 0), f"Negative category found in col {i}"
        assert np.all(col_vals < dim), f"Category index >= dim {dim} found in col {i}"
        assert np.all(col_vals == col_vals.astype(int)), f"Non-integer category in col {i}"

    # Test unseen category on transform
    X_new = X.iloc[:5].copy()
    X_new.loc[0, "ProductCD"] = "UNKNOWN_PRODUCT_999"
    X_new.loc[0, "DeviceType"] = "NEW_VR_HEADSET"

    X_new_arr = prep.transform(X_new)
    # The unseen category should map to dim - 1 (dedicated unknown bucket)
    prod_dim = prep.cat_dims[0]
    dev_dim = prep.cat_dims[5]
    assert X_new_arr[0, 0] == prod_dim - 1
    assert X_new_arr[0, 5] == dev_dim - 1


def test_train_validation_isolation(synthetic_ieee_df):
    """Verify scalers, encoders, and medians are fitted solely on training data and unchanged by validation transform."""
    X = synthetic_ieee_df.drop(columns=[TARGET_COL])
    X_train = X.iloc[:120].copy()
    X_val = X.iloc[120:].copy()

    prep = IEEEFraudPreprocessor(scale_numerical=True)
    prep.fit(X_train)

    # Record fitted parameters
    train_medians = prep.medians_.copy()
    train_cat_dims = list(prep.cat_dims)
    train_scaler_center = prep.scaler_.center_.copy() if prep.scaler_ else None
    train_scaler_scale = prep.scaler_.scale_.copy() if prep.scaler_ else None
    train_card_history_len = len(prep.card1_history_)

    # Transform validation data
    _ = prep.transform(X_val)

    # Assert fitted state on preprocessor did NOT change
    pd.testing.assert_series_equal(prep.medians_, train_medians)
    assert prep.cat_dims == train_cat_dims
    if prep.scaler_:
        np.testing.assert_array_equal(prep.scaler_.center_, train_scaler_center)
        np.testing.assert_array_equal(prep.scaler_.scale_, train_scaler_scale)
    assert len(prep.card1_history_) == train_card_history_len


def test_temporal_feature_leakage_prevention():
    """Verify that adding a future transaction does not alter features of earlier transactions."""
    df_base = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionDT": [1000, 2000],
        "TransactionAmt": [50.0, 100.0],
        "ProductCD": ["W", "W"],
        "card1": [10, 10],
        "card4": ["visa", "visa"],
        "card6": ["debit", "debit"],
        "P_emaildomain": ["gmail.com", "gmail.com"],
        "R_emaildomain": ["None", "None"],
        "DeviceType": ["desktop", "desktop"],
        "DeviceInfo": ["Windows", "Windows"],
        "addr1": [100.0, 100.0],
        "addr2": [87.0, 87.0],
        "M1": ["T", "T"], "M4": ["M0", "M0"], "M5": ["T", "T"], "M6": ["T", "T"],
        "C1": [1.0, 2.0], "C2": [1.0, 2.0], "C5": [1.0, 2.0], "C6": [1.0, 2.0],
        "C11": [1.0, 2.0], "C13": [1.0, 2.0], "C14": [1.0, 2.0],
        "D1": [10.0, 20.0], "D2": [10.0, 20.0], "D15": [10.0, 20.0],
        "V12": [1.0, 1.0], "V13": [1.0, 1.0], "V35": [1.0, 1.0], "V36": [1.0, 1.0],
        "V75": [1.0, 1.0], "V76": [1.0, 1.0], "V90": [1.0, 1.0], "V91": [1.0, 1.0],
        "V126": [0.0, 0.0], "V127": [0.0, 0.0], "V128": [0.0, 0.0],
        "V257": [1.0, 1.0], "V258": [1.0, 1.0], "V282": [1.0, 1.0], "V283": [1.0, 1.0],
    })

    # Future scenario: add transaction 3 at TransactionDT = 5000 with huge amount ($1,000,000)
    future_row = df_base.iloc[-1:].copy()
    future_row["TransactionID"] = 3
    future_row["TransactionDT"] = 5000
    future_row["TransactionAmt"] = 1_000_000.0
    df_extended = pd.concat([df_base, future_row], ignore_index=True)

    prep1 = IEEEFraudPreprocessor(scale_numerical=False, as_df=True)
    prep2 = IEEEFraudPreprocessor(scale_numerical=False, as_df=True)

    out_base = prep1.fit_transform(df_base)
    out_extended = prep2.fit_transform(df_extended)

    # Features for transaction 1 and transaction 2 must be IDENTICAL regardless of future transaction 3
    for col in IEEE_ENGINEERED_NUMERICAL_COLS:
        np.testing.assert_allclose(
            out_base[col].values,
            out_extended.iloc[:2][col].values,
            err_msg=f"Future transaction leaked into prior feature: {col}",
        )


def test_repeatable_transformations(synthetic_ieee_df):
    """Verify that transform is idempotent and yields identical outputs on identical inputs."""
    X = synthetic_ieee_df.drop(columns=[TARGET_COL])
    X_train = X.iloc[:100]
    X_val = X.iloc[100:]

    prep = IEEEFraudPreprocessor(as_df=False)
    prep.fit(X_train)

    out1 = prep.transform(X_val)
    out2 = prep.transform(X_val)

    np.testing.assert_array_equal(out1, out2)


def test_evaluation_splitters(synthetic_ieee_df):
    """Verify Stratified 5-Fold and Chronological holdout splitting semantics."""
    # 1. Stratified 5-Fold
    splits = get_ieee_stratified_cv_splits(synthetic_ieee_df, n_splits=5, shuffle=True, random_state=42)
    assert len(splits) == 5

    seen_val_indices = set()
    for train_idx, val_idx in splits:
        # No overlap between train and val
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        assert len(train_idx) + len(val_idx) == len(synthetic_ieee_df)
        seen_val_indices.update(val_idx)

    # Every row appears in validation exactly once
    assert seen_val_indices == set(range(len(synthetic_ieee_df)))

    # 2. Chronological Holdout Split
    train_idx, val_idx = get_ieee_chronological_split(synthetic_ieee_df, train_ratio=0.80)
    assert len(train_idx) == 160
    assert len(val_idx) == 40
    assert len(set(train_idx).intersection(set(val_idx))) == 0

    # Ensure train timestamps strictly precede or equal val timestamps
    max_train_dt = synthetic_ieee_df.loc[train_idx, TIME_COL].max()
    min_val_dt = synthetic_ieee_df.loc[val_idx, TIME_COL].min()
    assert max_train_dt <= min_val_dt, f"Future leak in chronological split: max train {max_train_dt} > min val {min_val_dt}"


def test_generate_ieee_folds_generator(synthetic_ieee_df):
    """Verify end-to-end fold generator yields clean, non-null preprocessed partitions."""
    fold_count = 0
    for fold_idx, X_tr, y_tr, X_va, y_va, prep in generate_ieee_folds(
        synthetic_ieee_df, strategy="stratified", n_splits=5
    ):
        fold_count += 1
        assert X_tr.shape[1] == 46
        assert X_va.shape[1] == 46
        assert len(y_tr) == X_tr.shape[0]
        assert len(y_va) == X_va.shape[0]
        assert not np.isnan(X_tr).any()
        assert not np.isnan(X_va).any()
        assert prep.is_fitted_ is True

    assert fold_count == 5


def test_live_data_preprocessing_pipeline():
    """Verify preprocessing on actual raw IEEE-CIS dataset sample from disk if available."""
    from ai.src.data_loader_ieee import check_dataset_availability, load_ieee_cis_raw

    status = check_dataset_availability()
    if not status["is_ready_for_loading"]:
        pytest.skip("Raw IEEE-CIS files not available, skipping live preprocessing test.")

    df = load_ieee_cis_raw(nrows=1000, load_identity=True, validate=True)
    assert len(df) == 1000

    X_raw = df.drop(columns=[TARGET_COL])
    prep = IEEEFraudPreprocessor(as_df=False, scale_numerical=True)
    X_mat = prep.fit_transform(X_raw)

    assert X_mat.shape == (1000, 46)
    assert not np.isnan(X_mat).any()
    assert len(prep.cat_dims) == 11
    assert prep.cat_idxs == list(range(11))

    # Test DataFrame output
    prep_df = IEEEFraudPreprocessor(as_df=True, scale_numerical=True)
    df_out = prep_df.fit_transform(X_raw)
    assert df_out.shape == (1000, 46)
    assert df_out.isna().sum().sum() == 0

    # Test chronological holdout split on live data
    train_idx, val_idx = get_ieee_chronological_split(df, train_ratio=0.80)
    assert len(train_idx) == 800
    assert len(val_idx) == 200
    assert df.loc[train_idx, TIME_COL].max() <= df.loc[val_idx, TIME_COL].min()

