"""
Unit and integration tests for Stage 1: Backend IEEE-CIS artifact loading,
preprocessing application, and multi-model inference.
"""

import pytest
import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.ml.artifact_loader import get_ml_registry, MLRegistry
from ai.src.preprocessing_ieee import (
    IEEE_FEATURE_COLS,
    IEEE_CATEGORICAL_COLS,
    IEEE_RAW_NUMERICAL_COLS,
)


@pytest.fixture(scope="module")
def registry():
    """Load and return singleton MLRegistry."""
    reg = get_ml_registry()
    assert reg.is_ieee_loaded, "IEEE-CIS artifacts must be loaded."
    return reg


@pytest.fixture
def sample_ieee_df():
    """Generate minimal valid single-row IEEE-CIS raw DataFrame."""
    row = {
        "TransactionDT": 86400,
        "TransactionAmt": 125.50,
        "ProductCD": "W",
        "card1": 13500,
        "card4": "visa",
        "card6": "debit",
        "addr1": 315.0,
        "addr2": 87.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "DeviceType": "mobile",
        "DeviceInfo": "iOS Device",
        "M1": "T",
        "M4": "M0",
        "M5": "F",
        "M6": "T",
    }
    # Velocity counters
    for col in ["C1", "C2", "C5", "C6", "C11", "C13", "C14"]:
        row[col] = 1.0
    # Timedeltas
    for col in ["D1", "D2", "D15"]:
        row[col] = 14.0
    # V-features
    for col in [
        "V12", "V13", "V35", "V36", "V75", "V76", "V90", "V91",
        "V126", "V127", "V128", "V257", "V258", "V282", "V283",
    ]:
        row[col] = 0.0

    return pd.DataFrame([row])


def test_ieee_artifacts_loaded(registry):
    """Verify that all IEEE-CIS artifacts are loaded and ready in MLRegistry."""
    assert registry.is_ieee_loaded is True
    assert registry.ieee_preprocessor is not None
    assert registry.ieee_tabnet is not None
    assert registry.ieee_xgboost is not None
    assert registry.ieee_random_forest is not None
    assert registry.ieee_logistic_regression is not None
    assert isinstance(registry.ieee_thresholds, dict)


def test_legacy_artifacts_unbroken(registry):
    """Verify that legacy synthetic artifacts are still preserved and loaded."""
    assert registry.is_loaded is True
    assert registry.preprocessor is not None
    assert registry.model is not None


def test_ieee_preprocessor_structure_and_dimensions(registry):
    """Verify preprocessor attributes, cardinalities, and feature ordering."""
    prep = registry.ieee_preprocessor
    assert prep.is_fitted_ is True

    # 11 Categoricals first, 35 Numericals second = 46 features
    assert len(prep.categorical_cols) == 11
    assert len(prep.numerical_cols) == 35
    assert len(prep.feature_cols) == 46
    assert prep.feature_cols == IEEE_FEATURE_COLS

    # Categorical indices and cardinalities
    assert prep.cat_idxs == list(range(11))
    assert len(prep.cat_dims) == 11
    for dim in prep.cat_dims:
        assert dim > 1


def test_preprocess_ieee_applied_exactly_once(registry, sample_ieee_df):
    """Verify that preprocess_ieee transforms DataFrame into exact (N, 46) float32 matrix."""
    X_proc = registry.preprocess_ieee(sample_ieee_df)

    assert isinstance(X_proc, np.ndarray)
    assert X_proc.dtype == np.float32
    assert X_proc.shape == (1, 46)
    assert not np.isnan(X_proc).any(), "Preprocessed matrix must have no NaN values."


def test_predict_ieee_proba_all_models(registry, sample_ieee_df):
    """Verify multi-model inference on preprocessed IEEE-CIS features."""
    X_proc = registry.preprocess_ieee(sample_ieee_df)

    # 1. TabNet
    p_tabnet = registry.predict_ieee_proba(X_proc, model_name="tabnet")
    assert p_tabnet.shape == (1, 2)
    assert 0.0 <= p_tabnet[0, 1] <= 1.0
    assert np.isclose(np.sum(p_tabnet[0]), 1.0, atol=1e-4)

    # 2. XGBoost
    p_xgb = registry.predict_ieee_proba(X_proc, model_name="xgboost")
    assert p_xgb.shape == (1, 2)
    assert 0.0 <= p_xgb[0, 1] <= 1.0
    assert np.isclose(np.sum(p_xgb[0]), 1.0, atol=1e-4)

    # 3. Random Forest
    p_rf = registry.predict_ieee_proba(X_proc, model_name="random_forest")
    assert p_rf.shape == (1, 2)
    assert 0.0 <= p_rf[0, 1] <= 1.0

    # 4. Logistic Regression
    p_lr = registry.predict_ieee_proba(X_proc, model_name="logistic_regression")
    assert p_lr.shape == (1, 2)
    assert 0.0 <= p_lr[0, 1] <= 1.0


def test_get_ieee_thresholds(registry):
    """Verify calibrated decision thresholds retrieved from Phase 9 evaluation."""
    th_tabnet = registry.get_ieee_threshold("tabnet")
    th_xgb = registry.get_ieee_threshold("xgboost")
    th_rf = registry.get_ieee_threshold("random_forest")
    th_lr = registry.get_ieee_threshold("logistic_regression")

    assert th_tabnet == 0.89
    assert th_xgb == 0.79
    assert th_rf == 0.57
    assert th_lr == 0.86


def test_preprocess_ieee_validation_errors(registry):
    """Verify error handling on invalid inputs."""
    with pytest.raises(TypeError):
        registry.preprocess_ieee("invalid_string_not_df")

    with pytest.raises(ValueError):
        registry.predict_ieee_proba(np.zeros((1, 46)), model_name="unsupported_model_123")


def test_missing_artifact_raises_clear_error_and_no_silent_substitution():
    """Verify that an unloaded model raises RuntimeError rather than silently substituting another model."""
    isolated_reg = MLRegistry()
    isolated_reg.is_ieee_loaded = True
    isolated_reg.ieee_tabnet = None
    isolated_reg.ieee_xgboost = None
    isolated_reg.ieee_random_forest = None
    isolated_reg.ieee_logistic_regression = None

    dummy_input = np.zeros((1, 46), dtype=np.float32)

    with pytest.raises(RuntimeError, match="TabNet model is not loaded"):
        isolated_reg.predict_ieee_proba(dummy_input, model_name="tabnet")

    with pytest.raises(RuntimeError, match="XGBoost model is not loaded"):
        isolated_reg.predict_ieee_proba(dummy_input, model_name="xgboost")

    with pytest.raises(RuntimeError, match="Random Forest model is not loaded"):
        isolated_reg.predict_ieee_proba(dummy_input, model_name="random_forest")

    with pytest.raises(RuntimeError, match="Logistic Regression model is not loaded"):
        isolated_reg.predict_ieee_proba(dummy_input, model_name="logistic_regression")
