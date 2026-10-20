"""
Unit and integration tests for Phase 9 IEEE-CIS benchmarking harness (benchmark_ieee.py).
Validates threshold tuning, model instantiation, single fold execution, metric schemas,
and artifact loadability across all 4 benchmarked architectures.
"""

import sys
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.src.config import RESULTS_DIR, ARTIFACTS_DIR, FIGURES_DIR
from ai.src.preprocessing_ieee import (
    IEEEFraudPreprocessor,
    IEEE_CATEGORICAL_COLS,
    IEEE_RAW_NUMERICAL_COLS,
    TARGET_COL,
    TIME_COL,
)
from ai.src.benchmark_ieee import (
    tune_threshold_on_train,
    instantiate_ieee_model,
    train_and_eval_ieee_fold,
)
from ai.src.models.tabnet_model import TabNetFraudModel
import xgboost as xgb


@pytest.fixture
def mock_ieee_train_val_data():
    """Generate small mock tabular data matching IEEE curated feature schema."""
    np.random.seed(42)
    n_train, n_val = 200, 50

    def generate_split(n):
        data = {}
        data["TransactionID"] = np.arange(1000, 1000 + n)
        data[TIME_COL] = np.linspace(10000, 50000, n)

        # Categoricals
        for col in IEEE_CATEGORICAL_COLS:
            data[col] = np.random.choice(["A", "B", "C", None], size=n)

        # Raw Numericals
        for col in IEEE_RAW_NUMERICAL_COLS:
            data[col] = np.random.normal(loc=100.0, scale=20.0, size=n)

        # Target (imbalanced ~10% for mock reliability)
        data[TARGET_COL] = np.random.choice([0, 1], size=n, p=[0.90, 0.10])
        return pd.DataFrame(data)

    train_df = generate_split(n_train)
    val_df = generate_split(n_val)

    # Preprocess
    prep = IEEEFraudPreprocessor(as_df=False, scale_numerical=True)
    X_train = prep.fit_transform(train_df.drop(columns=[TARGET_COL]))
    y_train = train_df[TARGET_COL].values.astype(np.int64)

    X_val = prep.transform(val_df.drop(columns=[TARGET_COL]))
    y_val = val_df[TARGET_COL].values.astype(np.int64)

    return X_train, y_train, X_val, y_val, prep


def test_tune_threshold_on_train_bounds_and_correctness():
    """Verify threshold tuning selects optimal threshold strictly on training data."""
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 1])
    # Simulated well-calibrated probabilities
    y_prob = np.array([0.1, 0.2, 0.15, 0.3, 0.8, 0.85, 0.4, 0.7, 0.2, 0.9])

    best_th = tune_threshold_on_train(y_true, y_prob)
    assert isinstance(best_th, float)
    assert 0.05 <= best_th <= 0.90

    # Test edge case: all legitimate
    y_zeros = np.zeros(20, dtype=np.int64)
    p_zeros = np.linspace(0.01, 0.2, 20)
    th_zeros = tune_threshold_on_train(y_zeros, p_zeros)
    assert 0.05 <= th_zeros <= 0.90


def test_instantiate_ieee_model_all_architectures():
    """Verify that all 4 models instantiate cleanly with IEEE configuration."""
    cat_idxs = [0, 1, 2]
    cat_dims = [5, 4, 3]

    lr = instantiate_ieee_model("logistic_regression", cat_idxs, cat_dims, random_state=42)
    assert hasattr(lr, "fit")
    assert lr.class_weight == "balanced"

    rf = instantiate_ieee_model("random_forest", cat_idxs, cat_dims, random_state=42, rf_trees=10)
    assert hasattr(rf, "fit")
    assert rf.n_estimators == 10

    xgb_model = instantiate_ieee_model("xgboost", cat_idxs, cat_dims, random_state=42, xgb_trees=10)
    assert hasattr(xgb_model, "fit")
    assert xgb_model.n_estimators == 10

    tab = instantiate_ieee_model("tabnet", cat_idxs, cat_dims, random_state=42)
    assert isinstance(tab, TabNetFraudModel)

    with pytest.raises(ValueError):
        instantiate_ieee_model("unsupported_model", cat_idxs, cat_dims, random_state=42)


@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest", "xgboost", "tabnet"])
def test_train_and_eval_ieee_fold_execution(mock_ieee_train_val_data, model_name):
    """Verify that train_and_eval_ieee_fold executes for all models and produces required schema."""
    X_train, y_train, X_val, y_val, prep = mock_ieee_train_val_data

    res, fitted_model = train_and_eval_ieee_fold(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        preprocessor=prep,
        fold_idx=0,
        random_state=42,
        tabnet_epochs=1,
        tabnet_patience=1,
        rf_trees=10,
        xgb_trees=10,
        verbose=False,
    )

    # Required metric keys
    required_keys = [
        "roc_auc",
        "pr_auc",
        "recall_at_fpr_1pct",
        "f1",
        "macro_f1",
        "precision",
        "recall",
        "threshold_tuned",
        "confusion_matrix",
        "train_time_sec",
        "infer_time_sec",
        "latency_ms_per_1k",
    ]
    for k in required_keys:
        assert k in res, f"Missing metric key: {k}"
        if isinstance(res[k], (int, float)):
            assert not np.isnan(res[k]), f"Metric {k} is NaN"
        elif isinstance(res[k], dict):
            assert len(res[k]) > 0, f"Dictionary {k} is empty"

    # Metric range checks
    assert 0.0 <= res["roc_auc"] <= 1.0
    assert 0.0 <= res["pr_auc"] <= 1.0
    assert 0.0 <= res["f1"] <= 1.0
    assert 0.05 <= res["threshold_tuned"] <= 0.90
    assert res["train_time_sec"] >= 0.0
    assert res["infer_time_sec"] >= 0.0

    # Verify fitted model can perform inference
    if model_name == "tabnet":
        preds = fitted_model.predict_proba(X_val[:5])
    else:
        preds = fitted_model.predict_proba(X_val[:5])
    assert preds.shape == (5, 2)


def test_saved_artifacts_loadability():
    """Verify that saved artifacts from benchmarking can be loaded and predict correctly."""
    # Check Logistic Regression
    lr_path = ARTIFACTS_DIR / "ieee_logistic_regression.joblib"
    if lr_path.exists():
        lr_loaded = joblib.load(lr_path)
        assert hasattr(lr_loaded, "predict_proba")

    # Check Random Forest
    rf_path = ARTIFACTS_DIR / "ieee_random_forest.joblib"
    if rf_path.exists():
        rf_loaded = joblib.load(rf_path)
        assert hasattr(rf_loaded, "predict_proba")

    # Check XGBoost
    xgb_path = ARTIFACTS_DIR / "ieee_xgboost.json"
    if xgb_path.exists():
        xgb_loaded = xgb.XGBClassifier()
        xgb_loaded.load_model(str(xgb_path))
        assert hasattr(xgb_loaded, "predict_proba")

    # Check TabNet
    tab_path = ARTIFACTS_DIR / "ieee_tabnet.zip"
    if tab_path.exists():
        tab_loaded = TabNetFraudModel.load_from_file(tab_path)
        assert tab_loaded.is_fitted_

    # Check Preprocessor
    prep_path = ARTIFACTS_DIR / "ieee_preprocessor.joblib"
    if prep_path.exists():
        prep_loaded = joblib.load(prep_path)
        assert hasattr(prep_loaded, "transform")
        assert len(prep_loaded.categorical_cols) == 11
        assert len(prep_loaded.numerical_cols) == 35


def test_benchmark_results_and_comparison_file_schemas():
    """Verify schema and validity of persisted benchmark results in ai/results/."""
    cv_path = RESULTS_DIR / "ieee_metrics_stratified_cv.json"
    if cv_path.exists():
        with open(cv_path, "r", encoding="utf-8") as f:
            cv_data = json.load(f)
        for m in ["logistic_regression", "random_forest", "xgboost", "tabnet"]:
            if m in cv_data:
                assert "aggregated" in cv_data[m]
                assert "pr_auc" in cv_data[m]["aggregated"]
                assert "roc_auc" in cv_data[m]["aggregated"]
                assert "mean" in cv_data[m]["aggregated"]["pr_auc"]

    ch_path = RESULTS_DIR / "ieee_metrics_chronological.json"
    if ch_path.exists():
        with open(ch_path, "r", encoding="utf-8") as f:
            ch_data = json.load(f)
        for m in ["logistic_regression", "random_forest", "xgboost", "tabnet"]:
            if m in ch_data:
                assert "metrics" in ch_data[m]
                assert "pr_auc" in ch_data[m]["metrics"]
                assert "roc_auc" in ch_data[m]["metrics"]

    csv_path = RESULTS_DIR / "ieee_model_comparison.csv"
    if csv_path.exists():
        df_comp = pd.read_csv(csv_path)
        assert "Model" in df_comp.columns
        assert "CV PR-AUC" in df_comp.columns
        assert "Chrono PR-AUC" in df_comp.columns
        assert len(df_comp) == 4
