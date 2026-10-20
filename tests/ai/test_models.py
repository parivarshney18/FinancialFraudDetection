"""
Unit tests for Phase 3: Model Architecture, Training, and Evaluation.
Verifies model construction, output shapes, metric computations, and TabNet explainability.
"""

import numpy as np
import pandas as pd
import pytest

from ai.src.evaluation import (
    compute_classification_metrics,
    compute_recall_at_fpr,
    aggregate_fold_metrics,
)
from ai.src.models.baseline_models import build_logistic_regression, build_random_forest
from ai.src.models.xgb_model import build_xgboost_model
from ai.src.models.tabnet_model import (
    build_tabnet_model,
    compute_recommended_embedding_dims,
    TabNetFraudModel,
)


@pytest.fixture
def synthetic_classification_data():
    """Generate small synthetic tabular classification dataset for testing."""
    np.random.seed(42)
    N = 100
    # 3 categorical features (integers 0..K-1) and 5 continuous numerical features
    cat_feats = np.column_stack([
        np.random.randint(0, 3, size=N),
        np.random.randint(0, 5, size=N),
        np.random.randint(0, 4, size=N),
    ])
    num_feats = np.random.randn(N, 5).astype(np.float32)
    X = np.hstack([cat_feats, num_feats]).astype(np.float32)

    # Imbalanced labels (5% fraud)
    y = np.zeros(N, dtype=np.int64)
    y[:6] = 1  # 6 positive cases
    np.random.shuffle(y)

    cat_idxs = [0, 1, 2]
    cat_dims = [4, 6, 5]  # including unknown slot (+1)
    return X, y, cat_idxs, cat_dims


def test_metric_computations_exactness():
    """Verify precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrix calculation."""
    y_true = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9], dtype=np.float64)

    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["roc_auc"] == 1.0, f"Expected perfect ROC-AUC, got {metrics['roc_auc']}"
    assert metrics["pr_auc"] == 1.0, f"Expected perfect PR-AUC, got {metrics['pr_auc']}"
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["confusion_matrix"]["tn"] == 4
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["fn"] == 0


def test_recall_at_fpr_boundary():
    """Verify Recall @ FPR <= 1% computation under known curves."""
    y_true = np.array([0] * 100 + [1] * 10)
    # Give all positives top score
    y_score = np.array([0.1] * 100 + [0.9] * 10)

    recall_at_1pct = compute_recall_at_fpr(y_true, y_score, max_fpr=0.01)
    assert recall_at_1pct == 1.0

    # Test with random noise
    np.random.seed(42)
    y_rand = np.random.rand(len(y_true))
    recall_rand = compute_recall_at_fpr(y_true, y_rand, max_fpr=0.01)
    assert 0.0 <= recall_rand <= 1.0


def test_aggregate_fold_metrics():
    """Verify aggregation computes correct sample mean and standard deviation."""
    fold_metrics = [
        {"pr_auc": 0.80, "roc_auc": 0.90, "f1": 0.70, "macro_f1": 0.85, "precision": 0.75, "recall": 0.65, "recall_at_fpr_1pct": 0.50},
        {"pr_auc": 0.90, "roc_auc": 0.92, "f1": 0.74, "macro_f1": 0.87, "precision": 0.77, "recall": 0.71, "recall_at_fpr_1pct": 0.54},
    ]

    agg = aggregate_fold_metrics(fold_metrics)
    assert pytest.approx(agg["pr_auc"]["mean"], abs=1e-5) == 0.85
    assert pytest.approx(agg["roc_auc"]["mean"], abs=1e-5) == 0.91
    assert "std" in agg["pr_auc"]
    assert agg["pr_auc"]["std"] > 0


def test_logistic_regression_pipeline(synthetic_classification_data):
    """Verify LogisticRegression baseline instantiation, fitting, and prediction shapes."""
    X, y, _, _ = synthetic_classification_data
    model = build_logistic_regression(max_iter=100)
    model.fit(X, y)

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_random_forest_pipeline(synthetic_classification_data):
    """Verify RandomForest baseline instantiation, fitting, and prediction shapes."""
    X, y, _, _ = synthetic_classification_data
    model = build_random_forest(n_estimators=10, max_depth=4)
    model.fit(X, y)

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_xgboost_pipeline(synthetic_classification_data):
    """Verify XGBoost classifier instantiation, fitting, and prediction shapes."""
    X, y, _, _ = synthetic_classification_data
    model = build_xgboost_model(n_estimators=10, max_depth=3)
    model.fit(X, y)

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_tabnet_wrapper_pipeline_and_explainability(synthetic_classification_data):
    """Verify TabNetFraudModel training, probability predictions, and attention mask extraction."""
    X, y, cat_idxs, cat_dims = synthetic_classification_data
    emb_dims = compute_recommended_embedding_dims(cat_dims)
    assert len(emb_dims) == len(cat_dims)

    tabnet = TabNetFraudModel(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        n_d=8,
        n_a=8,
        n_steps=3,
        seed=42,
        device_name="cpu",
    )

    # Fit on small sample with 2 epochs
    tabnet.fit(
        X_train=X[:80],
        y_train=y[:80],
        eval_set=[(X[80:], y[80:])],
        max_epochs=2,
        patience=2,
        batch_size=32,
        virtual_batch_size=16,
    )

    # Test predictions
    preds = tabnet.predict(X[80:])
    probs = tabnet.predict_proba(X[80:])
    assert len(preds) == 20
    assert probs.shape == (20, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    # Test explainability
    M_explain, masks = tabnet.explain(X[80:])
    assert M_explain.shape == (20, X.shape[1]), "Explain matrix shape mismatch."
    assert len(masks) == 3, "Expected 3 step masks for 3 decision steps."
    for step_idx, mask in masks.items():
        assert mask.shape == (20, X.shape[1]), f"Mask step {step_idx} shape mismatch."
        assert np.all(mask >= 0.0), "Sparsemax attention mask values must be non-negative."
