"""
Unit tests for TabNet Explainability and Interpretability Module (ai/src/explain.py).
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from ai.src.models.tabnet_model import TabNetFraudModel
from ai.src.explain import (
    TabNetExplainer,
    plot_global_feature_importance,
    plot_step_attention_heatmap,
    plot_case_studies,
)


@pytest.fixture
def synthetic_tabnet_setup():
    """Create a lightweight fitted TabNet model and synthetic feature matrix."""
    np.random.seed(42)
    n_samples = 60
    n_features = 8
    cat_idxs = [0, 1]
    cat_dims = [3, 4]
    feature_names = [f"feat_{i}" for i in range(n_features)]

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    X[:, 0] = np.random.randint(0, 3, n_samples)
    X[:, 1] = np.random.randint(0, 4, n_samples)
    y = np.random.randint(0, 2, n_samples).astype(np.int64)

    model = TabNetFraudModel(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        n_d=8,
        n_a=8,
        n_steps=3,
        seed=42,
        device_name="cpu",
    )
    model.fit(
        X_train=X[:40],
        y_train=y[:40],
        eval_set=[(X[40:], y[40:])],
        max_epochs=2,
        patience=2,
        batch_size=16,
        virtual_batch_size=8,
    )

    explainer = TabNetExplainer(model=model, feature_names=feature_names)
    return explainer, X[40:], y[40:], feature_names


def test_explain_batch_shapes_and_values(synthetic_tabnet_setup):
    """Verify that explain_batch produces correct shapes, step keys, and non-negative attention."""
    explainer, X_val, _, feature_names = synthetic_tabnet_setup
    n_val = len(X_val)
    n_feat = len(feature_names)

    M_explain, masks = explainer.explain_batch(X_val)

    assert M_explain.shape == (n_val, n_feat)
    assert len(masks) == 3
    for step_idx in [0, 1, 2]:
        assert step_idx in masks
        assert masks[step_idx].shape == (n_val, n_feat)
        assert np.all(masks[step_idx] >= 0.0)


def test_global_feature_importance(synthetic_tabnet_setup):
    """Verify global feature importance computation, column structure, and ranking."""
    explainer, X_val, _, feature_names = synthetic_tabnet_setup

    df_imp = explainer.get_global_feature_importance(X_val)

    assert list(df_imp.columns) == ["feature", "importance", "importance_pct", "rank"]
    assert len(df_imp) == len(feature_names)
    assert np.isclose(df_imp["importance_pct"].sum(), 100.0, atol=1e-2)
    assert df_imp["importance"].is_monotonic_decreasing
    assert list(df_imp["rank"]) == list(range(1, len(feature_names) + 1))


def test_step_feature_importance(synthetic_tabnet_setup):
    """Verify sequential step attention extraction across decision steps."""
    explainer, X_val, _, feature_names = synthetic_tabnet_setup

    step_df = explainer.get_step_feature_importance(X_val)

    assert list(step_df.index) == ["Step 0", "Step 1", "Step 2"]
    assert list(step_df.columns) == feature_names
    assert np.all(step_df.values >= 0.0)
    # Check that rows sum to approximately 1.0 when normalized
    assert np.allclose(step_df.sum(axis=1), 1.0, atol=1e-3)


def test_explain_single_instance_structure_and_narrative(synthetic_tabnet_setup):
    """Verify instance-level explanation dictionary, risk tier, and forensic narrative."""
    explainer, X_val, _, _ = synthetic_tabnet_setup
    sample_x = X_val[0]

    explanation = explainer.explain_instance(
        x=sample_x,
        top_k=3,
        raw_record={"feat_0": "TestVal", "feat_1": "POS"},
        fraud_probability=0.88,
        threshold=0.5,
    )

    assert explanation["fraud_probability"] == 0.88
    assert explanation["is_fraud_predicted"] == 1
    assert explanation["risk_tier"] == "CRITICAL"
    assert len(explanation["top_contributors"]) == 3

    top1 = explanation["top_contributors"][0]
    assert "feature" in top1
    assert "contribution_pct" in top1
    assert "primary_step" in top1
    assert 0 <= top1["primary_step"] <= 2

    # Narrative verification
    narrative = explanation["narrative_summary"]
    assert isinstance(narrative, str)
    assert "CRITICAL RISK" in narrative
    assert "Primary Forensic Drivers" in narrative


def test_explain_low_risk_instance_narrative(synthetic_tabnet_setup):
    """Verify narrative output for a legitimate/low-risk transaction."""
    explainer, X_val, _, _ = synthetic_tabnet_setup
    sample_x = X_val[1]

    explanation = explainer.explain_instance(
        x=sample_x,
        top_k=4,
        fraud_probability=0.03,
        threshold=0.5,
    )

    assert explanation["is_fraud_predicted"] == 0
    assert explanation["risk_tier"] == "LOW"
    narrative = explanation["narrative_summary"]
    assert "CLEARANCE" in narrative
    assert "LOW RISK" in narrative


def test_plotting_utilities_generate_files(synthetic_tabnet_setup, tmp_path):
    """Verify that plotting functions write valid image files to disk."""
    explainer, X_val, _, _ = synthetic_tabnet_setup

    df_imp = explainer.get_global_feature_importance(X_val)
    step_df = explainer.get_step_feature_importance(X_val)

    p1 = tmp_path / "global_imp.png"
    out1 = plot_global_feature_importance(df_imp, p1, top_n=5)
    assert out1.exists()
    assert out1.stat().st_size > 1000

    p2 = tmp_path / "step_heatmap.png"
    out2 = plot_step_attention_heatmap(step_df, p2, top_n=6)
    assert out2.exists()
    assert out2.stat().st_size > 1000

    # Test case study plotting
    cases = [
        explainer.explain_instance(X_val[0], top_k=3, fraud_probability=0.92),
        explainer.explain_instance(X_val[1], top_k=3, fraud_probability=0.04),
    ]
    p3 = tmp_path / "case_studies.png"
    out3 = plot_case_studies(cases, p3)
    assert out3.exists()
    assert out3.stat().st_size > 1000
