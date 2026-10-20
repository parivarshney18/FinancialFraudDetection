"""
Unit and integration tests for IEEE-CIS Explainability Module (ai/src/explain_ieee.py).
Tests TabNet Sparsemax attention extraction across 46 curated features,
sequential step breakdown, compliance narrative generation, and XGBoost limitation reporting.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from backend.app.ml.artifact_loader import get_ml_registry
from ai.src.explain_ieee import (
    IEEEFraudExplainer,
    plot_ieee_global_feature_importance,
    plot_ieee_step_attention_heatmap,
    IEEE_FEATURE_DESCRIPTIONS,
)


@pytest.fixture(scope="module")
def ieee_explainer_setup():
    """Load production IEEE-CIS registry, TabNet model, and preprocessor."""
    registry = get_ml_registry()
    assert registry.is_ieee_loaded, "IEEE-CIS production artifacts must be loaded."
    assert registry.ieee_explainer is not None, "IEEE explainer must be initialized."

    # Generate a realistic sample DataFrame and preprocessed array
    raw_dict = {
        "TransactionDT": 86400,
        "TransactionAmt": 250.0,
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
    for col in ["C1", "C2", "C5", "C6", "C11", "C13", "C14"]:
        raw_dict[col] = 1.0
    for col in ["D1", "D2", "D15"]:
        raw_dict[col] = 14.0
    for col in [
        "V12", "V13", "V35", "V36", "V75", "V76", "V90", "V91",
        "V126", "V127", "V128", "V257", "V258", "V282", "V283",
    ]:
        raw_dict[col] = 0.0

    df_sample = pd.DataFrame([raw_dict])
    X_proc = registry.preprocess_ieee(df_sample)
    assert X_proc.shape == (1, 46)

    return registry.ieee_explainer, X_proc, raw_dict


def test_ieee_explainer_initialization(ieee_explainer_setup):
    """Verify IEEEFraudExplainer correctly loads 46 curated features."""
    explainer, _, _ = ieee_explainer_setup
    assert len(explainer.feature_names) == 46
    assert "Amt_to_Card1_Mean_Ratio" in explainer.feature_names
    assert "Log_TransactionAmt" in explainer.feature_names
    assert "ProductCD" in explainer.feature_names

    # Check that invalid feature count raises ValueError
    with pytest.raises(ValueError, match="Expected 46 feature names"):
        IEEEFraudExplainer(model=explainer.model, feature_names=["feat_1", "feat_2"])


def test_ieee_explain_batch_shapes_and_values(ieee_explainer_setup):
    """Verify explain_batch produces correct shapes, 3 decision steps, and non-negative masks."""
    explainer, X_proc, _ = ieee_explainer_setup

    M_explain, masks = explainer.explain_batch(X_proc)

    assert M_explain.shape == (1, 46)
    assert len(masks) == 3
    for step_idx in [0, 1, 2]:
        assert step_idx in masks
        assert masks[step_idx].shape == (1, 46)
        assert np.all(masks[step_idx] >= 0.0)


def test_ieee_global_feature_importance(ieee_explainer_setup):
    """Verify global feature importance on 46 features with ranks and descriptions."""
    explainer, X_proc, _ = ieee_explainer_setup

    df_imp = explainer.get_global_feature_importance(X_proc)

    assert list(df_imp.columns) == ["feature", "description", "importance", "importance_pct", "rank"]
    assert len(df_imp) == 46
    assert np.isclose(df_imp["importance_pct"].sum(), 100.0, atol=1e-2)
    assert df_imp["importance"].is_monotonic_decreasing
    assert list(df_imp["rank"]) == list(range(1, 47))


def test_ieee_step_feature_importance(ieee_explainer_setup):
    """Verify sequential step attention extraction across 3 decision steps."""
    explainer, X_proc, _ = ieee_explainer_setup

    step_df = explainer.get_step_feature_importance(X_proc)

    assert list(step_df.index) == ["Step 0", "Step 1", "Step 2"]
    assert len(step_df.columns) == 46
    assert np.all(step_df.values >= 0.0)
    assert np.allclose(step_df.sum(axis=1), 1.0, atol=1e-3)


def test_ieee_explain_instance_structure_and_narrative(ieee_explainer_setup):
    """Verify single instance explanation dictionary, top-k contributors, and compliance narrative."""
    explainer, X_proc, raw_dict = ieee_explainer_setup

    result = explainer.explain_instance(
        x=X_proc[0],
        top_k=5,
        raw_record=raw_dict,
        fraud_probability=0.92,
        threshold=0.89,
        model_name="tabnet",
    )

    assert result["fraud_probability"] == 0.92
    assert result["is_fraud_predicted"] == 1
    assert result["risk_tier"] == "CRITICAL"
    assert len(result["top_contributors"]) == 5

    # Check top contributor fields
    top1 = result["top_contributors"][0]
    assert top1["feature"] in explainer.feature_names
    assert 0 <= top1["primary_step"] <= 2
    assert len(top1["step_attentions"]) == 3
    assert top1["contribution_pct"] >= 0.0

    # Step breakdown check
    assert "Step 0" in result["step_attention_breakdown"]
    assert "Step 1" in result["step_attention_breakdown"]
    assert "Step 2" in result["step_attention_breakdown"]

    # Compliance and causality checks in narrative
    narrative = result["narrative_summary"]
    assert "CRITICAL RISK" in narrative
    assert "Compliance Notice" in narrative
    assert "not deterministic or causal proof" in narrative.lower()

    # Explainer metadata check
    meta = result["explainer_metadata"]
    assert meta["dataset"] == "IEEE-CIS Fraud Detection"
    assert meta["n_features_evaluated"] == 46
    assert "Sparsemax" in meta["method"]


def test_ieee_explain_instance_invalid_shape_raises(ieee_explainer_setup):
    """Verify that feeding a vector with incorrect dimensions raises ValueError."""
    explainer, _, _ = ieee_explainer_setup
    bad_x = np.zeros(20, dtype=np.float32)

    with pytest.raises(ValueError, match="Expected input with 46 features"):
        explainer.explain_instance(bad_x)


def test_ieee_xgboost_limitation_reporting(ieee_explainer_setup):
    """Verify that XGBoost explanations explicitly report SHAP library availability without hallucinating."""
    explainer, X_proc, _ = ieee_explainer_setup

    xgb_exp = explainer.explain_xgboost(X_proc[0])

    assert xgb_exp["model_name"] == "xgboost"
    assert "explainer" in xgb_exp
    # In an environment without SHAP, it must report unavailable with clean explanation
    if not xgb_exp["available"]:
        assert "not installed" in xgb_exp["detail"]
        assert len(xgb_exp["top_contributors"]) == 0


def test_ieee_plotting_utilities(ieee_explainer_setup, tmp_path):
    """Verify that plotting utilities create valid image files for global importances and step heatmaps."""
    explainer, X_proc, _ = ieee_explainer_setup

    df_imp = explainer.get_global_feature_importance(X_proc)
    step_df = explainer.get_step_feature_importance(X_proc)

    p1 = tmp_path / "ieee_global_importance.png"
    out1 = plot_ieee_global_feature_importance(df_imp, p1, top_n=10)
    assert out1.exists()
    assert out1.stat().st_size > 1000

    p2 = tmp_path / "ieee_step_heatmap.png"
    out2 = plot_ieee_step_attention_heatmap(step_df, p2, top_n=10)
    assert out2.exists()
    assert out2.stat().st_size > 1000
