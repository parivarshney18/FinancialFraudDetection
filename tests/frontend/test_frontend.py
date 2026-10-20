"""
Unit tests for Streamlit frontend components, IEEE-CIS integration, and API client.
Location: tests/frontend/test_frontend.py
"""

from unittest.mock import patch, MagicMock
import pytest
import plotly.graph_objects as go

from frontend.components.gauge import render_risk_gauge
from frontend.components.contributors import render_contributors_chart
from frontend.components.heatmap import render_step_attention_heatmap
from frontend.services.api_client import APIClient, api_client


# ==============================================================================
# COMPONENT TESTS: RADIAL RISK GAUGE
# ==============================================================================

def test_render_risk_gauge_nominal():
    """Verify radial risk gauge construction and threshold marker for nominal tier."""
    fig = render_risk_gauge(fraud_probability=0.15, threshold=0.50)
    assert isinstance(fig, go.Figure)
    data = fig.data[0]
    assert data.value == 15.0
    assert data.gauge.threshold.value == 50.0
    assert data.gauge.bar.color == "#38a169"  # Green for low risk


def test_render_risk_gauge_moderate():
    """Verify moderate tier gauge coloring."""
    fig = render_risk_gauge(fraud_probability=0.45, threshold=0.50)
    assert isinstance(fig, go.Figure)
    data = fig.data[0]
    assert data.value == 45.0
    assert data.gauge.bar.color == "#d69e2e"  # Amber for moderate risk


def test_render_risk_gauge_high():
    """Verify high tier gauge coloring."""
    fig = render_risk_gauge(fraud_probability=0.75, threshold=0.50)
    assert isinstance(fig, go.Figure)
    data = fig.data[0]
    assert data.value == 75.0
    assert data.gauge.bar.color == "#dd6b20"  # Orange for high risk


def test_render_risk_gauge_critical():
    """Verify critical tier gauge coloring."""
    fig = render_risk_gauge(fraud_probability=0.88, threshold=0.50)
    assert isinstance(fig, go.Figure)
    data = fig.data[0]
    assert data.value == 88.0
    assert data.gauge.bar.color == "#e53e3e"  # Red for critical risk


# ==============================================================================
# COMPONENT TESTS: TOP CONTRIBUTORS & ATTENTION HEATMAP
# ==============================================================================

def test_render_contributors_chart_legacy():
    """Verify top contributors bar chart orientation and values with legacy synthetic features."""
    mock_contributors = [
        {"feature": "Amount_to_Balance_Ratio", "contribution_pct": 34.5, "importance": 0.45, "primary_step": 0},
        {"feature": "Tx_Hour", "contribution_pct": 22.1, "importance": 0.28, "primary_step": 1},
        {"feature": "Device_Mismatch_Flag", "contribution_pct": 18.0, "importance": 0.22, "primary_step": 0},
    ]

    fig = render_contributors_chart(mock_contributors)
    assert isinstance(fig, go.Figure)
    bar = fig.data[0]
    assert bar.orientation == "h"
    assert len(bar.x) == 3


def test_render_contributors_chart_ieee():
    """Verify top contributors bar chart with IEEE-CIS curated features."""
    mock_ieee_contributors = [
        {"feature": "Amt_to_Card1_Mean_Ratio", "contribution_pct": 28.4, "importance": 0.35, "primary_step": 0},
        {"feature": "V258", "contribution_pct": 21.2, "importance": 0.26, "primary_step": 1},
        {"feature": "C1", "contribution_pct": 16.5, "importance": 0.20, "primary_step": 2},
        {"feature": "TransactionAmt", "contribution_pct": 14.1, "importance": 0.17, "primary_step": 0},
        {"feature": "card1_tx_count_prior", "contribution_pct": 10.3, "importance": 0.13, "primary_step": 1},
    ]

    fig = render_contributors_chart(mock_ieee_contributors)
    assert isinstance(fig, go.Figure)
    bar = fig.data[0]
    assert bar.orientation == "h"
    assert len(bar.x) == 5
    # Features sorted ascending for horizontal display
    assert "Amt_to_Card1_Mean_Ratio" in bar.y[-1]


def test_render_step_attention_heatmap():
    """Verify sequential decision step attention heatmap structure across 3 TabNet steps."""
    mock_step_attns = {
        "Step 0": {"TransactionAmt": 0.4, "V258": 0.3, "C1": 0.3},
        "Step 1": {"TransactionAmt": 0.1, "V258": 0.7, "C1": 0.2},
        "Step 2": {"TransactionAmt": 0.2, "V258": 0.2, "C1": 0.6},
    }

    fig = render_step_attention_heatmap(mock_step_attns)
    assert isinstance(fig, go.Figure)
    heatmap = fig.data[0]
    assert heatmap.type == "heatmap"
    assert len(heatmap.y) == 3
    assert list(heatmap.y) == ["Step 0", "Step 1", "Step 2"]


# ==============================================================================
# API CLIENT TESTS: INITIALIZATION & LEGACY ENDPOINTS
# ==============================================================================

def test_api_client_initialization():
    """Verify API client default base URL and trailing slash stripping."""
    client = APIClient("http://localhost:8000/api/")
    assert client.base_url == "http://localhost:8000/api"


@patch("frontend.services.api_client.requests.get")
def test_api_client_health_check_success(mock_get):
    """Verify API client health check parsing on success."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "healthy", "model_loaded": True}
    mock_get.return_value = mock_resp

    client = APIClient("http://testserver/api")
    health = client.check_health()
    assert health["status"] == "healthy"
    assert health["model_loaded"] is True


@patch("frontend.services.api_client.requests.get")
def test_api_client_health_check_failure(mock_get):
    """Verify API client health check error handling when server is unreachable."""
    mock_get.side_effect = Exception("Connection refused")

    client = APIClient("http://testserver/api")
    health = client.check_health()
    assert health["status"] == "unreachable"
    assert health["model_loaded"] is False


@patch("frontend.services.api_client.requests.post")
def test_api_client_legacy_score_transaction(mock_post):
    """Verify legacy synthetic transaction scoring endpoint call."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "transaction_id": "TX_TEST_001",
        "fraud_probability": 0.12,
        "is_fraud_predicted": False,
        "risk_tier": "LOW",
    }
    mock_post.return_value = mock_resp

    client = APIClient("http://testserver/api")
    result = client.score_transaction({"Transaction_Amount": 150.0})
    assert result is not None
    assert result["risk_tier"] == "LOW"
    assert result["fraud_probability"] == 0.12


@patch("frontend.services.api_client.requests.get")
def test_api_client_legacy_benchmarks(mock_get):
    """Verify legacy benchmark endpoint retrieval."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"model_name": "TabNet", "cv_roc_auc": 0.89}]
    mock_get.return_value = mock_resp

    client = APIClient("http://testserver/api")
    res = client.get_benchmarks()
    assert len(res) == 1
    assert res[0]["model_name"] == "TabNet"


# ==============================================================================
# API CLIENT TESTS: IEEE-CIS PRODUCTION ENDPOINTS
# ==============================================================================

@patch("frontend.services.api_client.requests.post")
def test_api_client_score_ieee_success(mock_post):
    """Verify IEEE-CIS scoring with different model architecture queries."""
    for model_name in ["tabnet", "xgboost", "random_forest", "logistic_regression"]:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "transaction_id": "W3000001",
            "model_used": model_name,
            "fraud_probability": 0.82,
            "decision_threshold": 0.50,
            "is_fraud_predicted": True,
            "risk_tier": "CRITICAL",
        }
        mock_post.return_value = mock_resp

        client = APIClient("http://testserver/api")
        result = client.score_ieee({"TransactionAmt": 250.0}, model=model_name)

        assert result is not None
        assert result["model_used"] == model_name
        assert result["is_fraud_predicted"] is True
        assert result["risk_tier"] == "CRITICAL"


@patch("frontend.services.api_client.requests.post")
def test_api_client_score_ieee_error_handling(mock_post):
    """Verify IEEE-CIS scoring returns None on HTTP 500 error."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_post.return_value = mock_resp

    client = APIClient("http://testserver/api")
    result = client.score_ieee({"TransactionAmt": 100.0})
    assert result is None


@patch("frontend.services.api_client.requests.post")
def test_api_client_score_and_explain_ieee_tabnet(mock_post):
    """Verify IEEE-CIS score and explain with TabNet Sparsemax attention attribution."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "transaction_id": "W3000002",
        "model_used": "tabnet",
        "fraud_probability": 0.74,
        "decision_threshold": 0.50,
        "is_fraud_predicted": True,
        "risk_tier": "HIGH",
        "explainability_available": True,
        "explanation_method": "tabnet_sparsemax_attention",
        "top_contributors": [
            {"feature": "Amt_to_Card1_Mean_Ratio", "contribution_pct": 32.5, "importance": 0.45, "primary_step": 0},
            {"feature": "V258", "contribution_pct": 24.1, "importance": 0.33, "primary_step": 1},
        ],
        "step_attentions": {
            "Step 0": {"Amt_to_Card1_Mean_Ratio": 0.7, "V258": 0.3},
            "Step 1": {"Amt_to_Card1_Mean_Ratio": 0.2, "V258": 0.8},
            "Step 2": {"Amt_to_Card1_Mean_Ratio": 0.5, "V258": 0.5},
        },
        "forensic_report": "High risk detected. Primary factor: Amt_to_Card1_Mean_Ratio.",
    }
    mock_post.return_value = mock_resp

    client = APIClient("http://testserver/api")
    res = client.score_and_explain_ieee({"TransactionAmt": 300.0}, model="tabnet", top_k=2)

    assert res is not None
    assert res["model_used"] == "tabnet"
    assert res["explainability_available"] is True
    assert res["explanation_method"] == "tabnet_sparsemax_attention"
    assert len(res["top_contributors"]) == 2
    assert "Step 0" in res["step_attentions"]


@patch("frontend.services.api_client.requests.post")
def test_api_client_score_and_explain_ieee_xgboost_fallback(mock_post):
    """Verify IEEE-CIS score and explain for XGBoost returns prediction with explainability unavailable."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "transaction_id": "W3000003",
        "model_used": "xgboost",
        "fraud_probability": 0.89,
        "decision_threshold": 0.50,
        "is_fraud_predicted": True,
        "risk_tier": "CRITICAL",
        "explainability_available": False,
        "explanation_method": "unavailable",
        "top_contributors": [],
        "step_attentions": {},
        "forensic_report": "Instance-level explainability is unavailable for xgboost.",
    }
    mock_post.return_value = mock_resp

    client = APIClient("http://testserver/api")
    res = client.score_and_explain_ieee({"TransactionAmt": 400.0}, model="xgboost")

    assert res is not None
    assert res["model_used"] == "xgboost"
    assert res["explainability_available"] is False
    assert res["top_contributors"] == []


@patch("frontend.services.api_client.requests.post")
def test_api_client_score_batch_ieee(mock_post):
    """Verify IEEE-CIS batch prediction endpoint wrapper."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model_used": "tabnet",
        "total_scored": 2,
        "fraud_detected_count": 1,
        "fraud_prevalence_pct": 50.0,
        "predictions": [
            {"transaction_id": "TX1", "fraud_probability": 0.05, "is_fraud_predicted": False, "risk_tier": "LOW"},
            {"transaction_id": "TX2", "fraud_probability": 0.95, "is_fraud_predicted": True, "risk_tier": "CRITICAL"},
        ],
    }
    mock_post.return_value = mock_resp

    client = APIClient("http://testserver/api")
    batch_res = client.score_batch_ieee([{"TransactionAmt": 50.0}, {"TransactionAmt": 900.0}], model="tabnet")

    assert batch_res is not None
    assert batch_res["total_scored"] == 2
    assert batch_res["fraud_detected_count"] == 1
    assert len(batch_res["predictions"]) == 2


@patch("frontend.services.api_client.requests.get")
def test_api_client_get_ieee_benchmarks(mock_get):
    """Verify IEEE-CIS benchmark comparison retrieval."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"model_name": "XGBoost", "cv_pr_auc": 0.3845, "chrono_pr_auc": 0.3554},
        {"model_name": "TabNet", "cv_pr_auc": 0.2812, "chrono_pr_auc": 0.2554},
    ]
    mock_get.return_value = mock_resp

    client = APIClient("http://testserver/api")
    bench = client.get_ieee_benchmarks()

    assert len(bench) == 2
    assert bench[0]["model_name"] == "XGBoost"
    assert bench[1]["model_name"] == "TabNet"


@patch("frontend.services.api_client.requests.get")
def test_api_client_get_ieee_benchmarks_empty_on_error(mock_get):
    """Verify IEEE-CIS benchmarks returns empty list on network failure."""
    mock_get.side_effect = Exception("Connection timed out")

    client = APIClient("http://testserver/api")
    bench = client.get_ieee_benchmarks()
    assert bench == []


@patch("frontend.services.api_client.requests.get")
def test_api_client_get_ieee_chronological(mock_get):
    """Verify IEEE-CIS chronological holdout metrics retrieval."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "protocol": "chronological_holdout",
        "models": {
            "tabnet": {"pr_auc": 0.2554, "roc_auc": 0.8447},
            "xgboost": {"pr_auc": 0.3554, "roc_auc": 0.8838},
        },
    }
    mock_get.return_value = mock_resp

    client = APIClient("http://testserver/api")
    chrono = client.get_ieee_chronological()
    assert "models" in chrono
    assert chrono["models"]["tabnet"]["pr_auc"] == 0.2554


@patch("frontend.services.api_client.requests.get")
def test_api_client_get_ieee_metadata(mock_get):
    """Verify IEEE-CIS experiment metadata retrieval."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "dataset": "IEEE-CIS Fraud Detection",
        "num_features": 46,
        "sample_size_cv": 100000,
        "hardware": "16 GB RAM / 8-core CPU",
    }
    mock_get.return_value = mock_resp

    client = APIClient("http://testserver/api")
    meta = client.get_ieee_metadata()
    assert meta["dataset"] == "IEEE-CIS Fraud Detection"
    assert meta["num_features"] == 46


@patch("frontend.services.api_client.requests.get")
def test_api_client_get_ieee_models(mock_get):
    """Verify IEEE-CIS multi-model registry metadata retrieval."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "suite": "IEEE-CIS Fraud Detection",
        "models": {
            "tabnet": {"loaded": True, "type": "TabNetClassifier"},
            "xgboost": {"loaded": True, "type": "XGBClassifier"},
            "random_forest": {"loaded": True, "type": "RandomForestClassifier"},
            "logistic_regression": {"loaded": True, "type": "LogisticRegression"},
        },
    }
    mock_get.return_value = mock_resp

    client = APIClient("http://testserver/api")
    models = client.get_ieee_models()
    assert "models" in models
    assert len(models["models"]) == 4
    assert models["models"]["tabnet"]["loaded"] is True
