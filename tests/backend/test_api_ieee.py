"""
Unit and integration tests for Stage 2: IEEE-CIS API Integration.
Verifies /api/predictions/score-ieee across all 4 models, validation rules,
error handling, benchmarks, metadata, and legacy backward compatibility.
"""

from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.ml.artifact_loader import get_ml_registry


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_ieee_payload():
    """Valid IEEE-CIS transaction payload using snake_case field names."""
    payload = {
        "transaction_id": "TX-IEEE-TEST-0001",
        "transaction_dt": 86400.0,
        "transaction_amt": 150.0,
        "product_cd": "W",
        "card1": 13500,
        "card4": "visa",
        "card6": "debit",
        "p_emaildomain": "gmail.com",
        "r_emaildomain": "gmail.com",
        "device_type": "mobile",
        "device_info": "iOS Device",
        "m1": "T",
        "m4": "M0",
        "m5": "F",
        "m6": "T",
        "addr1": 315.0,
        "addr2": 87.0,
        # Velocity counters
        "c1": 1.0,
        "c2": 1.0,
        "c5": 0.0,
        "c6": 1.0,
        "c11": 1.0,
        "c13": 1.0,
        "c14": 1.0,
        # Timedeltas
        "d1": 14.0,
        "d2": 14.0,
        "d15": 14.0,
        # V-Features
        "v12": 1.0,
        "v13": 1.0,
        "v35": 1.0,
        "v36": 1.0,
        "v75": 1.0,
        "v76": 1.0,
        "v90": 0.0,
        "v91": 0.0,
        "v126": 0.0,
        "v127": 0.0,
        "v128": 0.0,
        "v257": 1.0,
        "v258": 1.0,
        "v282": 1.0,
        "v283": 1.0,
    }
    return payload


def test_score_ieee_all_four_models(client, sample_ieee_payload):
    """Verify single transaction scoring across all 4 IEEE-CIS models with calibrated thresholds."""
    expected_thresholds = {
        "tabnet": settings.IEEE_THRESHOLD_TABNET,
        "xgboost": settings.IEEE_THRESHOLD_XGBOOST,
        "random_forest": settings.IEEE_THRESHOLD_RANDOM_FOREST,
        "logistic_regression": settings.IEEE_THRESHOLD_LOGISTIC_REGRESSION,
    }

    for model_name, expected_thresh in expected_thresholds.items():
        resp = client.post(f"/api/predictions/score-ieee?model={model_name}", json=sample_ieee_payload)
        assert resp.status_code == 200, f"Scoring failed for model {model_name}: {resp.text}"

        data = resp.json()
        assert data["transaction_id"] == "TX-IEEE-TEST-0001"
        assert data["model_name"] == model_name
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert data["decision_threshold"] == pytest.approx(expected_thresh, rel=1e-3)
        assert data["is_fraud"] in [0, 1]
        assert data["is_fraud"] == int(data["fraud_probability"] >= data["decision_threshold"])
        assert data["risk_tier"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        assert data["latency_ms"] >= 0.0
        assert data["metadata"]["dataset"] == "IEEE-CIS Fraud Detection"
        assert data["metadata"]["n_processed_features"] == 46


def test_score_ieee_body_model_selection(client, sample_ieee_payload):
    """Verify model selection specified in JSON payload body."""
    payload = sample_ieee_payload.copy()
    payload["model_name"] = "xgboost"

    resp = client.post("/api/predictions/score-ieee", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "xgboost"
    assert data["decision_threshold"] == pytest.approx(settings.IEEE_THRESHOLD_XGBOOST, rel=1e-3)


def test_score_ieee_invalid_model_names(client, sample_ieee_payload):
    """Verify that unsupported model names return 400 Bad Request."""
    # Via query parameter
    resp_query = client.post("/api/predictions/score-ieee?model=deepseek", json=sample_ieee_payload)
    assert resp_query.status_code == 400
    assert "Invalid model" in resp_query.json()["detail"]

    # Via body
    bad_payload = sample_ieee_payload.copy()
    bad_payload["model_name"] = "transformer_v3"
    resp_body = client.post("/api/predictions/score-ieee", json=bad_payload)
    assert resp_body.status_code == 400
    assert "Invalid model" in resp_body.json()["detail"]


def test_score_ieee_missing_and_malformed_fields(client, sample_ieee_payload):
    """Verify strict validation rejects missing required fields and negative numbers."""
    # 1. Missing transaction_amt
    p_no_amt = sample_ieee_payload.copy()
    del p_no_amt["transaction_amt"]
    assert client.post("/api/predictions/score-ieee", json=p_no_amt).status_code == 422

    # 2. Missing transaction_dt
    p_no_dt = sample_ieee_payload.copy()
    del p_no_dt["transaction_dt"]
    assert client.post("/api/predictions/score-ieee", json=p_no_dt).status_code == 422

    # 3. Missing product_cd
    p_no_prod = sample_ieee_payload.copy()
    del p_no_prod["product_cd"]
    assert client.post("/api/predictions/score-ieee", json=p_no_prod).status_code == 422

    # 4. Missing card1
    p_no_card1 = sample_ieee_payload.copy()
    del p_no_card1["card1"]
    assert client.post("/api/predictions/score-ieee", json=p_no_card1).status_code == 422

    # 5. Invalid negative transaction_amt
    p_neg_amt = sample_ieee_payload.copy()
    p_neg_amt["transaction_amt"] = -50.0
    assert client.post("/api/predictions/score-ieee", json=p_neg_amt).status_code == 422

    # 6. Invalid zero transaction_amt
    p_zero_amt = sample_ieee_payload.copy()
    p_zero_amt["transaction_amt"] = 0.0
    assert client.post("/api/predictions/score-ieee", json=p_zero_amt).status_code == 422

    # 7. Invalid negative transaction_dt
    p_neg_dt = sample_ieee_payload.copy()
    p_neg_dt["transaction_dt"] = -100.0
    assert client.post("/api/predictions/score-ieee", json=p_neg_dt).status_code == 422

    # 8. Invalid non-positive card1
    p_bad_card = sample_ieee_payload.copy()
    p_bad_card["card1"] = 0
    assert client.post("/api/predictions/score-ieee", json=p_bad_card).status_code == 422


def test_score_ieee_nullable_fields_allowed(client, sample_ieee_payload):
    """Verify that optional/nullable C, D, V, and categorical fields default gracefully."""
    minimal_payload = {
        "transaction_dt": 120000.0,
        "transaction_amt": 89.99,
        "product_cd": "C",
        "card1": 9500,
    }
    resp = client.post("/api/predictions/score-ieee?model=tabnet", json=minimal_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["is_fraud"] in [0, 1]


def test_score_ieee_artifact_unloaded_error(client, sample_ieee_payload):
    """Verify 503 Service Unavailable when IEEE-CIS artifacts are reported not loaded."""
    registry = get_ml_registry()
    with patch.object(registry, "is_ieee_loaded", False):
        resp = client.post("/api/predictions/score-ieee", json=sample_ieee_payload)
        assert resp.status_code == 503
        assert "artifacts and preprocessor are not loaded" in resp.json()["detail"]


def test_score_batch_ieee_endpoint(client, sample_ieee_payload):
    """Verify batch IEEE-CIS scoring endpoint."""
    p1 = sample_ieee_payload.copy()
    p1["transaction_id"] = "TX-BATCH-001"
    p2 = sample_ieee_payload.copy()
    p2["transaction_id"] = "TX-BATCH-002"
    p2["transaction_amt"] = 4500.0

    batch_payload = {
        "transactions": [p1, p2],
        "model_name": "xgboost",
    }
    resp = client.post("/api/predictions/batch-ieee", json=batch_payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_transactions"] == 2
    assert data["model_used"] == "xgboost"
    assert len(data["results"]) == 2
    assert data["results"][0]["transaction_id"] == "TX-BATCH-001"
    assert data["results"][1]["transaction_id"] == "TX-BATCH-002"


def test_benchmarks_ieee_endpoints(client):
    """Verify retrieval of Phase 9 benchmark comparison, chronological metrics, and metadata."""
    # 1. IEEE comparison table
    b_resp = client.get("/api/benchmarks/ieee")
    assert b_resp.status_code == 200
    b_data = b_resp.json()
    assert isinstance(b_data, list)
    assert len(b_data) == 4
    model_names = [m["model_name"] for m in b_data]
    assert "Tabnet" in model_names
    assert "Xgboost" in model_names

    # Check structure
    for item in b_data:
        assert "cv_pr_auc" in item
        assert "chrono_pr_auc" in item
        assert "cv_roc_auc" in item
        assert "fit_time_s" in item
        assert "latency_ms_per_1k" in item

    # 2. Chronological metrics JSON
    c_resp = client.get("/api/benchmarks/ieee/chronological")
    assert c_resp.status_code == 200
    c_data = c_resp.json()
    assert "tabnet" in c_data
    assert "xgboost" in c_data
    assert "random_forest" in c_data
    assert "logistic_regression" in c_data
    assert c_data["tabnet"]["protocol"] == "chronological_holdout"

    # 3. IEEE experiment metadata
    m_resp = client.get("/api/benchmarks/ieee/metadata")
    assert m_resp.status_code == 200
    m_data = m_resp.json()
    assert m_data["dataset"] == "IEEE-CIS Fraud Detection"
    assert m_data["curated_features_count"] == 46
    assert m_data["n_splits_cv"] == 5

    # 4. IEEE models registry details
    models_resp = client.get("/api/models/ieee")
    assert models_resp.status_code == 200
    mod_data = models_resp.json()
    assert mod_data["status"] == "ready"
    assert mod_data["curated_features_count"] == 46
    assert mod_data["models"]["tabnet"]["available"] is True


def test_benchmarks_missing_file_error(client):
    """Verify that 404 is returned if benchmark files are missing."""
    with patch.object(settings, "IEEE_COMPARISON_CSV_PATH", Path("non_existent_file.csv")):
        resp = client.get("/api/benchmarks/ieee")
        assert resp.status_code == 404

    with patch.object(settings, "IEEE_CHRONO_METRICS_PATH", Path("non_existent_file.json")):
        resp = client.get("/api/benchmarks/ieee/chronological")
        assert resp.status_code == 404

    with patch.object(settings, "IEEE_METADATA_PATH", Path("non_existent_file.json")):
        resp = client.get("/api/benchmarks/ieee/metadata")
        assert resp.status_code == 404


def test_backward_compatibility_with_legacy_endpoints(client):
    """Verify that existing legacy endpoints remain 100% operational."""
    # 1. Health check returns both legacy and IEEE status
    h_resp = client.get("/api/health")
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert h_data["status"] == "healthy"
    assert h_data["model_loaded"] is True
    assert h_data["ieee_models_loaded"] is True

    # 2. Legacy benchmarks endpoint
    legacy_b_resp = client.get("/api/benchmarks/")
    assert legacy_b_resp.status_code == 200
    assert len(legacy_b_resp.json()) >= 1

    # 3. Active models endpoint
    act_resp = client.get("/api/models/active")
    assert act_resp.status_code == 200
    assert act_resp.json()["status"] == "ready"


def test_score_and_explain_ieee_tabnet(client, sample_ieee_payload):
    """Verify TabNet Sparsemax attention explanation endpoint on 46 IEEE-CIS features."""
    resp = client.post(
        "/api/predictions/score-and-explain-ieee?model=tabnet&top_k=5",
        json=sample_ieee_payload,
    )
    assert resp.status_code == 200, f"Score and explain failed: {resp.text}"
    data = resp.json()

    # 1. Check prediction block
    pred = data["prediction"]
    assert pred["model_name"] == "tabnet"
    assert 0.0 <= pred["fraud_probability"] <= 1.0
    assert pred["decision_threshold"] == pytest.approx(settings.IEEE_THRESHOLD_TABNET, rel=1e-3)
    assert pred["is_fraud"] in [0, 1]

    # 2. Check top contributors (top_k = 5)
    top_contribs = data["top_contributors"]
    assert len(top_contribs) == 5
    for c in top_contribs:
        assert "feature" in c
        assert "importance" in c
        assert "contribution_pct" in c
        assert 0 <= c["primary_step"] <= 2
        assert len(c["step_attentions"]) == 3

    # 3. Check step attention breakdown
    step_attns = data["step_attention_breakdown"]
    assert "Step 0" in step_attns
    assert "Step 1" in step_attns
    assert "Step 2" in step_attns
    assert len(step_attns["Step 0"]) == 5

    # 4. Check compliance narrative
    narrative = data["narrative_summary"]
    assert isinstance(narrative, str)
    assert "IEEE-CIS FORENSIC EXPLANATION REPORT" in narrative
    assert "Compliance Notice" in narrative

    # 5. Check explainer metadata
    meta = data["explainer_metadata"]
    assert meta["dataset"] == "IEEE-CIS Fraud Detection"
    assert meta["n_features_evaluated"] == 46


def test_score_and_explain_ieee_xgboost_limitation(client, sample_ieee_payload):
    """Verify that XGBoost explanation requests gracefully report limitation when SHAP is not installed."""
    resp = client.post(
        "/api/predictions/score-and-explain-ieee?model=xgboost",
        json=sample_ieee_payload,
    )
    assert resp.status_code == 200
    data = resp.json()

    pred = data["prediction"]
    assert pred["model_name"] == "xgboost"
    assert pred["decision_threshold"] == pytest.approx(settings.IEEE_THRESHOLD_XGBOOST, rel=1e-3)

    meta = data["explainer_metadata"]
    assert "limitation" in meta or "shap_installed" in meta
    narrative = data["narrative_summary"]
    assert "XGBOOST" in narrative


def test_score_and_explain_ieee_invalid_model(client, sample_ieee_payload):
    """Verify 400 Bad Request when requesting explanation for invalid model name."""
    resp = client.post(
        "/api/predictions/score-and-explain-ieee?model=unsupported_model",
        json=sample_ieee_payload,
    )
    assert resp.status_code == 400
    assert "Invalid model" in resp.json()["detail"]


def test_legacy_score_and_explain_unbroken(client):
    """Verify that the legacy synthetic score-and-explain endpoint remains 100% operational."""
    legacy_payload = {
        "Transaction_ID": "TX-LEGACY-001",
        "Age": 45,
        "Gender": "Male",
        "State": "Maharashtra",
        "City": "Mumbai",
        "Account_Type": "Savings",
        "Account_Balance": 120000.0,
        "Transaction_Type": "UPI",
        "Transaction_Amount": 8500.0,
        "Transaction_Date": "12-03-2025",
        "Transaction_Time": "11:22:33",
        "Merchant_Category": "Travel",
        "Device_Type": "Mobile",
        "Transaction_Device": "Mobile App",
        "Transaction_Description": "Flight booking",
    }
    resp = client.post("/api/predictions/score-and-explain?top_k=5", json=legacy_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"]["model_name"] == "TabNet"
    assert len(data["top_contributors"]) == 5

