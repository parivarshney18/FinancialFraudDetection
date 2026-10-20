"""
Integration tests for FinGuard AI FastAPI Microservice (tests/backend/test_api.py).
Tests authentication, real-time TabNet scoring, attention mask explanation, reviews, and benchmarks.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.db.init_db import init_db
from backend.app.ml.artifact_loader import get_ml_registry


@pytest.fixture(scope="session", autouse=True)
def setup_backend_service():
    """Ensure database and ML artifacts are loaded before running API tests."""
    init_db()
    registry = get_ml_registry()
    assert registry.is_loaded, "ML artifacts must be loaded for API testing."


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_transaction_payload():
    """Realistic sample banking transaction payload."""
    return {
        "Transaction_ID": "TX-TEST-0001",
        "Customer_ID": "CUST-9999",
        "Merchant_ID": "MCH-8888",
        "Customer_Name": "Rahul Sharma",
        "Customer_Contact": "9876543210",
        "Customer_Email": "rahul.sharma@example.com",
        "Age": 34,
        "Gender": "Male",
        "State": "Maharashtra",
        "City": "Mumbai",
        "Bank_Branch": "Mumbai Branch",
        "Account_Type": "Savings",
        "Account_Balance": 85000.0,
        "Transaction_Type": "UPI",
        "Transaction_Amount": 4200.0,
        "Transaction_Date": "15-01-2025",
        "Transaction_Time": "14:35:10",
        "Merchant_Category": "Retail",
        "Device_Type": "Mobile",
        "Transaction_Device": "Mobile Device",
        "Transaction_Location": "Mumbai, Maharashtra",
        "Transaction_Currency": "INR",
        "Transaction_Description": "Grocery store purchase",
    }


def test_health_check_endpoint(client):
    """Verify system health check and model loading status."""
    for path in ["/api/health", "/health"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "TabNet" in data["model_name"]


def test_auth_login_and_me_flow(client):
    """Verify user login, JWT issuance, and authenticated /me endpoint."""
    # 1. Login with valid seeded admin credentials
    login_resp = client.post(
        "/api/auth/login-json",
        json={"email": "admin@finguard.ai", "password": "admin123"},
    )
    assert login_resp.status_code == 200
    token_data = login_resp.json()
    assert "access_token" in token_data
    assert token_data["role"] == "admin"

    token = token_data["access_token"]

    # 2. Access /me endpoint with token
    headers = {"Authorization": f"Bearer {token}"}
    me_resp = client.get("/api/auth/me", headers=headers)
    assert me_resp.status_code == 200
    user_data = me_resp.json()
    assert user_data["email"] == "admin@finguard.ai"
    assert user_data["role"] == "admin"

    # 3. Test invalid password
    bad_login = client.post(
        "/api/auth/login-json",
        json={"email": "admin@finguard.ai", "password": "wrongpassword"},
    )
    assert bad_login.status_code == 401


def test_real_time_scoring_endpoint(client, sample_transaction_payload):
    """Verify single transaction scoring returns calibrated probability and risk tier."""
    response = client.post("/api/predictions/score", json=sample_transaction_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == "TX-TEST-0001"
    assert data["model_name"] == "TabNet"
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["is_fraud"] in [0, 1]
    assert data["risk_tier"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    assert data["latency_ms"] >= 0.0


def test_score_and_explain_endpoint(client, sample_transaction_payload):
    """Verify TabNet Sparsemax attention explanation, top contributors, and narrative."""
    response = client.post(
        "/api/predictions/score-and-explain?top_k=5",
        json=sample_transaction_payload,
    )
    assert response.status_code == 200
    data = response.json()

    # Prediction checks
    pred = data["prediction"]
    assert pred["transaction_id"] == "TX-TEST-0001"
    assert 0.0 <= pred["fraud_probability"] <= 1.0

    # Contributors checks
    top_contribs = data["top_contributors"]
    assert len(top_contribs) == 5
    for c in top_contribs:
        assert "feature" in c
        assert "importance" in c
        assert "contribution_pct" in c
        assert 0 <= c["primary_step"] <= 2

    # Step attention breakdown checks
    step_attns = data["step_attention_breakdown"]
    assert "Step 0" in step_attns
    assert "Step 1" in step_attns
    assert "Step 2" in step_attns

    # Forensic narrative checks
    narrative = data["narrative_summary"]
    assert isinstance(narrative, str)
    assert len(narrative) > 20


def test_batch_prediction_endpoint(client, sample_transaction_payload):
    """Verify high-throughput batch scoring."""
    p1 = sample_transaction_payload.copy()
    p1["Transaction_ID"] = "TX-BATCH-001"
    p2 = sample_transaction_payload.copy()
    p2["Transaction_ID"] = "TX-BATCH-002"
    p2["Transaction_Amount"] = 95000.0

    batch_payload = {"transactions": [p1, p2]}
    response = client.post("/api/predictions/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_transactions"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["transaction_id"] == "TX-BATCH-001"
    assert data["results"][1]["transaction_id"] == "TX-BATCH-002"


def test_benchmarks_and_feature_importance_endpoints(client):
    """Verify model comparison benchmarks and feature ranking endpoints."""
    # 1. Benchmarks
    b_resp = client.get("/api/benchmarks/")
    assert b_resp.status_code == 200
    b_data = b_resp.json()
    assert isinstance(b_data, list)
    assert len(b_data) >= 1
    model_names = [m["model_name"] for m in b_data]
    assert any("Tabnet" in name or "TabNet" in name for name in model_names)

    # 2. Feature Importance
    fi_resp = client.get("/api/benchmarks/feature-importance")
    assert fi_resp.status_code == 200
    fi_data = fi_resp.json()
    assert isinstance(fi_data, list)
    assert len(fi_data) == 22
    assert fi_data[0]["rank"] == 1
    assert "Amount_to_Balance_Ratio" in [item["feature"] for item in fi_data[:3]]


def test_transaction_detail_and_auditor_review_flow(client, sample_transaction_payload):
    """Verify transaction retrieval, review queue, and review submission."""
    # Score transaction first to ensure it exists in DB
    client.post("/api/predictions/score", json=sample_transaction_payload)

    # Retrieve transaction detail
    detail_resp = client.get("/api/transactions/TX-TEST-0001")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["transaction"]["transaction_id"] == "TX-TEST-0001"

    # Authenticate as auditor
    login_resp = client.post(
        "/api/auth/login-json",
        json={"email": "admin@finguard.ai", "password": "admin123"},
    )
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Submit auditor review
    review_resp = client.post(
        "/api/transactions/TX-TEST-0001/review",
        json={"action": "APPROVED", "notes": "Verified with cardholder over SMS."},
        headers=headers,
    )
    assert review_resp.status_code == 200
    rev_data = review_resp.json()
    assert rev_data["action"] == "APPROVED"
    assert rev_data["auditor_email"] == "admin@finguard.ai"


def test_executive_audit_report_endpoint(client):
    """Verify compliance summary statistics endpoint."""
    response = client.get("/api/reports/audit-summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_transactions_ingested" in data
    assert "total_predictions_scored" in data
    assert "average_inference_latency_ms" in data
    assert "risk_tier_breakdown" in data
