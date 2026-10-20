"""
HTTP Client for FinGuard AI FastAPI Microservice.
Provides resilient communication between Streamlit frontend and the backend API.
"""

from typing import Dict, Any, List, Optional
import os
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000/api")


class APIClient:
    """REST API wrapper for FinGuard AI backend."""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def check_health(self) -> Dict[str, Any]:
        """Check API and ML model availability."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return {"status": "unhealthy", "model_loaded": False}
        except Exception as e:
            return {"status": "unreachable", "model_loaded": False, "error": str(e)}

    def login(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with backend and retrieve JWT access token."""
        try:
            resp = requests.post(
                f"{self.base_url}/auth/login-json",
                json={"email": email, "password": password},
                timeout=5,
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    def get_me(self, token: str) -> Optional[Dict[str, Any]]:
        """Retrieve authenticated user profile."""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            resp = requests.get(f"{self.base_url}/auth/me", headers=headers, timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    def score_transaction(self, tx_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Score a single transaction."""
        try:
            resp = requests.post(f"{self.base_url}/predictions/score", json=tx_dict, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            print(f"[APIClient] Score error: {e}")
            return None

    def score_and_explain(self, tx_dict: Dict[str, Any], top_k: int = 5) -> Optional[Dict[str, Any]]:
        """Score transaction and retrieve TabNet Sparsemax attention explanation."""
        try:
            resp = requests.post(
                f"{self.base_url}/predictions/score-and-explain?top_k={top_k}",
                json=tx_dict,
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            print(f"[APIClient] Explain error: {e}")
            return None

    def score_batch(self, tx_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Score multiple transactions in batch mode."""
        try:
            resp = requests.post(
                f"{self.base_url}/predictions/batch",
                json={"transactions": tx_list},
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            print(f"[APIClient] Batch error: {e}")
            return None

    def list_transactions(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieve recent transactions from audit database."""
        try:
            resp = requests.get(
                f"{self.base_url}/transactions/?limit={limit}&offset={offset}",
                timeout=5,
            )
            if resp.status_code == 200:
                return resp.json().get("transactions", [])
            return []
        except Exception:
            return []

    def get_pending_review_queue(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve unreviewed high/critical risk transactions."""
        try:
            resp = requests.get(
                f"{self.base_url}/transactions/queue/pending?limit={limit}",
                timeout=5,
            )
            if resp.status_code == 200:
                return resp.json().get("queue", [])
            return []
        except Exception:
            return []

    def get_transaction_detail(self, tx_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full details, prediction, and explanations for transaction ID."""
        try:
            resp = requests.get(f"{self.base_url}/transactions/{tx_id}", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    def submit_review(
        self,
        tx_id: str,
        action: str,
        notes: str,
        token: str,
    ) -> bool:
        """Submit auditor review decision."""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            resp = requests.post(
                f"{self.base_url}/transactions/{tx_id}/review",
                json={"action": action, "notes": notes},
                headers=headers,
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def get_benchmarks(self) -> List[Dict[str, Any]]:
        """Retrieve 5-Fold CV model benchmark comparison."""
        try:
            resp = requests.get(f"{self.base_url}/benchmarks/", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception:
            return []

    def get_feature_importances(self) -> List[Dict[str, Any]]:
        """Retrieve TabNet global feature importance rankings."""
        try:
            resp = requests.get(f"{self.base_url}/benchmarks/feature-importance", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception:
            return []

    def get_active_model_info(self) -> Dict[str, Any]:
        """Retrieve active deployed model metadata."""
        try:
            resp = requests.get(f"{self.base_url}/models/active", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return {}
        except Exception:
            return {}

    def get_audit_summary(self) -> Dict[str, Any]:
        """Retrieve operational compliance audit summary."""
        try:
            resp = requests.get(f"{self.base_url}/reports/audit-summary", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return {}
        except Exception:
            return {}

    # --- IEEE-CIS Production Endpoints (Phase 9 & 10) ---

    def score_ieee(self, tx_dict: Dict[str, Any], model: str = "tabnet") -> Optional[Dict[str, Any]]:
        """Score a single IEEE-CIS transaction using chosen model architecture."""
        try:
            resp = requests.post(
                f"{self.base_url}/predictions/score-ieee?model={model}",
                json=tx_dict,
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            print(f"[APIClient] IEEE Score error: {e}")
            return None

    def score_and_explain_ieee(
        self, tx_dict: Dict[str, Any], model: str = "tabnet", top_k: int = 5
    ) -> Optional[Dict[str, Any]]:
        """Score IEEE-CIS transaction and retrieve TabNet Sparsemax attention attribution."""
        try:
            resp = requests.post(
                f"{self.base_url}/predictions/score-and-explain-ieee?model={model}&top_k={top_k}",
                json=tx_dict,
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            print(f"[APIClient] IEEE Explain error: {e}")
            return None

    def score_batch_ieee(
        self, tx_list: List[Dict[str, Any]], model: str = "tabnet"
    ) -> Optional[Dict[str, Any]]:
        """Score multiple IEEE-CIS transactions in batch mode."""
        try:
            resp = requests.post(
                f"{self.base_url}/predictions/batch-ieee?model={model}",
                json={"transactions": tx_list, "model_name": model},
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            print(f"[APIClient] IEEE Batch error: {e}")
            return None

    def get_ieee_benchmarks(self) -> List[Dict[str, Any]]:
        """Retrieve Phase 9 IEEE-CIS benchmark comparison table across CV and chronological holdout."""
        try:
            resp = requests.get(f"{self.base_url}/benchmarks/ieee", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception:
            return []

    def get_ieee_chronological(self) -> Dict[str, Any]:
        """Retrieve Phase 9 chronological holdout evaluation metrics for all 4 IEEE-CIS models."""
        try:
            resp = requests.get(f"{self.base_url}/benchmarks/ieee/chronological", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return {}
        except Exception:
            return {}

    def get_ieee_metadata(self) -> Dict[str, Any]:
        """Retrieve Phase 9 IEEE-CIS experiment metadata and run configuration."""
        try:
            resp = requests.get(f"{self.base_url}/benchmarks/ieee/metadata", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return {}
        except Exception:
            return {}

    def get_ieee_models(self) -> Dict[str, Any]:
        """Retrieve runtime metadata of deployed IEEE-CIS multi-model suite."""
        try:
            resp = requests.get(f"{self.base_url}/models/ieee", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return {}
        except Exception:
            return {}


# Default client instance
api_client = APIClient()
