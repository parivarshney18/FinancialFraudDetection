"""
Application configuration for FinGuard AI FastAPI Microservice.
Loads environment variables and sets defaults for database, security, and ML artifacts.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for backend API."""

    PROJECT_NAME: str = "FinGuard AI - TabNet Fraud Detection Microservice"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"

    # Security
    SECRET_KEY: str = "dev-secret-key-finguard-tabnet-2026-abesec"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # Database
    DATABASE_URL: str = "sqlite:///./fraud_app.db"

    # Legacy ML Artifact Paths (Phase 4 Synthetic)
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "ai" / "artifacts"
    MODEL_PATH: Path = ARTIFACTS_DIR / "tabnet_fraud_model.zip"
    PREPROCESSOR_PATH: Path = ARTIFACTS_DIR / "preprocessor.joblib"
    METADATA_PATH: Path = ARTIFACTS_DIR / "model_metadata.json"
    RESULTS_DIR: Path = PROJECT_ROOT / "ai" / "results"
    FIGURES_DIR: Path = PROJECT_ROOT / "ai" / "reports" / "figures"
    REPORTS_DIR: Path = PROJECT_ROOT / "ai" / "reports"

    # IEEE-CIS ML Artifact Paths (Phase 9)
    IEEE_TABNET_MODEL_PATH: Path = ARTIFACTS_DIR / "ieee_tabnet.zip"
    IEEE_XGBOOST_MODEL_PATH: Path = ARTIFACTS_DIR / "ieee_xgboost.json"
    IEEE_RANDOM_FOREST_MODEL_PATH: Path = ARTIFACTS_DIR / "ieee_random_forest.joblib"
    IEEE_LOGISTIC_REGRESSION_MODEL_PATH: Path = ARTIFACTS_DIR / "ieee_logistic_regression.joblib"
    IEEE_PREPROCESSOR_PATH: Path = ARTIFACTS_DIR / "ieee_preprocessor.joblib"
    IEEE_METADATA_PATH: Path = RESULTS_DIR / "ieee_metadata.json"
    IEEE_CHRONO_METRICS_PATH: Path = RESULTS_DIR / "ieee_metrics_chronological.json"
    IEEE_COMPARISON_CSV_PATH: Path = RESULTS_DIR / "ieee_model_comparison.csv"
    IEEE_FIGURE_PATH: Path = FIGURES_DIR / "15_ieee_model_comparison.png"

    # Operational Parameters
    DEFAULT_DECISION_THRESHOLD: float = 0.50

    # Calibrated IEEE-CIS Decision Thresholds (Optimized on Training F1 in Phase 9)
    IEEE_THRESHOLD_TABNET: float = 0.89
    IEEE_THRESHOLD_XGBOOST: float = 0.79
    IEEE_THRESHOLD_RANDOM_FOREST: float = 0.57
    IEEE_THRESHOLD_LOGISTIC_REGRESSION: float = 0.86
    IEEE_DEFAULT_THRESHOLD: float = 0.50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
