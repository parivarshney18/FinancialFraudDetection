"""
FinGuard AI - FastAPI Real-Time Fraud Detection Microservice.
Provides real-time scoring, TabNet Sparsemax attention explanations, review queuing, and benchmarking.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.db.init_db import init_db
from backend.app.ml.artifact_loader import get_ml_registry
from backend.app.routers import (
    auth,
    predictions,
    transactions,
    benchmarks,
    models,
    reports,
    users,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: Initialize DB tables and load ML artifacts upon startup."""
    print("[FastAPI] Running database initialization...")
    init_db()

    print("[FastAPI] Loading machine learning artifacts...")
    registry = get_ml_registry()
    if registry.is_loaded:
        print("[FastAPI] TabNet model and preprocessor successfully loaded.")
    else:
        print("[FastAPI] Warning: ML artifacts failed to load. Check ai/artifacts/ directory.")

    yield
    print("[FastAPI] Shutting down FinGuard AI API service.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production AI microservice for banking fraud detection utilizing PyTorch-TabNet "
        "and Sparsemax attention masks for explainable AI compliance."
    ),
    lifespan=lifespan,
)

# Configure CORS for Streamlit frontend and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers under /api
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(predictions.router, prefix=settings.API_V1_STR)
app.include_router(transactions.router, prefix=settings.API_V1_STR)
app.include_router(benchmarks.router, prefix=settings.API_V1_STR)
app.include_router(models.router, prefix=settings.API_V1_STR)
app.include_router(reports.router, prefix=settings.API_V1_STR)
app.include_router(users.router, prefix=settings.API_V1_STR)


@app.get("/health")
@app.get("/api/health")
def health():
    """System health check and ML model status."""
    registry = get_ml_registry()
    return {
        "status": "healthy",
        "service": "FinGuard AI",
        "model_loaded": registry.is_loaded,
        "ieee_models_loaded": registry.is_ieee_loaded,
        "available_ieee_models": ["tabnet", "xgboost", "random_forest", "logistic_regression"] if registry.is_ieee_loaded else [],
        "model_name": "TabNet Fraud Classifier",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/")
def root():
    """Root endpoint with service information."""
    return {
        "message": "Welcome to FinGuard AI - TabNet Fraud Detection Microservice",
        "docs_url": "/docs",
        "health_url": "/api/health",
        "version": settings.VERSION,
    }
