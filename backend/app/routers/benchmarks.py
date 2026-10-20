"""
Benchmark results and feature importance routes.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query
from backend.app.schemas.schemas import (
    BenchmarkModelItem,
    FeatureImportanceItem,
    IEEEBenchmarkItem,
)
from backend.app.services.model_registry_service import (
    get_benchmark_metrics,
    get_ieee_benchmark_metrics,
    get_ieee_chronological_metrics,
    get_ieee_metadata,
    get_feature_importances,
)

router = APIRouter(prefix="/benchmarks", tags=["Benchmarks & Interpretability"])


@router.get("/", response_model=List[BenchmarkModelItem])
def get_benchmarks(
    dataset: Optional[str] = Query(
        default=None,
        description="Dataset benchmark results to retrieve: 'legacy' or 'ieee'",
    ),
):
    """Retrieve evaluation benchmarks across all baseline and TabNet models."""
    return get_benchmark_metrics(dataset=dataset)


@router.get("/ieee", response_model=List[IEEEBenchmarkItem])
def get_ieee_benchmarks():
    """Retrieve Phase 9 IEEE-CIS benchmark comparison table across CV and chronological holdout."""
    return get_ieee_benchmark_metrics()


@router.get("/ieee/chronological")
def get_ieee_chronological():
    """Retrieve Phase 9 chronological holdout evaluation metrics for all 4 IEEE-CIS models."""
    return get_ieee_chronological_metrics()


@router.get("/ieee/metadata")
def get_ieee_meta():
    """Retrieve Phase 9 IEEE-CIS experiment metadata and run configuration."""
    return get_ieee_metadata()


@router.get("/feature-importance", response_model=List[FeatureImportanceItem])
def get_feature_importance_rankings():
    """Retrieve TabNet Attentive Transformer Sparsemax global feature importance rankings."""
    return get_feature_importances()

