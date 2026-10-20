"""
Model registry service.
Loads benchmark evaluation metrics, global feature importances, and model metadata.
"""

from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import pandas as pd
from fastapi import HTTPException

from backend.app.core.config import settings
from backend.app.schemas.schemas import (
    BenchmarkModelItem,
    FeatureImportanceItem,
    IEEEBenchmarkItem,
)
from backend.app.ml.artifact_loader import get_ml_registry


def get_benchmark_metrics(dataset: Optional[str] = None) -> List[BenchmarkModelItem]:
    """Load benchmark comparison table from CSV (supports 'legacy' or 'ieee')."""
    if dataset == "ieee":
        csv_path = settings.IEEE_COMPARISON_CSV_PATH
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="IEEE-CIS benchmark comparison file not found.")
        df = pd.read_csv(csv_path)
        items = []
        for _, row in df.iterrows():
            items.append(
                BenchmarkModelItem(
                    model_name=str(row["Model"]),
                    pr_auc=str(row["CV PR-AUC"]),
                    roc_auc=str(row["CV ROC-AUC"]),
                    recall_at_1pct_fpr=str(row["CV Recall@1%FPR"]),
                    f1_score=str(row["CV F1-Score"]),
                    macro_f1="N/A",
                    total_time_s=float(row["Fit Time (s)"]),
                    chrono_pr_auc=str(row["Chrono PR-AUC"]),
                    chrono_roc_auc=str(row["Chrono ROC-AUC"]),
                    chrono_recall_at_1pct_fpr=str(row["Chrono Recall@1%FPR"]),
                    chrono_f1_score=str(row["Chrono F1-Score"]),
                    latency_ms_per_1k=float(row["Latency (ms/1k)"]),
                )
            )
        return items

    csv_path = settings.RESULTS_DIR / "comparison_metrics.csv"
    if not csv_path.exists():
        # Fallback to IEEE if legacy file not present
        if settings.IEEE_COMPARISON_CSV_PATH.exists():
            return get_benchmark_metrics(dataset="ieee")
        return []

    try:
        df = pd.read_csv(csv_path)
        items = []
        for _, row in df.iterrows():
            items.append(
                BenchmarkModelItem(
                    model_name=str(row["Model"]),
                    pr_auc=str(row["PR-AUC"]),
                    roc_auc=str(row["ROC-AUC"]),
                    recall_at_1pct_fpr=str(row["Recall @ FPR<=1%"]),
                    f1_score=str(row["F1-Score"]),
                    macro_f1=str(row.get("Macro-F1", "N/A")),
                    total_time_s=float(row["Total Time (s)"]),
                )
            )
        return items
    except Exception as e:
        print(f"[ModelRegistry] Error parsing benchmark CSV: {e}")
        return []


def get_ieee_benchmark_metrics() -> List[IEEEBenchmarkItem]:
    """Retrieve Phase 9 IEEE-CIS benchmark comparison table across CV and chronological holdout."""
    csv_path = settings.IEEE_COMPARISON_CSV_PATH
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="IEEE-CIS benchmark comparison file not found.")

    try:
        df = pd.read_csv(csv_path)
        items = []
        for _, row in df.iterrows():
            items.append(
                IEEEBenchmarkItem(
                    model_name=str(row["Model"]),
                    cv_pr_auc=str(row["CV PR-AUC"]),
                    chrono_pr_auc=str(row["Chrono PR-AUC"]),
                    cv_roc_auc=str(row["CV ROC-AUC"]),
                    chrono_roc_auc=str(row["Chrono ROC-AUC"]),
                    cv_recall_at_1pct_fpr=str(row["CV Recall@1%FPR"]),
                    chrono_recall_at_1pct_fpr=str(row["Chrono Recall@1%FPR"]),
                    cv_f1_score=str(row["CV F1-Score"]),
                    chrono_f1_score=str(row["Chrono F1-Score"]),
                    fit_time_s=float(row["Fit Time (s)"]),
                    latency_ms_per_1k=float(row["Latency (ms/1k)"]),
                )
            )
        return items
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading IEEE-CIS benchmark CSV: {e}")


def get_ieee_chronological_metrics() -> Dict[str, Any]:
    """Retrieve Phase 9 chronological holdout evaluation metrics for all 4 models."""
    json_path = settings.IEEE_CHRONO_METRICS_PATH
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="IEEE-CIS chronological metrics file not found.")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading chronological metrics JSON: {e}")


def get_ieee_metadata() -> Dict[str, Any]:
    """Retrieve Phase 9 IEEE-CIS experiment metadata and run configuration."""
    json_path = settings.IEEE_METADATA_PATH
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="IEEE-CIS metadata file not found.")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading IEEE-CIS metadata JSON: {e}")


def get_feature_importances() -> List[FeatureImportanceItem]:
    """Load global TabNet Sparsemax feature importance rankings from CSV."""
    csv_path = settings.RESULTS_DIR / "feature_importance.csv"
    if not csv_path.exists():
        return []

    try:
        df = pd.read_csv(csv_path)
        items = []
        for _, row in df.iterrows():
            items.append(
                FeatureImportanceItem(
                    feature=str(row["feature"]),
                    importance=float(row["importance"]),
                    importance_pct=round(float(row["importance_pct"]), 2),
                    rank=int(row["rank"]),
                )
            )
        return items
    except Exception as e:
        print(f"[ModelRegistry] Error parsing feature importance CSV: {e}")
        return []


def get_active_model_details() -> Dict[str, Any]:
    """Retrieve runtime metadata of active deployed model."""
    registry = get_ml_registry()
    if not registry.is_loaded:
        return {
            "status": "unloaded",
            "message": "Model artifacts not yet loaded.",
        }

    return {
        "status": "ready",
        "model_name": "TabNet Fraud Classifier",
        "framework": "pytorch-tabnet",
        "metadata": registry.metadata,
        "n_features": len(registry.preprocessor.feature_names_) if registry.preprocessor else 0,
        "feature_names": registry.preprocessor.feature_names_ if registry.preprocessor else [],
        "ieee_models_ready": registry.is_ieee_loaded,
    }


def get_ieee_model_details() -> Dict[str, Any]:
    """Retrieve runtime metadata of deployed IEEE-CIS multi-model suite."""
    registry = get_ml_registry()
    if not registry.is_ieee_loaded:
        return {
            "status": "unloaded",
            "message": "IEEE-CIS model artifacts not yet loaded.",
        }

    models_info = {}
    for m in ["tabnet", "xgboost", "random_forest", "logistic_regression"]:
        models_info[m] = {
            "available": registry.ieee_models.get(m) is not None,
            "calibrated_threshold": registry.get_ieee_threshold(m),
        }

    prep = registry.ieee_preprocessor
    return {
        "status": "ready",
        "dataset": "IEEE-CIS Fraud Detection",
        "curated_features_count": 46,
        "raw_input_columns_count": 41,
        "models": models_info,
        "preprocessor": {
            "is_fitted": prep.is_fitted_ if prep else False,
            "categorical_features_count": len(prep.categorical_cols) if prep else 0,
            "numerical_features_count": len(prep.numerical_cols) if prep else 0,
            "total_output_features": len(prep.feature_cols) if prep else 0,
            "feature_names": prep.feature_cols if prep else [],
        },
    }
