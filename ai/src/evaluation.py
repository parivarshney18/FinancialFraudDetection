"""
Evaluation module for Financial Fraud Detection.
Computes PR-AUC, ROC-AUC, Recall@FPR<=1%, Macro F1, Precision, Recall, and Confusion Matrix.
Provides aggregation utilities across cross-validation folds.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)


def compute_recall_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, max_fpr: float = 0.01) -> float:
    """Compute maximum Recall (True Positive Rate) achieved while False Positive Rate <= max_fpr.

    Critical operational metric for banking fraud detection where false alarms must be strictly bounded.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probability of positive class (fraud).
        max_fpr: Maximum acceptable false positive rate (default 0.01 or 1%).

    Returns:
        float: True positive rate at or below max_fpr.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid_indices = np.where(fpr <= max_fpr)[0]

    if len(valid_indices) == 0:
        return 0.0

    # The last valid index corresponds to the maximum TPR with FPR <= max_fpr
    return float(tpr[valid_indices[-1]])


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute comprehensive performance metrics for a single fold evaluation.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probability for class 1 (fraud).
        threshold: Decision threshold for discrete classification (default 0.5).

    Returns:
        Dictionary containing all evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    # Clean any NaNs or infinities defensively
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    y_pred = (y_prob >= threshold).astype(np.int64)

    # Threshold-independent metrics
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float(np.mean(y_true))

    recall_at_fpr_1pct = compute_recall_at_fpr(y_true, y_prob, max_fpr=0.01)

    # Threshold-dependent metrics (at specified threshold)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "recall_at_fpr_1pct": recall_at_fpr_1pct,
        "f1": f1,
        "macro_f1": macro_f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "raw": cm.tolist(),
        },
        "threshold": threshold,
    }


def aggregate_fold_metrics(fold_metrics_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metric dictionaries across all cross-validation folds.

    Computes mean and sample standard deviation for each numerical metric.

    Args:
        fold_metrics_list: List of metric dicts returned by compute_classification_metrics.

    Returns:
        Dict mapping metric name to dict with 'mean' and 'std'.
    """
    keys_to_aggregate = [
        "pr_auc",
        "roc_auc",
        "recall_at_fpr_1pct",
        "f1",
        "macro_f1",
        "precision",
        "recall",
    ]

    aggregated = {}
    for key in keys_to_aggregate:
        values = [fold[key] for fold in fold_metrics_list if key in fold]
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }

    return aggregated


def format_metrics_table(aggregated_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Format aggregated metrics into a clean reporting DataFrame."""
    rows = []
    display_names = {
        "pr_auc": "PR-AUC (Average Precision)",
        "roc_auc": "ROC-AUC",
        "recall_at_fpr_1pct": "Recall @ FPR <= 1%",
        "f1": "F1-Score (Fraud Class)",
        "macro_f1": "Macro F1-Score",
        "precision": "Precision",
        "recall": "Recall (Sensitivity)",
    }

    for key, name in display_names.items():
        if key in aggregated_dict:
            mean_val = aggregated_dict[key]["mean"]
            std_val = aggregated_dict[key]["std"]
            rows.append({
                "Metric": name,
                "Mean": round(mean_val, 4),
                "Std Dev": round(std_val, 4),
                "Summary": f"{mean_val:.4f} ± {std_val:.4f}",
            })

    return pd.DataFrame(rows)
