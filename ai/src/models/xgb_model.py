"""
XGBoost baseline model wrapper for Financial Fraud Detection.
Configured with histogram-based boosting, class weighting, and PR-AUC early stopping.
"""

from typing import Optional, Any
import xgboost as xgb
from ai.src.config import RANDOM_SEED


def build_xgboost_model(
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    scale_pos_weight: float = 18.8255,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    tree_method: str = "hist",
    n_jobs: int = 4,
    random_state: int = RANDOM_SEED,
    **kwargs: Any,
) -> xgb.XGBClassifier:
    """Build tuned XGBoost classifier for imbalanced fraud classification.

    Args:
        n_estimators: Maximum boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Shrinkage factor per step.
        scale_pos_weight: Weighting factor for fraud class (189,912 / 10,088 = 18.8255).
        subsample: Row subsampling fraction per tree.
        colsample_bytree: Feature subsampling fraction per tree.
        tree_method: Histogram boosting ('hist') optimized for multi-core CPU efficiency.
        n_jobs: Number of parallel CPU workers.
        random_state: Seed for reproducibility.
        **kwargs: Additional parameters passed to XGBClassifier.

    Returns:
        Configured XGBClassifier instance.
    """
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method=tree_method,
        eval_metric="aucpr",
        n_jobs=n_jobs,
        random_state=random_state,
        **kwargs,
    )
