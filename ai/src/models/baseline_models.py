"""
Traditional machine learning baseline models for Financial Fraud Detection:
Logistic Regression and Random Forest.
"""

from typing import Optional, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ai.src.config import RANDOM_SEED


def build_logistic_regression(
    class_weight: Optional[str] = "balanced",
    max_iter: int = 500,
    C: float = 1.0,
    random_state: int = RANDOM_SEED,
    **kwargs: Any,
) -> LogisticRegression:
    """Build Logistic Regression baseline model with imbalanced class weighting.

    Args:
        class_weight: Strategy for handling class imbalance (default 'balanced').
        max_iter: Maximum optimization iterations.
        C: Inverse regularization strength.
        random_state: Seed for reproducibility.
        **kwargs: Additional parameters passed to LogisticRegression.

    Returns:
        Configured LogisticRegression estimator.
    """
    return LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        C=C,
        solver="lbfgs",
        random_state=random_state,
        **kwargs,
    )


def build_random_forest(
    n_estimators: int = 100,
    max_depth: Optional[int] = 12,
    class_weight: Optional[str] = "balanced_subsample",
    min_samples_split: int = 10,
    min_samples_leaf: int = 4,
    n_jobs: int = 4,
    random_state: int = RANDOM_SEED,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Build Random Forest baseline ensemble with balanced bootstrap subsampling.

    Args:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth (bounded for memory and speed on CPU).
        class_weight: Handling of class imbalance ('balanced_subsample' or 'balanced').
        min_samples_split: Minimum samples to split an internal node.
        min_samples_leaf: Minimum samples required at a leaf node.
        n_jobs: Number of CPU worker threads.
        random_state: Seed for reproducibility.
        **kwargs: Additional parameters passed to RandomForestClassifier.

    Returns:
        Configured RandomForestClassifier estimator.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state,
        **kwargs,
    )
