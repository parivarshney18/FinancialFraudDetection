"""
TabNet architecture wrapper for Financial Fraud Detection.
Integrates PyTorch-TabNet with automatic embedding configuration,
class-weighted objective, and native attention mask explainability.
"""

from typing import List, Optional, Tuple, Dict, Any, Union
from pathlib import Path
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from ai.src.config import RANDOM_SEED


def compute_recommended_embedding_dims(cat_dims: List[int], max_emb: int = 16) -> List[int]:
    """Compute standard tabular embedding dimensions based on cardinality.

    Rule of thumb: min(max_emb, (cardinality + 1) // 2), with minimum of 1.

    Args:
        cat_dims: List of category counts per categorical feature.
        max_emb: Upper cap on embedding dimensionality.

    Returns:
        List of embedding dimensions corresponding to each categorical feature.
    """
    return [max(1, min(max_emb, (dim + 1) // 2)) for dim in cat_dims]


def build_tabnet_model(
    cat_idxs: List[int],
    cat_dims: List[int],
    cat_emb_dim: Optional[List[int]] = None,
    n_d: int = 16,
    n_a: int = 16,
    n_steps: int = 5,
    gamma: float = 1.3,
    lambda_sparse: float = 1e-3,
    learning_rate: float = 2e-2,
    mask_type: str = "sparsemax",
    seed: int = RANDOM_SEED,
    device_name: str = "cpu",
    **kwargs: Any,
) -> TabNetClassifier:
    """Instantiate a TabNetClassifier configured for imbalanced tabular banking fraud detection.

    Args:
        cat_idxs: Indices of categorical columns in the feature matrix.
        cat_dims: Cardinality of each categorical feature.
        cat_emb_dim: Embedding dimensions (computed automatically if None).
        n_d: Dimensionality of prediction layer.
        n_a: Dimensionality of attention layer.
        n_steps: Number of sequential decision steps.
        gamma: Coefficient for feature reuse in masks.
        lambda_sparse: Sparsity regularization coefficient.
        learning_rate: Initial Adam learning rate.
        mask_type: Attention mask ('sparsemax' or 'entmax').
        seed: Random seed for reproducibility.
        device_name: Target compute device ('cpu' or 'cuda').
        **kwargs: Additional parameters passed to TabNetClassifier.

    Returns:
        Configured TabNetClassifier instance.
    """
    if cat_emb_dim is None:
        if len(cat_dims) > 0:
            cat_emb_dim = compute_recommended_embedding_dims(cat_dims)
        else:
            cat_emb_dim = []

    return TabNetClassifier(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=cat_emb_dim,
        lambda_sparse=lambda_sparse,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": learning_rate},
        mask_type=mask_type,
        seed=seed,
        verbose=0,
        device_name=device_name,
        **kwargs,
    )


class TabNetFraudModel:
    """High-level wrapper around TabNetClassifier for simplified training, inference, and explainability."""

    def __init__(
        self,
        cat_idxs: Optional[List[int]] = None,
        cat_dims: Optional[List[int]] = None,
        cat_emb_dim: Optional[List[int]] = None,
        n_d: int = 16,
        n_a: int = 16,
        n_steps: int = 5,
        gamma: float = 1.3,
        lambda_sparse: float = 1e-3,
        learning_rate: float = 2e-2,
        seed: int = RANDOM_SEED,
        device_name: str = "cpu",
    ):
        self.cat_idxs = list(cat_idxs) if cat_idxs is not None else []
        self.cat_dims = list(cat_dims) if cat_dims is not None else []
        self.cat_emb_dim = cat_emb_dim
        self.classifier = build_tabnet_model(
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=cat_emb_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            learning_rate=learning_rate,
            seed=seed,
            device_name=device_name,
        )
        self.is_fitted_ = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        max_epochs: int = 15,
        patience: int = 5,
        batch_size: int = 2048,
        virtual_batch_size: int = 256,
        weights: int = 1,
    ) -> "TabNetFraudModel":
        """Fit TabNet with early stopping and class-weighted cross-entropy.

        Args:
            X_train: Preprocessed training features.
            y_train: Training labels (0 or 1).
            eval_set: Optional validation sets [(X_val, y_val)].
            max_epochs: Maximum epochs.
            patience: Early stopping patience.
            batch_size: Mini-batch size.
            virtual_batch_size: Virtual mini-batch size for Ghost Batch Normalization.
            weights: 1 to auto-balance class weights in BCE loss, or 0 for unweighted.

        Returns:
            self
        """
        # Ensure float32 format
        X_tr = np.asarray(X_train, dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=np.int64)

        formatted_eval = None
        if eval_set:
            formatted_eval = [
                (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64))
                for X, y in eval_set
            ]

        self.classifier.fit(
            X_train=X_tr,
            y_train=y_tr,
            eval_set=formatted_eval,
            eval_name=["val"] if formatted_eval else None,
            eval_metric=["auc"],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            weights=weights,
            drop_last=False,
            pin_memory=False,
        )
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels using decision threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_arr = np.asarray(X, dtype=np.float32)
        return self.classifier.predict_proba(X_arr)

    def explain(
        self, X: np.ndarray, normalize: bool = False
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Extract instance-level feature importance and sequential decision masks.

        Args:
            X: Input feature array.
            normalize: Whether to normalize explain matrix so that sum of features equals 1.

        Returns:
            Tuple of:
            - explain_matrix: Aggregated importance per feature (shape [N, D]).
            - masks: Dictionary mapping step index (0..n_steps-1) to step mask (shape [N, D]).
        """
        X_arr = np.asarray(X, dtype=np.float32)
        return self.classifier.explain(X_arr, normalize=normalize)

    def save_model(self, filepath: Union[str, Path]) -> Path:
        """Save model state to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        # TabNetClassifier save_model adds .zip extension automatically
        saved_path = self.classifier.save_model(str(path))
        return Path(saved_path)

    def load_model(self, filepath: Union[str, Path]) -> None:
        """Load model state from disk."""
        self.classifier.load_model(str(filepath))
        self.cat_idxs = list(self.classifier.cat_idxs)
        self.cat_dims = list(self.classifier.cat_dims)
        self.is_fitted_ = True

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> "TabNetFraudModel":
        """Instantiate and load a TabNetFraudModel directly from a saved .zip file."""
        instance = cls(cat_idxs=[], cat_dims=[])
        instance.load_model(filepath)
        return instance

