"""
Machine learning artifact loader and singleton registry.
Caches TabNet model, fitted FraudPreprocessor, metadata, and TabNetExplainer for legacy synthetic dataset,
as well as production IEEE-CIS artifacts (TabNet, XGBoost, Random Forest, IEEEFraudPreprocessor).
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from backend.app.core.config import settings
from ai.src.models.tabnet_model import TabNetFraudModel
from ai.src.preprocessing import FraudPreprocessor
from ai.src.preprocessing_ieee import (
    IEEEFraudPreprocessor,
    IEEE_FEATURE_COLS,
    IEEE_CATEGORICAL_COLS,
    IEEE_NUMERICAL_COLS,
)
from ai.src.explain import TabNetExplainer


class MLRegistry:
    """Singleton container managing active machine learning artifacts for legacy and IEEE-CIS pipelines."""

    def __init__(self):
        # Legacy Synthetic Artifacts (Phase 4)
        self.model: Optional[TabNetFraudModel] = None
        self.preprocessor: Optional[FraudPreprocessor] = None
        self.explainer: Optional[TabNetExplainer] = None
        self.metadata: Dict[str, Any] = {}
        self.is_loaded: bool = False

        # IEEE-CIS Production Artifacts (Phase 9)
        self.ieee_tabnet: Optional[TabNetFraudModel] = None
        self.ieee_xgboost: Optional[xgb.XGBClassifier] = None
        self.ieee_random_forest: Optional[Any] = None
        self.ieee_logistic_regression: Optional[Any] = None
        self.ieee_preprocessor: Optional[IEEEFraudPreprocessor] = None
        self.ieee_metadata: Dict[str, Any] = {}
        self.ieee_explainer: Optional[Any] = None
        self.ieee_thresholds: Dict[str, float] = {
            "tabnet": settings.IEEE_THRESHOLD_TABNET,
            "xgboost": settings.IEEE_THRESHOLD_XGBOOST,
            "random_forest": settings.IEEE_THRESHOLD_RANDOM_FOREST,
            "logistic_regression": settings.IEEE_THRESHOLD_LOGISTIC_REGRESSION,
        }
        self.is_ieee_loaded: bool = False

    @property
    def ieee_models(self) -> Dict[str, Any]:
        """Dictionary mapping model names to loaded IEEE-CIS model instances."""
        return {
            "tabnet": self.ieee_tabnet,
            "xgboost": self.ieee_xgboost,
            "random_forest": self.ieee_random_forest,
            "logistic_regression": self.ieee_logistic_regression,
        }

    def load_legacy(self) -> bool:
        """Load legacy Phase 4 synthetic dataset model, preprocessor, and metadata from disk."""
        try:
            model_path = settings.MODEL_PATH
            prep_path = settings.PREPROCESSOR_PATH
            meta_path = settings.METADATA_PATH

            if not model_path.exists():
                print(f"[MLRegistry] Legacy model file not found at {model_path}")
                return False

            if not prep_path.exists():
                print(f"[MLRegistry] Legacy preprocessor file not found at {prep_path}")
                return False

            # Load Preprocessor
            self.preprocessor = joblib.load(prep_path)

            # Load TabNet Model
            self.model = TabNetFraudModel.load_from_file(model_path)

            # Load Metadata
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {
                    "model_type": "TabNetClassifier",
                    "n_steps": 3,
                    "optimal_threshold": settings.DEFAULT_DECISION_THRESHOLD,
                }

            # Initialize Explainer
            self.explainer = TabNetExplainer(
                model=self.model,
                feature_names=self.preprocessor.feature_names_,
                preprocessor=self.preprocessor,
            )

            self.is_loaded = True
            print("[MLRegistry] Successfully loaded legacy TabNet, preprocessor, and explainer artifacts.")
            return True
        except Exception as e:
            print(f"[MLRegistry] Error loading legacy ML artifacts: {e}")
            self.is_loaded = False
            return False

    def load_ieee(self) -> bool:
        """Load Phase 9 IEEE-CIS production artifacts (TabNet, XGBoost, Preprocessor, Metadata)."""
        try:
            prep_path = settings.IEEE_PREPROCESSOR_PATH
            tabnet_path = settings.IEEE_TABNET_MODEL_PATH
            xgb_path = settings.IEEE_XGBOOST_MODEL_PATH
            rf_path = settings.IEEE_RANDOM_FOREST_MODEL_PATH
            lr_path = settings.IEEE_LOGISTIC_REGRESSION_MODEL_PATH
            meta_path = settings.IEEE_METADATA_PATH
            chrono_path = settings.IEEE_CHRONO_METRICS_PATH

            if not prep_path.exists():
                print(f"[MLRegistry] IEEE-CIS preprocessor not found at {prep_path}")
                return False

            # 1. Load Preprocessor
            self.ieee_preprocessor = joblib.load(prep_path)
            if not getattr(self.ieee_preprocessor, "is_fitted_", False):
                print("[MLRegistry] IEEE-CIS preprocessor is not fitted!")
                return False

            # Validate feature count and ordering
            expected_count = len(IEEE_FEATURE_COLS)
            actual_count = len(self.ieee_preprocessor.feature_cols)
            if actual_count != expected_count:
                print(f"[MLRegistry] Preprocessor feature mismatch: expected {expected_count}, got {actual_count}")
                return False

            # 2. Load TabNet
            if tabnet_path.exists():
                self.ieee_tabnet = TabNetFraudModel.load_from_file(tabnet_path)
                print(f"[MLRegistry] Loaded IEEE-CIS TabNet from {tabnet_path.name}")
            else:
                print(f"[MLRegistry] IEEE-CIS TabNet not found at {tabnet_path}")

            # 3. Load XGBoost
            if xgb_path.exists():
                xgb_model = xgb.XGBClassifier()
                xgb_model.load_model(str(xgb_path))
                self.ieee_xgboost = xgb_model
                print(f"[MLRegistry] Loaded IEEE-CIS XGBoost from {xgb_path.name}")
            else:
                print(f"[MLRegistry] IEEE-CIS XGBoost not found at {xgb_path}")

            # 4. Load Random Forest (optional baseline)
            if rf_path.exists():
                self.ieee_random_forest = joblib.load(rf_path)
                print(f"[MLRegistry] Loaded IEEE-CIS Random Forest from {rf_path.name}")

            # 5. Load Logistic Regression (optional baseline)
            if lr_path.exists():
                self.ieee_logistic_regression = joblib.load(lr_path)
                print(f"[MLRegistry] Loaded IEEE-CIS Logistic Regression from {lr_path.name}")

            # 6. Load Metadata
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.ieee_metadata = json.load(f)

            # 7. Load Calibrated Thresholds from Chronological Evaluation
            if chrono_path.exists():
                try:
                    with open(chrono_path, "r", encoding="utf-8") as f:
                        chrono_data = json.load(f)
                    for m_key in ["tabnet", "xgboost", "random_forest", "logistic_regression"]:
                        if m_key in chrono_data and "metrics" in chrono_data[m_key]:
                            th = chrono_data[m_key]["metrics"].get("threshold_tuned")
                            if th is not None:
                                self.ieee_thresholds[m_key] = float(th)
                except Exception as ex:
                    print(f"[MLRegistry] Notice: Could not parse chronological thresholds ({ex}), using defaults.")

            # 8. Initialize IEEE-CIS Explainer Engine
            if self.ieee_tabnet is not None and self.ieee_preprocessor is not None:
                try:
                    from ai.src.explain_ieee import IEEEFraudExplainer
                    self.ieee_explainer = IEEEFraudExplainer(
                        model=self.ieee_tabnet,
                        preprocessor=self.ieee_preprocessor,
                        xgboost_model=self.ieee_xgboost,
                    )
                    print("[MLRegistry] Successfully initialized IEEE-CIS explainer engine.")
                except Exception as ex:
                    print(f"[MLRegistry] Warning: Could not initialize IEEE explainer: {ex}")

            # Require preprocessor and at least one model (TabNet or XGBoost)
            if self.ieee_tabnet is not None or self.ieee_xgboost is not None:
                self.is_ieee_loaded = True
                print("[MLRegistry] Successfully loaded IEEE-CIS production artifacts.")
                return True
            else:
                self.is_ieee_loaded = False
                return False

        except Exception as e:
            print(f"[MLRegistry] Error loading IEEE-CIS ML artifacts: {e}")
            self.is_ieee_loaded = False
            return False

    def load(self) -> bool:
        """Load both legacy and IEEE-CIS models and preprocessors from disk."""
        legacy_ok = self.load_legacy()
        ieee_ok = self.load_ieee()
        return legacy_ok or ieee_ok

    def preprocess_ieee(self, df: pd.DataFrame) -> np.ndarray:
        """Apply IEEE-CIS preprocessor exactly once to input DataFrame.

        Validates input structure, runs causal feature engineering and transforms,
        and returns preprocessed NumPy array of shape (N, 46) and dtype float32.
        """
        if not self.is_ieee_loaded or self.ieee_preprocessor is None:
            raise RuntimeError("IEEE-CIS preprocessor is not loaded.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(df)}")

        X_proc = self.ieee_preprocessor.transform(df)
        if isinstance(X_proc, pd.DataFrame):
            X_proc = X_proc.values

        X_arr = np.asarray(X_proc, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        expected_features = len(IEEE_FEATURE_COLS)
        if X_arr.shape[1] != expected_features:
            raise ValueError(
                f"Preprocessed feature dimension mismatch: Expected {expected_features} features, got {X_arr.shape[1]}."
            )

        return X_arr

    def predict_ieee_proba(
        self, X_proc: np.ndarray, model_name: str = "tabnet"
    ) -> np.ndarray:
        """Score preprocessed IEEE-CIS feature matrix with specified model.

        Args:
            X_proc: (N, 46) preprocessed float32 matrix.
            model_name: 'tabnet', 'xgboost', 'random_forest', or 'logistic_regression'.

        Returns:
            np.ndarray: Predicted probabilities of shape (N, 2).
        """
        model_key = model_name.lower().strip()
        if model_key == "tabnet":
            if self.ieee_tabnet is None:
                raise RuntimeError("IEEE-CIS TabNet model is not loaded.")
            return self.ieee_tabnet.predict_proba(X_proc)

        elif model_key == "xgboost":
            if self.ieee_xgboost is None:
                raise RuntimeError("IEEE-CIS XGBoost model is not loaded.")
            return self.ieee_xgboost.predict_proba(X_proc)

        elif model_key in ["random_forest", "rf"]:
            if self.ieee_random_forest is None:
                raise RuntimeError("IEEE-CIS Random Forest model is not loaded.")
            return self.ieee_random_forest.predict_proba(X_proc)

        elif model_key in ["logistic_regression", "lr"]:
            if self.ieee_logistic_regression is None:
                raise RuntimeError("IEEE-CIS Logistic Regression model is not loaded.")
            return self.ieee_logistic_regression.predict_proba(X_proc)

        else:
            raise ValueError(f"Unknown or unsupported IEEE-CIS model: {model_name}")

    def get_ieee_threshold(self, model_name: str = "tabnet") -> float:
        """Retrieve calibrated decision threshold for specified IEEE-CIS model."""
        model_key = model_name.lower().strip()
        return self.ieee_thresholds.get(model_key, settings.IEEE_DEFAULT_THRESHOLD)


# Global singleton instance
ml_registry = MLRegistry()


def get_ml_registry() -> MLRegistry:
    """Retrieve global MLRegistry instance, loading artifacts if needed."""
    if not ml_registry.is_loaded and not ml_registry.is_ieee_loaded:
        ml_registry.load()
    return ml_registry

