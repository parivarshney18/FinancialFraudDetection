"""
Feature engineering, scaling, and categorical encoding pipeline for Fraud Detection.
Implements leak-free preprocessing conforming to docs/preprocessing_plan.md.
"""

from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

from ai.src.config import (
    DROP_COLS,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    TARGET_COL,
)

# Compatibility map between high-level Device_Type and granular Transaction_Device
DEVICE_COMPATIBILITY: Dict[str, set] = {
    "ATM": {
        "ATM",
        "ATM Booth Kiosk",
        "Self-service Banking Machine",
        "Bank Branch",
    },
    "Mobile": {
        "Mobile Device",
        "POS Mobile App",
        "POS Mobile Device",
        "Tablet",
        "Wearable Device",
        "Banking Chatbot",
        "Voice Assistant",
        "QR Code Scanner",
    },
    "Desktop": {
        "Desktop/Laptop",
        "Web Browser",
        "Banking Chatbot",
    },
    "POS": {
        "POS Terminal",
        "POS Mobile Device",
        "POS Mobile App",
        "QR Code Scanner",
        "Biometric Scanner",
        "Debit/Credit Card",
        "Smart Card",
        "Virtual Card",
        "Payment Gateway Device",
    },
}

# Explicit feature taxonomy after feature engineering
PROCESSED_CATEGORICAL_COLS = list(CATEGORICAL_COLS)

PROCESSED_NUMERICAL_COLS = [
    "Age",
    "Transaction_Amount",
    "Account_Balance",
    "Amount_to_Balance_Ratio",
    "Tx_Day",
    "Tx_Hour",
    "Tx_DayOfWeek",
    "Hour_Sin",
    "Hour_Cos",
    "Amount_Exceeds_Balance",
    "Is_Weekend",
    "Is_Night",
    "Device_Mismatch_Flag",
]


class FraudPreprocessor(BaseEstimator, TransformerMixin):
    """Leak-free feature engineer, scaler, and ordinal encoder for tabular fraud detection.

    Designed for seamless integration with PyTorch TabNet and tree-based baselines.
    Ensures that all statistics, imputers, scalers, and encoders are fitted solely on
    training data, preventing target and validation data leakage.
    """

    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        drop_cols: Optional[List[str]] = None,
        target_col: str = TARGET_COL,
        scale_numerical: bool = True,
    ):
        self.categorical_cols = list(categorical_cols or PROCESSED_CATEGORICAL_COLS)
        self.numerical_cols = list(numerical_cols or PROCESSED_NUMERICAL_COLS)
        self.drop_cols = list(drop_cols or DROP_COLS)
        self.target_col = target_col
        self.scale_numerical = scale_numerical

        # Fitted transformer attributes
        self.scaler_: Optional[RobustScaler] = None
        self.encoder_: Optional[OrdinalEncoder] = None
        self.cat_dims: List[int] = []
        self.cat_idxs: List[int] = []
        self.unknown_category_values_: Dict[int, int] = {}
        self.feature_names_: List[str] = []

        # Training imputation baselines
        self.num_impute_values_: Dict[str, float] = {}
        self.cat_impute_values_: Dict[str, str] = {}
        self.date_default_: pd.Timestamp = pd.Timestamp("2025-01-15")
        self.hour_default_: float = 12.0

        self.is_fitted_: bool = False

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive interaction, financial, temporal, and device mismatch features.

        Guarantees input df is not mutated by copying. Excludes target and drop columns.
        """
        data = df.copy()

        # Strictly exclude target column from input features to avoid leakage
        if self.target_col in data.columns:
            data = data.drop(columns=[self.target_col])

        # Drop identifier and target-leakage columns if present
        cols_to_drop = [c for c in self.drop_cols if c in data.columns]
        if cols_to_drop:
            data = data.drop(columns=cols_to_drop)

        # 1. Financial Ratio & Balance Indicators
        if "Transaction_Amount" in data.columns and "Account_Balance" in data.columns:
            amt = pd.to_numeric(data["Transaction_Amount"], errors="coerce")
            bal = pd.to_numeric(data["Account_Balance"], errors="coerce")
            # Handle possible nulls or negatives safely
            bal_clipped = bal.fillna(0.0).clip(lower=0.0)
            amt_safe = amt.fillna(0.0)
            data["Amount_to_Balance_Ratio"] = (amt_safe / (bal_clipped + 1.0)).astype(float)
            data["Amount_Exceeds_Balance"] = (amt_safe > bal_clipped).astype(float)
        else:
            data["Amount_to_Balance_Ratio"] = 0.0
            data["Amount_Exceeds_Balance"] = 0.0

        # 2. Temporal Features
        if "Transaction_Date" in data.columns:
            parsed_dates = pd.to_datetime(data["Transaction_Date"], format="%d-%m-%Y", errors="coerce")
            if parsed_dates.isna().any():
                fallback_dates = pd.to_datetime(data["Transaction_Date"], errors="coerce", dayfirst=True)
                parsed_dates = parsed_dates.fillna(fallback_dates)
            parsed_dates = parsed_dates.fillna(self.date_default_)
            data["Tx_Day"] = parsed_dates.dt.day.astype(float)
            data["Tx_DayOfWeek"] = parsed_dates.dt.dayofweek.astype(float)
            data["Is_Weekend"] = (parsed_dates.dt.dayofweek >= 5).astype(float)
            data = data.drop(columns=["Transaction_Date"])
        else:
            data["Tx_Day"] = 15.0
            data["Tx_DayOfWeek"] = 2.0
            data["Is_Weekend"] = 0.0

        if "Transaction_Time" in data.columns:
            parsed_times = pd.to_datetime(data["Transaction_Time"], format="%H:%M:%S", errors="coerce")
            if parsed_times.isna().any():
                fallback_times = pd.to_datetime(data["Transaction_Time"], errors="coerce")
                parsed_times = parsed_times.fillna(fallback_times)
            hours = parsed_times.dt.hour.fillna(self.hour_default_).astype(float)
            data["Tx_Hour"] = hours
            data["Is_Night"] = ((hours >= 23.0) | (hours <= 5.0)).astype(float)
            data["Hour_Sin"] = np.sin(2.0 * np.pi * hours / 24.0).astype(float)
            data["Hour_Cos"] = np.cos(2.0 * np.pi * hours / 24.0).astype(float)
            data = data.drop(columns=["Transaction_Time"])
        else:
            data["Tx_Hour"] = self.hour_default_
            data["Is_Night"] = 0.0
            data["Hour_Sin"] = 0.0
            data["Hour_Cos"] = 1.0

        # 3. Contextual Device Mismatch Indicator
        if "Device_Type" in data.columns and "Transaction_Device" in data.columns:
            mismatch = pd.Series(0.0, index=data.index)
            for dev_type, valid_set in DEVICE_COMPATIBILITY.items():
                mask = data["Device_Type"] == dev_type
                invalid_dev = mask & (~data["Transaction_Device"].isin(valid_set))
                mismatch.loc[invalid_dev] = 1.0
            data["Device_Mismatch_Flag"] = mismatch
        else:
            data["Device_Mismatch_Flag"] = 0.0

        return data

    def fit(self, X: pd.DataFrame, y=None) -> "FraudPreprocessor":
        """Compute training imputation statistics, RobustScaler, and OrdinalEncoder parameters.

        Args:
            X: Training feature DataFrame.
            y: Optional target (ignored for unsupervised transformation, but target column
               is safely dropped if present in X).

        Returns:
            self: The fitted FraudPreprocessor instance.
        """
        # Feature engineering on training data
        df = self._engineer_features(X)

        # Record and apply imputation values
        self.num_impute_values_ = {}
        for col in self.numerical_cols:
            if col in df.columns:
                med = float(df[col].median(skipna=True))
                self.num_impute_values_[col] = 0.0 if np.isnan(med) else med
            else:
                self.num_impute_values_[col] = 0.0

        self.cat_impute_values_ = {}
        for col in self.categorical_cols:
            if col in df.columns:
                mode_series = df[col].dropna().mode()
                mode_val = str(mode_series.iloc[0]) if not mode_series.empty else "Missing"
                self.cat_impute_values_[col] = mode_val
            else:
                self.cat_impute_values_[col] = "Missing"

        # Impute missing values
        df_num = df[self.numerical_cols].fillna(self.num_impute_values_)
        df_cat = df[self.categorical_cols].astype(str).fillna("Missing")

        # Fit RobustScaler on numerical features
        if self.scale_numerical:
            self.scaler_ = RobustScaler()
            self.scaler_.fit(df_num)
        else:
            self.scaler_ = None

        # Fit OrdinalEncoder on categorical features
        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float64,
        )
        self.encoder_.fit(df_cat)

        # Configure TabNet categorical dimensions and unknown index mapping
        self.cat_dims = []
        self.unknown_category_values_ = {}
        for idx, categories in enumerate(self.encoder_.categories_):
            k = len(categories)
            # Reserve index k for any unseen/unknown categories encountered during inference
            self.unknown_category_values_[idx] = k
            self.cat_dims.append(k + 1)

        # Categorical columns are placed first, followed by numerical columns
        self.cat_idxs = list(range(len(self.categorical_cols)))
        self.feature_names_ = list(self.categorical_cols) + list(self.numerical_cols)

        self.is_fitted_ = True
        return self

    def transform(
        self,
        X: pd.DataFrame,
        as_df: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Transform input features using training-fitted encoders and scalers.

        Args:
            X: Input DataFrame to transform.
            as_df: If True, returns a pandas DataFrame with feature_names_ as columns.
                   If False, returns a 2D float32 NumPy array ready for TabNet / PyTorch.

        Returns:
            Transformed features as a 2D NumPy array or DataFrame.
        """
        if not self.is_fitted_:
            raise RuntimeError("FraudPreprocessor must be fitted before calling transform.")

        df = self._engineer_features(X)

        # Impute missing using training parameters
        df_num = df[self.numerical_cols].fillna(self.num_impute_values_)
        df_cat = df[self.categorical_cols].astype(str).fillna("Missing")

        # Transform categoricals with unknown category mapping
        cat_encoded = self.encoder_.transform(df_cat)
        for idx, k in self.unknown_category_values_.items():
            cat_encoded[:, idx] = np.where(cat_encoded[:, idx] == -1, k, cat_encoded[:, idx])

        # Scale numerical features with training-fitted scaler
        if self.scale_numerical and self.scaler_ is not None:
            num_scaled = self.scaler_.transform(df_num)
        else:
            num_scaled = df_num.values

        # Concatenate: [Categorical Features, Numerical Features]
        X_out = np.hstack([cat_encoded, num_scaled]).astype(np.float32)

        if as_df:
            return pd.DataFrame(X_out, columns=self.feature_names_, index=X.index)

        return X_out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y=None,
        as_df: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Fit preprocessor and transform features in a single step."""
        return self.fit(X, y).transform(X, as_df=as_df)

    def get_feature_names(self) -> List[str]:
        """Return the complete ordered list of feature names."""
        if not self.is_fitted_:
            return list(self.categorical_cols) + list(self.numerical_cols)
        return self.feature_names_
