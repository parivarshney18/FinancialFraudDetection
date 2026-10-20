"""
Safe Behavioral Feature Engineering Pipeline for Phase 5.
Extracts causally ordered, leakage-free temporal and contextual behavioral features:
1. Category-relative and City-relative spending deviation ratios.
2. Inter-transaction elapsed time within City and Merchant Category.
3. Causal rolling transaction velocity (1-hour density) by City and Category.
4. Intra-month temporal phase indicators.

Guarantees:
- Strictly causal: only transactions prior to the current transaction are used.
- Zero target leakage: Is_Fraud is never used to construct any feature.
- Preserves raw dataset immutability.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

from ai.src.config import (
    DROP_COLS,
    CATEGORICAL_COLS,
    TARGET_COL,
)
from ai.src.preprocessing import DEVICE_COMPATIBILITY


def engineer_causal_behavioral_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer causally sorted contextual and behavioral features from raw transaction DataFrame.

    Args:
        df: Raw banking transactions DataFrame.

    Returns:
        pd.DataFrame: Augmented DataFrame sorted chronologically with newly engineered behavioral features.
    """
    df_out = df.copy()

    # Parse full timestamp
    df_out["datetime"] = pd.to_datetime(
        df_out["Transaction_Date"] + " " + df_out["Transaction_Time"],
        format="%d-%m-%Y %H:%M:%S",
        dayfirst=True,
    )

    # Sort strictly chronologically to enforce causality
    df_out = df_out.sort_values("datetime").reset_index(drop=True)

    # 1. Month Phase (Early: 1-10, Mid: 11-20, Late: 21-31)
    day = df_out["datetime"].dt.day
    df_out["Tx_Day_Phase"] = np.where(day <= 10, 0, np.where(day <= 20, 1, 2))

    # 2. Causal Category Spending Ratio (Amount vs expanding prior mean in category)
    cat_cumsum = df_out.groupby("Merchant_Category")["Transaction_Amount"].cumsum() - df_out["Transaction_Amount"]
    cat_cumcount = df_out.groupby("Merchant_Category").cumcount()
    cat_prior_mean = np.where(cat_cumcount > 0, cat_cumsum / cat_cumcount, df_out["Transaction_Amount"])
    df_out["Cat_Amount_Ratio"] = df_out["Transaction_Amount"] / (cat_prior_mean + 1.0)

    # 3. Causal City Spending Ratio (Amount vs expanding prior mean in city)
    city_cumsum = df_out.groupby("City")["Transaction_Amount"].cumsum() - df_out["Transaction_Amount"]
    city_cumcount = df_out.groupby("City").cumcount()
    city_prior_mean = np.where(city_cumcount > 0, city_cumsum / city_cumcount, df_out["Transaction_Amount"])
    df_out["City_Amount_Ratio"] = df_out["Transaction_Amount"] / (city_prior_mean + 1.0)

    # 4. Inter-Transaction Elapsed Time (seconds) within same category
    prev_cat_time = df_out.groupby("Merchant_Category")["datetime"].shift(1)
    df_out["Time_Since_Prev_Cat_Tx"] = (df_out["datetime"] - prev_cat_time).dt.total_seconds().fillna(3600.0)

    # 5. Inter-Transaction Elapsed Time (seconds) within same city
    prev_city_time = df_out.groupby("City")["datetime"].shift(1)
    df_out["Time_Since_Prev_City_Tx"] = (df_out["datetime"] - prev_city_time).dt.total_seconds().fillna(86400.0)

    # 6. Causal 1-Hour Rolling Transaction Density by Category
    # Compute count of transactions in last 1 hour strictly before current transaction
    cat_density = []
    for cat, group in df_out.groupby("Merchant_Category"):
        # rolling 1h count on index
        counts = group.rolling("1h", on="datetime", closed="left")["Transaction_Amount"].count()
        cat_density.append(pd.Series(counts, index=group.index))
    df_out["Cat_Tx_Density_1h"] = pd.concat(cat_density).sort_index().fillna(0)

    # 7. Causal 1-Hour Rolling Transaction Density by City
    city_density = []
    for city, group in df_out.groupby("City"):
        counts = group.rolling("1h", on="datetime", closed="left")["Transaction_Amount"].count()
        city_density.append(pd.Series(counts, index=group.index))
    df_out["City_Tx_Density_1h"] = pd.concat(city_density).sort_index().fillna(0)

    return df_out


REVISED_CATEGORICAL_COLS = list(CATEGORICAL_COLS)

REVISED_NUMERICAL_COLS = [
    # Baseline numerical features
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
    # Newly engineered safe behavioral features
    "Tx_Day_Phase",
    "Cat_Amount_Ratio",
    "City_Amount_Ratio",
    "Time_Since_Prev_Cat_Tx",
    "Time_Since_Prev_City_Tx",
    "Cat_Tx_Density_1h",
    "City_Tx_Density_1h",
]


class RevisedFraudPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor integrating baseline features with newly engineered behavioral features.

    Maintains strict fold isolation: RobustScaler and OrdinalEncoder are fitted strictly on training data.
    """

    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        drop_cols: Optional[List[str]] = None,
        target_col: str = TARGET_COL,
        scale_numerical: bool = True,
    ):
        self.categorical_cols = list(categorical_cols or REVISED_CATEGORICAL_COLS)
        self.numerical_cols = list(numerical_cols or REVISED_NUMERICAL_COLS)
        self.drop_cols = list(drop_cols or DROP_COLS)
        self.target_col = target_col
        self.scale_numerical = scale_numerical

        self.scaler_: Optional[RobustScaler] = None
        self.encoder_: Optional[OrdinalEncoder] = None
        self.cat_dims: List[int] = []
        self.cat_idxs: List[int] = []
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "RevisedFraudPreprocessor":
        """Fit encoders and scalers strictly on training data."""
        # Derive baseline features
        X_eng = self._derive_baseline_features(X)

        # Categorical encoder
        cat_data = X_eng[self.categorical_cols].astype(str)
        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float32,
        )
        self.encoder_.fit(cat_data)
        self.cat_dims = [len(categories) for categories in self.encoder_.categories_]
        self.cat_idxs = list(range(len(self.categorical_cols)))

        # Numerical scaler
        if self.scale_numerical:
            num_data = X_eng[self.numerical_cols].values
            self.scaler_ = RobustScaler()
            self.scaler_.fit(num_data)

        self.feature_names_ = list(self.categorical_cols) + list(self.numerical_cols)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using training-fitted objects."""
        X_eng = self._derive_baseline_features(X)

        # Transform categoricals
        cat_data = X_eng[self.categorical_cols].astype(str)
        cat_encoded = self.encoder_.transform(cat_data)
        # Shift unknown (-1) to highest category index for embedding lookup
        for col_idx, cardinality in enumerate(self.cat_dims):
            col_vals = cat_encoded[:, col_idx]
            col_vals[col_vals == -1] = cardinality
            cat_encoded[:, col_idx] = col_vals

        # Transform numericals
        num_data = X_eng[self.numerical_cols].values.astype(np.float32)
        if self.scale_numerical and self.scaler_ is not None:
            num_scaled = self.scaler_.transform(num_data).astype(np.float32)
        else:
            num_scaled = num_data

        return np.hstack([cat_encoded.astype(np.float32), num_scaled.astype(np.float32)])

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _derive_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all baseline financial and temporal features are populated."""
        data = df.copy()

        # Balance indicators
        amt = pd.to_numeric(data["Transaction_Amount"], errors="coerce").fillna(0.0)
        bal = pd.to_numeric(data["Account_Balance"], errors="coerce").fillna(0.0)
        data["Amount_to_Balance_Ratio"] = (amt / (bal + 1.0)).astype(np.float32)
        data["Amount_Exceeds_Balance"] = (amt > bal).astype(np.float32)

        # Temporal indicators if not already extracted
        if "Tx_Hour" not in data.columns:
            dt = pd.to_datetime(
                data["Transaction_Date"] + " " + data["Transaction_Time"],
                format="%d-%m-%Y %H:%M:%S",
                dayfirst=True,
                errors="coerce",
            )
            data["Tx_Hour"] = dt.dt.hour.fillna(12).astype(np.float32)
            data["Tx_Day"] = dt.dt.day.fillna(15).astype(np.float32)
            data["Tx_DayOfWeek"] = dt.dt.dayofweek.fillna(0).astype(np.float32)
            data["Is_Weekend"] = (data["Tx_DayOfWeek"] >= 5).astype(np.float32)
            data["Is_Night"] = data["Tx_Hour"].isin([23, 0, 1, 2, 3, 4, 5]).astype(np.float32)
            hour_rad = (2.0 * np.pi * data["Tx_Hour"]) / 24.0
            data["Hour_Sin"] = np.sin(hour_rad).astype(np.float32)
            data["Hour_Cos"] = np.cos(hour_rad).astype(np.float32)

        # Device mismatch flag
        if "Device_Mismatch_Flag" not in data.columns:
            mismatches = []
            for dt_val, td_val in zip(data["Device_Type"], data["Transaction_Device"]):
                allowed = DEVICE_COMPATIBILITY.get(str(dt_val), set())
                mismatches.append(1.0 if str(td_val) not in allowed else 0.0)
            data["Device_Mismatch_Flag"] = np.array(mismatches, dtype=np.float32)

        return data
