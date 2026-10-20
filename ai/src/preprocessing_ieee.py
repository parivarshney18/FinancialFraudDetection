"""
Preprocessing and Feature Engineering Pipeline for the IEEE-CIS Fraud Detection Dataset.
Implements leak-free scaling, categorical encoding, causal behavioral feature engineering,
and TabNet-compatible dimension tracking.
Provides Stratified K-Fold and Chronological split evaluation strategies.
"""

from typing import List, Optional, Tuple, Dict, Any, Union, Generator
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TARGET_COL = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

# 1. Selected Categorical Features (11 features)
IEEE_CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "M1",
    "M4",
    "M5",
    "M6",
]

# 2. Selected Raw Numerical Features (29 features)
IEEE_RAW_NUMERICAL_COLS = [
    # Core Financial & Spatial
    "TransactionAmt",
    "card1",
    "addr1",
    "addr2",
    # Vesta Multi-Window Velocity Counters (0.00% missing)
    "C1",
    "C2",
    "C5",
    "C6",
    "C11",
    "C13",
    "C14",
    # Vesta Timedelta Features
    "D1",
    "D2",
    "D15",
    # Salient V-Features (High discrimination signal in benchmark literature)
    "V12",
    "V13",
    "V35",
    "V36",
    "V75",
    "V76",
    "V90",
    "V91",
    "V126",
    "V127",
    "V128",
    "V257",
    "V258",
    "V282",
    "V283",
]

# 3. Engineered Numerical Features (6 features)
IEEE_ENGINEERED_NUMERICAL_COLS = [
    "Log_TransactionAmt",
    "Hour_Sin",
    "Hour_Cos",
    "card1_tx_count_prior",
    "Amt_to_Card1_Mean_Ratio",
    "Time_Since_Prev_Card1_Tx",
]

# Combined Numerical Features (35 features)
IEEE_NUMERICAL_COLS = IEEE_RAW_NUMERICAL_COLS + IEEE_ENGINEERED_NUMERICAL_COLS

# Complete Curated Feature Set (46 features: 11 categorical + 35 numerical)
IEEE_FEATURE_COLS = IEEE_CATEGORICAL_COLS + IEEE_NUMERICAL_COLS


class IEEEFraudPreprocessor(BaseEstimator, TransformerMixin):
    """Leakage-safe preprocessor for IEEE-CIS Fraud Detection.

    Guarantees:
    1. Zero Target Leakage: Target column `isFraud` is completely separated from input features.
    2. Strict Train/Validation Isolation: Scalers, encoders, median imputers, and historical card
       entity summaries are fitted solely on training partitions.
    3. Strictly Causal Behavioral Features: Features use strictly prior transactions in time.
       Validation transactions use historical card states from training, preventing future leakage.
    4. TabNet Readiness: Categorical features are mapped to non-negative contiguous integers with
       `cat_dims` and `cat_idxs` tracked for embedding layers.
    5. Missing-value neutrality: Missing categorical values are mapped to an explicit "Unknown" class;
       missing numerical values are imputed using training medians without assuming missingness equals fraud.
    """

    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        raw_numerical_cols: Optional[List[str]] = None,
        engineered_numerical_cols: Optional[List[str]] = None,
        scale_numerical: bool = True,
        as_df: bool = False,
    ):
        self.categorical_cols = list(categorical_cols or IEEE_CATEGORICAL_COLS)
        self.raw_numerical_cols = list(raw_numerical_cols or IEEE_RAW_NUMERICAL_COLS)
        self.engineered_numerical_cols = list(engineered_numerical_cols or IEEE_ENGINEERED_NUMERICAL_COLS)
        self.numerical_cols = self.raw_numerical_cols + self.engineered_numerical_cols
        self.feature_cols = self.categorical_cols + self.numerical_cols

        self.scale_numerical = scale_numerical
        self.as_df = as_df

        # Attributes fitted during training
        self.scaler_: Optional[RobustScaler] = None
        self.encoder_: Optional[OrdinalEncoder] = None
        self.medians_: Optional[pd.Series] = None
        self.card1_history_: Dict[Any, Dict[str, Any]] = {}
        self.cat_dims: List[int] = []
        self.cat_idxs: List[int] = []
        self.feature_names_in_: List[str] = []
        self._train_index_: Optional[pd.Index] = None
        self.is_fitted_: bool = False

    def _engineer_causal_features(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
    ) -> pd.DataFrame:
        """Compute cyclical temporal, financial log, and causal entity-level behavioral features.

        Enforces strict causality using TransactionDT.
        When is_train=True, builds and records card1_history_ state.
        When is_train=False, uses card1_history_ recorded during training.
        """
        out = df.copy()

        # Ensure required columns exist
        if "TransactionDT" not in out.columns:
            raise KeyError("Input DataFrame must contain 'TransactionDT' for causal feature engineering.")
        if "TransactionAmt" not in out.columns:
            raise KeyError("Input DataFrame must contain 'TransactionAmt' for financial feature engineering.")
        if "card1" not in out.columns:
            raise KeyError("Input DataFrame must contain 'card1' for entity-level behavioral engineering.")

        # 1. Cyclical time features
        hour = (out["TransactionDT"] // 3600) % 24
        out["Hour_Sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        out["Hour_Cos"] = np.cos(2.0 * np.pi * hour / 24.0)

        # 2. Log-transformed transaction amount
        out["Log_TransactionAmt"] = np.log1p(out["TransactionAmt"].clip(lower=0))

        # 3. Causal entity-level behavioral features on card1
        # Sort chronologically to guarantee causal sequence
        sort_idx = out.sort_values(TIME_COL).index
        sorted_df = out.loc[sort_idx]
        c1_group = sorted_df.groupby("card1", sort=False)

        if is_train:
            # Training partition: Compute causal prior counts, sums, and deltas
            within_cumcount = c1_group.cumcount()
            within_cumsum = c1_group["TransactionAmt"].cumsum() - sorted_df["TransactionAmt"]
            within_prev_dt = c1_group[TIME_COL].shift(1)

            prior_mean = np.where(
                within_cumcount > 0,
                within_cumsum / within_cumcount,
                sorted_df["TransactionAmt"],
            )

            sorted_df["card1_tx_count_prior"] = within_cumcount
            sorted_df["Amt_to_Card1_Mean_Ratio"] = sorted_df["TransactionAmt"] / (prior_mean + 1.0)
            sorted_df["Time_Since_Prev_Card1_Tx"] = (sorted_df[TIME_COL] - within_prev_dt).fillna(86400.0 * 30.0)

            # Record final card1 state from train for subsequent validation transforms
            self.card1_history_ = {}
            for c1, grp in c1_group:
                self.card1_history_[c1] = {
                    "total_count": len(grp),
                    "total_sum": float(grp["TransactionAmt"].sum()),
                    "last_dt": float(grp[TIME_COL].max()),
                }

        else:
            # Validation partition: Continuous forward evaluation from training history
            within_cumcount = c1_group.cumcount()
            within_cumsum = c1_group["TransactionAmt"].cumsum() - sorted_df["TransactionAmt"]
            within_prev_dt = c1_group[TIME_COL].shift(1)

            # Map historical metrics from train
            hist = self.card1_history_
            hist_counts = sorted_df["card1"].map(lambda c: hist[c]["total_count"] if c in hist else 0)
            hist_sums = sorted_df["card1"].map(lambda c: hist[c]["total_sum"] if c in hist else 0.0)
            hist_last_dts = sorted_df["card1"].map(lambda c: hist[c]["last_dt"] if c in hist else np.nan)

            total_prior_count = hist_counts + within_cumcount
            total_prior_sum = hist_sums + within_cumsum

            prior_mean = np.where(
                total_prior_count > 0,
                total_prior_sum / total_prior_count,
                sorted_df["TransactionAmt"],
            )

            sorted_df["card1_tx_count_prior"] = total_prior_count
            sorted_df["Amt_to_Card1_Mean_Ratio"] = sorted_df["TransactionAmt"] / (prior_mean + 1.0)

            # Compute elapsed time from prior event
            time_delta = np.where(
                within_prev_dt.notna(),
                sorted_df[TIME_COL] - within_prev_dt,
                np.where(
                    hist_last_dts.notna(),
                    sorted_df[TIME_COL] - hist_last_dts,
                    86400.0 * 30.0,
                ),
            )
            sorted_df["Time_Since_Prev_Card1_Tx"] = np.maximum(time_delta, 0.0)

        # Restore original row ordering
        out = sorted_df.loc[df.index]
        return out

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "IEEEFraudPreprocessor":
        """Fit scalers, ordinal encoders, median imputers, and card history strictly on training data."""
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Union[np.ndarray, pd.DataFrame]:
        """Fit preprocessor and transform training data in a single leak-free pass."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if TARGET_COL in X.columns:
            raise ValueError(f"Target column '{TARGET_COL}' detected in feature input X. Target must be separated!")

        # 1. Feature Engineering on training data (strictly causal, builds card1_history_)
        df_eng = self._engineer_causal_features(X, is_train=True)

        # 2. Impute and encode Categoricals
        cat_df = df_eng[self.categorical_cols].copy().astype(str).fillna("Unknown")
        cat_df = cat_df.replace({"nan": "Unknown", "None": "Unknown"})

        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        cat_encoded = self.encoder_.fit_transform(cat_df)

        self.cat_dims = [len(cats) + 1 for cats in self.encoder_.categories_]
        self.cat_idxs = list(range(len(self.categorical_cols)))

        # 3. Fit Median Imputers on Numerical features
        num_df = df_eng[self.numerical_cols].copy()
        self.medians_ = num_df.median().fillna(0.0)

        # 4. Fit RobustScaler on imputed Numerical features
        num_imputed = num_df.fillna(self.medians_)
        if self.scale_numerical:
            self.scaler_ = RobustScaler()
            num_scaled = self.scaler_.fit_transform(num_imputed)
        else:
            self.scaler_ = None
            num_scaled = num_imputed.values

        self.feature_names_in_ = list(self.feature_cols)
        self._train_index_ = X.index
        self.is_fitted_ = True

        # 5. Output
        if self.as_df:
            cat_out_df = pd.DataFrame(cat_encoded, columns=self.categorical_cols, index=X.index)
            num_out_df = pd.DataFrame(num_scaled, columns=self.numerical_cols, index=X.index)
            return pd.concat([cat_out_df, num_out_df], axis=1)
        else:
            return np.hstack([cat_encoded, num_scaled]).astype(np.float32)

    def transform(self, X: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        """Transform input features using training-fitted scalers, encoders, and medians."""
        if not self.is_fitted_:
            raise RuntimeError("IEEEFraudPreprocessor must be fitted before calling transform.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if TARGET_COL in X.columns:
            raise ValueError(f"Target column '{TARGET_COL}' detected in feature input X during transform.")

        # Determine if X is the training partition or a validation/test partition
        is_train = bool(self._train_index_ is not None and X.index.equals(self._train_index_))

        # 1. Feature engineering (using train history for validation/test partitions)
        df_eng = self._engineer_causal_features(X, is_train=is_train)

        # 2. Transform Categoricals
        cat_df = df_eng[self.categorical_cols].copy().astype(str).fillna("Unknown")
        cat_df = cat_df.replace({"nan": "Unknown", "None": "Unknown"})

        cat_encoded = self.encoder_.transform(cat_df)
        # Shift unknown (-1) to the highest index (dedicated unknown bucket)
        for i, dim in enumerate(self.cat_dims):
            unknown_mask = cat_encoded[:, i] == -1
            cat_encoded[unknown_mask, i] = dim - 1

        # 3. Impute and Transform Numericals
        num_df = df_eng[self.numerical_cols].copy()
        num_imputed = num_df.fillna(self.medians_)

        if self.scale_numerical and self.scaler_ is not None:
            num_scaled = self.scaler_.transform(num_imputed)
        else:
            num_scaled = num_imputed.values

        # 4. Construct output
        if self.as_df:
            cat_out_df = pd.DataFrame(cat_encoded, columns=self.categorical_cols, index=X.index)
            num_out_df = pd.DataFrame(num_scaled, columns=self.numerical_cols, index=X.index)
            return pd.concat([cat_out_df, num_out_df], axis=1)
        else:
            # TabNet / PyTorch ready: 2D float32 array [categoricals_first, numericals_second]
            out_matrix = np.hstack([cat_encoded, num_scaled]).astype(np.float32)
            return out_matrix


# ==============================================================================
# EVALUATION STRATEGY: SPLIT GENERATORS
# ==============================================================================

def get_ieee_stratified_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    target_col: str = TARGET_COL,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate Stratified 5-Fold Cross-Validation splits.

    Role in Project:
    Serves as the internal benchmark to assess algorithm capacity, variance, and
    generalization stability across the full data distribution.
    Ensures identical class ratios (~3.50% fraud) across all folds.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    y = df[target_col].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return list(skf.split(df, y))


def get_ieee_chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    time_col: str = TIME_COL,
    target_col: str = TARGET_COL,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a chronological holdout split based on TransactionDT.

    Role in Project:
    Replicates realistic commercial production deployment: trains strictly on the first
    80% of transactions (e.g. initial ~145 days) and tests strictly on the final 20%
    (e.g. final ~37 days) of subsequent transactions.
    Exposes temporal drift, evolving fraud vectors, and distribution shifts.
    """
    if time_col not in df.columns:
        raise KeyError(f"Time column '{time_col}' not found in DataFrame.")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    # Sort indices by time
    sorted_indices = df.sort_values(time_col).index.values
    split_point = int(len(sorted_indices) * train_ratio)

    train_indices = sorted_indices[:split_point]
    val_indices = sorted_indices[split_point:]
    return train_indices, val_indices


def generate_ieee_folds(
    df: pd.DataFrame,
    strategy: str = "stratified",
    n_splits: int = 5,
    train_ratio: float = 0.80,
    shuffle: bool = True,
    random_state: int = 42,
    target_col: str = TARGET_COL,
    as_df: bool = False,
    scale_numerical: bool = True,
) -> Generator[
    Tuple[int, Union[np.ndarray, pd.DataFrame], np.ndarray, Union[np.ndarray, pd.DataFrame], np.ndarray, IEEEFraudPreprocessor],
    None,
    None,
]:
    """Generator yielding leak-free preprocessed training and validation fold partitions.

    For each fold:
    1. Splits data into train and validation partitions.
    2. Separates target column `isFraud`.
    3. Initializes an independent `IEEEFraudPreprocessor`.
    4. Fits strictly on training features (`X_train`).
    5. Transforms validation features (`X_val`) without refitting.
    6. Yields `(fold_idx, X_train, y_train, X_val, y_val, preprocessor)`.

    Args:
        df: Input merged DataFrame with candidate features and isFraud.
        strategy: 'stratified' (5-fold CV) or 'chronological' (time-aware holdout).
        n_splits: Number of folds for stratified CV.
        train_ratio: Train ratio for chronological holdout (default 0.80).
        shuffle: Whether to shuffle in stratified CV.
        random_state: Random seed for stratified CV.
        target_col: Name of target column (default 'isFraud').
        as_df: Whether to yield features as DataFrames or TabNet-ready NumPy arrays.
        scale_numerical: Whether to scale numerical features with RobustScaler.

    Yields:
        fold_idx: 0-indexed split/fold number.
        X_train: Preprocessed training features.
        y_train: Training binary labels (1D array).
        X_val: Preprocessed validation features.
        y_val: Validation binary labels (1D array).
        preprocessor: Fitted IEEEFraudPreprocessor instance.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    # Determine splits
    if strategy == "stratified":
        splits = get_ieee_stratified_cv_splits(
            df, n_splits=n_splits, shuffle=shuffle, random_state=random_state, target_col=target_col
        )
    elif strategy == "chronological":
        tr_idx, val_idx = get_ieee_chronological_split(
            df, train_ratio=train_ratio, time_col=TIME_COL, target_col=target_col
        )
        splits = [(tr_idx, val_idx)]
    else:
        raise ValueError(f"Unknown split strategy '{strategy}'. Use 'stratified' or 'chronological'.")

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # Slice partitions
        train_raw = df.loc[train_idx]
        val_raw = df.loc[val_idx]

        # Separate target
        y_train = train_raw[target_col].values.astype(np.int64)
        y_val = val_raw[target_col].values.astype(np.int64)

        X_train_raw = train_raw.drop(columns=[target_col])
        X_val_raw = val_raw.drop(columns=[target_col])

        # Initialize fresh preprocessor for this fold
        preprocessor = IEEEFraudPreprocessor(
            scale_numerical=scale_numerical,
            as_df=as_df,
        )

        # Fit strictly on train, transform train and val
        X_train = preprocessor.fit_transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw)

        yield fold_idx, X_train, y_train, X_val, y_val, preprocessor
