"""
Central configuration for AI-Based Financial Fraud Detection with TabNet.
Defines seeds, directory paths, column groupings, and modeling hyperparameters.
"""

from pathlib import Path

# Reproducibility
RANDOM_SEED = 42

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AI_DIR = PROJECT_ROOT / "ai"
RAW_DATA_PATH = AI_DIR / "data" / "raw" / "Bank_Transaction_Fraud_Detection.csv"
PROCESSED_DATA_DIR = AI_DIR / "data" / "processed"
REPORTS_DIR = AI_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = AI_DIR / "artifacts"
RESULTS_DIR = AI_DIR / "results"

# Target Definition
TARGET_COL = "Is_Fraud"

# Column Taxonomy
ID_COLS = [
    "Transaction_ID",
    "Customer_ID",
    "Merchant_ID",
]

PII_COLS = [
    "Customer_Name",
    "Customer_Contact",
    "Customer_Email",
]

REDUNDANT_COLS = [
    "Bank_Branch",            # 100% deterministic with City + ' Branch'
    "Transaction_Location",   # 100% deterministic with City + ', ' + State
    "Transaction_Currency",   # Constant value 'INR' (zero variance)
]

DROP_COLS = ID_COLS + PII_COLS + REDUNDANT_COLS

NUMERICAL_COLS = [
    "Age",
    "Transaction_Amount",
    "Account_Balance",
]

CATEGORICAL_COLS = [
    "Gender",
    "State",
    "City",
    "Account_Type",
    "Transaction_Type",
    "Merchant_Category",
    "Device_Type",
    "Transaction_Device",
    "Transaction_Description",
]

TEMPORAL_COLS = [
    "Transaction_Date",
    "Transaction_Time",
]

# Cross-Validation Configuration
CV_N_SPLITS = 5
CV_SHUFFLE = True

# Target Metric Weights
SCORING_METRICS = [
    "roc_auc",
    "pr_auc",
    "f1_macro",
    "recall_at_fpr_1pct",
]
