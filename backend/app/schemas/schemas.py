"""
Pydantic schemas for FinGuard AI API request validation and response serialization.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, AliasChoices


# --- Authentication Schemas ---

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    email: str


class TokenPayload(BaseModel):
    sub: Optional[str] = None
    role: Optional[str] = None
    exp: Optional[int] = None


class UserLogin(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "analyst"


# --- Transaction & Scoring Schemas ---

class TransactionInput(BaseModel):
    """Schema for incoming transaction payload for real-time inference."""

    Transaction_ID: Optional[str] = Field(default=None, description="Unique transaction ID")
    Customer_ID: Optional[str] = Field(default=None, description="Customer account ID")
    Merchant_ID: Optional[str] = Field(default=None, description="Merchant account ID")
    Customer_Name: Optional[str] = Field(default=None)
    Customer_Contact: Optional[str] = Field(default=None)
    Customer_Email: Optional[str] = Field(default=None)

    Age: int = Field(..., ge=18, le=100, description="Customer age in years")
    Gender: str = Field(..., description="Customer gender (Male, Female, Other)")
    State: str = Field(..., description="Indian state name")
    City: str = Field(..., description="City name")
    Bank_Branch: Optional[str] = Field(default=None)
    Account_Type: str = Field(..., description="Savings, Current, or Salary")
    Account_Balance: float = Field(..., ge=0.0, description="Customer balance in INR before transaction")
    Transaction_Type: str = Field(..., description="UPI, NEFT, RTGS, IMPS, Card, POS, ATM")
    Transaction_Amount: float = Field(..., gt=0.0, description="Transaction amount in INR")
    Transaction_Date: str = Field(..., description="Date in DD-MM-YYYY or YYYY-MM-DD")
    Transaction_Time: str = Field(..., description="Time in HH:MM:SS")
    Merchant_Category: str = Field(..., description="Retail, Grocery, Electronics, Travel, etc.")
    Device_Type: str = Field(..., description="Mobile, Desktop, POS, ATM")
    Transaction_Device: str = Field(..., description="Specific device used")
    Transaction_Location: Optional[str] = Field(default=None)
    Transaction_Currency: Optional[str] = Field(default="INR")
    Transaction_Description: str = Field(..., description="Text description of transaction")


class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput]


class TopContributor(BaseModel):
    feature: str
    importance: float
    contribution_pct: float
    primary_step: int
    step_attentions: Optional[List[float]] = None
    transformed_value: Optional[float] = None
    raw_value: Optional[Any] = None


class PredictionResponse(BaseModel):
    transaction_id: str
    model_name: str = "TabNet"
    fraud_probability: float
    is_fraud: int
    risk_tier: str
    decision_threshold: float
    latency_ms: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExplanationResponse(BaseModel):
    prediction: PredictionResponse
    top_contributors: List[TopContributor]
    step_attention_breakdown: Dict[str, Dict[str, float]]
    narrative_summary: str


class BatchPredictionResponse(BaseModel):
    total_transactions: int
    fraud_detected: int
    fraud_rate_pct: float
    results: List[PredictionResponse]


# --- Auditor Review Schemas ---

class ReviewCreate(BaseModel):
    action: str = Field(..., description="APPROVED, REJECTED, or ESCALATED")
    notes: Optional[str] = Field(default=None)


class ReviewResponse(BaseModel):
    id: int
    transaction_id: str
    auditor_email: str
    action: str
    notes: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --- Benchmark & Metrics Schemas ---

class BenchmarkModelItem(BaseModel):
    model_name: str
    pr_auc: str
    roc_auc: str
    recall_at_1pct_fpr: str
    f1_score: str
    macro_f1: Optional[str] = "N/A"
    total_time_s: float
    # Optional IEEE-CIS extensions
    chrono_pr_auc: Optional[str] = None
    chrono_roc_auc: Optional[str] = None
    chrono_recall_at_1pct_fpr: Optional[str] = None
    chrono_f1_score: Optional[str] = None
    latency_ms_per_1k: Optional[float] = None


class IEEEBenchmarkItem(BaseModel):
    model_name: str
    cv_pr_auc: str
    chrono_pr_auc: str
    cv_roc_auc: str
    chrono_roc_auc: str
    cv_recall_at_1pct_fpr: str
    chrono_recall_at_1pct_fpr: str
    cv_f1_score: str
    chrono_f1_score: str
    fit_time_s: float
    latency_ms_per_1k: float


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    importance_pct: float
    rank: int


# --- IEEE-CIS Production Schemas (41-Column Contract & 46 Features) ---

class IEEECISTransactionInput(BaseModel):
    """Schema for IEEE-CIS Fraud Detection transaction scoring payload (41 raw input columns)."""

    model_config = ConfigDict(populate_by_name=True)

    # Identifiers & Metadata
    transaction_id: Optional[Union[str, int]] = Field(
        default=None,
        description="Optional unique identifier for transaction tracking",
        validation_alias=AliasChoices("transaction_id", "TransactionID"),
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Optional model override: 'tabnet', 'xgboost', 'random_forest', 'logistic_regression'",
        validation_alias=AliasChoices("model_name", "model"),
    )

    # Temporal & Core Financial (Required)
    transaction_dt: float = Field(
        ...,
        ge=0.0,
        description="Timestep in seconds from reference date (TransactionDT)",
        validation_alias=AliasChoices("transaction_dt", "TransactionDT"),
    )
    transaction_amt: float = Field(
        ...,
        gt=0.0,
        description="Transaction payment amount in USD (TransactionAmt)",
        validation_alias=AliasChoices("transaction_amt", "TransactionAmt"),
    )
    product_cd: str = Field(
        ...,
        description="Product code category (ProductCD: W, C, R, H, S)",
        validation_alias=AliasChoices("product_cd", "ProductCD"),
    )
    card1: int = Field(
        ...,
        gt=0,
        description="Payment card entity identifier (card1)",
        validation_alias=AliasChoices("card1"),
    )

    # Categorical Attributes (Optional / Nullable)
    card4: Optional[str] = Field(
        default=None,
        description="Card network / issuer (visa, mastercard, discover, etc.)",
        validation_alias=AliasChoices("card4"),
    )
    card6: Optional[str] = Field(
        default=None,
        description="Card funding type (credit, debit)",
        validation_alias=AliasChoices("card6"),
    )
    p_emaildomain: Optional[str] = Field(
        default=None,
        description="Purchaser email domain (P_emaildomain)",
        validation_alias=AliasChoices("p_emaildomain", "P_emaildomain"),
    )
    r_emaildomain: Optional[str] = Field(
        default=None,
        description="Recipient email domain (R_emaildomain)",
        validation_alias=AliasChoices("r_emaildomain", "R_emaildomain"),
    )
    device_type: Optional[str] = Field(
        default=None,
        description="Device category: desktop or mobile (DeviceType)",
        validation_alias=AliasChoices("device_type", "DeviceType"),
    )
    device_info: Optional[str] = Field(
        default=None,
        description="Client device identification string (DeviceInfo)",
        validation_alias=AliasChoices("device_info", "DeviceInfo"),
    )
    m1: Optional[str] = Field(default=None, description="Match indicator M1", validation_alias=AliasChoices("m1", "M1"))
    m4: Optional[str] = Field(default=None, description="Match indicator M4", validation_alias=AliasChoices("m4", "M4"))
    m5: Optional[str] = Field(default=None, description="Match indicator M5", validation_alias=AliasChoices("m5", "M5"))
    m6: Optional[str] = Field(default=None, description="Match indicator M6", validation_alias=AliasChoices("m6", "M6"))

    # Spatial / Address Attributes
    addr1: Optional[float] = Field(default=None, description="Billing region / zipcode (addr1)", validation_alias=AliasChoices("addr1"))
    addr2: Optional[float] = Field(default=None, description="Billing country code (addr2)", validation_alias=AliasChoices("addr2"))

    # Velocity Counters (C-Features)
    c1: Optional[float] = Field(default=None, validation_alias=AliasChoices("c1", "C1"))
    c2: Optional[float] = Field(default=None, validation_alias=AliasChoices("c2", "C2"))
    c5: Optional[float] = Field(default=None, validation_alias=AliasChoices("c5", "C5"))
    c6: Optional[float] = Field(default=None, validation_alias=AliasChoices("c6", "C6"))
    c11: Optional[float] = Field(default=None, validation_alias=AliasChoices("c11", "C11"))
    c13: Optional[float] = Field(default=None, validation_alias=AliasChoices("c13", "C13"))
    c14: Optional[float] = Field(default=None, validation_alias=AliasChoices("c14", "C14"))

    # Timedelta Features (D-Features)
    d1: Optional[float] = Field(default=None, validation_alias=AliasChoices("d1", "D1"))
    d2: Optional[float] = Field(default=None, validation_alias=AliasChoices("d2", "D2"))
    d15: Optional[float] = Field(default=None, validation_alias=AliasChoices("d15", "D15"))

    # Salient V-Features
    v12: Optional[float] = Field(default=None, validation_alias=AliasChoices("v12", "V12"))
    v13: Optional[float] = Field(default=None, validation_alias=AliasChoices("v13", "V13"))
    v35: Optional[float] = Field(default=None, validation_alias=AliasChoices("v35", "V35"))
    v36: Optional[float] = Field(default=None, validation_alias=AliasChoices("v36", "V36"))
    v75: Optional[float] = Field(default=None, validation_alias=AliasChoices("v75", "V75"))
    v76: Optional[float] = Field(default=None, validation_alias=AliasChoices("v76", "V76"))
    v90: Optional[float] = Field(default=None, validation_alias=AliasChoices("v90", "V90"))
    v91: Optional[float] = Field(default=None, validation_alias=AliasChoices("v91", "V91"))
    v126: Optional[float] = Field(default=None, validation_alias=AliasChoices("v126", "V126"))
    v127: Optional[float] = Field(default=None, validation_alias=AliasChoices("v127", "V127"))
    v128: Optional[float] = Field(default=None, validation_alias=AliasChoices("v128", "V128"))
    v257: Optional[float] = Field(default=None, validation_alias=AliasChoices("v257", "V257"))
    v258: Optional[float] = Field(default=None, validation_alias=AliasChoices("v258", "V258"))
    v282: Optional[float] = Field(default=None, validation_alias=AliasChoices("v282", "V282"))
    v283: Optional[float] = Field(default=None, validation_alias=AliasChoices("v283", "V283"))

    def to_raw_dataframe(self):
        """Map validated payload into a 1-row DataFrame conforming to the 41-column raw contract."""
        import pandas as pd
        data = {
            "TransactionDT": self.transaction_dt,
            "TransactionAmt": self.transaction_amt,
            "ProductCD": self.product_cd,
            "card1": self.card1,
            "card4": self.card4,
            "card6": self.card6,
            "addr1": self.addr1,
            "addr2": self.addr2,
            "P_emaildomain": self.p_emaildomain,
            "R_emaildomain": self.r_emaildomain,
            "DeviceType": self.device_type,
            "DeviceInfo": self.device_info,
            "M1": self.m1,
            "M4": self.m4,
            "M5": self.m5,
            "M6": self.m6,
            "C1": self.c1,
            "C2": self.c2,
            "C5": self.c5,
            "C6": self.c6,
            "C11": self.c11,
            "C13": self.c13,
            "C14": self.c14,
            "D1": self.d1,
            "D2": self.d2,
            "D15": self.d15,
            "V12": self.v12,
            "V13": self.v13,
            "V35": self.v35,
            "V36": self.v36,
            "V75": self.v75,
            "V76": self.v76,
            "V90": self.v90,
            "V91": self.v91,
            "V126": self.v126,
            "V127": self.v127,
            "V128": self.v128,
            "V257": self.v257,
            "V258": self.v258,
            "V282": self.v282,
            "V283": self.v283,
        }
        if self.transaction_id is not None:
            data["TransactionID"] = self.transaction_id
        return pd.DataFrame([data])


class IEEEPredictionResponse(BaseModel):
    transaction_id: str
    model_name: str
    fraud_probability: float
    is_fraud: int
    decision_threshold: float
    risk_tier: str
    latency_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IEEEBatchTransactionInput(BaseModel):
    transactions: List[IEEECISTransactionInput]
    model_name: Optional[str] = "tabnet"


class IEEEBatchPredictionResponse(BaseModel):
    total_transactions: int
    fraud_detected: int
    fraud_rate_pct: float
    model_used: str
    results: List[IEEEPredictionResponse]


class IEEEExplanationResponse(BaseModel):
    prediction: IEEEPredictionResponse
    top_contributors: List[TopContributor]
    step_attention_breakdown: Dict[str, Dict[str, float]]
    narrative_summary: str
    explainer_metadata: Dict[str, Any] = Field(default_factory=dict)
