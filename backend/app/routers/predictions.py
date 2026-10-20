"""
Prediction and real-time fraud scoring routes.
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas.schemas import (
    TransactionInput,
    BatchTransactionInput,
    PredictionResponse,
    ExplanationResponse,
    BatchPredictionResponse,
    IEEECISTransactionInput,
    IEEEPredictionResponse,
    IEEEBatchTransactionInput,
    IEEEBatchPredictionResponse,
    IEEEExplanationResponse,
)
from backend.app.services.prediction_service import (
    score_single_transaction,
    score_batch_transactions,
    score_single_ieee_transaction,
    score_batch_ieee_transactions,
)
from backend.app.services.explanation_service import (
    explain_transaction,
    explain_ieee_transaction,
)

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post("/score", response_model=PredictionResponse)
def score_transaction(
    tx_input: TransactionInput,
    db: Session = Depends(get_db),
):
    """Real-time single transaction fraud scoring using TabNet."""
    return score_single_transaction(tx_input, db=db, save_db=True)


@router.post("/score-and-explain", response_model=ExplanationResponse)
def score_and_explain_transaction(
    tx_input: TransactionInput,
    top_k: int = Query(default=5, ge=1, le=22, description="Number of top features to attribute"),
    db: Session = Depends(get_db),
):
    """Real-time scoring plus TabNet Sparsemax attention mask explanation and forensic narrative."""
    return explain_transaction(tx_input, db=db, top_k=top_k, save_db=True)


@router.post("/batch", response_model=BatchPredictionResponse)
def score_batch(
    batch_input: BatchTransactionInput,
    db: Session = Depends(get_db),
):
    """High-throughput batch transaction scoring."""
    return score_batch_transactions(batch_input.transactions, db=db)


@router.post("/score-ieee", response_model=IEEEPredictionResponse)
def score_ieee(
    tx_input: IEEECISTransactionInput,
    model: Optional[str] = Query(
        default=None,
        description="Model architecture to use: 'tabnet', 'xgboost', 'random_forest', 'logistic_regression'",
    ),
    db: Session = Depends(get_db),
):
    """Real-time IEEE-CIS fraud scoring across TabNet, XGBoost, Random Forest, or Logistic Regression."""
    return score_single_ieee_transaction(tx_input, model_name=model, db=db)


@router.post("/batch-ieee", response_model=IEEEBatchPredictionResponse)
def score_batch_ieee(
    batch_input: IEEEBatchTransactionInput,
    model: Optional[str] = Query(
        default=None,
        description="Model architecture override",
    ),
    db: Session = Depends(get_db),
):
    """High-throughput batch scoring on IEEE-CIS transaction vectors."""
    chosen_model = model or batch_input.model_name
    return score_batch_ieee_transactions(batch_input.transactions, model_name=chosen_model, db=db)


@router.post("/score-and-explain-ieee", response_model=IEEEExplanationResponse)
def score_and_explain_ieee(
    tx_input: IEEECISTransactionInput,
    model: Optional[str] = Query(
        default="tabnet",
        description="Model architecture: 'tabnet', 'xgboost', 'random_forest', 'logistic_regression'",
    ),
    top_k: int = Query(default=5, ge=1, le=46, description="Number of top features to attribute"),
    db: Session = Depends(get_db),
):
    """Real-time scoring plus TabNet Sparsemax attention mask explanation on IEEE-CIS 46 features."""
    return explain_ieee_transaction(tx_input, model_name=model, top_k=top_k, db=db)

