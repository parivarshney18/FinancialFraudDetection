"""
Transaction history and compliance review queue routes.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.app.db.session import get_db
from backend.app.models.entities import TransactionRecord, PredictionRecord, ExplanationRecord, ReviewRecord, User
from backend.app.schemas.schemas import ReviewCreate, ReviewResponse
from backend.app.core.dependencies import get_current_user

router = APIRouter(prefix="/transactions", tags=["Transactions & Reviews"])


@router.get("/")
def list_transactions(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Retrieve recent transactions with associated predictions."""
    txs = (
        db.query(TransactionRecord)
        .order_by(desc(TransactionRecord.id))
        .offset(offset)
        .limit(limit)
        .all()
    )

    results = []
    for tx in txs:
        latest_pred = (
            db.query(PredictionRecord)
            .filter(PredictionRecord.transaction_id == tx.transaction_id)
            .order_by(desc(PredictionRecord.id))
            .first()
        )
        latest_review = (
            db.query(ReviewRecord)
            .filter(ReviewRecord.transaction_id == tx.transaction_id)
            .order_by(desc(ReviewRecord.id))
            .first()
        )

        results.append({
            "transaction_id": tx.transaction_id,
            "customer_id": tx.customer_id,
            "amount": tx.amount,
            "account_balance": tx.account_balance,
            "transaction_type": tx.transaction_type,
            "merchant_category": tx.merchant_category,
            "city": tx.city,
            "state": tx.state,
            "transaction_date": tx.transaction_date,
            "transaction_time": tx.transaction_time,
            "fraud_probability": latest_pred.fraud_probability if latest_pred else None,
            "is_fraud": latest_pred.is_fraud if latest_pred else None,
            "risk_tier": latest_pred.risk_tier if latest_pred else None,
            "review_status": latest_review.action if latest_review else "PENDING",
        })

    return {
        "count": len(results),
        "transactions": results,
    }


@router.get("/queue/pending")
def list_pending_review_queue(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Retrieve flagged HIGH/CRITICAL risk transactions pending auditor review."""
    flagged_preds = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.risk_tier.in_(["CRITICAL", "HIGH"]))
        .order_by(desc(PredictionRecord.id))
        .limit(limit)
        .all()
    )

    pending = []
    for pred in flagged_preds:
        # Check if already reviewed
        reviewed = db.query(ReviewRecord).filter(ReviewRecord.transaction_id == pred.transaction_id).first()
        if not reviewed:
            tx = db.query(TransactionRecord).filter(TransactionRecord.transaction_id == pred.transaction_id).first()
            if tx:
                pending.append({
                    "transaction_id": tx.transaction_id,
                    "customer_id": tx.customer_id,
                    "amount": tx.amount,
                    "account_balance": tx.account_balance,
                    "transaction_type": tx.transaction_type,
                    "merchant_category": tx.merchant_category,
                    "fraud_probability": pred.fraud_probability,
                    "risk_tier": pred.risk_tier,
                    "created_at": str(pred.created_at),
                })

    return {
        "pending_count": len(pending),
        "queue": pending,
    }


@router.get("/{tx_id}")
def get_transaction_detail(
    tx_id: str,
    db: Session = Depends(get_db),
):
    """Retrieve full detail, predictions, and attention explanation for a transaction."""
    tx = db.query(TransactionRecord).filter(TransactionRecord.transaction_id == tx_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found.")

    pred = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.transaction_id == tx_id)
        .order_by(desc(PredictionRecord.id))
        .first()
    )

    exp_data = None
    if pred:
        exp = db.query(ExplanationRecord).filter(ExplanationRecord.prediction_id == pred.id).first()
        if exp:
            import json
            exp_data = {
                "top_features": json.loads(exp.top_features_json),
                "step_attentions": json.loads(exp.step_attentions_json),
                "narrative_summary": exp.narrative_summary,
            }

    reviews = (
        db.query(ReviewRecord)
        .filter(ReviewRecord.transaction_id == tx_id)
        .order_by(desc(ReviewRecord.id))
        .all()
    )

    return {
        "transaction": {
            "transaction_id": tx.transaction_id,
            "customer_id": tx.customer_id,
            "merchant_id": tx.merchant_id,
            "amount": tx.amount,
            "account_balance": tx.account_balance,
            "transaction_type": tx.transaction_type,
            "merchant_category": tx.merchant_category,
            "device_type": tx.device_type,
            "transaction_device": tx.transaction_device,
            "state": tx.state,
            "city": tx.city,
            "transaction_date": tx.transaction_date,
            "transaction_time": tx.transaction_time,
        },
        "prediction": {
            "model_name": pred.model_name if pred else None,
            "fraud_probability": pred.fraud_probability if pred else None,
            "is_fraud": pred.is_fraud if pred else None,
            "risk_tier": pred.risk_tier if pred else None,
            "latency_ms": pred.latency_ms if pred else None,
        } if pred else None,
        "explanation": exp_data,
        "reviews": [
            {
                "id": r.id,
                "auditor_email": r.auditor_email,
                "action": r.action,
                "notes": r.notes,
                "created_at": str(r.created_at),
            }
            for r in reviews
        ],
    }


@router.post("/{tx_id}/review", response_model=ReviewResponse)
def submit_review(
    tx_id: str,
    review_in: ReviewCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Submit human compliance review decision (APPROVED, REJECTED, ESCALATED)."""
    tx = db.query(TransactionRecord).filter(TransactionRecord.transaction_id == tx_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found.")

    if review_in.action not in ["APPROVED", "REJECTED", "ESCALATED"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Action must be one of: APPROVED, REJECTED, ESCALATED",
        )

    review = ReviewRecord(
        transaction_id=tx_id,
        auditor_email=current_user.email,
        action=review_in.action,
        notes=review_in.notes,
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return review
