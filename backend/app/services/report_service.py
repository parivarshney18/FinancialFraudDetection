"""
Auditing and forensic compliance reporting service.
"""

from typing import Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.app.models.entities import TransactionRecord, PredictionRecord, ReviewRecord


def get_audit_summary(db: Session) -> Dict[str, Any]:
    """Aggregate statistics for executive and compliance dashboards."""
    total_tx = db.query(func.count(TransactionRecord.id)).scalar() or 0
    total_preds = db.query(func.count(PredictionRecord.id)).scalar() or 0
    total_fraud = db.query(func.count(PredictionRecord.id)).filter(PredictionRecord.is_fraud == 1).scalar() or 0

    # Risk tier distribution
    risk_tiers = (
        db.query(PredictionRecord.risk_tier, func.count(PredictionRecord.id))
        .group_by(PredictionRecord.risk_tier)
        .all()
    )
    risk_dict = {tier: count for tier, count in risk_tiers}

    # Review status distribution
    reviews = (
        db.query(ReviewRecord.action, func.count(ReviewRecord.id))
        .group_by(ReviewRecord.action)
        .all()
    )
    review_dict = {action: count for action, count in reviews}

    # Average scoring latency
    avg_latency = db.query(func.avg(PredictionRecord.latency_ms)).scalar() or 0.0

    return {
        "total_transactions_ingested": total_tx,
        "total_predictions_scored": total_preds,
        "total_fraud_alerts": total_fraud,
        "fraud_alert_rate_pct": round((total_fraud / total_preds * 100.0), 2) if total_preds > 0 else 0.0,
        "average_inference_latency_ms": round(float(avg_latency), 2),
        "risk_tier_breakdown": risk_dict,
        "auditor_review_breakdown": review_dict,
    }
