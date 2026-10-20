"""
Compliance and executive audit reporting routes.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app.services.report_service import get_audit_summary

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/audit-summary")
def get_executive_audit_summary(
    db: Session = Depends(get_db),
):
    """Retrieve operational compliance summary: total transactions, fraud rates, and auditor actions."""
    return get_audit_summary(db)
