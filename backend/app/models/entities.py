"""
SQLAlchemy ORM Entities for FinGuard AI backend database.
Tracks users, scored transactions, model predictions, TabNet explanations, and auditor reviews.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class User(Base):
    """System user entity (Compliance officers, fraud analysts, system admins)."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), default="analyst", nullable=False)  # admin, analyst, auditor
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)


class TransactionRecord(Base):
    """Raw transaction ingestion record."""

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(100), unique=True, index=True, nullable=False)
    customer_id = Column(String(100), index=True, nullable=False)
    merchant_id = Column(String(100), index=True, nullable=False)
    amount = Column(Float, nullable=False)
    account_balance = Column(Float, nullable=False)
    transaction_type = Column(String(50), nullable=False)
    merchant_category = Column(String(100), nullable=False)
    device_type = Column(String(50), nullable=False)
    transaction_device = Column(String(100), nullable=False)
    state = Column(String(100), nullable=False)
    city = Column(String(100), nullable=False)
    transaction_date = Column(String(50), nullable=False)
    transaction_time = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    predictions = relationship("PredictionRecord", back_populates="transaction", cascade="all, delete-orphan")


class PredictionRecord(Base):
    """Fraud scoring prediction record."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(100), ForeignKey("transactions.transaction_id"), index=True, nullable=False)
    model_name = Column(String(50), default="TabNet", nullable=False)
    fraud_probability = Column(Float, nullable=False)
    is_fraud = Column(Integer, nullable=False)  # 0 or 1
    risk_tier = Column(String(20), nullable=False)  # CRITICAL, HIGH, MEDIUM, LOW
    decision_threshold = Column(Float, default=0.50, nullable=False)
    latency_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    transaction = relationship("TransactionRecord", back_populates="predictions")
    explanation = relationship("ExplanationRecord", back_populates="prediction", uselist=False, cascade="all, delete-orphan")


class ExplanationRecord(Base):
    """TabNet Sparsemax feature attention attribution and forensic summary."""

    __tablename__ = "explanations"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), unique=True, index=True, nullable=False)
    top_features_json = Column(Text, nullable=False)  # JSON-serialized list of top contributors
    step_attentions_json = Column(Text, nullable=False)  # JSON-serialized step attention dict
    narrative_summary = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    prediction = relationship("PredictionRecord", back_populates="explanation")


class ReviewRecord(Base):
    """Human-in-the-loop compliance officer / auditor review decision."""

    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(100), index=True, nullable=False)
    auditor_email = Column(String(255), nullable=False)
    action = Column(String(50), nullable=False)  # APPROVED, REJECTED, ESCALATED
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
