"""
Transaction scoring and batch fraud prediction service.
Interfaces with MLRegistry to transform inputs and run low-latency TabNet inference.
"""

from typing import List, Dict, Any, Optional
import time
import uuid
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from fastapi import HTTPException

from backend.app.core.config import settings
from backend.app.ml.artifact_loader import get_ml_registry
from backend.app.schemas.schemas import (
    TransactionInput,
    PredictionResponse,
    BatchPredictionResponse,
    IEEECISTransactionInput,
    IEEEPredictionResponse,
    IEEEBatchTransactionInput,
    IEEEBatchPredictionResponse,
)
from backend.app.models.entities import TransactionRecord, PredictionRecord


def score_single_transaction(
    tx_input: TransactionInput,
    db: Optional[Session] = None,
    save_db: bool = True,
) -> PredictionResponse:
    """Score a single incoming transaction for fraud probability using TabNet."""
    registry = get_ml_registry()
    if not registry.is_loaded or registry.model is None or registry.preprocessor is None:
        raise HTTPException(status_code=503, detail="ML model and preprocessor are not loaded.")

    tx_dict = tx_input.model_dump()
    tx_id = tx_dict.get("Transaction_ID") or f"TX-{uuid.uuid4().hex[:12].upper()}"
    tx_dict["Transaction_ID"] = tx_id
    if not tx_dict.get("Customer_ID"):
        tx_dict["Customer_ID"] = f"CUST-{uuid.uuid4().hex[:8].upper()}"
    if not tx_dict.get("Merchant_ID"):
        tx_dict["Merchant_ID"] = f"MCH-{uuid.uuid4().hex[:8].upper()}"

    # Build 1-row DataFrame for pipeline transformation
    df_single = pd.DataFrame([tx_dict])

    start_time = time.time()
    X_proc = registry.preprocessor.transform(df_single)
    probs = registry.model.predict_proba(X_proc)[0]
    latency_ms = (time.time() - start_time) * 1000.0

    fraud_prob = float(probs[1])
    threshold = registry.metadata.get("optimal_threshold", settings.DEFAULT_DECISION_THRESHOLD)
    is_fraud = int(fraud_prob >= threshold)

    if fraud_prob >= 0.80:
        risk_tier = "CRITICAL"
    elif fraud_prob >= 0.50:
        risk_tier = "HIGH"
    elif fraud_prob >= 0.20:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    response = PredictionResponse(
        transaction_id=tx_id,
        model_name="TabNet",
        fraud_probability=round(fraud_prob, 4),
        is_fraud=is_fraud,
        risk_tier=risk_tier,
        decision_threshold=threshold,
        latency_ms=round(latency_ms, 2),
        created_at=datetime.now(timezone.utc),
    )

    # Persist in DB if session provided
    if save_db and db is not None:
        try:
            # Check if transaction already exists
            existing_tx = db.query(TransactionRecord).filter(TransactionRecord.transaction_id == tx_id).first()
            if not existing_tx:
                new_tx = TransactionRecord(
                    transaction_id=tx_id,
                    customer_id=tx_dict["Customer_ID"],
                    merchant_id=tx_dict["Merchant_ID"],
                    amount=float(tx_dict["Transaction_Amount"]),
                    account_balance=float(tx_dict["Account_Balance"]),
                    transaction_type=tx_dict["Transaction_Type"],
                    merchant_category=tx_dict["Merchant_Category"],
                    device_type=tx_dict["Device_Type"],
                    transaction_device=tx_dict["Transaction_Device"],
                    state=tx_dict["State"],
                    city=tx_dict["City"],
                    transaction_date=str(tx_dict["Transaction_Date"]),
                    transaction_time=str(tx_dict["Transaction_Time"]),
                )
                db.add(new_tx)
                db.flush()

            # Record prediction
            new_pred = PredictionRecord(
                transaction_id=tx_id,
                model_name="TabNet",
                fraud_probability=round(fraud_prob, 4),
                is_fraud=is_fraud,
                risk_tier=risk_tier,
                decision_threshold=threshold,
                latency_ms=round(latency_ms, 2),
            )
            db.add(new_pred)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"[PredictionService] DB persistence error: {e}")

    return response


def score_batch_transactions(
    tx_list: List[TransactionInput],
    db: Optional[Session] = None,
) -> BatchPredictionResponse:
    """Score a batch of transactions efficiently."""
    registry = get_ml_registry()
    if not registry.is_loaded or registry.model is None or registry.preprocessor is None:
        raise HTTPException(status_code=503, detail="ML model and preprocessor are not loaded.")

    if not tx_list:
        return BatchPredictionResponse(
            total_transactions=0,
            fraud_detected=0,
            fraud_rate_pct=0.0,
            results=[],
        )

    records = []
    for item in tx_list:
        d = item.model_dump()
        if not d.get("Transaction_ID"):
            d["Transaction_ID"] = f"TX-{uuid.uuid4().hex[:12].upper()}"
        if not d.get("Customer_ID"):
            d["Customer_ID"] = f"CUST-{uuid.uuid4().hex[:8].upper()}"
        if not d.get("Merchant_ID"):
            d["Merchant_ID"] = f"MCH-{uuid.uuid4().hex[:8].upper()}"
        records.append(d)

    df_batch = pd.DataFrame(records)

    start_time = time.time()
    X_proc = registry.preprocessor.transform(df_batch)
    probs = registry.model.predict_proba(X_proc)
    total_elapsed_ms = (time.time() - start_time) * 1000.0
    per_tx_latency = round(total_elapsed_ms / len(records), 2)

    threshold = registry.metadata.get("optimal_threshold", settings.DEFAULT_DECISION_THRESHOLD)
    results = []
    fraud_count = 0

    for i, rec in enumerate(records):
        prob = float(probs[i, 1])
        is_fraud = int(prob >= threshold)
        if is_fraud:
            fraud_count += 1

        if prob >= 0.80:
            tier = "CRITICAL"
        elif prob >= 0.50:
            tier = "HIGH"
        elif prob >= 0.20:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        res = PredictionResponse(
            transaction_id=rec["Transaction_ID"],
            model_name="TabNet",
            fraud_probability=round(prob, 4),
            is_fraud=is_fraud,
            risk_tier=tier,
            decision_threshold=threshold,
            latency_ms=per_tx_latency,
            created_at=datetime.now(timezone.utc),
        )
        results.append(res)

    fraud_rate_pct = round((fraud_count / len(records)) * 100.0, 2)
    return BatchPredictionResponse(
        total_transactions=len(records),
        fraud_detected=fraud_count,
        fraud_rate_pct=fraud_rate_pct,
        results=results,
    )


# --- IEEE-CIS Fraud Detection Services ---

SUPPORTED_IEEE_MODELS = {"tabnet", "xgboost", "random_forest", "logistic_regression"}


def score_single_ieee_transaction(
    tx_input: IEEECISTransactionInput,
    model_name: Optional[str] = None,
    db: Optional[Session] = None,
) -> IEEEPredictionResponse:
    """Score a single IEEE-CIS transaction using the specified model architecture."""
    # 1. Determine model architecture
    raw_model = model_name or tx_input.model_name or "tabnet"
    selected_model = raw_model.strip().lower()

    if selected_model not in SUPPORTED_IEEE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{selected_model}'. Supported models: {sorted(list(SUPPORTED_IEEE_MODELS))}",
        )

    # 2. Check ML registry readiness
    registry = get_ml_registry()
    if not registry.is_ieee_loaded or registry.ieee_preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="IEEE-CIS production model artifacts and preprocessor are not loaded.",
        )

    # 3. Create single-row DataFrame matching the 41-column raw input contract
    df_raw = tx_input.to_raw_dataframe()

    # 4. Preprocess exactly once
    start_time = time.perf_counter()
    try:
        X_proc = registry.preprocess_ieee(df_raw)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {str(e)}")

    # 5. Predict probabilities
    try:
        probs = registry.predict_ieee_proba(X_proc, model_name=selected_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error with model '{selected_model}': {str(e)}")

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    fraud_prob = float(probs[0, 1])
    threshold = registry.get_ieee_threshold(selected_model)
    is_fraud = int(fraud_prob >= threshold)

    if fraud_prob >= 0.80:
        risk_tier = "CRITICAL"
    elif fraud_prob >= 0.50:
        risk_tier = "HIGH"
    elif fraud_prob >= 0.20:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    tx_id = str(tx_input.transaction_id or f"TX-IEEE-{uuid.uuid4().hex[:10].upper()}")

    metadata = {
        "dataset": "IEEE-CIS Fraud Detection",
        "n_processed_features": X_proc.shape[1],
        "decision_rule": f"is_fraud = 1 if prob >= {threshold:.2f} else 0",
        "calibrated_threshold": threshold,
    }

    return IEEEPredictionResponse(
        transaction_id=tx_id,
        model_name=selected_model,
        fraud_probability=round(fraud_prob, 4),
        is_fraud=is_fraud,
        decision_threshold=threshold,
        risk_tier=risk_tier,
        latency_ms=round(latency_ms, 2),
        metadata=metadata,
        created_at=datetime.now(timezone.utc),
    )


def score_batch_ieee_transactions(
    tx_list: List[IEEECISTransactionInput],
    model_name: Optional[str] = None,
    db: Optional[Session] = None,
) -> IEEEBatchPredictionResponse:
    """Score a batch of IEEE-CIS transactions efficiently using the chosen model."""
    raw_model = model_name or (tx_list[0].model_name if tx_list and tx_list[0].model_name else "tabnet")
    selected_model = raw_model.strip().lower()

    if selected_model not in SUPPORTED_IEEE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{selected_model}'. Supported models: {sorted(list(SUPPORTED_IEEE_MODELS))}",
        )

    registry = get_ml_registry()
    if not registry.is_ieee_loaded or registry.ieee_preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="IEEE-CIS production model artifacts and preprocessor are not loaded.",
        )

    if not tx_list:
        return IEEEBatchPredictionResponse(
            total_transactions=0,
            fraud_detected=0,
            fraud_rate_pct=0.0,
            model_used=selected_model,
            results=[],
        )

    # Concatenate all rows into a single DataFrame
    dfs = [item.to_raw_dataframe() for item in tx_list]
    df_batch = pd.concat(dfs, ignore_index=True)

    start_time = time.perf_counter()
    X_proc = registry.preprocess_ieee(df_batch)
    probs = registry.predict_ieee_proba(X_proc, model_name=selected_model)
    total_latency_ms = (time.perf_counter() - start_time) * 1000.0
    per_tx_latency = round(total_latency_ms / len(tx_list), 2)

    threshold = registry.get_ieee_threshold(selected_model)
    results = []
    fraud_count = 0

    for i, item in enumerate(tx_list):
        prob = float(probs[i, 1])
        is_fraud = int(prob >= threshold)
        if is_fraud:
            fraud_count += 1

        if prob >= 0.80:
            tier = "CRITICAL"
        elif prob >= 0.50:
            tier = "HIGH"
        elif prob >= 0.20:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        tx_id = str(item.transaction_id or f"TX-IEEE-{uuid.uuid4().hex[:10].upper()}")
        res = IEEEPredictionResponse(
            transaction_id=tx_id,
            model_name=selected_model,
            fraud_probability=round(prob, 4),
            is_fraud=is_fraud,
            decision_threshold=threshold,
            risk_tier=tier,
            latency_ms=per_tx_latency,
            metadata={
                "dataset": "IEEE-CIS Fraud Detection",
                "n_processed_features": X_proc.shape[1],
                "decision_rule": f"is_fraud = 1 if prob >= {threshold:.2f} else 0",
                "calibrated_threshold": threshold,
            },
            created_at=datetime.now(timezone.utc),
        )
        results.append(res)

    fraud_rate_pct = round((fraud_count / len(tx_list)) * 100.0, 2)
    return IEEEBatchPredictionResponse(
        total_transactions=len(tx_list),
        fraud_detected=fraud_count,
        fraud_rate_pct=fraud_rate_pct,
        model_used=selected_model,
        results=results,
    )

