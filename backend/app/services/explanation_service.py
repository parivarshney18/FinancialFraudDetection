"""
TabNet explanation service.
Generates instance-level Sparsemax attention masks, top-k feature attributions, and forensic narratives.
"""

from typing import Optional, Dict, Any
import json
import time
import uuid
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy.orm import Session
from fastapi import HTTPException

from backend.app.core.config import settings
from backend.app.ml.artifact_loader import get_ml_registry
from backend.app.schemas.schemas import (
    TransactionInput,
    PredictionResponse,
    ExplanationResponse,
    TopContributor,
    IEEECISTransactionInput,
    IEEEPredictionResponse,
    IEEEExplanationResponse,
)
from backend.app.models.entities import TransactionRecord, PredictionRecord, ExplanationRecord


def explain_transaction(
    tx_input: TransactionInput,
    db: Optional[Session] = None,
    top_k: int = 5,
    save_db: bool = True,
) -> ExplanationResponse:
    """Score a transaction and generate full TabNet attention mask explanations."""
    registry = get_ml_registry()
    if not registry.is_loaded or registry.explainer is None:
        raise HTTPException(status_code=503, detail="ML explainer engine is not initialized.")

    tx_dict = tx_input.model_dump()
    tx_id = tx_dict.get("Transaction_ID") or f"TX-{uuid.uuid4().hex[:12].upper()}"
    tx_dict["Transaction_ID"] = tx_id
    if not tx_dict.get("Customer_ID"):
        tx_dict["Customer_ID"] = f"CUST-{uuid.uuid4().hex[:8].upper()}"
    if not tx_dict.get("Merchant_ID"):
        tx_dict["Merchant_ID"] = f"MCH-{uuid.uuid4().hex[:8].upper()}"

    # Preprocess
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

    prediction = PredictionResponse(
        transaction_id=tx_id,
        model_name="TabNet",
        fraud_probability=round(fraud_prob, 4),
        is_fraud=is_fraud,
        risk_tier=risk_tier,
        decision_threshold=threshold,
        latency_ms=round(latency_ms, 2),
        created_at=datetime.now(timezone.utc),
    )

    # Compute explanation
    raw_explanation = registry.explainer.explain_instance(
        x=X_proc[0],
        top_k=top_k,
        raw_record=tx_dict,
        fraud_probability=fraud_prob,
        threshold=threshold,
    )

    top_contributors = [
        TopContributor(
            feature=c["feature"],
            importance=c["importance"],
            contribution_pct=c["contribution_pct"],
            primary_step=c["primary_step"],
            step_attentions=c["step_attentions"],
            transformed_value=c["transformed_value"],
            raw_value=c["raw_value"],
        )
        for c in raw_explanation["top_contributors"]
    ]

    response = ExplanationResponse(
        prediction=prediction,
        top_contributors=top_contributors,
        step_attention_breakdown=raw_explanation["step_attention_breakdown"],
        narrative_summary=raw_explanation["narrative_summary"],
    )

    # Persist in DB if requested
    if save_db and db is not None:
        try:
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
            db.flush()

            new_exp = ExplanationRecord(
                prediction_id=new_pred.id,
                top_features_json=json.dumps([c.model_dump() for c in top_contributors]),
                step_attentions_json=json.dumps(raw_explanation["step_attention_breakdown"]),
                narrative_summary=raw_explanation["narrative_summary"],
            )
            db.add(new_exp)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"[ExplanationService] DB persistence error: {e}")

    return response


def explain_ieee_transaction(
    tx_input: IEEECISTransactionInput,
    model_name: Optional[str] = None,
    top_k: int = 5,
    db: Optional[Session] = None,
) -> IEEEExplanationResponse:
    """Score an IEEE-CIS transaction and generate TabNet Sparsemax attention mask explanations."""
    raw_model = model_name or tx_input.model_name or "tabnet"
    selected_model = raw_model.strip().lower()

    valid_models = {"tabnet", "xgboost", "random_forest", "logistic_regression"}
    if selected_model not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{selected_model}'. Supported models: {sorted(list(valid_models))}",
        )

    registry = get_ml_registry()
    if not registry.is_ieee_loaded or registry.ieee_preprocessor is None:
        raise HTTPException(status_code=503, detail="IEEE-CIS model artifacts are not loaded.")

    # 1. Preprocess raw input exactly once
    df_raw = tx_input.to_raw_dataframe()
    start_time = time.perf_counter()
    try:
        X_proc = registry.preprocess_ieee(df_raw)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {str(e)}")

    # 2. Predict probability with selected model
    try:
        probs = registry.predict_ieee_proba(X_proc, model_name=selected_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

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

    prediction = IEEEPredictionResponse(
        transaction_id=tx_id,
        model_name=selected_model,
        fraud_probability=round(fraud_prob, 4),
        is_fraud=is_fraud,
        decision_threshold=threshold,
        risk_tier=risk_tier,
        latency_ms=round(latency_ms, 2),
        metadata={
            "dataset": "IEEE-CIS Fraud Detection",
            "n_processed_features": X_proc.shape[1],
            "decision_rule": f"is_fraud = 1 if prob >= {threshold:.2f} else 0",
            "calibrated_threshold": threshold,
        },
        created_at=datetime.now(timezone.utc),
    )

    # 3. Generate explanations
    if registry.ieee_explainer is None:
        raise HTTPException(status_code=503, detail="IEEE-CIS explainer engine is not initialized.")

    if selected_model == "tabnet":
        raw_explanation = registry.ieee_explainer.explain_instance(
            x=X_proc[0],
            top_k=top_k,
            raw_record=df_raw.iloc[0].to_dict(),
            fraud_probability=fraud_prob,
            threshold=threshold,
            model_name="tabnet",
        )

        top_contributors = [
            TopContributor(
                feature=c["feature"],
                importance=c["importance"],
                contribution_pct=c["contribution_pct"],
                primary_step=c["primary_step"],
                step_attentions=c["step_attentions"],
                transformed_value=c["transformed_value"],
                raw_value=c["raw_value"],
            )
            for c in raw_explanation["top_contributors"]
        ]

        step_breakdown = raw_explanation["step_attention_breakdown"]
        narrative = raw_explanation["narrative_summary"]
        explainer_meta = raw_explanation["explainer_metadata"]

    elif selected_model == "xgboost":
        xgb_info = registry.ieee_explainer.explain_xgboost(X_proc[0], top_k=top_k)
        top_contributors = []
        step_breakdown = {}
        narrative = (
            f"**IEEE-CIS EXPLANATION REPORT (XGBOOST)**\n"
            f"- Fraud Probability: **{fraud_prob * 100.0:.2f}%**\n"
            f"- Decision Threshold: **{threshold:.2f}**\n"
            f"- Assessment: **{risk_tier} RISK** ({'FLAGGED' if is_fraud else 'APPROVED'})\n\n"
            f"> [Limitation]: {xgb_info['detail']}\n"
            f"For fine-grained instance-level attention masks, utilize TabNet."
        )
        explainer_meta = {
            "dataset": "IEEE-CIS Fraud Detection",
            "method": "XGBoost (Instance explanation limitation)",
            "shap_installed": xgb_info.get("available", False),
            "limitation": xgb_info["detail"],
        }
    else:
        top_contributors = []
        step_breakdown = {}
        narrative = (
            f"**IEEE-CIS EXPLANATION REPORT ({selected_model.upper()})**\n"
            f"- Fraud Probability: **{fraud_prob * 100.0:.2f}%**\n"
            f"- Decision Threshold: **{threshold:.2f}**\n"
            f"- Assessment: **{risk_tier} RISK** ({'FLAGGED' if is_fraud else 'APPROVED'})\n\n"
            f"> [Notice]: Instance-level Sparsemax step attention masks are an architectural "
            f"feature of TabNet. Model '{selected_model}' provides prediction classification and calibrated "
            f"thresholding without sequential decision masks."
        )
        explainer_meta = {
            "dataset": "IEEE-CIS Fraud Detection",
            "method": f"{selected_model} (Decision step masks unavailable)",
        }

    return IEEEExplanationResponse(
        prediction=prediction,
        top_contributors=top_contributors,
        step_attention_breakdown=step_breakdown,
        narrative_summary=narrative,
        explainer_metadata=explainer_meta,
    )

