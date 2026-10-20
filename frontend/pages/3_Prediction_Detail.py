"""
FinGuard AI - Transaction Detail & Forensic Audit History.
Lookup full transaction profile, TabNet attention masks, and historical review determinations.
"""

import streamlit as st
import pandas as pd
from frontend.services.api_client import api_client
from frontend.components.gauge import render_risk_gauge
from frontend.components.contributors import render_contributors_chart
from frontend.components.heatmap import render_step_attention_heatmap

st.set_page_config(page_title="Prediction Detail | FinGuard AI", page_icon="🔎", layout="wide")

st.title("🔎 Transaction Forensic Investigation Detail")
st.markdown("Search and inspect the complete scoring record, TabNet decision steps, and review audit trail.")

# Search Input
search_id = st.text_input("Enter Transaction ID (e.g. TX-TEST-0001)", value="TX-TEST-0001")

if search_id:
    detail = api_client.get_transaction_detail(search_id)
    if not detail:
        st.warning(f"No records found for transaction ID: `{search_id}`")
    else:
        tx = detail["transaction"]
        pred = detail.get("prediction") or {}
        exp = detail.get("explanation") or {}
        reviews = detail.get("reviews", [])

        st.markdown("---")
        # Header Metrics
        col1, col2, col3, col4 = st.columns(4)
        prob = (pred.get("fraud_probability", 0.0) or 0.0) * 100.0

        with col1:
            st.metric("Transaction ID", tx["transaction_id"])
        with col2:
            st.metric("Fraud Risk Score", f"{prob:.1f}%")
        with col3:
            st.metric("Risk Classification", pred.get("risk_tier", "N/A"))
        with col4:
            st.metric("Scoring Model", pred.get("model_name", "TabNet"))

        # Details Layout
        left_col, right_col = st.columns([1, 1.2])

        with left_col:
            st.subheader("📋 Ingested Transaction Attributes")
            st.json(tx)

            if prob > 0:
                fig_gauge = render_risk_gauge(pred.get("fraud_probability", 0.0))
                st.plotly_chart(fig_gauge, use_container_width=True)

        with right_col:
            if exp and "top_features" in exp:
                st.subheader("📊 TabNet Feature Attributions")
                fig_bars = render_contributors_chart(exp["top_features"])
                st.plotly_chart(fig_bars, use_container_width=True)

                if "step_attentions" in exp:
                    fig_heat = render_step_attention_heatmap(exp["step_attentions"])
                    st.plotly_chart(fig_heat, use_container_width=True)

            if exp and "narrative_summary" in exp:
                st.subheader("📝 Forensic Narrative")
                st.code(exp["narrative_summary"], language="markdown")

        # Auditor Reviews History
        st.markdown("---")
        st.subheader("📜 Compliance & Auditor Adjudication History")
        if not reviews:
            st.info("No compliance reviews submitted yet for this transaction.")
        else:
            df_reviews = pd.DataFrame(reviews)
            st.dataframe(df_reviews, use_container_width=True)
