"""
FinGuard AI - Compliance Officer & Auditor Review Queue.
Triage flagged transactions, inspect risk factors, and submit audited regulatory decisions.
"""

import streamlit as st
import pandas as pd
from frontend.services.api_client import api_client

st.set_page_config(page_title="Review Queue | FinGuard AI", page_icon="⚖️", layout="wide")

st.title("⚖️ Compliance Officer Review Queue")
st.markdown(
    "Transactions identified with **CRITICAL** or **HIGH** risk tiers by the TabNet decision engine "
    "are routed here for human-in-the-loop compliance review pursuant to RBI guidelines."
)

token = st.session_state.get("token")
if not token:
    st.warning("🔒 Please sign in on the sidebar or main page to submit compliance review decisions.")

# Fetch queue
queue_items = api_client.get_pending_review_queue(limit=50)

if not queue_items:
    st.success("🎉 No pending transactions in the fraud alert review queue!")
else:
    st.subheader(f"📌 Flagged Transactions Awaiting Review ({len(queue_items)})")
    df_queue = pd.DataFrame(queue_items)

    st.dataframe(
        df_queue[[
            "transaction_id", "customer_id", "amount", "account_balance",
            "transaction_type", "merchant_category", "fraud_probability", "risk_tier"
        ]],
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("🔍 Review & Adjudicate Selected Transaction")

    tx_options = [item["transaction_id"] for item in queue_items]
    selected_tx_id = st.selectbox("Select Transaction to Review", options=tx_options)

    if selected_tx_id:
        detail = api_client.get_transaction_detail(selected_tx_id)
        if detail:
            tx = detail["transaction"]
            pred = detail.get("prediction") or {}
            exp = detail.get("explanation") or {}

            c1, c2, c3 = st.columns(3)
            with c1:
                st.write(f"**Customer ID:** `{tx['customer_id']}`")
                st.write(f"**Amount:** ₹{tx['amount']:,.2f}")
                st.write(f"**Account Balance:** ₹{tx['account_balance']:,.2f}")
            with c2:
                st.write(f"**Type:** `{tx['transaction_type']}`")
                st.write(f"**Merchant Category:** `{tx['merchant_category']}`")
                st.write(f"**Location:** {tx['city']}, {tx['state']}")
            with c3:
                st.write(f"**Fraud Probability:** `{pred.get('fraud_probability', 0.0)*100:.1f}%`")
                st.write(f"**Risk Tier:** `{pred.get('risk_tier', 'HIGH')}`")

            if exp and "narrative_summary" in exp:
                st.markdown("#### 📜 TabNet Forensic Explanation")
                st.info(exp["narrative_summary"])

            # Review Action Form
            st.markdown("#### ✍️ Compliance Decision")
            with st.form("auditor_review_form"):
                action = st.radio(
                    "Review Determination",
                    options=["APPROVED", "REJECTED", "ESCALATED"],
                    horizontal=True,
                    help="APPROVED: Clear transaction. REJECTED: Confirm fraud and freeze. ESCALATED: Transfer to senior AML committee.",
                )
                notes = st.text_area("Auditor Case Notes", placeholder="Customer confirmed transaction via OTP; verified legitimate merchant.")
                submit_review_btn = st.form_submit_button("Submit Compliance Decision", use_container_width=True)

                if submit_review_btn:
                    if not token:
                        st.error("Authentication token required. Please sign in.")
                    else:
                        success = api_client.submit_review(
                            tx_id=selected_tx_id,
                            action=action,
                            notes=notes,
                            token=token,
                        )
                        if success:
                            st.success(f"Transaction {selected_tx_id} successfully marked as {action}!")
                            st.rerun()
                        else:
                            st.error("Failed to submit review decision to backend.")
