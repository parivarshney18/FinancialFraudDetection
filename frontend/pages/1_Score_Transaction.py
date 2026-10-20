"""
FinGuard AI - Real-Time Transaction Scoring & Explainability Interface.
Allows risk officers to evaluate live transactions, select between TabNet, XGBoost,
Random Forest, and Logistic Regression, and inspect TabNet Sparsemax attention attributions.
"""

from datetime import datetime
import streamlit as st
import pandas as pd

from frontend.services.api_client import api_client
from frontend.components.gauge import render_risk_gauge
from frontend.components.contributors import render_contributors_chart
from frontend.components.heatmap import render_step_attention_heatmap

st.set_page_config(page_title="Score Transaction | FinGuard AI", page_icon="💳", layout="wide")

st.title("💳 Real-Time Transaction Scoring & Explainability")
st.markdown(
    "Evaluate live banking transactions across **TabNet**, **XGBoost**, **Random Forest**, "
    "and **Logistic Regression** trained on the **IEEE-CIS Fraud Detection dataset** (46 features). "
    "TabNet provides full sequential Sparsemax attention masks for regulatory audit compliance."
)

# --- Mode Selection ---
dataset_mode = st.radio(
    "Dataset Architecture Mode",
    options=["IEEE-CIS Fraud Detection (Production)", "Legacy Synthetic Dataset (Historical)"],
    index=0,
    horizontal=True,
)

if dataset_mode == "IEEE-CIS Fraud Detection (Production)":
    # =========================================================================
    # IEEE-CIS Production Scoring Interface
    # =========================================================================

    # Model selector
    st.subheader("🤖 Model Selection & Calibrated Thresholds")
    col_mod, col_thresh = st.columns([1.5, 2.5])

    with col_mod:
        selected_model_label = st.selectbox(
            "Inference Model",
            options=[
                "TabNet (Primary Deep Tabular)",
                "XGBoost (Gradient Boosted Trees)",
                "Random Forest (Ensemble Trees)",
                "Logistic Regression (Linear Baseline)",
            ],
            index=0,
        )

    model_mapping = {
        "TabNet (Primary Deep Tabular)": "tabnet",
        "XGBoost (Gradient Boosted Trees)": "xgboost",
        "Random Forest (Ensemble Trees)": "random_forest",
        "Logistic Regression (Linear Baseline)": "logistic_regression",
    }
    selected_model = model_mapping[selected_model_label]

    # Dynamically query active backend model registry for authoritative calibrated threshold
    models_metadata = api_client.get_ieee_models()
    backend_thresh = None
    if models_metadata and "models" in models_metadata and selected_model in models_metadata["models"]:
        backend_thresh = models_metadata["models"][selected_model].get("calibrated_threshold")

    threshold_docs = {
        "tabnet": ("0.89", "TabNet Sparsemax decision steps; calibrated via F1 maximization on chronological training partition."),
        "xgboost": ("0.79", "Gradient boosted trees; calibrated via F1 maximization on chronological training partition."),
        "random_forest": ("0.57", "Ensemble bagging trees; calibrated via F1 maximization on chronological training partition."),
        "logistic_regression": ("0.86", "L2-regularized linear baseline; calibrated via F1 maximization on chronological training partition."),
    }
    fallback_thresh, thresh_note = threshold_docs[selected_model]
    cal_thresh = f"{backend_thresh:.2f}" if backend_thresh is not None else fallback_thresh

    with col_thresh:
        source_label = "Live Backend API" if backend_thresh is not None else "Phase 9 Chronological Holdout"
        st.info(f"**Calibrated Decision Threshold:** `{cal_thresh}` (Source: {source_label})\n\n*{thresh_note}*")

    # Preset Scenarios
    st.subheader("⚡ Quick Load IEEE-CIS Scenarios")
    sc1, sc2, sc3 = st.columns(3)
    preset_ieee = None

    if sc1.button("📋 Scenario 1: Nominal Verified Transaction", use_container_width=True):
        preset_ieee = {
            "transaction_id": "TX-NOMINAL-001",
            "transaction_dt": 86400.0,
            "transaction_amt": 45.50,
            "product_cd": "W",
            "card1": 13500,
            "card4": "visa",
            "card6": "debit",
            "p_emaildomain": "gmail.com",
            "r_emaildomain": "gmail.com",
            "device_type": "mobile",
            "device_info": "iOS Device",
            "addr1": 315.0,
            "addr2": 87.0,
            "m1": "T", "m4": "M0", "m5": "F", "m6": "T",
            "c1": 1.0, "c2": 1.0, "c5": 0.0, "c6": 1.0, "c11": 1.0, "c13": 1.0, "c14": 1.0,
            "d1": 14.0, "d2": 14.0, "d15": 14.0,
            "v12": 1.0, "v13": 1.0, "v35": 1.0, "v36": 1.0, "v75": 1.0, "v76": 1.0, "v90": 0.0, "v91": 0.0,
            "v126": 0.0, "v127": 0.0, "v128": 0.0, "v257": 1.0, "v258": 1.0, "v282": 1.0, "v283": 1.0,
        }

    if sc2.button("🚨 Scenario 2: High Velocity Bot Spike", use_container_width=True):
        preset_ieee = {
            "transaction_id": "TX-VELOCITY-002",
            "transaction_dt": 120500.0,
            "transaction_amt": 520.00,
            "product_cd": "C",
            "card1": 9500,
            "card4": "mastercard",
            "card6": "credit",
            "p_emaildomain": "protonmail.com",
            "r_emaildomain": "anonymous.com",
            "device_type": "desktop",
            "device_info": "Linux",
            "addr1": 299.0,
            "addr2": 87.0,
            "m1": "F", "m4": "M2", "m5": "F", "m6": "F",
            "c1": 15.0, "c2": 18.0, "c5": 0.0, "c6": 12.0, "c11": 14.0, "c13": 24.0, "c14": 8.0,
            "d1": 0.0, "d2": 0.0, "d15": 0.0,
            "v12": 0.0, "v13": 0.0, "v35": 0.0, "v36": 0.0, "v75": 0.0, "v76": 0.0, "v90": 0.0, "v91": 0.0,
            "v126": 520.0, "v127": 520.0, "v128": 520.0, "v257": 4.0, "v258": 5.0, "v282": 3.0, "v283": 4.0,
        }

    if sc3.button("⚠️ Scenario 3: Cross-Border High-Value Transfer", use_container_width=True):
        preset_ieee = {
            "transaction_id": "TX-CROSSBORDER-003",
            "transaction_dt": 172800.0,
            "transaction_amt": 1850.00,
            "product_cd": "R",
            "card1": 18200,
            "card4": "visa",
            "card6": "credit",
            "p_emaildomain": "hotmail.com",
            "r_emaildomain": "yandex.ru",
            "device_type": "mobile",
            "device_info": "Android 14",
            "addr1": 126.0,
            "addr2": 60.0,
            "m1": "F", "m4": "M1", "m5": "T", "m6": "F",
            "c1": 4.0, "c2": 3.0, "c5": 1.0, "c6": 4.0, "c11": 3.0, "c13": 5.0, "c14": 2.0,
            "d1": 45.0, "d2": 45.0, "d15": 45.0,
            "v12": 1.0, "v13": 1.0, "v35": 0.0, "v36": 0.0, "v75": 1.0, "v76": 1.0, "v90": 1.0, "v91": 1.0,
            "v126": 1850.0, "v127": 1850.0, "v128": 1850.0, "v257": 2.0, "v258": 2.0, "v282": 1.0, "v283": 1.0,
        }

    if preset_ieee:
        st.session_state["ieee_form"] = preset_ieee

    cur_ieee = st.session_state.get("ieee_form", {})

    with st.form("ieee_scoring_form"):
        st.markdown("### 📝 IEEE-CIS Transaction Parameters")
        c1, c2, c3 = st.columns(3)

        with c1:
            tx_amt = st.number_input(
                "Transaction Amount ($ USD)",
                min_value=0.01,
                max_value=50000.0,
                value=float(cur_ieee.get("transaction_amt", 150.0)),
                step=10.0,
            )
            card1 = st.number_input(
                "Payment Card Entity ID (card1)",
                min_value=1,
                max_value=50000,
                value=int(cur_ieee.get("card1", 13500)),
                step=1,
            )
            product_cd = st.selectbox(
                "Product Category Code (ProductCD)",
                options=["W", "C", "R", "H", "S"],
                index=["W", "C", "R", "H", "S"].index(cur_ieee.get("product_cd", "W")),
            )
            tx_dt = st.number_input(
                "Transaction Timestep (TransactionDT in seconds)",
                min_value=0.0,
                max_value=31536000.0,
                value=float(cur_ieee.get("transaction_dt", 86400.0)),
                step=3600.0,
            )

        with c2:
            card4 = st.selectbox(
                "Card Network (card4)",
                options=["visa", "mastercard", "discover", "american express"],
                index=["visa", "mastercard", "discover", "american express"].index(cur_ieee.get("card4", "visa")),
            )
            card6 = st.selectbox(
                "Card Funding Type (card6)",
                options=["debit", "credit"],
                index=["debit", "credit"].index(cur_ieee.get("card6", "debit")),
            )
            p_email = st.selectbox(
                "Purchaser Email Domain (P_emaildomain)",
                options=["gmail.com", "yahoo.com", "hotmail.com", "protonmail.com", "anonymous.com", "other"],
                index=0 if "p_emaildomain" not in cur_ieee else [
                    "gmail.com", "yahoo.com", "hotmail.com", "protonmail.com", "anonymous.com", "other"
                ].index(cur_ieee.get("p_emaildomain", "gmail.com")),
            )
            r_email = st.selectbox(
                "Recipient Email Domain (R_emaildomain)",
                options=["gmail.com", "yahoo.com", "hotmail.com", "yandex.ru", "anonymous.com", "other"],
                index=0 if "r_emaildomain" not in cur_ieee else [
                    "gmail.com", "yahoo.com", "hotmail.com", "yandex.ru", "anonymous.com", "other"
                ].index(cur_ieee.get("r_emaildomain", "gmail.com")),
            )

        with c3:
            dev_type = st.selectbox(
                "Device Category (DeviceType)",
                options=["mobile", "desktop", "unknown"],
                index=["mobile", "desktop", "unknown"].index(cur_ieee.get("device_type", "mobile")),
            )
            dev_info = st.text_input("Device OS/Hardware (DeviceInfo)", value=str(cur_ieee.get("device_info", "iOS Device")))
            addr1 = st.number_input("Billing Region Code (addr1)", min_value=1.0, max_value=600.0, value=float(cur_ieee.get("addr1", 315.0)))
            addr2 = st.number_input("Billing Country Code (addr2)", min_value=1.0, max_value=200.0, value=float(cur_ieee.get("addr2", 87.0)))

        # Velocity and Vesta Indicators expander
        with st.expander("⚙️ Advanced Vesta Velocity & Timedelta Features (C, D, V, M)", expanded=False):
            v_c1, v_c2, v_c3 = st.columns(3)
            with v_c1:
                c1_val = st.number_input("Velocity Count C1", value=float(cur_ieee.get("c1", 1.0)))
                c11_val = st.number_input("Velocity Count C11", value=float(cur_ieee.get("c11", 1.0)))
                c13_val = st.number_input("Velocity Count C13", value=float(cur_ieee.get("c13", 1.0)))
                d1_val = st.number_input("Timedelta D1 (days)", value=float(cur_ieee.get("d1", 14.0)))
            with v_c2:
                c2_val = st.number_input("Velocity Count C2", value=float(cur_ieee.get("c2", 1.0)))
                c14_val = st.number_input("Velocity Count C14", value=float(cur_ieee.get("c14", 1.0)))
                d2_val = st.number_input("Timedelta D2 (days)", value=float(cur_ieee.get("d2", 14.0)))
                d15_val = st.number_input("Timedelta D15 (days)", value=float(cur_ieee.get("d15", 14.0)))
            with v_c3:
                m1_val = st.selectbox("Match Indicator M1", ["T", "F"], index=0 if cur_ieee.get("m1", "T") == "T" else 1)
                m4_val = st.selectbox("Match Indicator M4", ["M0", "M1", "M2"], index=["M0", "M1", "M2"].index(cur_ieee.get("m4", "M0")))
                m5_val = st.selectbox("Match Indicator M5", ["T", "F"], index=0 if cur_ieee.get("m5", "F") == "T" else 1)
                m6_val = st.selectbox("Match Indicator M6", ["T", "F"], index=0 if cur_ieee.get("m6", "T") == "T" else 1)

        submit_ieee = st.form_submit_button("🚀 Score Transaction & Extract Explanations", use_container_width=True)

    if submit_ieee:
        payload_ieee = {
            "transaction_id": cur_ieee.get("transaction_id", f"TX-LIVE-{datetime.now().strftime('%H%M%S')}"),
            "transaction_dt": tx_dt,
            "transaction_amt": tx_amt,
            "product_cd": product_cd,
            "card1": card1,
            "card4": card4,
            "card6": card6,
            "addr1": addr1,
            "addr2": addr2,
            "p_emaildomain": p_email,
            "r_emaildomain": r_email,
            "device_type": dev_type,
            "device_info": dev_info,
            "m1": m1_val,
            "m4": m4_val,
            "m5": m5_val,
            "m6": m6_val,
            "c1": c1_val,
            "c2": c2_val,
            "c5": 0.0,
            "c6": 1.0,
            "c11": c11_val,
            "c13": c13_val,
            "c14": c14_val,
            "d1": d1_val,
            "d2": d2_val,
            "d15": d15_val,
            "v12": cur_ieee.get("v12", 1.0),
            "v13": cur_ieee.get("v13", 1.0),
            "v35": cur_ieee.get("v35", 1.0),
            "v36": cur_ieee.get("v36", 1.0),
            "v75": cur_ieee.get("v75", 1.0),
            "v76": cur_ieee.get("v76", 1.0),
            "v90": cur_ieee.get("v90", 0.0),
            "v91": cur_ieee.get("v91", 0.0),
            "v126": cur_ieee.get("v126", 0.0),
            "v127": cur_ieee.get("v127", 0.0),
            "v128": cur_ieee.get("v128", 0.0),
            "v257": cur_ieee.get("v257", 1.0),
            "v258": cur_ieee.get("v258", 1.0),
            "v282": cur_ieee.get("v282", 1.0),
            "v283": cur_ieee.get("v283", 1.0),
        }

        with st.spinner(f"Evaluating transaction with {selected_model_label}..."):
            result_ieee = api_client.score_and_explain_ieee(payload_ieee, model=selected_model, top_k=5)

        if not result_ieee:
            st.error("Error communicating with FinGuard AI backend microservice.")
        else:
            pred = result_ieee["prediction"]
            top_contribs = result_ieee["top_contributors"]
            step_attns = result_ieee["step_attention_breakdown"]
            narrative = result_ieee["narrative_summary"]
            explainer_meta = result_ieee.get("explainer_metadata", {})

            st.markdown("---")
            st.subheader(f"🎯 Scoring Assessment ({pred['model_name'].upper()})")

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            prob_pct = pred["fraud_probability"] * 100.0

            with m1:
                st.metric("Fraud Probability", f"{prob_pct:.2f}%")
            with m2:
                is_fraud = pred["is_fraud"]
                st.metric(
                    "Model Classification",
                    "🚨 FRAUD ALERT" if is_fraud else "✅ APPROVED",
                    delta=f"Threshold: {pred['decision_threshold']:.2f}",
                    delta_color="inverse" if is_fraud else "normal",
                )
            with m3:
                st.metric("Risk Classification", pred["risk_tier"])
            with m4:
                st.metric("Inference Latency", f"{pred['latency_ms']:.2f} ms")

            # Visualizations
            col_gauge, col_bars = st.columns([1, 1.2])
            with col_gauge:
                fig_gauge = render_risk_gauge(pred["fraud_probability"], threshold=pred["decision_threshold"])
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_bars:
                if top_contribs:
                    fig_bars = render_contributors_chart(top_contribs)
                    st.plotly_chart(fig_bars, use_container_width=True)
                else:
                    st.info(
                        "**Model Attribution Notice:**\n"
                        f"Instance-level feature attribution masks are an architectural feature of **TabNet**. "
                        f"Model **{selected_model.upper()}** produces calibrated classification probabilities without "
                        f"sequential decision masks."
                    )

            # Step Attention Heatmap (TabNet)
            if step_attns and any(len(v) > 0 for v in step_attns.values()):
                st.subheader("🔬 TabNet Sequential Decision Steps (Sparsemax Attention)")
                st.markdown(
                    "TabNet's Attentive Transformer computes a sparse attention mask at each sequential step. "
                    "Values show feature prioritization by the neural network across sequential steps."
                )
                fig_heat = render_step_attention_heatmap(step_attns)
                st.plotly_chart(fig_heat, use_container_width=True)

            # Forensic Narrative Box
            st.subheader("📋 Auditor Forensic Report & Regulatory Notice")
            st.markdown(narrative)

else:
    # =========================================================================
    # Legacy Synthetic Dataset Scoring Interface (Historical Phase 3)
    # =========================================================================
    st.subheader("⚡ Quick Load Legacy Scenarios")
    scen_col1, scen_col2, scen_col3 = st.columns(3)
    preset_legacy = None

    if scen_col1.button("📋 Scenario A: Nominal UPI Transaction", use_container_width=True):
        preset_legacy = {
            "Age": 28, "Gender": "Female", "State": "Delhi", "City": "New Delhi",
            "Account_Type": "Savings", "Account_Balance": 120000.0, "Transaction_Type": "UPI",
            "Transaction_Amount": 850.0, "Transaction_Date": datetime.now().strftime("%d-%m-%Y"),
            "Transaction_Time": "11:20:00", "Merchant_Category": "Grocery", "Device_Type": "Mobile",
            "Transaction_Device": "Mobile Device", "Transaction_Description": "Supermarket grocery payment",
        }

    if scen_col2.button("🚨 Scenario B: Suspicious Night High-Ratio", use_container_width=True):
        preset_legacy = {
            "Age": 45, "Gender": "Male", "State": "Maharashtra", "City": "Mumbai",
            "Account_Type": "Savings", "Account_Balance": 25000.0, "Transaction_Type": "NEFT",
            "Transaction_Amount": 98000.0, "Transaction_Date": datetime.now().strftime("%d-%m-%Y"),
            "Transaction_Time": "02:45:00", "Merchant_Category": "Electronics", "Device_Type": "Desktop",
            "Transaction_Device": "Web Browser", "Transaction_Description": "High-value electronics transfer",
        }

    if scen_col3.button("⚠️ Scenario C: Device Mismatch Anomaly", use_container_width=True):
        preset_legacy = {
            "Age": 52, "Gender": "Male", "State": "Karnataka", "City": "Bengaluru",
            "Account_Type": "Current", "Account_Balance": 15000.0, "Transaction_Type": "POS",
            "Transaction_Amount": 45000.0, "Transaction_Date": datetime.now().strftime("%d-%m-%Y"),
            "Transaction_Time": "23:15:00", "Merchant_Category": "Jewelry", "Device_Type": "ATM",
            "Transaction_Device": "POS Mobile Device", "Transaction_Description": "Unusual point of sale device match",
        }

    if preset_legacy:
        st.session_state["tx_form"] = preset_legacy

    current_preset = st.session_state.get("tx_form", {})

    with st.form("legacy_transaction_scoring_form"):
        st.markdown("### 📝 Legacy Synthetic Parameters")
        c1, c2, c3 = st.columns(3)

        with c1:
            tx_amount = st.number_input("Transaction Amount (₹)", min_value=1.0, value=float(current_preset.get("Transaction_Amount", 5000.0)))
            acc_balance = st.number_input("Account Balance (₹)", min_value=0.0, value=float(current_preset.get("Account_Balance", 45000.0)))
            tx_type = st.selectbox("Transaction Type", ["UPI", "NEFT", "RTGS", "IMPS", "Card", "POS", "ATM"], index=0)
            acc_type = st.selectbox("Account Type", ["Savings", "Current", "Salary"], index=0)

        with c2:
            mch_cat = st.selectbox("Merchant Category", ["Retail", "Grocery", "Electronics", "Travel", "Dining", "Entertainment", "Healthcare", "Jewelry", "Utility Bills", "Other"], index=0)
            dev_type = st.selectbox("Device Type", ["Mobile", "Desktop", "POS", "ATM"], index=0)
            tx_device = st.selectbox("Transaction Device", ["Mobile Device", "Web Browser", "Desktop/Laptop", "POS Terminal", "ATM"], index=0)
            age = st.number_input("Age", min_value=18, max_value=100, value=int(current_preset.get("Age", 32)))

        with c3:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
            state = st.selectbox("State", ["Maharashtra", "Karnataka", "Delhi", "Tamil Nadu", "Other"], index=0)
            city = st.text_input("City", value=str(current_preset.get("City", "Mumbai")))
            tx_date = st.text_input("Date (DD-MM-YYYY)", value=str(current_preset.get("Transaction_Date", "15-01-2025")))
            tx_time = st.text_input("Time (HH:MM:SS)", value=str(current_preset.get("Transaction_Time", "14:30:00")))

        tx_desc = st.text_input("Description", value=str(current_preset.get("Transaction_Description", "Standard merchant purchase")))
        submit_legacy = st.form_submit_button("🚀 Score Legacy Transaction", use_container_width=True)

    if submit_legacy:
        legacy_payload = {
            "Transaction_Amount": tx_amount,
            "Account_Balance": acc_balance,
            "Transaction_Type": tx_type,
            "Account_Type": acc_type,
            "Merchant_Category": mch_cat,
            "Device_Type": dev_type,
            "Transaction_Device": tx_device,
            "Age": age,
            "Gender": gender,
            "State": state,
            "City": city,
            "Transaction_Date": tx_date,
            "Transaction_Time": tx_time,
            "Transaction_Description": tx_desc,
            "Transaction_Currency": "INR",
        }

        with st.spinner("Scoring legacy synthetic transaction..."):
            res = api_client.score_and_explain(legacy_payload, top_k=5)

        if res:
            p = res["prediction"]
            st.metric("Fraud Probability", f"{p['fraud_probability'] * 100.0:.2f}%")
            fig_g = render_risk_gauge(p["fraud_probability"], threshold=p["decision_threshold"])
            st.plotly_chart(fig_g, use_container_width=True)
            fig_c = render_contributors_chart(res["top_contributors"])
            st.plotly_chart(fig_c, use_container_width=True)
            st.markdown(res["narrative_summary"])
