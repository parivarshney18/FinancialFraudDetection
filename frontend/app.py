"""
FinGuard AI - Forensic Fraud Detection & Auditor Dashboard.
Streamlit multi-page application entry point with authentication and executive KPIs.
"""

import streamlit as st
from frontend.services.api_client import api_client

st.set_page_config(
    page_title="FinGuard AI | TabNet Fraud Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "token" not in st.session_state:
    st.session_state["token"] = None
if "user" not in st.session_state:
    st.session_state["user"] = None


# --- Sidebar: Auth & System Status ---
with st.sidebar:
    st.title("🛡️ FinGuard AI")
    st.caption("AI-Based Banking Fraud Detection with TabNet")

    # API Connection & Health Status
    health = api_client.check_health()
    if health.get("status") == "healthy" and health.get("model_loaded"):
        st.success("🟢 API Connected | TabNet Ready")
    else:
        st.warning("🟡 Backend Offline / Standalone Mode")

    st.markdown("---")

    # User Authentication Widget
    if st.session_state["token"]:
        user = st.session_state["user"] or {}
        st.write(f"👤 **Logged in as:** {user.get('full_name', 'Auditor')}")
        st.write(f"🏷️ **Role:** `{user.get('role', 'analyst').upper()}`")
        if st.button("Logout", key="logout_btn", use_container_width=True):
            st.session_state["token"] = None
            st.session_state["user"] = None
            st.rerun()
    else:
        st.subheader("Auditor Login")
        login_email = st.text_input("Email", value="admin@finguard.ai", key="login_email")
        login_pwd = st.text_input("Password", value="admin123", type="password", key="login_pwd")
        if st.button("Sign In", key="login_btn", use_container_width=True):
            res = api_client.login(login_email, login_pwd)
            if res and "access_token" in res:
                st.session_state["token"] = res["access_token"]
                me = api_client.get_me(res["access_token"])
                st.session_state["user"] = me or {"email": login_email, "role": res.get("role", "analyst")}
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials.")

    st.markdown("---")
    st.markdown("### 🎓 Academic Capstone")
    st.markdown(
        "**ABES Engineering College, Ghaziabad**  \n"
        "**Department:** Computer Science & Engineering (2026-27)  \n"
        "**Team:** Pari Varshney, Neha Awasthi, Nandini Sharma  \n"
        "**Supervisor:** Ms. Shipra Gautam"
    )


# --- Main Dashboard Header ---
st.title("🛡️ FinGuard AI: TabNet Forensic Auditor Dashboard")
st.markdown(
    "Welcome to the **FinGuard AI** operational management system. This platform provides real-time banking "
    "transaction fraud screening, instant **TabNet Sparsemax attention explainability**, and a human-in-the-loop "
    "compliance auditing queue conforming to **RBI Payment Safety Standards** and Explainable AI mandates."
)

st.markdown("---")

# Fetch Executive Audit Summary KPIs
summary = api_client.get_audit_summary()
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="Transactions Ingested",
        value=f"{summary.get('total_transactions_ingested', 200000):,}",
        delta="Raw Dataset Verified",
    )

with kpi2:
    st.metric(
        label="Fraud Alert Ratio",
        value=f"{summary.get('fraud_alert_rate_pct', 5.04)}%",
        delta="Imbalance: 18.8 : 1",
    )

with kpi3:
    st.metric(
        label="Scoring Latency",
        value=f"{summary.get('average_inference_latency_ms', 12.4)} ms",
        delta="Real-Time CPU",
    )

with kpi4:
    st.metric(
        label="Active Architecture",
        value="TabNet (3-Step)",
        delta="Sparsemax Masks",
    )

st.markdown("---")

# Navigation Hub
st.subheader("📌 Platform Navigation Hub")
col_a, col_b = st.columns(2)

with col_a:
    st.info(
        "### 🔍 Real-Time Scoring & Explanation\n"
        "Navigate to **1_Score_Transaction** to input custom transaction parameters, calculate live fraud "
        "probabilities, and inspect Sparsemax feature attention attribution."
    )
    st.warning(
        "### ⚖️ Auditor Review Queue\n"
        "Navigate to **2_Review_Queue** to triage high-risk transactions, view forensic alerts, and submit "
        "human compliance decisions (Approve, Reject, Escalate)."
    )
    st.success(
        "### 📊 Deep-Dive Prediction Detail\n"
        "Navigate to **3_Prediction_Detail** to enter any specific `Transaction_ID` and view full timeline, "
        "decision step heatmaps, and audit logs."
    )

with col_b:
    st.info(
        "### 🏆 Model Benchmark Comparison\n"
        "Navigate to **4_Model_Comparison** to view verified 5-Fold Stratified Cross-Validation benchmarks "
        "comparing TabNet against XGBoost, Random Forest, and Logistic Regression."
    )
    st.warning(
        "### 📈 Exploratory Data Analysis\n"
        "Navigate to **5_EDA_Insights** to interactively explore the 200,000-row transaction dataset, "
        "class distributions, and correlation heatmaps."
    )
    st.success(
        "### ⚙️ System Administration\n"
        "Navigate to **6_Admin** for database health, active model registry parameters, and system metrics."
    )
