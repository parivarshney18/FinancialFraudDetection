"""
FinGuard AI - Exploratory Data Analysis (EDA) Insights & Dataset Profiling.
Provides verified ground-truth statistical audits of the IEEE-CIS Fraud Detection dataset (Phases 7–8)
alongside historical diagnostic figures from the Phase 1 synthetic transaction baseline.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="EDA Insights | FinGuard AI", page_icon="📈", layout="wide")

st.title("📈 Exploratory Data Analysis & Empirical Dataset Profiling")
st.markdown(
    "Ground-truth statistical properties, table topologies, missingness profiles, "
    "and feature architectures for financial fraud detection across the real-world **IEEE-CIS Fraud Detection dataset** "
    "and the legacy synthetic baseline."
)

# Navigation tabs: IEEE-CIS Production vs Legacy Synthetic
dataset_tab1, dataset_tab2 = st.tabs([
    "🌐 IEEE-CIS Dataset Profiling (Phases 7–8 Ground Truth)",
    "🏛️ Legacy Synthetic Dataset Diagnostic Figures (Phase 1)"
])

# ==============================================================================
# TAB 1: IEEE-CIS DATASET PROFILING (GROUND TRUTH)
# ==============================================================================
with dataset_tab1:
    st.subheader("🌐 Real-World Payment Fraud: IEEE-CIS Benchmark")
    st.markdown(
        "Empirical profiling of the **IEEE-CIS Fraud Detection dataset**, provided by **Vesta Corporation** "
        "(leading payment fraud solution provider) and the **IEEE Computational Intelligence Society**. "
        "Comprises authentic Card-Not-Present (CNP) e-commerce transactions across a 182-day operating window."
    )

    # Key Dataset Statistics
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total Transactions", "590,540")
    with s2:
        st.metric("Legitimate Class", "569,877 (96.501%)")
    with s3:
        st.metric("Fraudulent Class", "20,663 (3.499%)")
    with s4:
        st.metric("Imbalance Ratio", "27.58 : 1")

    st.markdown("---")

    # Table Topology & Referential Integrity
    st.subheader("📋 Table Topology & Relational Architecture")
    st.markdown(
        "The raw dataset is partitioned into a transaction log and a digital identity telemetry table, "
        "linked by a strictly unique primary key (`TransactionID`)."
    )

    topo_data = {
        "Dataset Table": ["train_transaction.csv", "train_identity.csv", "Merged Dataset (Left Join)"],
        "Raw File Size": ["651.69 MB", "25.30 MB", "~1.94 GB (Extrapolated RAM)"],
        "Total Rows": ["590,540", "144,233", "590,540"],
        "Total Columns": ["394", "41", "434 (394 + 41 - 1 Key)"],
        "Primary Key": ["TransactionID", "TransactionID", "TransactionID"],
        "Key Duplicates": ["0 (Strictly Unique)", "0 (Strictly Unique)", "0"],
        "Identity Coverage": ["—", "—", "144,233 (24.42%)"],
    }
    st.dataframe(pd.DataFrame(topo_data), use_container_width=True)

    st.info(
        "**Referential Integrity Confirmed:** 100% of identity IDs exist in the transaction table. "
        "24.42% of transactions possess digital identity signatures (e.g. `DeviceType`, `DeviceInfo`), "
        "while 75.58% represent guest checkout or transactions without device telemetry."
    )

    # Visualizations: Class Balance and Missingness
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### ⚖️ Ground-Truth Class Distribution")
        df_class = pd.DataFrame({
            "Class": ["Legitimate (isFraud=0)", "Fraudulent (isFraud=1)"],
            "Count": [569877, 20663],
            "Percentage": [96.501, 3.499],
        })
        fig_class = px.bar(
            df_class,
            x="Class",
            y="Count",
            text="Count",
            color="Class",
            color_discrete_sequence=["#2b6cb0", "#e53e3e"],
            title="IEEE-CIS Class Breakdown (590,540 Total Records)",
        )
        fig_class.update_traces(texttemplate="%{text:,.0f} (%{customdata[0]:.2f}%)", customdata=df_class[["Percentage"]])
        fig_class.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_class, use_container_width=True)

    with col_chart2:
        st.markdown("#### 🔍 Missingness Profile by Feature Group")
        missing_data = pd.DataFrame({
            "Feature Block": [
                "Count Features (C1–C14)",
                "V-Features (V1–V339)",
                "Time Deltas (D1–D15)",
                "Match Flags (M1–M9)",
                "Identity Telemetry (id_01–id_38)",
            ],
            "Null Rate (%)": [0.00, 43.27, 64.53, 65.91, 78.44],
            "Column Count": [14, 339, 15, 9, 40],
        })
        fig_missing = px.bar(
            missing_data,
            x="Null Rate (%)",
            y="Feature Block",
            orientation="h",
            color="Null Rate (%)",
            color_continuous_scale="Reds",
            title="Missing Value Percentage Across Feature Blocks",
            text="Null Rate (%)",
        )
        fig_missing.update_traces(texttemplate="%{text:.1f}%")
        fig_missing.update_layout(height=360, yaxis={"autorange": "reversed"})
        st.plotly_chart(fig_missing, use_container_width=True)

    st.caption(
        "💡 **Key Discovery:** Vesta Count Features (`C1`–`C14`) exhibit **0.00% missing values**, "
        "providing uninterrupted multi-window velocity counters across all transactions."
    )

    st.markdown("---")

    # Curated 46-Feature Research Architecture
    st.subheader("🔬 Curated 46-Feature Research Architecture (Phase 8)")
    st.markdown(
        "To ensure CPU feasibility (16 GB RAM / 8-core CPU) and prevent memory exhaustion during TabNet attention computation, "
        "Phase 8 established a **curated 46-feature research subset** (11 Categorical + 35 Numerical). "
        "This achieves an **89.4% dimensionality reduction** from the 434 raw columns while preserving high-signal fraud discriminators."
    )

    feature_blocks = [
        {"Block": "Categorical Entity & Channel", "Count": 7, "Features": "ProductCD, card4, card6, P_emaildomain, R_emaildomain, DeviceType, DeviceInfo", "Type": "Discrete (Ordinal)", "Role": "Channel, card scheme, domain risk, hardware model."},
        {"Block": "Verification Match Flags", "Count": 4, "Features": "M1, M4, M5, M6", "Type": "Binary / String", "Role": "Name, billing address, cardholder match flags."},
        {"Block": "Core Financial & Spatial", "Count": 4, "Features": "TransactionAmt, card1, addr1, addr2", "Type": "Continuous / ID", "Role": "Transaction amount (USD), issuer bank ID, billing region."},
        {"Block": "Vesta Velocity Counters", "Count": 7, "Features": "C1, C2, C5, C6, C11, C13, C14", "Type": "Continuous", "Role": "0.00% null rate! Multi-window transaction counters."},
        {"Block": "Vesta Timedelta Telemetry", "Count": 3, "Features": "D1, D2, D15", "Type": "Continuous", "Role": "Days elapsed since prior card/account activity."},
        {"Block": "Top Salient V-Features", "Count": 15, "Features": "V12, V13, V35, V36, V75, V76, V90, V91, V126, V127, V128, V257, V258, V282, V283", "Type": "Continuous", "Role": "High-importance interaction clusters tied to product codes."},
        {"Block": "Engineered Cyclical Temporal", "Count": 2, "Features": "Hour_Sin, Hour_Cos", "Type": "Continuous [-1, 1]", "Role": "Trigonometric encoding of hour-of-day: sin/cos(2π·h / 24)."},
        {"Block": "Engineered Monetary Deviation", "Count": 2, "Features": "Log_TransactionAmt, Amt_to_Card1_Mean_Ratio", "Type": "Continuous", "Role": "Log1p scale compression & ratio to cardholder expanding mean."},
        {"Block": "Engineered Cardholder Recency", "Count": 2, "Features": "card1_tx_count_prior, Time_Since_Prev_Card1_Tx", "Type": "Continuous", "Role": "Prior tx count and inter-transaction elapsed seconds (Δt)."},
    ]
    df_feat = pd.DataFrame(feature_blocks)
    st.dataframe(df_feat, use_container_width=True)

    # Sample vs Full-Dataset Comparison
    st.markdown("---")
    st.subheader("⏱️ Sample Telemetry vs. Full-Dataset Temporal Drift")
    st.markdown(
        "Comparing the 50,000-row chronological sample (used in exploratory development) against the full 590,540-row dataset "
        "demonstrates significant temporal concept drift."
    )

    drift_data = {
        "Metric / Dimension": [
            "Total Transactions",
            "Legitimate Count",
            "Fraudulent Count",
            "Fraud Prevalence (%)",
            "Imbalance Ratio",
            "Identity Match Rate (%)",
            "Temporal Window",
        ],
        "50,000-Row Chronological Sample": [
            "50,000",
            "48,643 (97.286%)",
            "1,357 (2.714%)",
            "2.714%",
            "35.85 : 1",
            "32.14%",
            "Initial ~15.4 operating days",
        ],
        "Complete Dataset (590,540 Rows)": [
            "590,540",
            "569,877 (96.501%)",
            "20,663 (3.499%)",
            "3.499%",
            "27.58 : 1",
            "24.42%",
            "Full 182 operating days",
        ],
        "Analytical Finding": [
            "Sample represents earliest cohort",
            "Nominal majority throughout",
            "Fraud density spikes in later months",
            "Fraud rate delta: +0.785% over time",
            "Higher fraud concentration in later periods",
            "Early cohort had slightly higher identity capture",
            "Validates necessity of chronological holdout test",
        ],
    }
    st.dataframe(pd.DataFrame(drift_data), use_container_width=True)

    st.warning(
        "⚠️ **Methodological Notice:** The temporal increase in fraud prevalence (+0.785%) highlights adversarial concept drift. "
        "Standard random cross-validation overestimates performance; chronological holdout validation provides the true production baseline."
    )

# ==============================================================================
# TAB 2: LEGACY SYNTHETIC DATASET DIAGNOSTIC FIGURES (PHASE 1)
# ==============================================================================
with dataset_tab2:
    st.subheader("🏛️ Historical Diagnostic Figures: Phase 1 Synthetic Baseline")
    st.markdown(
        "Exploratory analysis conducted on the initial **200,000 synthetic banking transaction dataset** (Phase 1). "
        "These figures documented the structural limitations that motivated migrating to the real-world IEEE-CIS dataset."
    )

    # Legacy Dataset Statistics
    l1, l2, l3, l4 = st.columns(4)
    with l1:
        st.metric("Total Rows", "200,000")
    with l2:
        st.metric("Legitimate Class", "189,912 (94.96%)")
    with l3:
        st.metric("Fraud Class", "10,088 (5.04%)")
    with l4:
        st.metric("Imbalance Ratio", "18.83 : 1")

    st.markdown("---")

    figures = [
        ("01_target_distribution.png", "Target Class Distribution (Is_Fraud)", "Demonstrates severe class imbalance with exactly 5.044% fraud cases."),
        ("02_numerical_distributions.png", "Numerical Feature Distributions", "Transaction amount and account balance distributions across fraud vs legitimate classes."),
        ("03_categorical_distributions.png", "Categorical Attribute Frequencies", "High-cardinality representations across State, City, Merchant Category, and Transaction Type."),
        ("04_temporal_analysis.png", "Temporal Fraud Velocity & Hourly Patterns", "Analysis of nocturnal transaction volume and fraud probabilities across hours and days."),
        ("05_amount_vs_balance.png", "Transaction Amount vs Account Balance", "Scatter analysis revealing fraud cases where transaction amount equals or exceeds account balance."),
        ("06_correlation_heatmap.png", "Linear Correlation Heatmap", "Verifies near-zero marginal linear correlation between raw features and fraud target, necessitating deep tabular models."),
        ("07_geographic_fraud_rate.png", "Geographic Fraud Rates Across Indian States", "Distribution of fraud concentrations across Indian states and union territories."),
        ("08_device_matrix_heatmap.png", "Device Type vs Transaction Device Compatibility", "Matrix identifying contextual device mismatches (e.g. ATM device types executing POS mobile transactions)."),
    ]

    fig_dir = Path("ai/reports/figures")
    tab_labels = [f"Fig {i+1}" for i in range(len(figures))]
    tabs = st.tabs(tab_labels)

    for idx, (filename, title, description) in enumerate(figures):
        with tabs[idx]:
            st.markdown(f"### {title}")
            st.write(description)
            file_path = fig_dir / filename
            if file_path.exists():
                st.image(str(file_path), caption=f"Figure {idx+1}: {title}", use_container_width=True)
            else:
                st.warning(f"Figure file {filename} not found.")
