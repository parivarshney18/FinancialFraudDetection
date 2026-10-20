"""
FinGuard AI - Multi-Model Benchmark Comparison & Evaluation Suite.
Displays verified Phase 9 IEEE-CIS benchmark results (Stratified 5-Fold Cross-Validation
and Chronological Holdout) across TabNet, XGBoost, Random Forest, and Logistic Regression.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

from frontend.services.api_client import api_client

st.set_page_config(page_title="Model Benchmarks | FinGuard AI", page_icon="🏆", layout="wide")

st.title("🏆 IEEE-CIS Multi-Model Benchmark Comparison")
st.markdown(
    "Empirical evaluation of **TabNet**, **XGBoost**, **Random Forest**, and **Logistic Regression** "
    "on the **IEEE-CIS Fraud Detection dataset** across both **Stratified 5-Fold Cross-Validation** (400,000 train / 100,000 val) "
    "and **Chronological Holdout Validation** (40,000 train / 10,000 forward-test). "
    "Evaluated under strict leakage-safe protocols."
)

# Fetch IEEE-CIS Benchmarks
benchmarks = api_client.get_ieee_benchmarks()

if not benchmarks:
    st.error("Failed to retrieve IEEE-CIS benchmark metrics from backend microservice. Check /api/benchmarks/ieee.")
else:
    df_ieee = pd.DataFrame(benchmarks)

    st.subheader("📊 Phase 9 Benchmark Performance Metrics")
    # Rename columns for clear presentation
    display_cols = {
        "model_name": "Model",
        "cv_pr_auc": "CV PR-AUC",
        "chrono_pr_auc": "Chrono PR-AUC",
        "cv_roc_auc": "CV ROC-AUC",
        "chrono_roc_auc": "Chrono ROC-AUC",
        "cv_recall_at_1pct_fpr": "CV Recall@1%FPR",
        "chrono_recall_at_1pct_fpr": "Chrono Recall@1%FPR",
        "cv_f1_score": "CV F1-Score",
        "chrono_f1_score": "Chrono F1-Score",
        "fit_time_s": "Fit Time (s)",
        "latency_ms_per_1k": "Latency (ms/1k)",
    }
    df_display = df_ieee.rename(columns=display_cols)
    st.dataframe(df_display, use_container_width=True)

    # Key Takeaways Cards
    st.markdown("### 🔍 Architectural Observations")
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.info(
            "**XGBoost**\n\n"
            "- **Chrono PR-AUC:** 0.3554\n"
            "- **Chrono ROC-AUC:** 0.8838\n"
            "- **Recall @ 1% FPR:** 33.77%\n"
            "- Fastest fit time (0.92s) and high discrimination."
        )
    with k2:
        st.success(
            "**TabNet (Deep Tabular)**\n\n"
            "- **Chrono PR-AUC:** 0.2554\n"
            "- **Chrono ROC-AUC:** 0.8447\n"
            "- **Recall @ 1% FPR:** 20.18%\n"
            "- **Regulatory Compliant:** Instance-level Sparsemax attention masks."
        )
    with k3:
        st.warning(
            "**Random Forest**\n\n"
            "- **Chrono PR-AUC:** 0.3030\n"
            "- **Chrono ROC-AUC:** 0.8703\n"
            "- **Recall @ 1% FPR:** 31.58%\n"
            "- Solid ensemble baseline with strong tree discrimination."
        )
    with k4:
        st.error(
            "**Logistic Regression**\n\n"
            "- **Chrono PR-AUC:** 0.1287\n"
            "- **Chrono ROC-AUC:** 0.6948\n"
            "- **Recall @ 1% FPR:** 15.35%\n"
            "- Linear baseline demonstrates need for non-linear models."
        )

    # Plotly interactive comparison
    st.markdown("---")
    st.subheader("📈 Chronological Holdout vs. Cross-Validation Tradeoffs")

    plot_rows = []
    for item in benchmarks:
        def parse_val(v):
            if isinstance(v, (int, float)):
                return float(v)
            try:
                return float(str(v).split("+/-")[0].split("±")[0].strip())
            except Exception:
                return 0.0

        plot_rows.append({
            "Model": item["model_name"],
            "CV PR-AUC": parse_val(item["cv_pr_auc"]),
            "Chrono PR-AUC": parse_val(item["chrono_pr_auc"]),
            "CV ROC-AUC": parse_val(item["cv_roc_auc"]),
            "Chrono ROC-AUC": parse_val(item["chrono_roc_auc"]),
            "Recall @ 1% FPR": parse_val(item["chrono_recall_at_1pct_fpr"]),
            "Chrono F1-Score": parse_val(item["chrono_f1_score"]),
        })

    df_p = pd.DataFrame(plot_rows)
    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        df_auc = df_p[["Model", "CV PR-AUC", "Chrono PR-AUC", "Recall @ 1% FPR"]].melt(
            id_vars=["Model"], var_name="Evaluation Metric", value_name="Score"
        )
        fig_pr = px.bar(
            df_auc,
            x="Model",
            y="Score",
            color="Evaluation Metric",
            barmode="group",
            title="Precision-Recall AUC & High-Confidence Recall (FPR <= 1%)",
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"],
        )
        fig_pr.update_layout(height=380, yaxis_range=[0, 0.6])
        st.plotly_chart(fig_pr, use_container_width=True)

    with c_chart2:
        df_roc = df_p[["Model", "CV ROC-AUC", "Chrono ROC-AUC"]].melt(
            id_vars=["Model"], var_name="Protocol", value_name="ROC-AUC"
        )
        fig_roc = px.bar(
            df_roc,
            x="Model",
            y="ROC-AUC",
            color="Protocol",
            barmode="group",
            title="ROC-AUC Comparison: Stratified CV vs. Forward Chronological Holdout",
            color_discrete_sequence=["#3182bd", "#9ecae1"],
        )
        fig_roc.update_layout(height=380, yaxis_range=[0.5, 1.0])
        st.plotly_chart(fig_roc, use_container_width=True)

# Publication Figure Gallery
st.markdown("---")
st.subheader("🖼️ Publication Benchmark Figures")

fig15_path = Path("ai/reports/figures/15_ieee_model_comparison.png")
if fig15_path.exists():
    st.image(
        str(fig15_path),
        caption="Figure 15: Cross-Validation & Chronological Holdout Benchmarking Across TabNet and Baselines (IEEE-CIS)",
        use_container_width=True,
    )
else:
    st.warning("Figure 15 (15_ieee_model_comparison.png) not found.")

# Historical Legacy Comparison (Preserved in Expander)
with st.expander("📁 View Historical Legacy Synthetic Benchmark (Phase 3)", expanded=False):
    st.markdown("Historical benchmark table evaluated on the legacy 200,000 synthetic transaction dataset.")
    legacy_bench = api_client.get_benchmarks()
    if legacy_bench:
        st.dataframe(pd.DataFrame(legacy_bench), use_container_width=True)
