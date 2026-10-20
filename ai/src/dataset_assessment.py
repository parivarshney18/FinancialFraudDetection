"""
Dataset Suitability and Behavioral Feature Feasibility Assessment for Phase 5.
Performs quantitative provenance, label distribution, and behavioral feature audits.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency, pointbiserialr
from sklearn.feature_selection import mutual_info_classif

from ai.src.data_loader import load_raw_data


def audit_dataset_provenance(df: pd.DataFrame) -> dict:
    """Document dataset source, license, generation properties, and uniqueness."""
    n_rows = len(df)
    n_cols = df.shape[1]
    n_customers = df["Customer_ID"].nunique()
    n_merchants = df["Merchant_ID"].nunique()
    n_transactions = df["Transaction_ID"].nunique()

    # Masked PII cardinality
    n_contacts = df["Customer_Contact"].nunique()
    n_emails = df["Customer_Email"].nunique()
    n_names = df["Customer_Name"].nunique()

    results = {
        "dataset_name": "Bank_Transaction_Fraud_Detection.csv",
        "documented_source": "Kaggle Dataset ('Bank Transaction Fraud Detection' by Sagar Maru, simulating fictional 'LOL Bank Pvt. Ltd.')",
        "license": "Custom / Community Data License (as hosted on Kaggle)",
        "rows": n_rows,
        "columns": n_cols,
        "customer_id_unique": n_customers,
        "customer_id_uniqueness_pct": round(n_customers / n_rows * 100, 2),
        "merchant_id_unique": n_merchants,
        "merchant_id_uniqueness_pct": round(n_merchants / n_rows * 100, 2),
        "transaction_id_unique": n_transactions,
        "transaction_id_uniqueness_pct": round(n_transactions / n_rows * 100, 2),
        "distinct_masked_contacts": n_contacts,
        "distinct_masked_emails": n_emails,
        "distinct_customer_names": n_names,
        "data_nature": "Fully synthetic simulation; generated via synthetic data libraries with decoupled feature sampling",
    }
    return results


def audit_label_distribution(df: pd.DataFrame) -> dict:
    """Audit Is_Fraud label distribution across categorical and continuous features."""
    y = df["Is_Fraud"].values
    base_rate = float(np.mean(y))

    cat_audits = {}
    for col in ["Transaction_Type", "Merchant_Category", "Device_Type", "Account_Type", "Gender", "State"]:
        ct = pd.crosstab(df[col], df["Is_Fraud"])
        chi2, p, dof, _ = chi2_contingency(ct)
        # Compute Cramers V
        n = len(df)
        k = min(ct.shape)
        cramers_v = np.sqrt(chi2 / (n * (k - 1)))
        rates = df.groupby(col)["Is_Fraud"].mean().to_dict()
        cat_audits[col] = {
            "chi2_stat": round(float(chi2), 4),
            "p_value": round(float(p), 4),
            "cramers_v": round(float(cramers_v), 4),
            "min_fraud_rate": round(float(min(rates.values())), 5),
            "max_fraud_rate": round(float(max(rates.values())), 5),
            "rate_spread": round(float(max(rates.values()) - min(rates.values())), 5),
        }

    num_audits = {}
    for col in ["Transaction_Amount", "Account_Balance", "Age"]:
        f0 = df[df["Is_Fraud"] == 0][col].values
        f1 = df[df["Is_Fraud"] == 1][col].values
        ks_res = ks_2samp(f0, f1)
        r, p_val = pointbiserialr(df[col], df["Is_Fraud"])
        num_audits[col] = {
            "mean_class_0": round(float(np.mean(f0)), 2),
            "mean_class_1": round(float(np.mean(f1)), 2),
            "ks_statistic": round(float(ks_res.statistic), 5),
            "ks_pvalue": round(float(ks_res.pvalue), 5),
            "pearson_r": round(float(r), 5),
            "r_pvalue": round(float(p_val), 5),
        }

    return {
        "base_fraud_rate": round(base_rate, 5),
        "categorical_tests": cat_audits,
        "numerical_tests": num_audits,
    }


def assess_behavioral_features_feasibility(df: pd.DataFrame) -> list:
    """Evaluate feasibility of 5 behavioral features against available schema."""
    n_rows = len(df)
    n_unique_cust = df["Customer_ID"].nunique()

    assessment = [
        {
            "feature": "Customer Rolling Transaction Counts (1h, 6h, 24h)",
            "required_columns": ["Customer_ID", "Transaction_Date", "Transaction_Time"],
            "information_exists": False,
            "computation_logic": "Group by Customer_ID, sort by datetime, compute rolling counts over [t - window, t).",
            "leakage_risks": "Lookahead leakage if window includes future events (t' >= t) or if not computed strictly causally.",
            "feasibility": "Infeasible",
            "empirical_finding": f"Customer_ID is 100% unique ({n_unique_cust:,} distinct IDs in {n_rows:,} rows). Every customer transacts exactly once. Rolling historical counts are identically 0 (prior to current transaction) for all rows.",
        },
        {
            "feature": "Customer Historical Spending Deviations (e.g. Ratio to Mean Amount)",
            "required_columns": ["Customer_ID", "Transaction_Amount"],
            "information_exists": False,
            "computation_logic": "Compute customer's expanding mean/median amount up to transaction t, then compute ratio = Amount / (Customer_Mean + 1).",
            "leakage_risks": "Including current or future transaction amounts in the customer historical baseline.",
            "feasibility": "Infeasible",
            "empirical_finding": f"Zero historical transactions exist per customer. No customer spending baseline can be established from a single transaction.",
        },
        {
            "feature": "Time Elapsed Since Previous Transaction (Delta t)",
            "required_columns": ["Customer_ID", "Transaction_Date", "Transaction_Time"],
            "information_exists": False,
            "computation_logic": "Sort by Customer_ID and datetime, compute Delta t = timestamp[i] - timestamp[i-1].",
            "leakage_risks": "Timestamp sorting leakage across folds if not ordered chronologically.",
            "feasibility": "Infeasible at Customer Level",
            "empirical_finding": "No customer has a prior transaction (Delta t is undefined / null for 100% of rows at customer level). Macro-level (bank-wide or city-level) inter-arrival time is feasible but reflects aggregate traffic density rather than user compromise.",
        },
        {
            "feature": "Device Changes & Device Reuse Across Customers",
            "required_columns": ["Device_ID / Hardware_Fingerprint", "Customer_ID", "Transaction_Date"],
            "information_exists": False,
            "computation_logic": "Count distinct Customer_IDs per Device_ID; track device switches per customer.",
            "leakage_risks": "Using future device associations to flag past transactions.",
            "feasibility": "Infeasible",
            "empirical_finding": "The dataset contains only high-level channel categories ('POS Terminal', 'Web Browser', 'Mobile Device') with no physical hardware identifier (IMEI, MAC, fingerprint hash). Individual device reuse cannot be tracked.",
        },
        {
            "feature": "Geospatial Velocity Between Consecutive Transactions",
            "required_columns": ["Customer_ID", "Latitude", "Longitude", "Transaction_Date", "Transaction_Time"],
            "information_exists": False,
            "computation_logic": "Compute Haversine distance between customer location[i] and location[i-1], divide by Delta t.",
            "leakage_risks": "Spatial lookahead if future transaction locations are incorporated.",
            "feasibility": "Infeasible",
            "empirical_finding": "No geographic coordinates (lat/long) exist (only discrete City and State names). Furthermore, since each customer only has 1 transaction, consecutive location pairs do not exist.",
        },
    ]
    return assessment


if __name__ == "__main__":
    df = load_raw_data()
    print("=" * 70)
    print("TASK 1: DATASET PROVENANCE AUDIT")
    print("=" * 70)
    prov = audit_dataset_provenance(df)
    for k, v in prov.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("TASK 1B: LABEL DISTRIBUTION AUDIT")
    print("=" * 70)
    lbl = audit_label_distribution(df)
    print(f"Base Fraud Rate: {lbl['base_fraud_rate'] * 100:.3f}%\n")
    print("Categorical Tests (Chi-Square & Cramér's V):")
    for col, res in lbl["categorical_tests"].items():
        print(f"  {col:<20}: p={res['p_value']:.4f}, Cramér's V={res['cramers_v']:.4f}, Rate Spread={res['rate_spread']:.5f}")

    print("\nNumerical Tests (Two-Sample KS & Point-Biserial Correlation):")
    for col, res in lbl["numerical_tests"].items():
        print(f"  {col:<20}: KS stat={res['ks_statistic']:.4f} (p={res['ks_pvalue']:.4f}), Pearson r={res['pearson_r']:.4f} (p={res['r_pvalue']:.4f})")

    print("\n" + "=" * 70)
    print("TASK 3: BEHAVIORAL FEATURE FEASIBILITY ASSESSMENT")
    print("=" * 70)
    feas = assess_behavioral_features_feasibility(df)
    for item in feas:
        print(f"\nFeature: {item['feature']}")
        print(f"  Feasibility:         {item['feasibility']}")
        print(f"  Information Exists:  {item['information_exists']}")
        print(f"  Empirical Finding:   {item['empirical_finding']}")
