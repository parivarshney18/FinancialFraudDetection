"""
Comprehensive Exploratory Data Analysis (EDA) module for Financial Fraud Detection.

This module inspects the raw transaction dataset without modifying it,
computes verified summary statistics, statistical tests, data quality audits,
and saves diagnostic visualization figures and reports under ai/reports/.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal, mannwhitneyu

# Set visualization parameters
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

RAW_DATA_PATH = Path("ai/data/raw/Bank_Transaction_Fraud_Detection.csv")
REPORTS_DIR = Path("ai/reports")
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "eda_metrics.json"


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw dataset in read-only mode."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    print(f"[EDA] Loading raw dataset from: {path} ...")
    df = pd.read_csv(path)
    print(f"[EDA] Successfully loaded {len(df):,} rows and {df.shape[1]} columns.")
    return df


def audit_missing_and_duplicates(df: pd.DataFrame) -> dict:
    """Compute exact missing value and duplicate counts."""
    null_counts = df.isnull().sum().to_dict()
    null_percentages = (df.isnull().sum() / len(df) * 100).round(4).to_dict()
    exact_duplicates = int(df.duplicated().sum())
    
    unique_tx_ids = int(df['Transaction_ID'].nunique()) if 'Transaction_ID' in df.columns else None
    tx_id_duplicates = int(df['Transaction_ID'].duplicated().sum()) if 'Transaction_ID' in df.columns else None
    unique_cust_ids = int(df['Customer_ID'].nunique()) if 'Customer_ID' in df.columns else None
    unique_merch_ids = int(df['Merchant_ID'].nunique()) if 'Merchant_ID' in df.columns else None

    return {
        "null_counts": null_counts,
        "null_percentages": null_percentages,
        "exact_duplicates": exact_duplicates,
        "unique_transaction_ids": unique_tx_ids,
        "duplicate_transaction_ids": tx_id_duplicates,
        "unique_customer_ids": unique_cust_ids,
        "unique_merchant_ids": unique_merch_ids,
    }


def analyze_target(df: pd.DataFrame) -> dict:
    """Analyze the target variable Is_Fraud."""
    if 'Is_Fraud' not in df.columns:
        raise KeyError("Column 'Is_Fraud' missing from dataset.")

    counts = df['Is_Fraud'].value_counts().to_dict()
    total = len(df)
    n_non_fraud = int(counts.get(0, 0))
    n_fraud = int(counts.get(1, 0))
    pct_non_fraud = round(n_non_fraud / total * 100, 4)
    pct_fraud = round(n_fraud / total * 100, 4)
    imbalance_ratio = round(n_non_fraud / n_fraud, 4) if n_fraud > 0 else None

    return {
        "total_records": total,
        "non_fraud_count": n_non_fraud,
        "fraud_count": n_fraud,
        "non_fraud_percentage": pct_non_fraud,
        "fraud_percentage": pct_fraud,
        "imbalance_ratio": imbalance_ratio,
    }


def analyze_numerical_features(df: pd.DataFrame) -> dict:
    """Compute comprehensive descriptive statistics for numerical columns."""
    num_cols = ['Age', 'Transaction_Amount', 'Account_Balance']
    results = {}

    for col in num_cols:
        series = df[col]
        non_fraud_s = df.loc[df['Is_Fraud'] == 0, col]
        fraud_s = df.loc[df['Is_Fraud'] == 1, col]

        # Normality and statistical tests
        t_stat, t_pval = stats.ttest_ind(fraud_s, non_fraud_s)
        u_stat, u_pval = stats.mannwhitneyu(fraud_s, non_fraud_s)
        f_stat, f_pval = f_oneway(non_fraud_s, fraud_s)
        k_stat, k_pval = kruskal(non_fraud_s, fraud_s)

        results[col] = {
            "overall": {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "median": float(series.median()),
                "min": float(series.min()),
                "max": float(series.max()),
                "p25": float(np.percentile(series, 25)),
                "p75": float(np.percentile(series, 75)),
                "p99": float(np.percentile(series, 99)),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "negative_count": int((series < 0).sum()),
                "zero_count": int((series == 0).sum()),
            },
            "non_fraud": {
                "mean": float(non_fraud_s.mean()),
                "std": float(non_fraud_s.std()),
                "median": float(non_fraud_s.median()),
                "min": float(non_fraud_s.min()),
                "max": float(non_fraud_s.max()),
                "p25": float(np.percentile(non_fraud_s, 25)),
                "p75": float(np.percentile(non_fraud_s, 75)),
                "p99": float(np.percentile(non_fraud_s, 99)),
            },
            "fraud": {
                "mean": float(fraud_s.mean()),
                "std": float(fraud_s.std()),
                "median": float(fraud_s.median()),
                "min": float(fraud_s.min()),
                "max": float(fraud_s.max()),
                "p25": float(np.percentile(fraud_s, 25)),
                "p75": float(np.percentile(fraud_s, 75)),
                "p99": float(np.percentile(fraud_s, 99)),
            },
            "tests": {
                "t_statistic": float(t_stat),
                "t_pvalue": float(t_pval),
                "mannwhitney_pvalue": float(u_pval),
                "anova_f_stat": float(f_stat),
                "anova_f_pvalue": float(f_pval),
                "kruskal_h_stat": float(k_stat),
                "kruskal_h_pvalue": float(k_pval),
            }
        }

    # Correlations
    corr_pearson = df[num_cols + ['Is_Fraud']].corr().to_dict()
    corr_spearman = df[num_cols + ['Is_Fraud']].corr(method='spearman').to_dict()

    return {
        "features": results,
        "pearson_correlation": corr_pearson,
        "spearman_correlation": corr_spearman,
    }


def analyze_categorical_features(df: pd.DataFrame) -> dict:
    """Analyze distributions, fraud rates, and contingency tests for categorical columns."""
    cat_cols = ['Gender', 'Account_Type', 'Transaction_Type', 'Merchant_Category',
                'Device_Type', 'Transaction_Device', 'State', 'Transaction_Currency']
    results = {}

    for col in cat_cols:
        agg = df.groupby(col)['Is_Fraud'].agg(
            total='count',
            fraud_count='sum',
            fraud_rate=lambda x: float(x.mean() * 100)
        ).reset_index()
        agg['percentage_of_all'] = (agg['total'] / len(df) * 100).round(2)

        # Chi-square contingency
        ct = pd.crosstab(df[col], df['Is_Fraud'])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p_val, dof, _ = chi2_contingency(ct)
            cramers_v = float(np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1))))
        else:
            chi2, p_val, dof, cramers_v = 0.0, 1.0, 0, 0.0

        results[col] = {
            "categories": agg.to_dict(orient='records'),
            "cardinality": int(df[col].nunique()),
            "chi2_stat": float(chi2),
            "chi2_pvalue": float(p_val),
            "degrees_of_freedom": int(dof),
            "cramers_v": float(cramers_v),
        }

    return results


def analyze_temporal_features(df: pd.DataFrame) -> dict:
    """Analyze temporal attributes: dates, days of week, and hours."""
    dates = pd.to_datetime(df['Transaction_Date'], format='%d-%m-%Y', errors='coerce')
    times = pd.to_datetime(df['Transaction_Time'], format='%H:%M:%S', errors='coerce')
    hours = times.dt.hour

    min_date = str(dates.min().date())
    max_date = str(dates.max().date())
    date_range_days = int((dates.max() - dates.min()).days + 1)

    # By Day of week
    df_temp = pd.DataFrame({
        'Date': dates,
        'DayOfWeek': dates.dt.day_name(),
        'Hour': hours,
        'Is_Fraud': df['Is_Fraud']
    })

    dow_agg = df_temp.groupby('DayOfWeek')['Is_Fraud'].agg(
        total='count',
        fraud_count='sum',
        fraud_rate=lambda x: float(x.mean() * 100)
    ).reset_index().to_dict(orient='records')

    # By Hour
    hour_agg = df_temp.groupby('Hour')['Is_Fraud'].agg(
        total='count',
        fraud_count='sum',
        fraud_rate=lambda x: float(x.mean() * 100)
    ).reset_index().to_dict(orient='records')

    # Chi-square tests
    ct_dow = pd.crosstab(df_temp['DayOfWeek'], df_temp['Is_Fraud'])
    chi2_dow, p_dow, _, _ = chi2_contingency(ct_dow)

    ct_hr = pd.crosstab(df_temp['Hour'], df_temp['Is_Fraud'])
    chi2_hr, p_hr, _, _ = chi2_contingency(ct_hr)

    return {
        "min_date": min_date,
        "max_date": max_date,
        "date_range_days": date_range_days,
        "day_of_week_distribution": dow_agg,
        "day_of_week_chi2_pvalue": float(p_dow),
        "hourly_distribution": hour_agg,
        "hourly_chi2_pvalue": float(p_hr),
    }


def audit_leakage_and_quality(df: pd.DataFrame) -> dict:
    """Audit data quality, redundancy, high cardinality, and leakage vectors."""
    n_total = len(df)

    # Redundancy checks
    branch_redundancy = bool((df['Bank_Branch'] == df['City'] + ' Branch').all())
    location_redundancy = bool((df['Transaction_Location'] == df['City'] + ', ' + df['State']).all())
    currency_constant = bool((df['Transaction_Currency'] == 'INR').all())

    # High cardinality
    tx_id_unq = int(df['Transaction_ID'].nunique())
    cust_id_unq = int(df['Customer_ID'].nunique())
    merch_id_unq = int(df['Merchant_ID'].nunique())
    contact_unq = int(df['Customer_Contact'].nunique())
    email_unq = int(df['Customer_Email'].nunique())

    # Inconsistency checks
    amount_gt_balance_count = int((df['Transaction_Amount'] > df['Account_Balance']).sum())
    amount_gt_balance_pct = round(amount_gt_balance_count / n_total * 100, 2)
    amount_gt_balance_fraud_rate = float(df.loc[df['Transaction_Amount'] > df['Account_Balance'], 'Is_Fraud'].mean() * 100)
    amount_le_balance_fraud_rate = float(df.loc[df['Transaction_Amount'] <= df['Account_Balance'], 'Is_Fraud'].mean() * 100)

    # Device mismatch check
    # Check if Device_Type cross Transaction_Device behaves like independent product
    device_cross = pd.crosstab(df['Device_Type'], df['Transaction_Device'])
    chi2_dev, p_dev, _, _ = chi2_contingency(device_cross)

    return {
        "branch_is_city_branch_redundant": branch_redundancy,
        "location_is_city_state_redundant": location_redundancy,
        "currency_is_constant_inr": currency_constant,
        "unique_transaction_ids": tx_id_unq,
        "unique_customer_ids": cust_id_unq,
        "unique_merchant_ids": merch_id_unq,
        "unique_customer_contacts": contact_unq,
        "unique_customer_emails": email_unq,
        "amount_gt_balance_count": amount_gt_balance_count,
        "amount_gt_balance_pct": amount_gt_balance_pct,
        "amount_gt_balance_fraud_rate": amount_gt_balance_fraud_rate,
        "amount_le_balance_fraud_rate": amount_le_balance_fraud_rate,
        "device_crosstab_chi2_pvalue": float(p_dev),
    }


def generate_all_visualizations(df: pd.DataFrame, output_dir: Path = FIGURES_DIR):
    """Generate high-resolution diagnostic plots for reports and presentations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[EDA] Generating figures in: {output_dir} ...")

    # 1. Target Distribution
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    counts = df['Is_Fraud'].value_counts()
    colors = ['#2b5c8f', '#d9534f']
    
    # Bar plot
    bars = axes[0].bar(['Legitimate (0)', 'Fraud (1)'], [counts[0], counts[1]], color=colors, width=0.5, edgecolor='black', linewidth=0.8)
    axes[0].set_title('Target Class Frequency Distribution', fontweight='bold')
    axes[0].set_ylabel('Number of Transactions')
    axes[0].set_ylim(0, 220000)
    for bar in bars:
        h = bar.get_height()
        axes[0].annotate(f'{h:,}\n({h/len(df)*100:.2f}%)',
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points",
                         ha='center', va='bottom', fontweight='bold')

    # Donut plot
    axes[1].pie([counts[0], counts[1]], labels=['Legitimate\n94.96%', 'Fraud\n5.04%'],
                colors=colors, startangle=140, explode=[0, 0.1],
                wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2))
    axes[1].set_title('Target Class Imbalance (18.83 : 1)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / "01_target_distribution.png", dpi=300)
    plt.close(fig)
    print("  Saved: 01_target_distribution.png")

    # 2. Numerical Distributions & Box Plots
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    features = [
        ('Transaction_Amount', 'Transaction Amount (INR)'),
        ('Account_Balance', 'Account Balance (INR)'),
        ('Age', 'Customer Age (Years)')
    ]

    for i, (col, title) in enumerate(features):
        # KDE / Hist
        sns.histplot(data=df, x=col, hue='Is_Fraud', bins=40, kde=True,
                     palette={0: '#2b5c8f', 1: '#d9534f'}, ax=axes[i, 0],
                     stat="density", common_norm=False, alpha=0.4)
        axes[i, 0].set_title(f'{title} — Density by Fraud Status', fontweight='bold')
        axes[i, 0].set_xlabel(title)
        axes[i, 0].legend(['Fraud (1)', 'Legitimate (0)'])

        # Boxplot
        sns.boxplot(data=df, x='Is_Fraud', y=col, ax=axes[i, 1],
                    hue='Is_Fraud', palette={0: '#2b5c8f', 1: '#d9534f'},
                    width=0.4, boxprops=dict(alpha=0.8), legend=False)
        axes[i, 1].set_title(f'{title} — Boxplot by Class', fontweight='bold')
        axes[i, 1].set_xticks([0, 1])
        axes[i, 1].set_xticklabels(['Legitimate (0)', 'Fraud (1)'])
        axes[i, 1].set_xlabel('')
        axes[i, 1].set_ylabel(title)

    plt.tight_layout()
    fig.savefig(output_dir / "02_numerical_distributions.png", dpi=300)
    plt.close(fig)
    print("  Saved: 02_numerical_distributions.png")

    # 3. Categorical Distributions and Fraud Rates
    cat_pairs = [
        ('Transaction_Type', 'Transaction Type'),
        ('Merchant_Category', 'Merchant Category'),
        ('Device_Type', 'Device Type'),
        ('Account_Type', 'Account Type')
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for idx, (col, label) in enumerate(cat_pairs):
        agg = df.groupby(col)['Is_Fraud'].agg(total='count', fraud_rate=lambda x: x.mean() * 100).reset_index()
        ax = axes[idx]
        ax2 = ax.twinx()

        # Bar for volume
        x = np.arange(len(agg))
        width = 0.4
        bars = ax.bar(x - width/2, agg['total'], width, color='#3470a3', alpha=0.85, label='Volume', edgecolor='black', linewidth=0.5)
        # Line for fraud rate
        line = ax2.plot(x + width/2, agg['fraud_rate'], color='#d9534f', marker='o', linewidth=2.2, label='Fraud Rate (%)')

        ax.set_xticks(x)
        ax.set_xticklabels(agg[col], rotation=25, ha='right')
        ax.set_title(f'{label}: Volume vs Fraud Rate', fontweight='bold')
        ax.set_ylabel('Transaction Count', color='#3470a3')
        ax2.set_ylabel('Fraud Rate (%)', color='#d9534f')
        ax2.set_ylim(0, 10)  # show clearly relative to baseline 5%
        ax2.axhline(5.04, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)

    plt.tight_layout()
    fig.savefig(output_dir / "03_categorical_distributions.png", dpi=300)
    plt.close(fig)
    print("  Saved: 03_categorical_distributions.png")

    # 4. Temporal Analysis
    dates = pd.to_datetime(df['Transaction_Date'], format='%d-%m-%Y')
    hours = pd.to_datetime(df['Transaction_Time'], format='%H:%M:%S').dt.hour
    df_temp = pd.DataFrame({'Date': dates, 'Day': dates.dt.day, 'DayOfWeek': dates.dt.day_name(), 'Hour': hours, 'Is_Fraud': df['Is_Fraud']})

    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    # Daily volume and fraud rate over January 2025
    daily = df_temp.groupby('Day')['Is_Fraud'].agg(total='count', fraud_rate=lambda x: x.mean() * 100).reset_index()
    ax_d1 = axes[0]
    ax_d2 = ax_d1.twinx()
    ax_d1.bar(daily['Day'], daily['total'], color='#5c82a6', alpha=0.7, edgecolor='black', linewidth=0.5, label='Tx Count')
    ax_d2.plot(daily['Day'], daily['fraud_rate'], color='#c9302c', marker='s', linewidth=1.8, label='Fraud Rate (%)')
    ax_d1.set_title('Daily Transaction Volume and Fraud Rate Across January 2025', fontweight='bold')
    ax_d1.set_xlabel('Day of Month (January 2025)')
    ax_d1.set_ylabel('Daily Transactions')
    ax_d2.set_ylabel('Fraud Rate (%)')
    ax_d2.set_ylim(0, 10)
    ax_d2.axhline(5.04, color='gray', linestyle='--', linewidth=0.8)

    # Day of Week
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow = df_temp.groupby('DayOfWeek')['Is_Fraud'].agg(total='count', fraud_rate=lambda x: x.mean() * 100).reindex(dow_order).reset_index()
    ax_w1 = axes[1]
    ax_w2 = ax_w1.twinx()
    ax_w1.bar(dow['DayOfWeek'], dow['total'], color='#6baed6', width=0.45, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax_w2.plot(dow['DayOfWeek'], dow['fraud_rate'], color='#d9534f', marker='o', linewidth=2.2)
    ax_w1.set_title('Weekly Transaction Pattern & Fraud Rate', fontweight='bold')
    ax_w1.set_ylabel('Transaction Volume')
    ax_w2.set_ylabel('Fraud Rate (%)')
    ax_w2.set_ylim(0, 10)
    ax_w2.axhline(5.04, color='gray', linestyle='--', linewidth=0.8)

    # Hourly pattern
    hourly = df_temp.groupby('Hour')['Is_Fraud'].agg(total='count', fraud_rate=lambda x: x.mean() * 100).reset_index()
    ax_h1 = axes[2]
    ax_h2 = ax_h1.twinx()
    ax_h1.bar(hourly['Hour'], hourly['total'], color='#9ecae1', width=0.6, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax_h2.plot(hourly['Hour'], hourly['fraud_rate'], color='#bd0026', marker='^', linewidth=2)
    ax_h1.set_title('Hourly Transaction Distribution (00:00 - 23:00) & Fraud Rate', fontweight='bold')
    ax_h1.set_xlabel('Hour of Day')
    ax_h1.set_ylabel('Hourly Transactions')
    ax_h2.set_ylabel('Fraud Rate (%)')
    ax_h2.set_ylim(0, 10)
    ax_h2.axhline(5.04, color='gray', linestyle='--', linewidth=0.8)
    ax_h1.set_xticks(range(24))

    plt.tight_layout()
    fig.savefig(output_dir / "04_temporal_analysis.png", dpi=300)
    plt.close(fig)
    print("  Saved: 04_temporal_analysis.png")

    # 5. Amount vs Balance Scatter with Ratio Diagonal
    fig, ax = plt.subplots(figsize=(9, 7))
    sample_df = df.sample(n=10000, random_state=42)
    non_f = sample_df[sample_df['Is_Fraud'] == 0]
    f = sample_df[sample_df['Is_Fraud'] == 1]

    ax.scatter(non_f['Account_Balance'], non_f['Transaction_Amount'], c='#2b5c8f', alpha=0.25, s=15, label='Legitimate (0)')
    ax.scatter(f['Account_Balance'], f['Transaction_Amount'], c='#d9534f', alpha=0.75, s=25, edgecolor='black', linewidth=0.5, label='Fraud (1)')
    ax.plot([0, 100000], [0, 100000], color='black', linestyle='--', linewidth=1.5, label='Amount = Balance Line')
    ax.set_title('Transaction Amount vs Account Balance (10,000 Sample Points)', fontweight='bold')
    ax.set_xlabel('Account Balance (INR)')
    ax.set_ylabel('Transaction Amount (INR)')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 105000)
    ax.set_ylim(0, 105000)
    plt.tight_layout()
    fig.savefig(output_dir / "05_amount_vs_balance.png", dpi=300)
    plt.close(fig)
    print("  Saved: 05_amount_vs_balance.png")

    # 6. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(7, 5.5))
    num_cols = ['Transaction_Amount', 'Account_Balance', 'Age', 'Is_Fraud']
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.4f', cmap='Blues', vmin=-0.05, vmax=0.05, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title('Correlation Heatmap: Numerical Features & Is_Fraud', fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / "06_correlation_heatmap.png", dpi=300)
    plt.close(fig)
    print("  Saved: 06_correlation_heatmap.png")

    # 7. Geographic Fraud Rates (Top 15 States)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    state_agg = df.groupby('State')['Is_Fraud'].agg(total='count', fraud_rate=lambda x: x.mean() * 100).sort_values(by='total', ascending=False).head(15).reset_index()
    x = np.arange(len(state_agg))
    bars = ax.bar(x, state_agg['fraud_rate'], color='#4575b4', edgecolor='black', linewidth=0.6, width=0.5)
    ax.axhline(5.04, color='#d73027', linestyle='--', linewidth=1.5, label='National Average (5.04%)')
    ax.set_xticks(x)
    ax.set_xticklabels(state_agg['State'], rotation=40, ha='right')
    ax.set_title('Fraud Rate Across Top 15 Indian States by Volume', fontweight='bold')
    ax.set_ylabel('Fraud Rate (%)')
    ax.set_ylim(0, 8)
    ax.legend()
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "07_geographic_fraud_rate.png", dpi=300)
    plt.close(fig)
    print("  Saved: 07_geographic_fraud_rate.png")

    # 8. Device Matrix Heatmap
    fig, ax = plt.subplots(figsize=(12, 4.5))
    dev_ct = pd.crosstab(df['Device_Type'], df['Transaction_Device'])
    sns.heatmap(dev_ct, cmap='YlGnBu', annot=True, fmt='d', ax=ax, cbar_kws={'label': 'Transaction Count'})
    ax.set_title('Cross-Tabulation Matrix: Device_Type vs Transaction_Device', fontweight='bold')
    ax.set_xlabel('Transaction_Device (20 granular categories)')
    ax.set_ylabel('Device_Type (4 high-level categories)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(output_dir / "08_device_matrix_heatmap.png", dpi=300)
    plt.close(fig)
    print("  Saved: 08_device_matrix_heatmap.png")


def run_full_eda():
    """Execute complete end-to-end EDA pipeline."""
    df = load_dataset()

    print("[EDA] Computing missing values and duplicates...")
    missing_dup = audit_missing_and_duplicates(df)

    print("[EDA] Computing target distribution...")
    target_stats = analyze_target(df)

    print("[EDA] Computing numerical feature statistics...")
    num_stats = analyze_numerical_features(df)

    print("[EDA] Computing categorical feature statistics...")
    cat_stats = analyze_categorical_features(df)

    print("[EDA] Computing temporal patterns...")
    temp_stats = analyze_temporal_features(df)

    print("[EDA] Auditing data quality and leakage...")
    leakage_stats = audit_leakage_and_quality(df)

    all_metrics = {
        "dataset_shape": {"rows": len(df), "columns": df.shape[1]},
        "missing_and_duplicates": missing_dup,
        "target_distribution": target_stats,
        "numerical_features": num_stats,
        "categorical_features": cat_stats,
        "temporal_features": temp_stats,
        "leakage_and_quality": leakage_stats,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[EDA] Saved exact metrics to {METRICS_PATH}")

    generate_all_visualizations(df, FIGURES_DIR)
    print("[EDA] All EDA pipeline tasks finished successfully!")


if __name__ == '__main__':
    run_full_eda()
