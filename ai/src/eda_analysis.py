"""
Exploratory Data Analysis (EDA) Script for Financial Fraud Detection.
Analyzes Bank_Transaction_Fraud_Detection.csv without modifying the raw data.
Computes comprehensive metrics and generates diagnostic visualizations.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.autolayout'] = True

RAW_DATA_PATH = Path("ai/data/raw/Bank_Transaction_Fraud_Detection.csv")
REPORTS_DIR = Path("ai/reports")
FIGURES_DIR = REPORTS_DIR / "figures"

def run_preliminary_audit():
    print(f"Loading dataset from: {RAW_DATA_PATH}")
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded successfully. Shape: {df.shape}")

    # Basic Info
    print("\n" + "="*50)
    print("1. DATASET OVERVIEW & DTYPES")
    print("="*50)
    print(df.info())

    # Nulls
    print("\n" + "="*50)
    print("2. MISSING VALUES PER COLUMN")
    print("="*50)
    nulls = df.isnull().sum()
    null_pct = (df.isnull().sum() / len(df) * 100).round(4)
    null_df = pd.DataFrame({"Null_Count": nulls, "Null_Pct": null_pct})
    print(null_df)

    # Duplicates
    print("\n" + "="*50)
    print("3. DUPLICATE ANALYSIS")
    print("="*50)
    exact_dups = df.duplicated().sum()
    print(f"Exact duplicate rows: {exact_dups}")
    tx_id_dups = df['Transaction_ID'].duplicated().sum() if 'Transaction_ID' in df.columns else "N/A"
    print(f"Duplicate Transaction_IDs: {tx_id_dups}")
    unique_tx = df['Transaction_ID'].nunique() if 'Transaction_ID' in df.columns else "N/A"
    print(f"Unique Transaction_IDs: {unique_tx}")
    unique_cust = df['Customer_ID'].nunique() if 'Customer_ID' in df.columns else "N/A"
    print(f"Unique Customer_IDs: {unique_cust}")

    # Target Distribution
    print("\n" + "="*50)
    print("4. TARGET DISTRIBUTION (Is_Fraud)")
    print("="*50)
    if 'Is_Fraud' in df.columns:
        counts = df['Is_Fraud'].value_counts()
        pcts = df['Is_Fraud'].value_counts(normalize=True) * 100
        target_df = pd.DataFrame({"Count": counts, "Percentage": pcts.round(4)})
        print(target_df)
        fraud_ratio = counts.get(1, 0) / counts.get(0, 1)
        imbalance_ratio = counts.get(0, 1) / counts.get(1, 1)
        print(f"Imbalance ratio (Non-fraud : Fraud): {imbalance_ratio:.2f} : 1")

    # Cardinality
    print("\n" + "="*50)
    print("5. COLUMN CARDINALITY & UNIQUE VALUES")
    print("="*50)
    for col in df.columns:
        n_unique = df[col].nunique()
        sample_vals = df[col].dropna().unique()[:3].tolist()
        print(f"{col:25} | Dtype: {str(df[col].dtype):10} | Uniques: {n_unique:7} | Samples: {sample_vals}")

    # Numerical Summary
    print("\n" + "="*50)
    print("6. NUMERICAL FEATURES SUMMARY")
    print("="*50)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("Numerical columns:", num_cols)
    num_summary = df[num_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999]).T
    print(num_summary[['count', 'mean', 'std', 'min', '1%', '25%', '50%', '75%', '99%', 'max']])

    # Check for negatives
    print("\nChecking for unexpected negative/zero values in numerical features:")
    for c in ['Transaction_Amount', 'Account_Balance', 'Age']:
        if c in df.columns:
            neg_count = (df[c] < 0).sum()
            zero_count = (df[c] == 0).sum()
            print(f"{c}: {neg_count} negative values, {zero_count} zero values")

    # Bivariate analysis: Numerical by Target
    print("\n" + "="*50)
    print("7. NUMERICAL FEATURES BY TARGET (Is_Fraud)")
    print("="*50)
    for col in ['Transaction_Amount', 'Account_Balance', 'Age']:
        if col in df.columns and 'Is_Fraud' in df.columns:
            grouped = df.groupby('Is_Fraud')[col].agg(['count', 'mean', 'std', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75), lambda x: np.percentile(x, 99)])
            grouped.columns = ['count', 'mean', 'std', 'median', 'p25', 'p75', 'p99']
            print(f"\nFeature: {col} grouped by Is_Fraud:")
            print(grouped.round(2))

    # Categorical Fraud Rates
    print("\n" + "="*50)
    print("8. CATEGORICAL FRAUD RATES")
    print("="*50)
    cat_cols_to_check = ['Transaction_Type', 'Merchant_Category', 'Device_Type', 'Transaction_Device', 
                         'Account_Type', 'Gender', 'Transaction_Currency']
    for cat in cat_cols_to_check:
        if cat in df.columns and 'Is_Fraud' in df.columns:
            agg = df.groupby(cat)['Is_Fraud'].agg(
                Total_Transactions='count',
                Fraud_Transactions='sum',
                Fraud_Rate=lambda x: (x.mean() * 100)
            ).reset_index()
            agg['Pct_of_All_Tx'] = (agg['Total_Transactions'] / len(df) * 100).round(2)
            agg = agg.sort_values(by='Total_Transactions', ascending=False)
            print(f"\nCategorical Fraud Rates for: {cat}")
            print(agg.to_string(index=False))

    # Check Device_Type vs Transaction_Device
    if 'Device_Type' in df.columns and 'Transaction_Device' in df.columns:
        print("\nCross-tab of Device_Type vs Transaction_Device:")
        print(pd.crosstab(df['Device_Type'], df['Transaction_Device']))

    # Date and Time analysis
    print("\n" + "="*50)
    print("9. TEMPORAL ANALYSIS (Transaction_Date & Transaction_Time)")
    print("="*50)
    if 'Transaction_Date' in df.columns:
        print("Date samples:", df['Transaction_Date'].head(5).tolist())
        # Try parsing dates
        try:
            dates = pd.to_datetime(df['Transaction_Date'], errors='coerce')
            print(f"Date range: {dates.min()} to {dates.max()}")
            print(f"Failed date parses: {dates.isnull().sum()}")
            df['Parsed_Date'] = dates
            df['DayOfWeek'] = dates.dt.day_name()
            print("\nDay of Week Fraud Rate:")
            dow_agg = df.groupby('DayOfWeek')['Is_Fraud'].agg(['count', 'sum', lambda x: x.mean()*100]).reset_index()
            dow_agg.columns = ['DayOfWeek', 'Total', 'Fraud', 'Fraud_Rate_Pct']
            print(dow_agg)
        except Exception as e:
            print("Date parse error:", e)

    if 'Transaction_Time' in df.columns:
        print("Time samples:", df['Transaction_Time'].head(5).tolist())
        try:
            # Extract hour
            hours = pd.to_datetime(df['Transaction_Time'], format='%H:%M:%S', errors='coerce').dt.hour
            if hours.isnull().all():
                # Try generic parsing
                hours = pd.to_datetime(df['Transaction_Time'], errors='coerce').dt.hour
            print(f"Failed time parses: {hours.isnull().sum()}")
            df['Hour'] = hours
            hour_agg = df.groupby('Hour')['Is_Fraud'].agg(['count', 'sum', lambda x: x.mean()*100]).reset_index()
            hour_agg.columns = ['Hour', 'Total', 'Fraud', 'Fraud_Rate_Pct']
            print("\nHourly Fraud Distribution (Head/Tail):")
            print(hour_agg.head(6))
            print(hour_agg.tail(6))
        except Exception as e:
            print("Time parse error:", e)

    # Check high-cardinality & potential leakage columns
    print("\n" + "="*50)
    print("10. HIGH CARDINALITY & IDENTIFIER ANALYSIS (Leakage audit)")
    print("="*50)
    id_cols = ['Customer_ID', 'Customer_Name', 'Customer_Contact', 'Customer_Email', 'Merchant_ID', 'Transaction_ID', 'Bank_Branch', 'City', 'State']
    for c in id_cols:
        if c in df.columns:
            n_unq = df[c].nunique()
            print(f"{c:20}: {n_unq:7} unique values out of {len(df)} rows ({n_unq/len(df)*100:.2f}%)")
            # If Customer_ID, how many repeat customers?
            if c == 'Customer_ID':
                tx_per_cust = df['Customer_ID'].value_counts()
                print(f"  Customer_ID tx count distribution: min={tx_per_cust.min()}, median={tx_per_cust.median()}, mean={tx_per_cust.mean():.2f}, max={tx_per_cust.max()}")
                print(f"  Customers with >1 transaction: {(tx_per_cust > 1).sum()} ({(tx_per_cust > 1).sum()/len(tx_per_cust)*100:.2f}%)")
            if c == 'Merchant_ID':
                tx_per_merch = df['Merchant_ID'].value_counts()
                print(f"  Merchant_ID tx count distribution: min={tx_per_merch.min()}, median={tx_per_merch.median()}, mean={tx_per_merch.mean():.2f}, max={tx_per_merch.max()}")
                print(f"  Merchants with >1 transaction: {(tx_per_merch > 1).sum()} ({(tx_per_merch > 1).sum()/len(tx_per_merch)*100:.2f}%)")

    return df

if __name__ == '__main__':
    run_preliminary_audit()
