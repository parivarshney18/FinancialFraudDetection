"""
Tests for Phase 1: Exploratory Data Analysis (EDA).
Verifies dataset integrity, schema compliance, figure generation, and metric persistence.
"""

from pathlib import Path
import json
import pytest
import pandas as pd


@pytest.fixture(scope="module")
def project_root():
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def raw_data_path(project_root):
    return project_root / "ai" / "data" / "raw" / "Bank_Transaction_Fraud_Detection.csv"


@pytest.fixture(scope="module")
def reports_dir(project_root):
    return project_root / "ai" / "reports"


def test_raw_dataset_exists_and_unmodified(raw_data_path):
    """Verify raw dataset exists and has expected dimensions."""
    assert raw_data_path.exists(), f"Raw dataset not found at {raw_data_path}"
    # Read only shape without mutating
    df = pd.read_csv(raw_data_path, nrows=10)
    assert df.shape[1] == 24, f"Expected 24 columns, got {df.shape[1]}"


def test_raw_dataset_target_distribution(raw_data_path):
    """Verify target column values and distribution without loading all columns into memory."""
    target_series = pd.read_csv(raw_data_path, usecols=['Is_Fraud'])['Is_Fraud']
    assert len(target_series) == 200000
    counts = target_series.value_counts().to_dict()
    assert counts[0] == 189912
    assert counts[1] == 10088
    assert set(counts.keys()) == {0, 1}


def test_eda_metrics_file_validity(reports_dir):
    """Verify eda_metrics.json exists and contains calculated ground-truth values."""
    metrics_path = reports_dir / "eda_metrics.json"
    assert metrics_path.exists(), f"Metrics file missing at {metrics_path}"

    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    assert metrics["dataset_shape"]["rows"] == 200000
    assert metrics["dataset_shape"]["columns"] == 24
    assert metrics["missing_and_duplicates"]["exact_duplicates"] == 0
    assert metrics["target_distribution"]["non_fraud_count"] == 189912
    assert metrics["target_distribution"]["fraud_count"] == 10088
    assert metrics["target_distribution"]["imbalance_ratio"] == 18.8255
    assert metrics["leakage_and_quality"]["branch_is_city_branch_redundant"] is True
    assert metrics["leakage_and_quality"]["location_is_city_state_redundant"] is True
    assert metrics["leakage_and_quality"]["currency_is_constant_inr"] is True


def test_all_eda_figures_exist(reports_dir):
    """Verify all 8 figures were generated and have valid file sizes (>10KB)."""
    fig_dir = reports_dir / "figures"
    assert fig_dir.exists(), f"Figures directory missing at {fig_dir}"

    expected_figures = [
        "01_target_distribution.png",
        "02_numerical_distributions.png",
        "03_categorical_distributions.png",
        "04_temporal_analysis.png",
        "05_amount_vs_balance.png",
        "06_correlation_heatmap.png",
        "07_geographic_fraud_rate.png",
        "08_device_matrix_heatmap.png",
    ]

    for fig_name in expected_figures:
        fig_path = fig_dir / fig_name
        assert fig_path.exists(), f"Expected figure {fig_name} was not found"
        assert fig_path.stat().st_size > 10000, f"Figure {fig_name} is unexpectedly small: {fig_path.stat().st_size} bytes"


def test_eda_summary_report_and_data_dictionary(project_root, reports_dir):
    """Verify eda_summary.md and data_dictionary.md exist and have non-trivial size."""
    summary_path = reports_dir / "eda_summary.md"
    assert summary_path.exists(), f"EDA summary report missing at {summary_path}"
    assert summary_path.stat().st_size > 1000, "EDA summary report is too short"

    dict_path = project_root / "ai" / "data" / "data_dictionary.md"
    assert dict_path.exists(), f"Data dictionary missing at {dict_path}"
    assert dict_path.stat().st_size > 1000, "Data dictionary is too short"


def test_data_loader_module(raw_data_path):
    """Verify data_loader functions load and validate schema correctly."""
    from ai.src.data_loader import load_raw_data, validate_dataset
    df = load_raw_data(raw_data_path, validate=True)
    assert df.shape == (200000, 24)
    # validate_dataset should not raise on valid df
    validate_dataset(df)


