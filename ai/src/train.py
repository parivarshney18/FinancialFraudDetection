"""
Training and cross-validation orchestration harness for Financial Fraud Detection.
Evaluates Logistic Regression, Random Forest, XGBoost, and TabNet across Stratified 5-Fold CV.
Computes comprehensive metrics, persists experiment logs, and generates comparison artifacts.
"""

import sys
import time
import json
import argparse
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ai.src.config import (
    RANDOM_SEED,
    CV_N_SPLITS,
    CV_SHUFFLE,
    TARGET_COL,
    RESULTS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    ARTIFACTS_DIR,
)
from ai.src.data_loader import load_raw_data
from ai.src.cv_pipeline import generate_cv_folds, get_cv_split_indices
from ai.src.evaluation import (
    compute_classification_metrics,
    aggregate_fold_metrics,
    format_metrics_table,
)
from ai.src.models.baseline_models import build_logistic_regression, build_random_forest
from ai.src.models.xgb_model import build_xgboost_model
from ai.src.models.tabnet_model import TabNetFraudModel


def train_and_eval_fold(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    preprocessor: Any,
    fold_idx: int,
    verbose: bool = True,
    **model_kwargs: Any,
) -> Tuple[Dict[str, Any], float]:
    """Train a single model on a single cross-validation fold and return metrics and elapsed time."""
    start_time = time.time()

    if model_name == "logistic_regression":
        model = build_logistic_regression(**model_kwargs)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]

    elif model_name == "random_forest":
        model = build_random_forest(**model_kwargs)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]

    elif model_name == "xgboost":
        model = build_xgboost_model(**model_kwargs)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]

    elif model_name == "tabnet":
        cat_idxs = preprocessor.cat_idxs
        cat_dims = preprocessor.cat_dims

        # TabNet configuration optimized for CPU stability and memory constraints
        tabnet_kwargs = {
            "n_d": model_kwargs.get("n_d", 8),
            "n_a": model_kwargs.get("n_a", 8),
            "n_steps": model_kwargs.get("n_steps", 3),
            "gamma": model_kwargs.get("gamma", 1.3),
            "lambda_sparse": model_kwargs.get("lambda_sparse", 1e-3),
            "learning_rate": model_kwargs.get("learning_rate", 2e-2),
            "seed": RANDOM_SEED + fold_idx,
            "device_name": "cpu",
        }
        tabnet_wrapper = TabNetFraudModel(cat_idxs=cat_idxs, cat_dims=cat_dims, **tabnet_kwargs)

        max_epochs = model_kwargs.get("max_epochs", 10)
        patience = model_kwargs.get("patience", 3)
        batch_size = model_kwargs.get("batch_size", 2048)
        virtual_batch_size = model_kwargs.get("virtual_batch_size", 256)

        tabnet_wrapper.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            weights=1,
        )
        probs = tabnet_wrapper.predict_proba(X_val)[:, 1]

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    elapsed = time.time() - start_time
    metrics = compute_classification_metrics(y_val, probs)
    metrics["fold"] = fold_idx
    metrics["elapsed_seconds"] = round(elapsed, 2)

    if verbose:
        print(f"  [Fold {fold_idx}] PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | Recall@FPR<=1%: {metrics['recall_at_fpr_1pct']:.4f} | F1: {metrics['f1']:.4f} ({elapsed:.1f}s)")

    return metrics, elapsed


def run_cv_experiment(
    model_name: str,
    df: pd.DataFrame,
    n_splits: int = CV_N_SPLITS,
    random_state: int = RANDOM_SEED,
    model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute complete 5-fold cross-validation experiment for a single model architecture."""
    if model_kwargs is None:
        model_kwargs = {}

    print(f"\n{'='*70}")
    print(f"STARTING EXPERIMENT: {model_name.upper()} ({n_splits}-Fold Stratified CV, N={len(df):,})")
    print(f"{'='*70}")

    fold_metrics_list = []
    total_start = time.time()

    cv_gen = generate_cv_folds(
        df=df,
        n_splits=n_splits,
        shuffle=CV_SHUFFLE,
        random_state=random_state,
        target_col=TARGET_COL,
        as_df=False,
    )

    for fold_idx, X_train, y_train, X_val, y_val, preprocessor in cv_gen:
        metrics, _ = train_and_eval_fold(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            preprocessor=preprocessor,
            fold_idx=fold_idx,
            verbose=verbose,
            **model_kwargs,
        )
        fold_metrics_list.append(metrics)

    total_elapsed = time.time() - total_start
    aggregated = aggregate_fold_metrics(fold_metrics_list)

    summary_df = format_metrics_table(aggregated)
    print(f"\n{'-'*70}")
    print(f"AGGREGATED RESULTS: {model_name.upper()} (Total Time: {total_elapsed:.1f}s)")
    print(f"{'-'*70}")
    print(summary_df.to_string(index=False))

    experiment_record = {
        "model_name": model_name,
        "n_samples": len(df),
        "n_splits": n_splits,
        "random_state": random_state,
        "model_kwargs": model_kwargs,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "fold_metrics": fold_metrics_list,
        "aggregated": aggregated,
    }

    # Persist fold metrics JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / f"metrics_{model_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(experiment_record, f, indent=2)

    return experiment_record


def plot_model_comparison(comparison_data: Dict[str, Dict[str, Any]], output_path: Path):
    """Generate high-resolution comparison chart of PR-AUC, ROC-AUC, and F1 across models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = list(comparison_data.keys())
    metrics_to_plot = ["pr_auc", "roc_auc", "recall_at_fpr_1pct", "f1"]
    metric_labels = ["PR-AUC", "ROC-AUC", "Recall @ FPR <= 1%", "Fraud F1-Score"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.2

    colors = ["#2b5c8f", "#41b6c4", "#e7298a", "#fd8d3c"]

    for i, (m_key, m_label) in enumerate(zip(metrics_to_plot, metric_labels)):
        means = [comparison_data[m]["aggregated"][m_key]["mean"] for m in models]
        stds = [comparison_data[m]["aggregated"][m_key]["std"] for m in models]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4, label=m_label, color=colors[i], edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Metric Value (0.0 to 1.0)")
    ax.set_title("Cross-Validation Model Benchmark Comparison (5-Fold Stratified CV)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models], fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved benchmark figure to: {output_path}")


def compile_persisted_results() -> pd.DataFrame:
    """Read all persisted model metric JSON files, print comparison table, save CSV, and plot figure."""
    models = ["logistic_regression", "random_forest", "xgboost", "tabnet"]
    all_results = {}

    for m in models:
        path = RESULTS_DIR / f"metrics_{m}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                all_results[m] = json.load(f)

    if not all_results:
        print("[TRAIN] No persisted model metrics found in", RESULTS_DIR)
        return pd.DataFrame()

    comp_rows = []
    for m_name, res in all_results.items():
        agg = res["aggregated"]
        comp_rows.append({
            "Model": m_name.replace("_", " ").title(),
            "PR-AUC": f"{agg['pr_auc']['mean']:.4f} +/- {agg['pr_auc']['std']:.4f}",
            "ROC-AUC": f"{agg['roc_auc']['mean']:.4f} +/- {agg['roc_auc']['std']:.4f}",
            "Recall @ FPR<=1%": f"{agg['recall_at_fpr_1pct']['mean']:.4f} +/- {agg['recall_at_fpr_1pct']['std']:.4f}",
            "F1-Score": f"{agg['f1']['mean']:.4f} +/- {agg['f1']['std']:.4f}",
            "Macro-F1": f"{agg['macro_f1']['mean']:.4f} +/- {agg['macro_f1']['std']:.4f}",
            "Precision": f"{agg['precision']['mean']:.4f} +/- {agg['precision']['std']:.4f}",
            "Recall": f"{agg['recall']['mean']:.4f} +/- {agg['recall']['std']:.4f}",
            "Total Time (s)": res["total_elapsed_seconds"],
        })
    comp_df = pd.DataFrame(comp_rows)
    print("\n" + "="*95)
    print("FINAL 5-FOLD CV BENCHMARK COMPARISON TABLE (N=200,000)")
    print("="*95)
    print(comp_df.to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(RESULTS_DIR / "comparison_metrics.csv", index=False)
    plot_model_comparison(all_results, FIGURES_DIR / "09_model_comparison.png")
    return comp_df


def main():
    parser = argparse.ArgumentParser(description="Financial Fraud Detection Phase 3 Model Training Harness")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on 5,000 samples with 3 folds")
    parser.add_argument("--compile-results", action="store_true", help="Compile and plot all persisted 5-fold CV results")
    parser.add_argument("--model", type=str, default="all", choices=["all", "logistic_regression", "random_forest", "xgboost", "tabnet"])
    parser.add_argument("--folds", type=int, default=CV_N_SPLITS, help="Number of folds (default 5)")
    args = parser.parse_args()

    if args.compile_results:
        compile_persisted_results()
        return

    # Load dataset
    df = load_raw_data(validate=True)

    if args.smoke_test:
        print("[TRAIN] Running in SMOKE TEST mode (5,000 samples, 3 folds)...")
        df = df.iloc[:5000].copy()
        n_splits = 3
        tabnet_epochs = 2
    else:
        n_splits = args.folds
        tabnet_epochs = 10

    models_to_run = (
        ["logistic_regression", "random_forest", "xgboost", "tabnet"]
        if args.model == "all"
        else [args.model]
    )

    all_results = {}
    for model_name in models_to_run:
        m_kwargs = {}
        if model_name == "tabnet":
            m_kwargs["max_epochs"] = tabnet_epochs
            m_kwargs["patience"] = 3
            m_kwargs["batch_size"] = 2048
            m_kwargs["virtual_batch_size"] = 256
        elif model_name == "random_forest":
            m_kwargs["n_estimators"] = 50 if args.smoke_test else 100
            m_kwargs["max_depth"] = 10
            m_kwargs["n_jobs"] = 4
        elif model_name == "xgboost":
            m_kwargs["n_estimators"] = 50 if args.smoke_test else 100
            m_kwargs["max_depth"] = 4
            m_kwargs["n_jobs"] = 4

        res = run_cv_experiment(
            model_name=model_name,
            df=df,
            n_splits=n_splits,
            model_kwargs=m_kwargs,
        )
        all_results[model_name] = res

    # Build comparison summary table
    if len(all_results) > 1:
        comp_rows = []
        for m_name, res in all_results.items():
            agg = res["aggregated"]
            comp_rows.append({
                "Model": m_name.replace("_", " ").title(),
                "PR-AUC": f"{agg['pr_auc']['mean']:.4f} ± {agg['pr_auc']['std']:.4f}",
                "ROC-AUC": f"{agg['roc_auc']['mean']:.4f} ± {agg['roc_auc']['std']:.4f}",
                "Recall @ FPR<=1%": f"{agg['recall_at_fpr_1pct']['mean']:.4f} ± {agg['recall_at_fpr_1pct']['std']:.4f}",
                "F1-Score": f"{agg['f1']['mean']:.4f} ± {agg['f1']['std']:.4f}",
                "Macro-F1": f"{agg['macro_f1']['mean']:.4f} ± {agg['macro_f1']['std']:.4f}",
                "Total Time (s)": res["total_elapsed_seconds"],
            })
        comp_df = pd.DataFrame(comp_rows)
        print("\n" + "="*80)
        print("OVERALL MODEL BENCHMARK COMPARISON TABLE")
        print("="*80)
        print(comp_df.to_string(index=False))

        # Save summary CSV
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        comp_df.to_csv(RESULTS_DIR / "comparison_metrics.csv", index=False)

        # Plot figure
        plot_model_comparison(all_results, FIGURES_DIR / "09_model_comparison.png")


if __name__ == "__main__":
    main()
