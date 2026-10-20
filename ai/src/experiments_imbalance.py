"""
Class-Imbalance Experiments Harness for Phase 4.5.
Evaluates the impact of SMOTENC oversampling on mixed tabular banking data across
Stratified 5-Fold Cross-Validation for Logistic Regression, Random Forest, XGBoost, and TabNet.

Key Experimental Controls:
1. Feature Type Integrity: Categorical features (indices 0..8) are handled via SMOTENC
   to strictly preserve discrete category indices for TabNet embedding layers.
2. Leakage-Free Isolation: SMOTENC is fitted strictly on (X_train, y_train) inside each fold.
   Validation folds (X_val, y_val) remain 100% natural, unmodified fraud prevalence (~5.04%).
3. Baseline Preservation: Baseline results from Phase 3 (class-weighted, no-SMOTE) are
   preserved and loaded for rigorous side-by-side comparison.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC

from ai.src.config import (
    RANDOM_SEED,
    CV_N_SPLITS,
    CV_SHUFFLE,
    TARGET_COL,
    RESULTS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
)
from ai.src.data_loader import load_raw_data
from ai.src.cv_pipeline import generate_cv_folds
from ai.src.evaluation import (
    compute_classification_metrics,
    aggregate_fold_metrics,
    format_metrics_table,
)
from ai.src.models.baseline_models import build_logistic_regression, build_random_forest
from ai.src.models.xgb_model import build_xgboost_model
from ai.src.models.tabnet_model import TabNetFraudModel


def get_smotenc_resampler(cat_idxs: List[int], random_state: int) -> SMOTENC:
    """Instantiate SMOTENC configured with categorical feature indices and fixed random seed."""
    return SMOTENC(
        categorical_features=cat_idxs,
        sampling_strategy="auto",  # 1:1 balance on training fold
        random_state=random_state,
        k_neighbors=5,
    )


def train_and_eval_smotenc_fold(
    model_name: str,
    X_train_res: np.ndarray,
    y_train_res: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    preprocessor: Any,
    fold_idx: int,
    verbose: bool = True,
    smoke_test: bool = False,
) -> Tuple[Dict[str, Any], float]:
    """Train a single model on SMOTENC-resampled training data and evaluate on untouched validation data."""
    start_time = time.time()

    if model_name == "logistic_regression":
        # On balanced resampled data, class_weight=None is standard
        model = build_logistic_regression(class_weight=None, max_iter=300 if smoke_test else 500)
        model.fit(X_train_res, y_train_res)
        probs = model.predict_proba(X_val)[:, 1]

    elif model_name == "random_forest":
        # On balanced resampled data, class_weight=None is standard
        n_est = 10 if smoke_test else 100
        max_d = 6 if smoke_test else 12
        model = build_random_forest(
            n_estimators=n_est,
            max_depth=max_d,
            class_weight=None,
            n_jobs=4,
        )
        model.fit(X_train_res, y_train_res)
        probs = model.predict_proba(X_val)[:, 1]

    elif model_name == "xgboost":
        # On balanced resampled data, scale_pos_weight=1.0
        n_est = 10 if smoke_test else 100
        max_d = 3 if smoke_test else 5
        model = build_xgboost_model(
            n_estimators=n_est,
            max_depth=max_d,
            scale_pos_weight=1.0,
            n_jobs=4,
        )
        model.fit(X_train_res, y_train_res)
        probs = model.predict_proba(X_val)[:, 1]

    elif model_name == "tabnet":
        cat_idxs = preprocessor.cat_idxs
        cat_dims = preprocessor.cat_dims

        tabnet_kwargs = {
            "n_d": 8,
            "n_a": 8,
            "n_steps": 3,
            "gamma": 1.3,
            "lambda_sparse": 1e-3,
            "learning_rate": 2e-2,
            "seed": RANDOM_SEED + fold_idx,
            "device_name": "cpu",
        }
        tabnet_wrapper = TabNetFraudModel(cat_idxs=cat_idxs, cat_dims=cat_dims, **tabnet_kwargs)

        max_epochs = 2 if smoke_test else 10
        patience = 2 if smoke_test else 3
        batch_size = 1024 if smoke_test else 2048
        virtual_batch_size = 128 if smoke_test else 256

        tabnet_wrapper.fit(
            X_train=X_train_res,
            y_train=y_train_res,
            eval_set=[(X_val, y_val)],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            weights=0,  # Unweighted since resampled data is 1:1
        )
        probs = tabnet_wrapper.predict_proba(X_val)[:, 1]

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    elapsed = time.time() - start_time
    metrics = compute_classification_metrics(y_val, probs)
    metrics["fold"] = fold_idx
    metrics["elapsed_seconds"] = round(elapsed, 2)

    if verbose:
        print(f"    [{model_name.upper():<19}] PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | Recall@FPR<=1%: {metrics['recall_at_fpr_1pct']:.4f} | F1: {metrics['f1']:.4f} ({elapsed:.1f}s)")

    return metrics, elapsed


def run_smotenc_cv_experiments(
    models_to_run: List[str],
    df: pd.DataFrame,
    n_splits: int = CV_N_SPLITS,
    random_state: int = RANDOM_SEED,
    smoke_test: bool = False,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Execute leak-free SMOTENC cross-validation experiments for specified models."""
    print("=" * 75)
    mode_str = "SMOKE TEST MODE (Sampled subset)" if smoke_test else "FULL EVALUATION MODE (200,000 samples)"
    print(f"PHASE 4.5: SMOTENC CLASS-IMBALANCE EXPERIMENTS ({mode_str})")
    print(f"Target Models: {[m.upper() for m in models_to_run]}")
    print(f"Cross-Validation: {n_splits} Folds, Stratified, Fixed Seed={random_state}")
    print("=" * 75)

    results_by_model: Dict[str, Dict[str, Any]] = {
        m: {
            "model_name": f"smotenc_{m}",
            "base_model": m,
            "technique": "SMOTENC_oversampling",
            "n_samples": len(df),
            "n_splits": n_splits,
            "random_state": random_state,
            "smoke_test": smoke_test,
            "fold_metrics": [],
            "resampling_times": [],
            "training_times": [],
        }
        for m in models_to_run
    }

    cv_gen = generate_cv_folds(
        df=df,
        n_splits=n_splits,
        shuffle=CV_SHUFFLE,
        random_state=random_state,
        target_col=TARGET_COL,
        as_df=False,
    )

    for fold_idx, X_train, y_train, X_val, y_val, preprocessor in cv_gen:
        print(f"\n>>> PROCESSING FOLD {fold_idx + 1}/{n_splits} (Train: {len(X_train):,}, Val: {len(X_val):,})")
        n_pos_orig = int(np.sum(y_train == 1))
        n_neg_orig = int(np.sum(y_train == 0))
        print(f"    Original Train Distribution: Non-Fraud={n_neg_orig:,}, Fraud={n_pos_orig:,} ({n_pos_orig/len(y_train)*100:.2f}%)")

        # Apply SMOTENC strictly to training data
        cat_idxs = preprocessor.cat_idxs
        smote = get_smotenc_resampler(cat_idxs=cat_idxs, random_state=random_state + fold_idx)

        t_resample_start = time.time()
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        resample_elapsed = time.time() - t_resample_start

        n_pos_res = int(np.sum(y_train_res == 1))
        n_neg_res = int(np.sum(y_train_res == 0))
        print(f"    SMOTENC Resampled Train:     Non-Fraud={n_neg_res:,}, Fraud={n_pos_res:,} ({resample_elapsed:.2f}s)")
        print(f"    Validation Distribution:     Non-Fraud={int(np.sum(y_val==0)):,}, Fraud={int(np.sum(y_val==1)):,} (Untouched: {np.mean(y_val)*100:.2f}%)")

        # Verify categorical columns remained discrete integers
        for c_idx in cat_idxs:
            vals = X_train_res[:, c_idx]
            assert np.all(np.equal(np.mod(vals, 1), 0)), f"SMOTENC corrupted categorical feature index {c_idx} with non-integers!"

        for m in models_to_run:
            metrics, fit_time = train_and_eval_smotenc_fold(
                model_name=m,
                X_train_res=X_train_res,
                y_train_res=y_train_res,
                X_val=X_val,
                y_val=y_val,
                preprocessor=preprocessor,
                fold_idx=fold_idx,
                verbose=verbose,
                smoke_test=smoke_test,
            )
            results_by_model[m]["fold_metrics"].append(metrics)
            results_by_model[m]["resampling_times"].append(resample_elapsed)
            results_by_model[m]["training_times"].append(fit_time)

    # Aggregate and persist results
    print("\n" + "=" * 75)
    print("EXPERIMENT SUMMARY & AGGREGATION")
    print("=" * 75)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for m in models_to_run:
        res = results_by_model[m]
        res["aggregated"] = aggregate_fold_metrics(res["fold_metrics"])
        res["total_resample_seconds"] = round(sum(res["resampling_times"]), 2)
        res["total_training_seconds"] = round(sum(res["training_times"]), 2)
        res["total_elapsed_seconds"] = round(res["total_resample_seconds"] + res["total_training_seconds"], 2)

        print(f"\n--- Model: SMOTENC + {m.upper()} ---")
        summary_df = format_metrics_table(res["aggregated"])
        print(summary_df.to_string(index=False))
        print(f"Total Time: {res['total_elapsed_seconds']}s (Resampling: {res['total_resample_seconds']}s, Training: {res['total_training_seconds']}s)")

        if not smoke_test:
            out_path = RESULTS_DIR / f"metrics_smotenc_{m}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"Persisted: {out_path}")

    return results_by_model


def compile_imbalance_comparison_table() -> pd.DataFrame:
    """Load baseline and SMOTENC results, compile side-by-side comparison CSV."""
    models = ["logistic_regression", "random_forest", "xgboost", "tabnet"]
    records = []

    for m in models:
        # 1. Baseline (Class-Weighted)
        base_path = RESULTS_DIR / f"metrics_{m}.json"
        if base_path.exists():
            with open(base_path, "r", encoding="utf-8") as f:
                b_data = json.load(f)
            b_agg = b_data["aggregated"]
            records.append({
                "Model": m.replace("_", " ").title(),
                "Strategy": "Baseline (Class-Weighted)",
                "PR-AUC": f"{b_agg['pr_auc']['mean']:.4f} +/- {b_agg['pr_auc']['std']:.4f}",
                "ROC-AUC": f"{b_agg['roc_auc']['mean']:.4f} +/- {b_agg['roc_auc']['std']:.4f}",
                "Recall @ FPR<=1%": f"{b_agg['recall_at_fpr_1pct']['mean']:.4f} +/- {b_agg['recall_at_fpr_1pct']['std']:.4f}",
                "Fraud F1": f"{b_agg['f1']['mean']:.4f} +/- {b_agg['f1']['std']:.4f}",
                "Macro F1": f"{b_agg['macro_f1']['mean']:.4f} +/- {b_agg['macro_f1']['std']:.4f}",
                "Precision": f"{b_agg['precision']['mean']:.4f} +/- {b_agg['precision']['std']:.4f}",
                "Recall": f"{b_agg['recall']['mean']:.4f} +/- {b_agg['recall']['std']:.4f}",
                "Total Time (s)": b_data.get("total_elapsed_seconds", "N/A"),
            })

        # 2. SMOTENC Resampled
        smote_path = RESULTS_DIR / f"metrics_smotenc_{m}.json"
        if smote_path.exists():
            with open(smote_path, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            s_agg = s_data["aggregated"]
            records.append({
                "Model": m.replace("_", " ").title(),
                "Strategy": "SMOTENC Resampled",
                "PR-AUC": f"{s_agg['pr_auc']['mean']:.4f} +/- {s_agg['pr_auc']['std']:.4f}",
                "ROC-AUC": f"{s_agg['roc_auc']['mean']:.4f} +/- {s_agg['roc_auc']['std']:.4f}",
                "Recall @ FPR<=1%": f"{s_agg['recall_at_fpr_1pct']['mean']:.4f} +/- {s_agg['recall_at_fpr_1pct']['std']:.4f}",
                "Fraud F1": f"{s_agg['f1']['mean']:.4f} +/- {s_agg['f1']['std']:.4f}",
                "Macro F1": f"{s_agg['macro_f1']['mean']:.4f} +/- {s_agg['macro_f1']['std']:.4f}",
                "Precision": f"{s_agg['precision']['mean']:.4f} +/- {s_agg['precision']['std']:.4f}",
                "Recall": f"{s_agg['recall']['mean']:.4f} +/- {s_agg['recall']['std']:.4f}",
                "Total Time (s)": s_data.get("total_elapsed_seconds", "N/A"),
            })

    comp_df = pd.DataFrame(records)
    if not comp_df.empty:
        csv_path = RESULTS_DIR / "class_imbalance_comparison.csv"
        comp_df.to_csv(csv_path, index=False)
        print(f"\nPersisted comparison table to: {csv_path}")
        print("\n" + comp_df.to_string(index=False))

    return comp_df


def plot_imbalance_comparison_figure(output_path: Path):
    """Generate high-resolution diagnostic plot comparing Baseline vs SMOTENC across models."""
    models = ["logistic_regression", "random_forest", "xgboost", "tabnet"]
    model_labels = ["Logistic Reg.", "Random Forest", "XGBoost", "TabNet"]

    base_pr, smote_pr = [], []
    base_roc, smote_roc = [], []
    base_rec1, smote_rec1 = [], []
    base_f1, smote_f1 = [], []

    for m in models:
        bp = RESULTS_DIR / f"metrics_{m}.json"
        sp = RESULTS_DIR / f"metrics_smotenc_{m}.json"

        if bp.exists() and sp.exists():
            with open(bp, "r", encoding="utf-8") as f:
                b = json.load(f)["aggregated"]
            with open(sp, "r", encoding="utf-8") as f:
                s = json.load(f)["aggregated"]

            base_pr.append(b["pr_auc"]["mean"])
            smote_pr.append(s["pr_auc"]["mean"])
            base_roc.append(b["roc_auc"]["mean"])
            smote_roc.append(s["roc_auc"]["mean"])
            base_rec1.append(b["recall_at_fpr_1pct"]["mean"])
            smote_rec1.append(s["recall_at_fpr_1pct"]["mean"])
            base_f1.append(b["f1"]["mean"])
            smote_f1.append(s["f1"]["mean"])
        else:
            print(f"Skipping plot entry for {m} (missing metric file)")
            return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(len(models))
    width = 0.35

    def draw_bar(ax, base_vals, smote_vals, title, ylabel, baseline_val=None, ylim=(0, 1.0)):
        ax.bar(x - width/2, base_vals, width, label="Baseline (Class-Weighted)", color="#1f77b4", edgecolor="black", alpha=0.9)
        ax.bar(x + width/2, smote_vals, width, label="SMOTENC Oversampling", color="#ff7f0e", edgecolor="black", alpha=0.9)
        if baseline_val is not None:
            ax.axhline(baseline_val, color="red", linestyle="--", linewidth=1.2, label=f"Random Chance ({baseline_val:.4f})")
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9)

    # 1. PR-AUC
    draw_bar(axes[0, 0], base_pr, smote_pr, "PR-AUC (Precision-Recall AUC)", "PR-AUC", baseline_val=0.05044, ylim=(0, 0.12))
    # 2. ROC-AUC
    draw_bar(axes[0, 1], base_roc, smote_roc, "ROC-AUC", "ROC-AUC", baseline_val=0.5000, ylim=(0.4, 0.6))
    # 3. Recall @ FPR <= 1%
    draw_bar(axes[1, 0], base_rec1, smote_rec1, "Operational Recall @ FPR <= 1%", "Recall", baseline_val=0.0100, ylim=(0, 0.03))
    # 4. Fraud F1-Score
    draw_bar(axes[1, 1], base_f1, smote_f1, "Fraud F1-Score", "F1-Score", ylim=(0, 0.15))

    fig.suptitle("Phase 4.5: Class-Imbalance Strategy Comparison\nBaseline Class-Weighting vs. SMOTENC Oversampling (5-Fold Stratified CV)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved imbalance comparison figure to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4.5 Class-Imbalance Experiments")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "logistic_regression", "random_forest", "xgboost", "tabnet"],
        help="Model to evaluate (default: all)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run rapid smoke test on small sample subset",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    args = parser.parse_args()

    df = load_raw_data()

    if args.smoke_test:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=2500, random_state=RANDOM_SEED)
        idx, _ = next(sss.split(df, df[TARGET_COL]))
        df_exp = df.iloc[idx].copy().reset_index(drop=True)
        n_splits = 2
    else:
        df_exp = df
        n_splits = args.splits

    models_to_run = (
        ["logistic_regression", "random_forest", "xgboost", "tabnet"]
        if args.model == "all"
        else [args.model]
    )

    run_smotenc_cv_experiments(
        models_to_run=models_to_run,
        df=df_exp,
        n_splits=n_splits,
        random_state=RANDOM_SEED,
        smoke_test=args.smoke_test,
    )

    if not args.smoke_test:
        compile_imbalance_comparison_table()
        fig_path = FIGURES_DIR / "13_class_imbalance_comparison.png"
        plot_imbalance_comparison_figure(fig_path)


if __name__ == "__main__":
    main()
