"""
Evaluation harness for Phase 5: Revised Behavioral Feature Set.
Trains and evaluates Logistic Regression, Random Forest, XGBoost, and TabNet across
Stratified 5-Fold Cross-Validation on the causally engineered 29-feature dataset.

Key Experimental Controls:
1. Strict Fold Isolation: Preprocessor fitted strictly on training partition inside each fold.
2. Identical CV Protocol: 5-Fold Stratified splits with fixed random_state=42.
3. Metric Fidelity: PR-AUC, ROC-AUC, Recall@FPR<=1%, Precision, Recall, and F1 computed without leakage.
4. Non-Destructive Persistence: Persists metrics separately under ai/results/.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from ai.src.config import (
    RANDOM_SEED,
    CV_N_SPLITS,
    TARGET_COL,
    RESULTS_DIR,
    FIGURES_DIR,
)
from ai.src.data_loader import load_raw_data
from ai.src.behavioral_features import (
    engineer_causal_behavioral_dataframe,
    RevisedFraudPreprocessor,
)
from ai.src.evaluation import (
    compute_classification_metrics,
    aggregate_fold_metrics,
    format_metrics_table,
)
from ai.src.models.baseline_models import build_logistic_regression, build_random_forest
from ai.src.models.xgb_model import build_xgboost_model
from ai.src.models.tabnet_model import TabNetFraudModel


def train_and_eval_revised_fold(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    preprocessor: Any,
    fold_idx: int,
    verbose: bool = True,
    smoke_test: bool = False,
) -> Tuple[Dict[str, Any], float]:
    """Train single model on revised feature matrix and evaluate on validation fold."""
    start_time = time.time()

    if model_name == "logistic_regression":
        model = build_logistic_regression(
            class_weight="balanced",
            max_iter=300 if smoke_test else 500,
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]

    elif model_name == "random_forest":
        n_est = 10 if smoke_test else 100
        max_d = 6 if smoke_test else 12
        model = build_random_forest(
            n_estimators=n_est,
            max_depth=max_d,
            class_weight="balanced_subsample",
            n_jobs=4,
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]

    elif model_name == "xgboost":
        n_est = 10 if smoke_test else 100
        max_d = 3 if smoke_test else 5
        model = build_xgboost_model(
            n_estimators=n_est,
            max_depth=max_d,
            scale_pos_weight=18.8255,
            n_jobs=4,
        )
        model.fit(X_train, y_train)
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
        print(f"    [{model_name.upper():<19}] PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | Recall@FPR<=1%: {metrics['recall_at_fpr_1pct']:.4f} | F1: {metrics['f1']:.4f} ({elapsed:.1f}s)")

    return metrics, elapsed


def run_revised_features_experiment(
    models_to_run: List[str],
    df: pd.DataFrame,
    n_splits: int = CV_N_SPLITS,
    random_state: int = RANDOM_SEED,
    smoke_test: bool = False,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Execute complete 5-fold cross-validation experiment with revised behavioral feature pipeline."""
    print("=" * 75)
    mode_str = "SMOKE TEST MODE (Sampled subset)" if smoke_test else "FULL EVALUATION MODE (200,000 samples)"
    print(f"PHASE 5: REVISED BEHAVIORAL FEATURE EXPERIMENTS ({mode_str})")
    print(f"Target Models: {[m.upper() for m in models_to_run]}")
    print(f"Cross-Validation: {n_splits} Folds, Stratified, Fixed Seed={random_state}")
    print("=" * 75)

    # 1. Feature Engineering
    t0_feat = time.time()
    df_aug = engineer_causal_behavioral_dataframe(df)
    feat_time = time.time() - t0_feat
    print(f"Causal behavioral feature engineering complete in {feat_time:.2f}s. Shape: {df_aug.shape}")

    results_by_model: Dict[str, Dict[str, Any]] = {
        m: {
            "model_name": f"revised_{m}",
            "base_model": m,
            "feature_set": "revised_behavioral_29_features",
            "n_samples": len(df),
            "n_splits": n_splits,
            "random_state": random_state,
            "smoke_test": smoke_test,
            "fold_metrics": [],
            "training_times": [],
        }
        for m in models_to_run
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(df_aug, df_aug[TARGET_COL]))

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n>>> PROCESSING FOLD {fold_idx + 1}/{n_splits} (Train: {len(train_idx):,}, Val: {len(val_idx):,})")

        train_df = df_aug.iloc[train_idx].copy().reset_index(drop=True)
        val_df = df_aug.iloc[val_idx].copy().reset_index(drop=True)

        preprocessor = RevisedFraudPreprocessor()
        X_train = preprocessor.fit_transform(train_df)
        X_val = preprocessor.transform(val_df)
        y_train = train_df[TARGET_COL].values.astype(np.int64)
        y_val = val_df[TARGET_COL].values.astype(np.int64)

        print(f"    Feature Matrix Shape: Train {X_train.shape}, Val {X_val.shape} ({len(preprocessor.feature_names_)} features)")

        for m in models_to_run:
            metrics, fit_time = train_and_eval_revised_fold(
                model_name=m,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                preprocessor=preprocessor,
                fold_idx=fold_idx,
                verbose=verbose,
                smoke_test=smoke_test,
            )
            results_by_model[m]["fold_metrics"].append(metrics)
            results_by_model[m]["training_times"].append(fit_time)

    # Aggregation & Persistence
    print("\n" + "=" * 75)
    print("REVISED FEATURE EXPERIMENT SUMMARY & AGGREGATION")
    print("=" * 75)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for m in models_to_run:
        res = results_by_model[m]
        res["aggregated"] = aggregate_fold_metrics(res["fold_metrics"])
        res["total_elapsed_seconds"] = round(sum(res["training_times"]), 2)

        print(f"\n--- Model: Revised Features + {m.upper()} ---")
        summary_df = format_metrics_table(res["aggregated"])
        print(summary_df.to_string(index=False))
        print(f"Total Time: {res['total_elapsed_seconds']}s")

        if not smoke_test:
            out_path = RESULTS_DIR / f"metrics_revised_features_{m}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"Persisted: {out_path}")

    return results_by_model


def compile_behavioral_comparison_table() -> pd.DataFrame:
    """Load baseline, SMOTENC, and revised feature results into a side-by-side comparison table."""
    models = ["logistic_regression", "random_forest", "xgboost", "tabnet"]
    records = []

    for m in models:
        m_title = m.replace("_", " ").title()

        # 1. Baseline (Phase 3, 22 features)
        bp = RESULTS_DIR / f"metrics_{m}.json"
        if bp.exists():
            with open(bp, "r", encoding="utf-8") as f:
                b_agg = json.load(f)["aggregated"]
            records.append({
                "Model": m_title,
                "Experiment": "Baseline (22 features)",
                "PR-AUC": f"{b_agg['pr_auc']['mean']:.4f} +/- {b_agg['pr_auc']['std']:.4f}",
                "ROC-AUC": f"{b_agg['roc_auc']['mean']:.4f} +/- {b_agg['roc_auc']['std']:.4f}",
                "Recall @ FPR<=1%": f"{b_agg['recall_at_fpr_1pct']['mean']:.4f} +/- {b_agg['recall_at_fpr_1pct']['std']:.4f}",
                "Fraud F1": f"{b_agg['f1']['mean']:.4f} +/- {b_agg['f1']['std']:.4f}",
                "Precision": f"{b_agg['precision']['mean']:.4f} +/- {b_agg['precision']['std']:.4f}",
                "Recall": f"{b_agg['recall']['mean']:.4f} +/- {b_agg['recall']['std']:.4f}",
            })

        # 2. SMOTENC (Phase 4.5, 22 features + 1:1 oversampling)
        sp = RESULTS_DIR / f"metrics_smotenc_{m}.json"
        if sp.exists():
            with open(sp, "r", encoding="utf-8") as f:
                s_agg = json.load(f)["aggregated"]
            records.append({
                "Model": m_title,
                "Experiment": "SMOTENC Oversampling",
                "PR-AUC": f"{s_agg['pr_auc']['mean']:.4f} +/- {s_agg['pr_auc']['std']:.4f}",
                "ROC-AUC": f"{s_agg['roc_auc']['mean']:.4f} +/- {s_agg['roc_auc']['std']:.4f}",
                "Recall @ FPR<=1%": f"{s_agg['recall_at_fpr_1pct']['mean']:.4f} +/- {s_agg['recall_at_fpr_1pct']['std']:.4f}",
                "Fraud F1": f"{s_agg['f1']['mean']:.4f} +/- {s_agg['f1']['std']:.4f}",
                "Precision": f"{s_agg['precision']['mean']:.4f} +/- {s_agg['precision']['std']:.4f}",
                "Recall": f"{s_agg['recall']['mean']:.4f} +/- {s_agg['recall']['std']:.4f}",
            })

        # 3. Revised Behavioral Features (Phase 5, 29 features)
        rp = RESULTS_DIR / f"metrics_revised_features_{m}.json"
        if rp.exists():
            with open(rp, "r", encoding="utf-8") as f:
                r_agg = json.load(f)["aggregated"]
            records.append({
                "Model": m_title,
                "Experiment": "Revised Behavioral (29 features)",
                "PR-AUC": f"{r_agg['pr_auc']['mean']:.4f} +/- {r_agg['pr_auc']['std']:.4f}",
                "ROC-AUC": f"{r_agg['roc_auc']['mean']:.4f} +/- {r_agg['roc_auc']['std']:.4f}",
                "Recall @ FPR<=1%": f"{r_agg['recall_at_fpr_1pct']['mean']:.4f} +/- {r_agg['recall_at_fpr_1pct']['std']:.4f}",
                "Fraud F1": f"{r_agg['f1']['mean']:.4f} +/- {r_agg['f1']['std']:.4f}",
                "Precision": f"{r_agg['precision']['mean']:.4f} +/- {r_agg['precision']['std']:.4f}",
                "Recall": f"{r_agg['recall']['mean']:.4f} +/- {r_agg['recall']['std']:.4f}",
            })

    comp_df = pd.DataFrame(records)
    if not comp_df.empty:
        csv_path = RESULTS_DIR / "behavioral_feature_comparison.csv"
        comp_df.to_csv(csv_path, index=False)
        print(f"\nPersisted comparison table to: {csv_path}")
        print("\n" + comp_df.to_string(index=False))

    return comp_df


def plot_behavioral_comparison_figure(output_path: Path):
    """Plot diagnostic chart comparing Baseline vs Revised Behavioral features across models."""
    models = ["logistic_regression", "random_forest", "xgboost", "tabnet"]
    model_labels = ["Logistic Reg.", "Random Forest", "XGBoost", "TabNet"]

    base_pr, rev_pr = [], []
    base_roc, rev_roc = [], []
    base_rec1, rev_rec1 = [], []
    base_f1, rev_f1 = [], []

    for m in models:
        bp = RESULTS_DIR / f"metrics_{m}.json"
        rp = RESULTS_DIR / f"metrics_revised_features_{m}.json"

        if bp.exists() and rp.exists():
            with open(bp, "r", encoding="utf-8") as f:
                b = json.load(f)["aggregated"]
            with open(rp, "r", encoding="utf-8") as f:
                r = json.load(f)["aggregated"]

            base_pr.append(b["pr_auc"]["mean"])
            rev_pr.append(r["pr_auc"]["mean"])
            base_roc.append(b["roc_auc"]["mean"])
            rev_roc.append(r["roc_auc"]["mean"])
            base_rec1.append(b["recall_at_fpr_1pct"]["mean"])
            rev_rec1.append(r["recall_at_fpr_1pct"]["mean"])
            base_f1.append(b["f1"]["mean"])
            rev_f1.append(r["f1"]["mean"])
        else:
            print(f"Skipping plot entry for {m} (missing metric file)")
            return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(len(models))
    width = 0.35

    def draw_bar(ax, base_vals, rev_vals, title, ylabel, baseline_val=None, ylim=(0, 1.0)):
        ax.bar(x - width/2, base_vals, width, label="Baseline (22 features)", color="#2b5c8f", edgecolor="black", alpha=0.9)
        ax.bar(x + width/2, rev_vals, width, label="Revised Behavioral (29 features)", color="#2ca02c", edgecolor="black", alpha=0.9)
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
    draw_bar(axes[0, 0], base_pr, rev_pr, "PR-AUC (Precision-Recall AUC)", "PR-AUC", baseline_val=0.05044, ylim=(0, 0.12))
    # 2. ROC-AUC
    draw_bar(axes[0, 1], base_roc, rev_roc, "ROC-AUC", "ROC-AUC", baseline_val=0.5000, ylim=(0.4, 0.6))
    # 3. Recall @ FPR <= 1%
    draw_bar(axes[1, 0], base_rec1, rev_rec1, "Operational Recall @ FPR <= 1%", "Recall", baseline_val=0.0100, ylim=(0, 0.03))
    # 4. Fraud F1-Score
    draw_bar(axes[1, 1], base_f1, rev_f1, "Fraud F1-Score", "F1-Score", ylim=(0, 0.15))

    fig.suptitle("Phase 5: Feature Engineering Impact Assessment\nBaseline Feature Set (22 features) vs. Revised Behavioral Set (29 features)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved behavioral comparison figure to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Revised Behavioral Feature Experiments")
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

    run_revised_features_experiment(
        models_to_run=models_to_run,
        df=df_exp,
        n_splits=n_splits,
        random_state=RANDOM_SEED,
        smoke_test=args.smoke_test,
    )

    if not args.smoke_test:
        compile_behavioral_comparison_table()
        fig_path = FIGURES_DIR / "14_behavioral_feature_comparison.png"
        plot_behavioral_comparison_figure(fig_path)


if __name__ == "__main__":
    main()
