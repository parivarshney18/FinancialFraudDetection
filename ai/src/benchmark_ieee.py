"""
IEEE-CIS Fraud Detection Model Benchmarking and Training Harness.
Evaluates Logistic Regression, Random Forest, XGBoost, and TabNet across:
  Protocol A: Stratified 5-Fold Cross-Validation (In-Distribution / Algorithmic Capacity)
  Protocol B: Chronological 80/20 Holdout Split (Out-of-Time / Concept Drift Generalization)

Metrics:
  - ROC-AUC
  - PR-AUC (Average Precision)
  - Recall @ FPR <= 1%
  - F1-Score (Fraud Class)
  - Macro F1-Score
  - Precision and Recall
  - Confusion Matrix
  - Training and Inference Latency

Discipline:
  - Strict target separation
  - Preprocessor fitted exclusively on training partitions
  - Causal feature engineering with training history transfer
  - Decision threshold selected on training fold only; never tuned on validation/test sets
  - Preserves all historical Phase 1-8 reports, results, and artifacts
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.src.config import (
    RANDOM_SEED,
    CV_N_SPLITS,
    RESULTS_DIR,
    FIGURES_DIR,
    ARTIFACTS_DIR,
)
from ai.src.data_loader_ieee import (
    load_ieee_cis_raw,
    IEEE_RAW_DIR,
    TRANSACTION_FILE,
    IDENTITY_FILE,
)
from ai.src.preprocessing_ieee import (
    IEEEFraudPreprocessor,
    TARGET_COL,
    TIME_COL,
    generate_ieee_folds,
)
from ai.src.evaluation import (
    compute_classification_metrics,
    aggregate_fold_metrics,
)
from ai.src.models.baseline_models import build_logistic_regression, build_random_forest
from ai.src.models.xgb_model import build_xgboost_model
from ai.src.models.tabnet_model import TabNetFraudModel


def tune_threshold_on_train(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the decision threshold in [0.05, 0.90] that maximizes F1 on training data.
    
    Guarantees that test/validation sets are never consulted during threshold calibration.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    thresholds = np.linspace(0.05, 0.90, 86)
    best_score = -1.0
    best_th = 0.5

    for th in thresholds:
        pred = (y_prob_arr >= th).astype(np.int64)
        score = f1_score(y_true_arr, pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_th = float(th)

    return round(best_th, 4)


def load_ieee_benchmark_data(
    sample_size: Optional[int] = 50000,
    strategy: str = "stratified",
    random_state: int = RANDOM_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load IEEE-CIS data for benchmarking with reproducible sampling.
    
    Args:
        sample_size: Number of transactions to sample. If None or >= 590,540, loads full dataset.
        strategy: 'stratified' (samples maintaining exact 3.499% fraud rate) or
                  'chronological' (first N transactions ordered by TransactionDT).
        random_state: Seed for stratified sampling.
        verbose: Whether to print loading progress.

    Returns:
        pd.DataFrame: Merged transaction and identity table.
    """
    tx_file = IEEE_RAW_DIR / TRANSACTION_FILE
    id_file = IEEE_RAW_DIR / IDENTITY_FILE

    if not tx_file.exists():
        raise FileNotFoundError(f"IEEE-CIS transaction file not found at: {tx_file}")

    if sample_size is None or sample_size >= 590540:
        if verbose:
            print(f"[DataLoader] Loading full IEEE-CIS dataset (590,540 rows)...")
        return load_ieee_cis_raw(load_identity=True, validate=False)

    if strategy == "chronological":
        if verbose:
            print(f"[DataLoader] Loading first {sample_size:,} chronological transactions...")
        tx_df = pd.read_csv(tx_file, nrows=sample_size)
        if id_file.exists():
            id_df = pd.read_csv(id_file)
            merged = pd.merge(tx_df, id_df, on="TransactionID", how="left")
        else:
            merged = tx_df
        return merged

    elif strategy == "stratified":
        if verbose:
            print(f"[DataLoader] Sampling {sample_size:,} stratified transactions preserving population fraud rate...")
        t0 = time.time()
        targets = pd.read_csv(tx_file, usecols=["isFraud"])["isFraud"]

        rng = np.random.RandomState(random_state)
        fraud_indices = targets[targets == 1].index.values
        legit_indices = targets[targets == 0].index.values

        target_fraud_count = int(sample_size * (len(fraud_indices) / len(targets)))
        target_legit_count = sample_size - target_fraud_count

        sampled_fraud_idx = rng.choice(fraud_indices, size=target_fraud_count, replace=False)
        sampled_legit_idx = rng.choice(legit_indices, size=target_legit_count, replace=False)
        sampled_set = set(sampled_fraud_idx).union(set(sampled_legit_idx))

        chunks = []
        current_idx = 0
        for chunk in pd.read_csv(tx_file, chunksize=100000):
            chunk_indices = np.arange(current_idx, current_idx + len(chunk))
            mask = [i in sampled_set for i in chunk_indices]
            if any(mask):
                chunks.append(chunk.iloc[mask])
            current_idx += len(chunk)

        tx_sampled = pd.concat(chunks, ignore_index=True)
        if id_file.exists():
            id_df = pd.read_csv(id_file)
            merged = pd.merge(tx_sampled, id_df, on="TransactionID", how="left")
        else:
            merged = tx_sampled

        elapsed = time.time() - t0
        if verbose:
            fraud_cnt = int(merged["isFraud"].sum())
            legit_cnt = len(merged) - fraud_cnt
            print(f"[DataLoader] Loaded {len(merged):,} rows in {elapsed:.1f}s (Fraud: {fraud_cnt} [{fraud_cnt/len(merged)*100:.3f}%], Legit: {legit_cnt})")
        return merged

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


def instantiate_ieee_model(
    model_name: str,
    cat_idxs: List[int],
    cat_dims: List[int],
    random_state: int,
    scale_pos_weight: float = 27.58,
    tabnet_epochs: int = 12,
    tabnet_patience: int = 4,
    rf_trees: int = 100,
    xgb_trees: int = 100,
) -> Any:
    """Instantiate model architecture configured for IEEE-CIS fraud detection."""
    if model_name == "logistic_regression":
        return build_logistic_regression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=random_state,
        )
    elif model_name == "random_forest":
        return build_random_forest(
            n_estimators=rf_trees,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=4,
            random_state=random_state,
        )
    elif model_name == "xgboost":
        return build_xgboost_model(
            n_estimators=xgb_trees,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=4,
            random_state=random_state,
        )
    elif model_name == "tabnet":
        return TabNetFraudModel(
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            n_d=16,
            n_a=16,
            n_steps=3,
            gamma=1.3,
            lambda_sparse=1e-4,
            learning_rate=0.01,
            seed=random_state,
            device_name="cpu",
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")


def train_and_eval_ieee_fold(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    preprocessor: IEEEFraudPreprocessor,
    fold_idx: int,
    random_state: int = RANDOM_SEED,
    tabnet_epochs: int = 12,
    tabnet_patience: int = 4,
    rf_trees: int = 100,
    xgb_trees: int = 100,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Any]:
    """Train a single model on training data, calibrate threshold, and evaluate on validation data."""
    n_train_pos = int(y_train.sum())
    n_train_neg = len(y_train) - n_train_pos
    pos_weight = float(n_train_neg / max(1, n_train_pos))

    model = instantiate_ieee_model(
        model_name=model_name,
        cat_idxs=preprocessor.cat_idxs,
        cat_dims=preprocessor.cat_dims,
        random_state=random_state + fold_idx,
        scale_pos_weight=pos_weight,
        tabnet_epochs=tabnet_epochs,
        tabnet_patience=tabnet_patience,
        rf_trees=rf_trees,
        xgb_trees=xgb_trees,
    )

    # 1. Measure Training Time
    t_train_start = time.time()
    if model_name == "tabnet":
        n_samples_train = len(X_train)
        batch_sz = 1024 if n_samples_train >= 10000 else 256
        v_batch_sz = 128 if n_samples_train >= 10000 else 32
        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)],
            max_epochs=tabnet_epochs,
            patience=tabnet_patience,
            batch_size=batch_sz,
            virtual_batch_size=v_batch_sz,
            weights=1,
        )
    else:
        model.fit(X_train, y_train)
    train_time_sec = time.time() - t_train_start

    # 2. Determine Train Predictions for Threshold Selection
    # (strictly on training set, zero test leakage)
    if model_name == "tabnet":
        p_train = model.predict_proba(X_train)[:, 1]
    else:
        p_train = model.predict_proba(X_train)[:, 1]

    tuned_threshold = tune_threshold_on_train(y_train, p_train)

    # 3. Measure Inference Time on Validation Data
    t_infer_start = time.time()
    if model_name == "tabnet":
        p_val = model.predict_proba(X_val)[:, 1]
    else:
        p_val = model.predict_proba(X_val)[:, 1]
    infer_time_sec = time.time() - t_infer_start

    n_val = len(X_val)
    latency_ms_per_1k = (infer_time_sec / max(1, n_val)) * 1000 * 1000
    throughput_tx_per_sec = n_val / max(1e-6, infer_time_sec)

    # 4. Compute Metrics on Validation Data
    metrics_tuned = compute_classification_metrics(y_val, p_val, threshold=tuned_threshold)
    metrics_default = compute_classification_metrics(y_val, p_val, threshold=0.5)

    result_dict = {
        "fold": fold_idx,
        "roc_auc": metrics_tuned["roc_auc"],
        "pr_auc": metrics_tuned["pr_auc"],
        "recall_at_fpr_1pct": metrics_tuned["recall_at_fpr_1pct"],
        "f1": metrics_tuned["f1"],
        "macro_f1": metrics_tuned["macro_f1"],
        "precision": metrics_tuned["precision"],
        "recall": metrics_tuned["recall"],
        "threshold_tuned": tuned_threshold,
        "confusion_matrix": metrics_tuned["confusion_matrix"],
        "metrics_at_default_05": {
            "f1": metrics_default["f1"],
            "macro_f1": metrics_default["macro_f1"],
            "precision": metrics_default["precision"],
            "recall": metrics_default["recall"],
            "confusion_matrix": metrics_default["confusion_matrix"],
        },
        "train_time_sec": round(train_time_sec, 3),
        "infer_time_sec": round(infer_time_sec, 3),
        "latency_ms_per_1k": round(latency_ms_per_1k, 2),
        "throughput_tx_per_sec": round(throughput_tx_per_sec, 1),
    }

    if verbose:
        print(
            f"  [Fold {fold_idx}] PR-AUC: {result_dict['pr_auc']:.4f} | "
            f"ROC-AUC: {result_dict['roc_auc']:.4f} | "
            f"Recall@1%FPR: {result_dict['recall_at_fpr_1pct']:.4f} | "
            f"F1: {result_dict['f1']:.4f} (th={tuned_threshold:.2f}) | "
            f"Fit: {train_time_sec:.2f}s | Infer: {latency_ms_per_1k:.1f}ms/1k"
        )

    return result_dict, model


def run_stratified_cv_benchmark(
    df: pd.DataFrame,
    models: List[str],
    n_splits: int = CV_N_SPLITS,
    random_state: int = RANDOM_SEED,
    tabnet_epochs: int = 12,
    tabnet_patience: int = 4,
    rf_trees: int = 100,
    xgb_trees: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute Protocol A: Stratified 5-Fold Cross-Validation across all models."""
    print("\n" + "=" * 80)
    print(f"PROTOCOL A: STRATIFIED {n_splits}-FOLD CROSS-VALIDATION (N={len(df):,})")
    print("Objective: Assess intrinsic algorithmic capacity under stationary class distribution.")
    print("=" * 80)

    results: Dict[str, Any] = {}

    for model_name in models:
        print(f"\n>>> Running Stratified CV for Model: {model_name.upper()}...")
        fold_records = []
        t0 = time.time()

        cv_gen = generate_ieee_folds(
            df=df,
            strategy="stratified",
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
            target_col=TARGET_COL,
            as_df=False,
            scale_numerical=True,
        )

        for fold_idx, X_tr, y_tr, X_va, y_va, prep in cv_gen:
            fold_res, _ = train_and_eval_ieee_fold(
                model_name=model_name,
                X_train=X_tr,
                y_train=y_tr,
                X_val=X_va,
                y_val=y_va,
                preprocessor=prep,
                fold_idx=fold_idx,
                random_state=random_state,
                tabnet_epochs=tabnet_epochs,
                tabnet_patience=tabnet_patience,
                rf_trees=rf_trees,
                xgb_trees=xgb_trees,
                verbose=verbose,
            )
            fold_records.append(fold_res)

        total_time = time.time() - t0
        aggregated = aggregate_fold_metrics(fold_records)

        # Also aggregate latency & train time
        train_times = [r["train_time_sec"] for r in fold_records]
        infer_latencies = [r["latency_ms_per_1k"] for r in fold_records]
        aggregated["train_time_sec"] = {
            "mean": float(np.mean(train_times)),
            "std": float(np.std(train_times, ddof=1)) if len(train_times) > 1 else 0.0,
        }
        aggregated["latency_ms_per_1k"] = {
            "mean": float(np.mean(infer_latencies)),
            "std": float(np.std(infer_latencies, ddof=1)) if len(infer_latencies) > 1 else 0.0,
        }

        results[model_name] = {
            "model_name": model_name,
            "protocol": "stratified_cv",
            "n_samples": len(df),
            "n_splits": n_splits,
            "total_wall_clock_sec": round(total_time, 2),
            "fold_metrics": fold_records,
            "aggregated": aggregated,
        }

        print(f"Summary for {model_name.upper()}:")
        print(f"  PR-AUC:             {aggregated['pr_auc']['mean']:.4f} +/- {aggregated['pr_auc']['std']:.4f}")
        print(f"  ROC-AUC:            {aggregated['roc_auc']['mean']:.4f} +/- {aggregated['roc_auc']['std']:.4f}")
        print(f"  Recall @ 1% FPR:    {aggregated['recall_at_fpr_1pct']['mean']:.4f} +/- {aggregated['recall_at_fpr_1pct']['std']:.4f}")
        print(f"  F1-Score:           {aggregated['f1']['mean']:.4f} +/- {aggregated['f1']['std']:.4f}")
        print(f"  Mean Fit Time:      {aggregated['train_time_sec']['mean']:.2f}s")
        print(f"  Mean Latency / 1k:  {aggregated['latency_ms_per_1k']['mean']:.1f}ms")

    return results


def run_chronological_holdout_benchmark(
    df: pd.DataFrame,
    models: List[str],
    train_ratio: float = 0.80,
    tabnet_epochs: int = 12,
    tabnet_patience: int = 4,
    rf_trees: int = 100,
    xgb_trees: int = 100,
    save_artifacts: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute Protocol B: Chronological 80/20 Holdout Split across all models."""
    print("\n" + "=" * 80)
    print(f"PROTOCOL B: CHRONOLOGICAL 80/20 HOLDOUT SPLIT (N={len(df):,})")
    print("Objective: Assess real-world temporal generalization under concept drift.")
    print("=" * 80)

    chrono_gen = generate_ieee_folds(
        df=df,
        strategy="chronological",
        train_ratio=train_ratio,
        target_col=TARGET_COL,
        as_df=False,
        scale_numerical=True,
    )

    fold_idx, X_train, y_train, X_val, y_val, preprocessor = next(chrono_gen)
    print(f"Train samples (past): {len(X_train):,} | Fraud: {y_train.sum()} ({y_train.mean()*100:.3f}%)")
    print(f"Val samples (future): {len(X_val):,} | Fraud: {y_val.sum()} ({y_val.mean()*100:.3f}%)")

    results: Dict[str, Any] = {}

    for model_name in models:
        print(f"\n>>> Running Chronological Holdout for Model: {model_name.upper()}...")
        eval_res, fitted_model = train_and_eval_ieee_fold(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            preprocessor=preprocessor,
            fold_idx=0,
            tabnet_epochs=tabnet_epochs,
            tabnet_patience=tabnet_patience,
            rf_trees=rf_trees,
            xgb_trees=xgb_trees,
            verbose=verbose,
        )

        results[model_name] = {
            "model_name": model_name,
            "protocol": "chronological_holdout",
            "n_samples": len(df),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "train_ratio": train_ratio,
            "metrics": eval_res,
        }

        # Save production model artifacts
        if save_artifacts:
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            if model_name == "logistic_regression":
                art_path = ARTIFACTS_DIR / "ieee_logistic_regression.joblib"
                joblib.dump(fitted_model, art_path)
                print(f"  Saved artifact: {art_path}")
            elif model_name == "random_forest":
                art_path = ARTIFACTS_DIR / "ieee_random_forest.joblib"
                joblib.dump(fitted_model, art_path)
                print(f"  Saved artifact: {art_path}")
            elif model_name == "xgboost":
                art_path = ARTIFACTS_DIR / "ieee_xgboost.json"
                fitted_model.save_model(str(art_path))
                print(f"  Saved artifact: {art_path}")
            elif model_name == "tabnet":
                art_path = ARTIFACTS_DIR / "ieee_tabnet.zip"
                fitted_model.save_model(ARTIFACTS_DIR / "ieee_tabnet")
                print(f"  Saved artifact: {art_path}")

    if save_artifacts:
        prep_path = ARTIFACTS_DIR / "ieee_preprocessor.joblib"
        joblib.dump(preprocessor, prep_path)
        print(f"  Saved fitted preprocessor: {prep_path}")

    return results


def compile_and_export_ieee_results(
    cv_results: Dict[str, Any],
    chrono_results: Dict[str, Any],
    output_csv: Path,
    output_fig: Path,
) -> pd.DataFrame:
    """Build unified comparison summary table and high-resolution multi-panel visualization."""
    models = list(cv_results.keys())
    rows = []

    for m in models:
        cv_agg = cv_results[m]["aggregated"]
        ch_m = chrono_results[m]["metrics"]

        rows.append({
            "Model": m.replace("_", " ").title(),
            "CV PR-AUC": f"{cv_agg['pr_auc']['mean']:.4f} +/- {cv_agg['pr_auc']['std']:.4f}",
            "Chrono PR-AUC": f"{ch_m['pr_auc']:.4f}",
            "CV ROC-AUC": f"{cv_agg['roc_auc']['mean']:.4f} +/- {cv_agg['roc_auc']['std']:.4f}",
            "Chrono ROC-AUC": f"{ch_m['roc_auc']:.4f}",
            "CV Recall@1%FPR": f"{cv_agg['recall_at_fpr_1pct']['mean']:.4f} +/- {cv_agg['recall_at_fpr_1pct']['std']:.4f}",
            "Chrono Recall@1%FPR": f"{ch_m['recall_at_fpr_1pct']:.4f}",
            "CV F1-Score": f"{cv_agg['f1']['mean']:.4f} +/- {cv_agg['f1']['std']:.4f}",
            "Chrono F1-Score": f"{ch_m['f1']:.4f}",
            "Fit Time (s)": round(cv_agg["train_time_sec"]["mean"], 2),
            "Latency (ms/1k)": round(cv_agg["latency_ms_per_1k"]["mean"], 1),
        })

    summary_df = pd.DataFrame(rows)
    print("\n" + "=" * 105)
    print("UNIFIED IEEE-CIS BENCHMARK COMPARISON TABLE (PROTOCOL A vs PROTOCOL B)")
    print("=" * 105)
    print(summary_df.to_string(index=False))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    print(f"\nPersisted comparison CSV to: {output_csv}")

    # Plot Multi-panel Comparison Figure
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("IEEE-CIS Model Benchmark: Algorithmic Capacity (CV) vs Concept Drift (Chronological)", fontsize=13, fontweight="bold")

    x = np.arange(len(models))
    width = 0.35
    model_labels = [m.replace("_", " ").title() for m in models]

    # Subplot 1: PR-AUC
    ax = axes[0, 0]
    cv_vals = [cv_results[m]["aggregated"]["pr_auc"]["mean"] for m in models]
    cv_errs = [cv_results[m]["aggregated"]["pr_auc"]["std"] for m in models]
    ch_vals = [chrono_results[m]["metrics"]["pr_auc"] for m in models]
    ax.bar(x - width/2, cv_vals, width, yerr=cv_errs, capsize=4, label="Stratified CV", color="#1f77b4", edgecolor="black", linewidth=0.7)
    ax.bar(x + width/2, ch_vals, width, label="Chronological Holdout", color="#ff7f0e", edgecolor="black", linewidth=0.7)
    ax.set_ylabel("PR-AUC (Average Precision)")
    ax.set_title("PR-AUC (Higher is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha="right")
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    # Subplot 2: ROC-AUC
    ax = axes[0, 1]
    cv_vals = [cv_results[m]["aggregated"]["roc_auc"]["mean"] for m in models]
    cv_errs = [cv_results[m]["aggregated"]["roc_auc"]["std"] for m in models]
    ch_vals = [chrono_results[m]["metrics"]["roc_auc"] for m in models]
    ax.bar(x - width/2, cv_vals, width, yerr=cv_errs, capsize=4, label="Stratified CV", color="#1f77b4", edgecolor="black", linewidth=0.7)
    ax.bar(x + width/2, ch_vals, width, label="Chronological Holdout", color="#ff7f0e", edgecolor="black", linewidth=0.7)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC (Higher is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 3: Recall @ FPR <= 1%
    ax = axes[0, 2]
    cv_vals = [cv_results[m]["aggregated"]["recall_at_fpr_1pct"]["mean"] for m in models]
    cv_errs = [cv_results[m]["aggregated"]["recall_at_fpr_1pct"]["std"] for m in models]
    ch_vals = [chrono_results[m]["metrics"]["recall_at_fpr_1pct"] for m in models]
    ax.bar(x - width/2, cv_vals, width, yerr=cv_errs, capsize=4, label="Stratified CV", color="#1f77b4", edgecolor="black", linewidth=0.7)
    ax.bar(x + width/2, ch_vals, width, label="Chronological Holdout", color="#ff7f0e", edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Recall @ FPR <= 1%")
    ax.set_title("Operational Recall at Bounded 1% FPR")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha="right")
    ax.set_ylim(0, 0.65)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 4: F1-Score
    ax = axes[1, 0]
    cv_vals = [cv_results[m]["aggregated"]["f1"]["mean"] for m in models]
    cv_errs = [cv_results[m]["aggregated"]["f1"]["std"] for m in models]
    ch_vals = [chrono_results[m]["metrics"]["f1"] for m in models]
    ax.bar(x - width/2, cv_vals, width, yerr=cv_errs, capsize=4, label="Stratified CV", color="#1f77b4", edgecolor="black", linewidth=0.7)
    ax.bar(x + width/2, ch_vals, width, label="Chronological Holdout", color="#ff7f0e", edgecolor="black", linewidth=0.7)
    ax.set_ylabel("F1-Score")
    ax.set_title("F1-Score (Train-Calibrated Threshold)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha="right")
    ax.set_ylim(0, 0.65)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 5: Mean Fit Time
    ax = axes[1, 1]
    fit_times = [cv_results[m]["aggregated"]["train_time_sec"]["mean"] for m in models]
    fit_errs = [cv_results[m]["aggregated"]["train_time_sec"]["std"] for m in models]
    ax.bar(x, fit_times, width=0.5, yerr=fit_errs, capsize=4, color="#2ca02c", edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Wall Clock Training Time (s)")
    ax.set_title("Training Time per 40k Partition")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 6: Inference Latency
    ax = axes[1, 2]
    latencies = [cv_results[m]["aggregated"]["latency_ms_per_1k"]["mean"] for m in models]
    lat_errs = [cv_results[m]["aggregated"]["latency_ms_per_1k"]["std"] for m in models]
    ax.bar(x, latencies, width=0.5, yerr=lat_errs, capsize=4, color="#9467bd", edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Latency (ms per 1,000 transactions)")
    ax.set_title("Inference Latency on CPU")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_fig, dpi=300)
    plt.close(fig)
    print(f"Persisted comparison chart to: {output_fig}")

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="IEEE-CIS Model Benchmarking and Training Harness")
    parser.add_argument("--sample-size", type=int, default=50000, help="Transaction sample size (default 50,000)")
    parser.add_argument("--smoke-test", action="store_true", help="Quick sanity check on 5,000 samples with 3 folds")
    parser.add_argument("--protocol", type=str, default="all", choices=["all", "stratified", "chronological"])
    parser.add_argument("--model", type=str, default="all", choices=["all", "logistic_regression", "random_forest", "xgboost", "tabnet"])
    args = parser.parse_args()

    if args.smoke_test:
        sample_size = 5000
        n_splits = 3
        tabnet_epochs = 2
        rf_trees = 30
        xgb_trees = 30
    else:
        sample_size = args.sample_size
        n_splits = CV_N_SPLITS
        tabnet_epochs = 12
        rf_trees = 100
        xgb_trees = 100

    models_to_run = (
        ["logistic_regression", "random_forest", "xgboost", "tabnet"]
        if args.model == "all"
        else [args.model]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    cv_results = {}
    chrono_results = {}

    # 1. Stratified CV Benchmark (Protocol A)
    if args.protocol in ["all", "stratified"]:
        df_strat = load_ieee_benchmark_data(sample_size=sample_size, strategy="stratified", random_state=RANDOM_SEED)
        cv_results = run_stratified_cv_benchmark(
            df=df_strat,
            models=models_to_run,
            n_splits=n_splits,
            random_state=RANDOM_SEED,
            tabnet_epochs=tabnet_epochs,
            tabnet_patience=4,
            rf_trees=rf_trees,
            xgb_trees=xgb_trees,
        )
        cv_out_path = RESULTS_DIR / "ieee_metrics_stratified_cv.json"
        with open(cv_out_path, "w", encoding="utf-8") as f:
            json.dump(cv_results, f, indent=2)
        print(f"\nPersisted Stratified CV metrics to: {cv_out_path}")

    # 2. Chronological Holdout Benchmark (Protocol B)
    if args.protocol in ["all", "chronological"]:
        df_chrono = load_ieee_benchmark_data(sample_size=sample_size, strategy="chronological", random_state=RANDOM_SEED)
        chrono_results = run_chronological_holdout_benchmark(
            df=df_chrono,
            models=models_to_run,
            train_ratio=0.80,
            tabnet_epochs=tabnet_epochs,
            tabnet_patience=4,
            rf_trees=rf_trees,
            xgb_trees=xgb_trees,
            save_artifacts=True,
        )
        chrono_out_path = RESULTS_DIR / "ieee_metrics_chronological.json"
        with open(chrono_out_path, "w", encoding="utf-8") as f:
            json.dump(chrono_results, f, indent=2)
        print(f"\nPersisted Chronological Holdout metrics to: {chrono_out_path}")

    # 3. Export Summary Table and Visualization
    if cv_results and chrono_results and args.model == "all":
        compile_and_export_ieee_results(
            cv_results=cv_results,
            chrono_results=chrono_results,
            output_csv=RESULTS_DIR / "ieee_model_comparison.csv",
            output_fig=FIGURES_DIR / "15_ieee_model_comparison.png",
        )

        metadata = {
            "dataset": "IEEE-CIS Fraud Detection",
            "curated_features_count": 46,
            "sample_size": sample_size,
            "n_splits_cv": n_splits,
            "chronological_train_ratio": 0.80,
            "models_evaluated": models_to_run,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(RESULTS_DIR / "ieee_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
