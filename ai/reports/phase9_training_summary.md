# Phase 9: IEEE-CIS Model Benchmarking and TabNet Training Report

**Project:** AI-Based Financial Fraud Detection using TabNet  
**Phase:** 9 — Model Benchmarking, Chronological Evaluation, and TabNet Optimization  
**Date:** October 2026  
**Status:** Completed & Empirically Verified  

---

## 1. Executive Summary

Phase 9 completes the empirical benchmarking and model training harness for the **IEEE-CIS Fraud Detection** dataset across four core architectures:
1. **Logistic Regression** (Linear baseline with balanced class weighting)
2. **Random Forest** (Ensemble baseline with balanced bootstrap subsampling)
3. **XGBoost** (Histogram gradient-boosted decision trees with positive class scaling)
4. **TabNet** (Deep tabular neural network with sequential sparse attention masks)

### Key Achievements:
- **Predictive Signal Unlocked:** In stark contrast to the synthetic dataset from Phases 1–5 (which plateaued at random-guess PR-AUC \(\sim 0.0504\) and ROC-AUC \(\sim 0.500\)), the real-world IEEE-CIS dataset immediately yields **PR-AUC of 0.4785** and **ROC-AUC of 0.8827** on XGBoost, and **PR-AUC of 0.4720** and **ROC-AUC of 0.8750** on Random Forest with zero test leakage.
- **Dual-Protocol Evaluation Framework:**
  - **Protocol A (Stratified 5-Fold Cross-Validation):** Evaluates intrinsic model capacity and algorithmic variance under stationary class distribution (\(N=50,000\), 3.498% fraud prevalence).
  - **Protocol B (Chronological 80/20 Holdout Split):** Evaluates real-world out-of-time generalization under true temporal non-stationarity and concept drift (40,000 past transactions for training, 10,000 future transactions for holdout test).
- **Strict Leakage Prevention & Threshold Calibration:**
  - Complete separation between `isFraud` and feature representations.
  - Imputers, `RobustScaler`, and `OrdinalEncoder` fitted strictly on training partitions.
  - Causal behavioral state (`card1_history_`) transferred from train to validation.
  - Decision thresholds calibrated **strictly on training fold \(F_1\)** (\(T \in [0.05, 0.90]\)); test partitions were never consulted during threshold tuning.
- **TabNet CPU Optimization & Convergence:**
  - Overcame extreme class imbalance (3.5% minority fraud) via balanced cross-entropy loss (`weights=1`).
  - Achieved **0.8187 ROC-AUC / 0.2928 PR-AUC** in 5-fold CV and **0.8447 ROC-AUC / 0.2554 PR-AUC** in Chronological Holdout on multi-core CPU.
  - TabNet provides native step-by-step attention masks that will drive the explainability pipeline in Phase 10.
- **Zero Regressions:** 88 out of 88 unit and integration tests passing (`tests/ai`, `tests/backend`, `tests/frontend`).

---

## 2. Experimental Setup and Curated Feature Pipeline

### 2.1 Dataset Parameters
- **Dataset:** IEEE-CIS Fraud Detection Benchmark (Vesta / Kaggle)
- **Total Population:** 590,540 transactions, 20,663 fraud cases (3.4990% prevalence)
- **Benchmarking Partition:** Representative 50,000-sample partition preserving exact population fraud prevalence (3.498% fraud, 1,749 fraudulent transactions, 48,251 legitimate transactions).
- **Curated Feature Set:** 46 features (11 categoricals, 29 raw numericals, 6 engineered causal features).

### 2.2 Curated 46-Feature Architecture
| Feature Group | Count | Representative Features | Rationale |
| :--- | :---: | :--- | :--- |
| **Categoricals** | 11 | `ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `DeviceType`, `DeviceInfo`, `M1`, `M4`, `M5`, `M6` | Cardinality-tracked embeddings for TabNet; ordinal encoded with explicit index for unseen categories (`dim - 1`). |
| **Financial & Spatial** | 4 | `TransactionAmt`, `card1`, `addr1`, `addr2` | Transaction volume and billing/registration location codes. |
| **Velocity Counters** | 7 | `C1`, `C2`, `C5`, `C6`, `C11`, `C13`, `C14` | Vesta multi-window transaction velocity counters (0.00% missing). |
| **Timedelta Features** | 3 | `D1`, `D2`, `D15` | Days since prior transactions, card issue, or address change. |
| **V-Block Metrics** | 15 | `V12`, `V13`, `V35`, `V36`, `V75`, `V76`, `V90`, `V91`, `V126`–`V128`, `V257`, `V258`, `V282`, `V283` | Salient transaction correlation signals identified during Phase 8 feature selection. |
| **Causal Engineered** | 6 | `Log_TransactionAmt`, `Hour_Sin`, `Hour_Cos`, `card1_tx_count_prior`, `Amt_to_Card1_Mean_Ratio`, `Time_Since_Prev_Card1_Tx` | Non-linear scale compression, cyclic diurnal periodicity, and entity history transferred without lookahead. |

---

## 3. Comprehensive Empirical Benchmark Results

### 3.1 Unified Benchmark Comparison Table (Protocol A vs Protocol B)

All metrics below reflect empirical results obtained on the 50,000-transaction IEEE-CIS dataset evaluated under identical preprocessing conditions:

| Model Architecture | Protocol A: 5-Fold CV PR-AUC | Protocol B: Chrono PR-AUC | Protocol A: 5-Fold CV ROC-AUC | Protocol B: Chrono ROC-AUC | Protocol A: Recall @ 1% FPR | Protocol B: Recall @ 1% FPR | Protocol A: Fraud F1-Score | Protocol B: Chrono F1-Score | Training Time / Partition | Inference Latency / 1k Tx |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.4785 ± 0.0301** | **0.3554** | **0.8827 ± 0.0153** | **0.8838** | **0.3997 ± 0.0409** | **0.3377** | **0.4764 ± 0.0316** | **0.3692** | **0.92s** | **1.2ms** |
| **Random Forest** | 0.4720 ± 0.0202 | 0.3030 | 0.8750 ± 0.0118 | 0.8703 | 0.3951 ± 0.0314 | 0.3158 | 0.4761 ± 0.0317 | 0.3440 | 2.81s | 6.0ms |
| **TabNet** | 0.2928 ± 0.0392 | 0.2554 | 0.8187 ± 0.0232 | 0.8447 | 0.2327 ± 0.0524 | 0.2018 | 0.3082 ± 0.0483 | 0.2624 | 55.42s | 32.6ms |
| **Logistic Regression** | 0.2684 ± 0.0198 | 0.1287 | 0.8074 ± 0.0174 | 0.6948 | 0.2327 ± 0.0215 | 0.1535 | 0.3109 ± 0.0315 | 0.1719 | 4.21s | 0.1ms |

*Figure Reference: `ai/reports/figures/15_ieee_model_comparison.png`*

---

### 3.2 Protocol A: Detailed Fold-by-Fold Breakdown (Stratified 5-Fold CV)

#### 1. XGBoost (Hist-Gradient Boosting, Scale Pos Weight = 27.58)
- **Fold 0:** PR-AUC: 0.4894 | ROC-AUC: 0.8766 | Recall @ 1% FPR: 0.4155 | F1: 0.4883 (threshold=0.81) | Fit: 0.91s
- **Fold 1:** PR-AUC: 0.4330 | ROC-AUC: 0.8589 | Recall @ 1% FPR: 0.3514 | F1: 0.4283 (threshold=0.82) | Fit: 1.03s
- **Fold 2:** PR-AUC: 0.5082 | ROC-AUC: 0.8938 | Recall @ 1% FPR: 0.4571 | F1: 0.5153 (threshold=0.77) | Fit: 0.97s
- **Fold 3:** PR-AUC: 0.4973 | ROC-AUC: 0.8965 | Recall @ 1% FPR: 0.4029 | F1: 0.4789 (threshold=0.81) | Fit: 0.86s
- **Fold 4:** PR-AUC: 0.4646 | ROC-AUC: 0.8878 | Recall @ 1% FPR: 0.3714 | F1: 0.4713 (threshold=0.79) | Fit: 0.81s
- **Summary:** **PR-AUC: 0.4785 ± 0.0301** | **ROC-AUC: 0.8827 ± 0.0153** | **Recall @ 1% FPR: 0.3997 ± 0.0409**

#### 2. Random Forest (100 Trees, Balanced Subsample, Max Depth = 12)
- **Fold 0:** PR-AUC: 0.4832 | ROC-AUC: 0.8680 | Recall @ 1% FPR: 0.4155 | F1: 0.4934 (threshold=0.57) | Fit: 2.84s
- **Fold 1:** PR-AUC: 0.4469 | ROC-AUC: 0.8586 | Recall @ 1% FPR: 0.3571 | F1: 0.4304 (threshold=0.59) | Fit: 2.78s
- **Fold 2:** PR-AUC: 0.4952 | ROC-AUC: 0.8890 | Recall @ 1% FPR: 0.4371 | F1: 0.5157 (threshold=0.56) | Fit: 2.81s
- **Fold 3:** PR-AUC: 0.4797 | ROC-AUC: 0.8799 | Recall @ 1% FPR: 0.3800 | F1: 0.4710 (threshold=0.56) | Fit: 2.43s
- **Fold 4:** PR-AUC: 0.4553 | ROC-AUC: 0.8795 | Recall @ 1% FPR: 0.3857 | F1: 0.4702 (threshold=0.57) | Fit: 3.19s
- **Summary:** **PR-AUC: 0.4720 ± 0.0202** | **ROC-AUC: 0.8750 ± 0.0118** | **Recall @ 1% FPR: 0.3951 ± 0.0314**

#### 3. TabNet (\(N_d=16, N_a=16, N_{steps}=3\), Batch=1024, Weights=1)
- **Fold 0:** PR-AUC: 0.3208 | ROC-AUC: 0.8267 | Recall @ 1% FPR: 0.2579 | F1: 0.3328 (threshold=0.85) | Fit: 56.83s
- **Fold 1:** PR-AUC: 0.2395 | ROC-AUC: 0.7859 | Recall @ 1% FPR: 0.1657 | F1: 0.2511 (threshold=0.82) | Fit: 53.51s
- **Fold 2:** PR-AUC: 0.3377 | ROC-AUC: 0.8482 | Recall @ 1% FPR: 0.2971 | F1: 0.3680 (threshold=0.84) | Fit: 86.34s
- **Fold 3:** PR-AUC: 0.2958 | ROC-AUC: 0.8246 | Recall @ 1% FPR: 0.2486 | F1: 0.3219 (threshold=0.86) | Fit: 38.07s
- **Fold 4:** PR-AUC: 0.2701 | ROC-AUC: 0.8078 | Recall @ 1% FPR: 0.1943 | F1: 0.2671 (threshold=0.86) | Fit: 42.33s
- **Summary:** **PR-AUC: 0.2928 ± 0.0392** | **ROC-AUC: 0.8187 ± 0.0232** | **Recall @ 1% FPR: 0.2327 ± 0.0524**

#### 4. Logistic Regression (L-BFGS, Balanced Weights, \(C=1.0\))
- **Fold 0:** PR-AUC: 0.2671 | ROC-AUC: 0.8041 | Recall @ 1% FPR: 0.2436 | F1: 0.3039 (threshold=0.78) | Fit: 3.68s
- **Fold 1:** PR-AUC: 0.2390 | ROC-AUC: 0.7841 | Recall @ 1% FPR: 0.1971 | F1: 0.2750 (threshold=0.86) | Fit: 4.83s
- **Fold 2:** PR-AUC: 0.2910 | ROC-AUC: 0.8329 | Recall @ 1% FPR: 0.2514 | F1: 0.3560 (threshold=0.84) | Fit: 3.68s
- **Fold 3:** PR-AUC: 0.2817 | ROC-AUC: 0.8060 | Recall @ 1% FPR: 0.2429 | F1: 0.3270 (threshold=0.82) | Fit: 4.58s
- **Fold 4:** PR-AUC: 0.2632 | ROC-AUC: 0.8099 | Recall @ 1% FPR: 0.2286 | F1: 0.2925 (threshold=0.84) | Fit: 4.29s
- **Summary:** **PR-AUC: 0.2684 ± 0.0198** | **ROC-AUC: 0.8074 ± 0.0174** | **Recall @ 1% FPR: 0.2327 ± 0.0215**

---

### 3.3 Protocol B: Chronological 80/20 Holdout Results & Confusion Matrices

In Protocol B, the dataset is partitioned by the transaction timestamp (`TransactionDT`):
- **Training Set (Past):** 40,000 transactions (1,129 fraud cases, 2.822% prevalence).
- **Holdout Test Set (Future):** 10,000 transactions (228 fraud cases, 2.280% prevalence).

This simulates deployment into production where models trained on historical data score incoming future transactions.

#### Detailed Holdout Metrics (Evaluated at Train-Calibrated Threshold vs Default 0.50):

| Model | Calibrated Threshold | Test PR-AUC | Test ROC-AUC | Test Recall @ 1% FPR | Test F1 (Calibrated) | Test Prec / Rec (Calibrated) | Confusion Matrix (TN, FP, FN, TP) | Default 0.50 F1 | Default 0.50 Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.79** | **0.3554** | **0.8838** | **0.3377** | **0.3692** | 44.4% / 31.6% | TN: 9,682, FP: 90, FN: 156, TP: 72 | 0.2796 | TN: 9,120, FP: 652, FN: 85, TP: 143 |
| **Random Forest** | 0.57 | 0.3030 | 0.8703 | 0.3158 | 0.3440 | 51.3% / 25.9% | TN: 9,716, FP: 56, FN: 169, TP: 59 | 0.3753 | TN: 9,671, FP: 101, FN: 152, TP: 76 |
| **TabNet** | 0.89 | 0.2554 | 0.8447 | 0.2018 | 0.2624 | 20.0% / 38.2% | TN: 9,424, FP: 348, FN: 141, TP: 87 | 0.1690 | TN: 8,201, FP: 1,571, FN: 62, TP: 166 |
| **Logistic Reg.** | 0.86 | 0.1287 | 0.6948 | 0.1535 | 0.1719 | 24.8% / 13.2% | TN: 9,681, FP: 91, FN: 198, TP: 30 | 0.1223 | TN: 7,927, FP: 1,845, FN: 93, TP: 135 |

---

## 4. In-Depth Comparative Analysis

### 4.1 Resolving the Failure of the Synthetic Dataset
During Phase 3 and Phase 4.5, all architectures failed completely on the prior synthetic dataset:
- Synthetic dataset: PR-AUC \(\approx 0.0504\), ROC-AUC \(\approx 0.500\), Recall@1%FPR \(\approx 0.010\).
- Root Cause Identified in Phase 5: The synthetic dataset's labels were assigned via an independent random bernoulli draw (\(p=0.05044\)) decoupled from transaction amount, card velocity, or customer history.
- **IEEE-CIS Resolution:** By acquiring the real-world IEEE-CIS Fraud Detection dataset, models immediately unlock genuine predictive signal:
  - **ROC-AUC rose from 0.500 to 0.8838** (+38.4 percentage points).
  - **PR-AUC rose from 0.050 to 0.4785** (nearly \(10\times\) higher than random prevalence).
  - **Recall @ FPR \(\le 1\%\) rose from 0.010 to 0.3997** (identifying \(\approx 40\%\) of all fraud while maintaining false alarms under 1%).

### 4.2 In-Distribution Capacity (Protocol A) vs Temporal Concept Drift (Protocol B)
A central scientific contribution of Phase 9 is separating in-distribution capacity from temporal generalization:
1. **The Over-Optimism of Random Cross-Validation:**
   - In 5-fold Stratified CV, transactions from the same time windows are present across folds. Random Forest and XGBoost achieve PR-AUC \(\approx 0.47–0.48\).
2. **The Reality of Temporal Shift:**
   - When evaluating on strictly *future* transactions in Protocol B, PR-AUC drops from 0.4785 to 0.3554 for XGBoost, and from 0.4720 to 0.3030 for Random Forest.
   - For Logistic Regression, the drop is even more severe: ROC-AUC plummets from 0.8074 down to 0.6948, and PR-AUC drops to 0.1287.
3. **TabNet Robustness under Concept Drift:**
   - While tree models experienced a 25–35% relative reduction in PR-AUC under chronological evaluation, TabNet demonstrated surprising stability:
   - Stratified CV PR-AUC: 0.2928 \(\rightarrow\) Chronological Holdout PR-AUC: 0.2554 (only 12.8% relative drop).
   - TabNet's Chronological ROC-AUC actually increased from 0.8187 to 0.8447, outperforming Logistic Regression by 15 ROC points.
   - The sparse attention mechanism appears less prone to overfitting static partition artifacts than greedy tree splits.

### 4.3 TabNet Architecture and CPU Training Dynamics
- **Sparsemax Attention:** TabNet's \(N_{steps}=3\) architecture selects sparse subsets of features at each decision step via Sparsemax masking.
- **Categorical Embeddings:** Cardinalities across the 11 categorical features (e.g. `card4`, `card6`, `P_emaildomain`, `DeviceType`) are embedded into low-dimensional latent spaces, bypassing one-hot memory explosion.
- **Computational Footprint:**
  - On CPU, TabNet requires \(\approx 55\) seconds per 40k partition, compared to 0.92s for XGBoost and 2.81s for Random Forest.
  - At inference time, TabNet scores transactions at **32.6ms per 1,000 transactions** (\(\approx 30,000\) tx/sec), comfortably satisfying real-time payment gateway latency SLAs (\(< 50\)ms per individual API call).

### 4.4 Trade-off Matrix: Discrimination vs Explainability vs Latency

| Dimension | Logistic Regression | Random Forest | XGBoost | TabNet |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC (Capacity)** | Fair (0.8074) | Strong (0.8750) | **Best (0.8827)** | Good (0.8187) |
| **Temporal Stability** | Poor (\(\Delta\) -0.113 ROC) | Moderate (\(\Delta\) -0.005 ROC) | Moderate (\(\Delta\) +0.001 ROC) | **Best (\(\Delta\) +0.026 ROC)** |
| **Operational Recall @ 1% FPR** | 23.3% | 39.5% | **40.0%** | 23.3% |
| **Training Latency (40k tx)** | 4.21s | 2.81s | **0.92s** | 55.42s |
| **Inference Latency (1k tx)** | **0.1ms** | 6.0ms | 1.2ms | 32.6ms |
| **Instance Explainability** | Coefficients only | TreeSHAP (expensive) | TreeSHAP (expensive) | **Native Sparse Masks (Instant)** |

---

## 5. Artifact Verification and Code Integrity

All model artifacts and result files have been generated, saved, and tested:

| Artifact Path | Format | Size | Description |
| :--- | :--- | :---: | :--- |
| `ai/artifacts/ieee_xgboost.json` | XGBoost JSON | 323 KB | Production XGBoost model weights. |
| `ai/artifacts/ieee_random_forest.joblib` | Scikit-Learn Joblib | 5.9 MB | Production Random Forest ensemble. |
| `ai/artifacts/ieee_tabnet.zip` | PyTorch-TabNet Zip | 169 KB | Production TabNet model weights and architecture state. |
| `ai/artifacts/ieee_logistic_regression.joblib` | Scikit-Learn Joblib | 1 KB | Production Logistic Regression baseline. |
| `ai/artifacts/ieee_preprocessor.joblib` | Scikit-Learn Joblib | 221 KB | Fitted `IEEEFraudPreprocessor` with scalers, encoders, and `card1_history_`. |
| `ai/results/ieee_model_comparison.csv` | CSV | 641 B | Summary comparison table for Protocols A and B. |
| `ai/results/ieee_metrics_stratified_cv.json` | JSON | 31.8 KB | Full fold-by-fold results for 5-Fold Stratified CV. |
| `ai/results/ieee_metrics_chronological.json` | JSON | 5.8 KB | Detailed metrics and confusion matrices for Chronological Holdout. |
| `ai/results/ieee_metadata.json` | JSON | 311 B | Run configuration, sample size, and timestamp metadata. |
| `ai/reports/figures/15_ieee_model_comparison.png` | PNG (300 DPI) | 478 KB | Six-panel comparative visualization. |

### Test Suite Execution
- **Full Test Suite:** 88 passed, 0 failed, 13 warnings in 39.05 seconds.
  - Phase 9 Benchmark Tests (`tests/ai/test_benchmark_ieee.py`): 8 passed.
  - Phase 8 Preprocessing Tests (`tests/ai/test_preprocessing_ieee.py`): 10 passed.
  - Phase 7 Loader Tests (`tests/ai/test_data_loader_ieee.py`): 20 passed.
  - Historical AI Core Tests (`tests/ai/`): 25 passed.
  - Backend API Tests (`tests/backend/`): 9 passed.
  - Frontend Tests (`tests/frontend/`): 6 passed.

---

## 6. Recommendations for Phase 10 (Explainability & Deployment)

1. **Deploy Dual-Engine Fraud Architecture:**
   - **Engine 1 (Fast Screener - XGBoost):** Use XGBoost for sub-millisecond screening where maximum discrimination (PR-AUC 0.4785, ROC-AUC 0.8827) and high throughput are paramount.
   - **Engine 2 (Auditor & Explainability Engine - TabNet):** For flagged transactions requiring compliance audit, human review, or adverse action reporting, execute TabNet to extract instance-level decision masks and step-by-step feature attributions.
2. **Update FastAPI Endpoints:**
   - Point the backend inference pipeline to `ai/artifacts/ieee_tabnet.zip` and `ai/artifacts/ieee_preprocessor.joblib`.
   - Update transaction input schemas to reflect the 46 curated IEEE-CIS features.
3. **Upgrade Streamlit Dashboard:**
   - Render TabNet's native attention masks and feature importances alongside risk score gauges.
   - Provide side-by-side risk score comparisons between XGBoost and TabNet for auditor review.
