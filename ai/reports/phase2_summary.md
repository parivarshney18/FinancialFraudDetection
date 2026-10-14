# Phase 2: Preprocessing and Cross-Validation Pipeline Summary Report

**Project:** AI-Based Financial Fraud Detection using TabNet  
**Institution:** B.Tech Final Year Project — ABESEC Ghaziabad, CSE (Session 2026-27)  
**Execution Date:** 2026-10-14  
**Pipeline Author:** Autonomous Senior AI/ML Engineer  
**Status:** Completed & 100% Tested (15/15 tests passing)  

---

## 1. Executive Overview

Phase 2 established the core feature engineering, data transformation, and cross-validation foundation for the fraud detection project. The pipeline operationalizes all recommendations formulated during Phase 1 EDA and formally documented in [`docs/preprocessing_plan.md`](file:///C:/Projects/FinancialFraudDetection/financial-fraud-tabnet/docs/preprocessing_plan.md).

All transformers, encoders, and scalers strictly enforce **zero data leakage**: statistics are computed solely on training folds, and validation folds are transformed using training-fitted objects. PyTorch TabNet categorical compatibility is guaranteed through automatic tracking of categorical indices (`cat_idxs`) and cardinalities (`cat_dims`) with dedicated unknown-category index reservation.

---

## 2. Implemented Features & Pipeline Architecture

### 2.1 Feature Engineering ([`ai/src/preprocessing.py`](file:///C:/Projects/FinancialFraudDetection/financial-fraud-tabnet/ai/src/preprocessing.py))

1. **Leakage & Redundancy Pruning:**
   - Discarded 9 columns before feature transformation:
     - High-cardinality identifiers: `Transaction_ID`, `Customer_ID`, `Merchant_ID` (each 100% unique UUIDs).
     - Direct & masked PII: `Customer_Name`, `Customer_Contact`, `Customer_Email`.
     - Deterministic duplicates: `Bank_Branch` (`City + ' Branch'`), `Transaction_Location` (`City + ', ' + State`).
     - Constant zero-variance feature: `Transaction_Currency` (`INR`).
   - Automatically drops target `Is_Fraud` if accidentally passed in the feature set.
   - Operates on defensive copies (`df.copy()`) ensuring the raw input dataset is never modified in-place.

2. **Domain Banking Indicators:**
   - **`Amount_to_Balance_Ratio`:** Continuous measure of transaction weight relative to available liquidity:
     $$\text{Amount\_to\_Balance\_Ratio} = \frac{\text{Transaction\_Amount}}{\text{Account\_Balance} + 1.0}$$
   - **`Amount_Exceeds_Balance`:** Binary indicator for transactions exceeding account balance ($\mathbb{I}(\text{Amount} > \text{Balance})$).

3. **Temporal & Cyclical Features:**
   - Robust date and time parsing with automatic format inference and fallback (`%d-%m-%Y`, `%H:%M:%S`).
   - Resilient missing/corrupt timestamp handling: missing dates default to training median (`2025-01-15`) and missing times default to midday (`12:00:00`).
   - **`Tx_Day`:** Calendar day of the month ($1 \le d \le 31$).
   - **`Tx_DayOfWeek`:** Day of the week ($0 = \text{Monday}, \dots, 6 = \text{Sunday}$).
   - **`Is_Weekend`:** Binary flag for Saturday or Sunday ($\mathbb{I}(\text{Tx\_DayOfWeek} \ge 5)$).
   - **`Tx_Hour`:** Hour of the day ($0 \le h \le 23$).
   - **`Is_Night`:** Binary flag for high-risk nighttime window ($\mathbb{I}(h \ge 23 \lor h \le 5)$).
   - **`Hour_Sin` & `Hour_Cos`:** Continuous circular projection:
     $$\text{Hour\_Sin} = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad \text{Hour\_Cos} = \cos\left(\frac{2\pi \cdot h}{24}\right)$$

4. **Contextual Device Mismatch Indicator:**
   - **`Device_Mismatch_Flag`:** Vectorized verification checking physical compatibility between high-level `Device_Type` (`POS`, `ATM`, `Mobile`, `Desktop`) and granular `Transaction_Device` (20 types). Physical incongruities (e.g. `ATM` paired with `Web Browser`) are flagged as $1.0$.

### 2.2 Scaling & Encoding Strategy

- **`RobustScaler` on Numerical Features:**
  - Applied to all 13 numerical/continuous features (`Age`, `Transaction_Amount`, `Account_Balance`, `Amount_to_Balance_Ratio`, `Tx_Day`, `Tx_Hour`, `Tx_DayOfWeek`, `Hour_Sin`, `Hour_Cos`, `Amount_Exceeds_Balance`, `Is_Weekend`, `Is_Night`, `Device_Mismatch_Flag`).
  - Centers by median and scales by Interquartile Range (IQR $[Q_1, Q_3]$).
  - Robust to uniform extremes and outliers without distorting scale.
- **`OrdinalEncoder` for Categorical Features:**
  - Applied to 9 categorical columns (`Gender`, `State`, `City`, `Account_Type`, `Transaction_Type`, `Merchant_Category`, `Device_Type`, `Transaction_Device`, `Transaction_Description`).
  - Configured with `handle_unknown='use_encoded_value', unknown_value=-1`.
  - For column $j$ with $K_j$ categories in the training fold:
    - Known training categories mapped to integers in $[0, K_j - 1]$.
    - Any unseen/out-of-vocabulary category during validation or inference is dynamically mapped to index $K_j$.
    - TabNet embedding cardinality is explicitly recorded as `cat_dims[j] = K_j + 1`.
  - Categorical column indices are ordered first in the matrix (`cat_idxs = [0, 1, ..., 8]`), matching TabNet convention.

---

## 3. Stratified 5-Fold Cross-Validation Architecture ([`ai/src/cv_pipeline.py`](file:///C:/Projects/FinancialFraudDetection/financial-fraud-tabnet/ai/src/cv_pipeline.py))

```
Raw Transaction Dataset (200,000 records, 5.044% fraud)
  │
  ├── StratifiedKFold (n_splits=5, shuffle=True, random_state=42)
  │
  ├── Fold 0: Train (160,000) ────► Fit Preprocessor ────► Transform Val (40,000)
  ├── Fold 1: Train (160,000) ────► Fit Preprocessor ────► Transform Val (40,000)
  ├── Fold 2: Train (160,000) ────► Fit Preprocessor ────► Transform Val (40,000)
  ├── Fold 3: Train (160,000) ────► Fit Preprocessor ────► Transform Val (40,000)
  └── Fold 4: Train (160,000) ────► Fit Preprocessor ────► Transform Val (40,000)
```

### Verified Cross-Validation Properties
- **Exact Coverage:** Every single record in the 200,000-row dataset appears in a validation fold exactly once ($\bigcup \text{val\_folds} = \text{Dataset}$, $\sum |\text{val\_folds}| = 200,000$).
- **Strict Disjointness:** In every fold, $\text{train\_indices} \cap \text{val\_indices} = \emptyset$.
- **Class Balance Preservation:** Every fold's train and validation partitions preserve the global fraud rate ($5.044\% \pm 0.05\%$).
- **Independent Preprocessor per Fold:** A fresh `FraudPreprocessor` instance is initialized for each fold and fitted solely on `train_df`. Scaler parameters (`center_`, `scale_`) and encoder categories differ across folds, demonstrating fold independence.
- **No Resampling (SMOTE) Leakage:** No synthetic oversampling is applied to validation folds or before the CV split.
- **Reproducibility:** Splits and transformed matrices are 100% bit-exact across independent executions when using `random_state=42`.

---

## 4. Test Verification & Results

Automated tests in [`tests/ai/test_cv_pipeline.py`](file:///C:/Projects/FinancialFraudDetection/financial-fraud-tabnet/tests/ai/test_cv_pipeline.py) and [`tests/ai/test_eda.py`](file:///C:/Projects/FinancialFraudDetection/financial-fraud-tabnet/tests/ai/test_eda.py) verify the pipeline.

### Test Execution Command
```bash
& C:\Projects\FinancialFraudDetection\financial-fraud-tabnet\venv\Scripts\pytest.exe -v
```

### Test Results
```
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Projects\FinancialFraudDetection\financial-fraud-tabnet
configfile: pytest.ini
testpaths: tests

tests/ai/test_cv_pipeline.py::test_cv_splits_coverage_and_disjointness PASSED [  6%]
tests/ai/test_cv_pipeline.py::test_stratification_and_class_presence PASSED [ 13%]
tests/ai/test_cv_pipeline.py::test_target_exclusion_from_features PASSED [ 20%]
tests/ai/test_cv_pipeline.py::test_raw_data_immutability PASSED          [ 26%]
tests/ai/test_cv_pipeline.py::test_train_validation_isolation_and_no_leakage PASSED [ 33%]
tests/ai/test_cv_pipeline.py::test_unseen_categorical_value_handling PASSED [ 40%]
tests/ai/test_cv_pipeline.py::test_missing_and_invalid_datetime_handling PASSED [ 46%]
tests/ai/test_cv_pipeline.py::test_reproducibility_with_fixed_seed PASSED [ 53%]
tests/ai/test_cv_pipeline.py::test_cv_integrity_utility PASSED           [ 60%]
tests/ai/test_eda.py::test_raw_dataset_exists_and_unmodified PASSED      [ 66%]
tests/ai/test_eda.py::test_raw_dataset_target_distribution PASSED        [ 73%]
tests/ai/test_eda.py::test_eda_metrics_file_validity PASSED              [ 80%]
tests/ai/test_eda.py::test_all_eda_figures_exist PASSED                  [ 86%]
tests/ai/test_eda.py::test_eda_summary_report_and_data_dictionary PASSED [ 93%]
tests/ai/test_eda.py::test_data_loader_module PASSED                     [100%]

============================= 15 passed in 7.74s ==============================
```

---

## 5. Files Changed or Created

| File | Status | Description |
| :--- | :--- | :--- |
| [`ai/src/preprocessing.py`](file:///C:/Projects/FinancialFraudDetection/financial-fraud-tabnet/ai/src/preprocessing.py) | **Created / Implemented** | Full `FraudPreprocessor` with feature engineering, `RobustScaler`, `OrdinalEncoder`, and TabNet dimension metadata. |
| [`ai/src/cv_pipeline.py`](file:///C:/Projects/FinancialFraudDetection/financial-fraud-tabnet/ai/src/cv_pipeline.py) | **Created / Implemented** | `generate_cv_folds`, `get_cv_split_indices`, and `validate_cv_integrity` with strict fold isolation. |
| [`tests/ai/test_cv_pipeline.py`](file:///C:/Projects/FinancialFraudDetection/financial-fraud-tabnet/tests/ai/test_cv_pipeline.py) | **Created / Implemented** | 9 unit and integration tests covering stratification, immutability, unseen categories, and leakage prevention. |
| [`ai/reports/phase2_summary.md`](file:///C:/Projects/FinancialFraudDetection/financial-fraud-tabnet/ai/reports/phase2_summary.md) | **Created** | Comprehensive documentation of Phase 2 implementation and verification. |

---

## 6. Unresolved Issues & Blocker Audit

- **Unresolved issues:** None.
- **Blockers:** None.
- **Warnings:** Zero deprecation or runtime warnings in pytest.

---

## 7. Next Steps (Transition to Phase 3: Model Architecture & Training)

Phase 2 establishes all prerequisites for Phase 3. The exact upcoming tasks are:
1. **Implement Baseline Models** in `ai/src/models/`:
   - Logistic Regression / Random Forest baseline
   - XGBoost baseline (`ai/src/models/xgb_model.py`)
   - 1D-CNN baseline (`ai/src/models/cnn1d_model.py`)
   - LSTM baseline (`ai/src/models/lstm_model.py`)
2. **Implement PyTorch TabNet Architecture** in `ai/src/models/tabnet_model.py`:
   - Wire `cat_idxs` and `cat_dims` from `FraudPreprocessor` directly into TabNet embedding layers.
   - Configure attention transformers, sparsemax masks, and decision steps.
3. **Evaluation & Metric Logging** in `ai/src/evaluation.py`:
   - Compute PR-AUC, ROC-AUC, Recall@FPR $\le 1\%$, and F1-macro across all 5 folds.
   - Log results into `docs/experiment_log.md`.
