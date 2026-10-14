# Phase 2 Preprocessing & Feature Engineering Plan

**Project:** AI-Based Financial Fraud Detection using TabNet  
**Dataset:** `ai/data/raw/Bank_Transaction_Fraud_Detection.csv` (200,000 rows, 24 raw columns)  
**Target:** `Is_Fraud` (Binary classification: 0 = 189,912, 1 = 10,088; 5.044% fraud rate)  
**Status:** Approved for Implementation (Transition from Phase 1 EDA)  

---

## 1. Objectives

1. Eliminate all identified data leakage vectors (IDs, direct/masked PII) and deterministic redundancies.
2. Engineer domain-relevant banking interaction signals from transaction amount, account balance, and timestamps.
3. Prepare dual encoding pipelines:
   - **TabNet Pipeline:** Ordinal encoding with categorical cardinality tracking for PyTorch TabNet embedding layers.
   - **Tree/Baseline Pipeline:** One-hot / Target encoding for XGBoost and traditional ML baselines.
4. Enforce strict Stratified 5-Fold Cross-Validation isolation so that scalers and encoders are fitted exclusively on training folds, guaranteeing zero validation leakage.

---

## 2. Feature Pruning Strategy (Leakage & Redundancy Removal)

Based on verified Phase 1 findings, the following 9 columns will be discarded prior to feature engineering:

| Feature Name | Category | Reason for Removal |
| :--- | :--- | :--- |
| `Transaction_ID` | Identifier | 100% unique UUID (200,000 distinct values); arbitrary memorization risk. |
| `Customer_ID` | Identifier | 100% unique UUID (1 transaction per customer); cannot generalize to future customers. |
| `Merchant_ID` | Identifier | 100% unique UUID; zero repeat merchants in dataset. |
| `Customer_Name` | PII | Arbitrary synthetic names; high risk of spurious tree splits. |
| `Customer_Contact`| PII | Masked synthetic phone numbers (`+919XXXXXXXXXX`); arbitrary prefix grouping. |
| `Customer_Email` | PII | Masked synthetic email addresses derived from customer first names. |
| `Bank_Branch` | Redundant | 100% deterministic duplicate of `City + ' Branch'`. |
| `Transaction_Location` | Redundant | 100% deterministic duplicate of `City + ', ' + State`. |
| `Transaction_Currency` | Constant | Zero-variance constant feature (`INR` across all 200,000 rows). |

---

## 3. Feature Engineering Specifications

From the retained raw columns (`Transaction_Date`, `Transaction_Time`, `Transaction_Amount`, `Account_Balance`, `Device_Type`, `Transaction_Device`), the following new features will be derived:

### 3.1 Financial Ratios & Balance Indicators
1. **`Amount_to_Balance_Ratio` (Continuous):**
   $$\text{Amount\_to\_Balance\_Ratio} = \frac{\text{Transaction\_Amount}}{\text{Account\_Balance} + 1.0}$$
   Measures the financial burden of the transaction relative to available liquid funds.
2. **`Amount_Exceeds_Balance` (Binary):**
   $$\text{Amount\_Exceeds\_Balance} = \mathbb{I}(\text{Transaction\_Amount} > \text{Account\_Balance})$$
   Flags overdraw transactions (observed in 47.04% of raw records).

### 3.2 Temporal & Cyclical Features
1. **`Tx_Hour` (Integer: $0 \le h \le 23$):** Hour of day extracted from `Transaction_Time`.
2. **`Tx_DayOfWeek` (Integer: $0 \le dow \le 6$):** Day of week (0 = Monday, 6 = Sunday) extracted from `Transaction_Date`.
3. **`Tx_Day` (Integer: $1 \le d \le 31$):** Calendar day of month.
4. **`Is_Weekend` (Binary):** $\mathbb{I}(\text{Tx\_DayOfWeek} \ge 5)$.
5. **`Is_Night` (Binary):** $\mathbb{I}(\text{Tx\_Hour} \in \{23, 0, 1, 2, 3, 4, 5\})$.
6. **Cyclical Sine/Cosine Encodings:**
   $$\text{Hour\_Sin} = \sin\left(\frac{2\pi \cdot \text{Tx\_Hour}}{24}\right), \quad \text{Hour\_Cos} = \cos\left(\frac{2\pi \cdot \text{Tx\_Hour}}{24}\right)$$
   Preserves temporal continuity across the midnight boundary.

### 3.3 Contextual Device Match Indicator
1. **`Device_Mismatch_Flag` (Binary):**
   Flags physical incongruities (e.g. `Device_Type == 'ATM'` paired with `Transaction_Device == 'Web Browser'`).

---

## 4. Modeling Feature Taxonomy

After feature engineering, the model input space will consist of:

- **Continuous Numerical Features (5):**
  - `Age`
  - `Transaction_Amount`
  - `Account_Balance`
  - `Amount_to_Balance_Ratio`
  - `Hour_Sin`, `Hour_Cos`
- **Binary Indicator Features (3):**
  - `Amount_Exceeds_Balance`
  - `Is_Weekend`
  - `Is_Night`
- **Categorical Features (9):**
  - `Gender` (2 unique)
  - `State` (34 unique)
  - `City` (145 unique)
  - `Account_Type` (3 unique)
  - `Transaction_Type` (5 unique)
  - `Merchant_Category` (6 unique)
  - `Device_Type` (4 unique)
  - `Transaction_Device` (20 unique)
  - `Transaction_Description` (172 unique)

---

## 5. Scaling & Encoding Architecture

### 5.1 TabNet Transformer Architecture Requirements
PyTorch TabNet accepts categorical features through learned entity embeddings:
- Categorical features must be encoded as integer indices: $0, 1, \dots, K-1$.
- Unseen categories during validation/inference must map to an "unknown" index ($K$).
- For each categorical feature $j$, record:
  - `cat_idxs`: List of column indices corresponding to categorical features.
  - `cat_dims`: List of unique cardinalities for each categorical feature (including unknown token).
- Numerical features must be scaled using `RobustScaler` (centers by median and scales by IQR $[Q_1, Q_3]$).

### 5.2 Tree-Based Baseline Architecture (XGBoost / LightGBM)
- Categorical features encoded with `TargetEncoder` or frequency / ordinal encoding.
- Missing handling natively supported by XGBoost.

---

## 6. Leakage-Free Stratified 5-Fold Cross-Validation Protocol

```
Raw Data (200,000 rows)
   │
   ▼
[ StratifiedKFold (n_splits=5, shuffle=True, random_state=42) ]
   │
   ├── Fold 1: Train (160,000) ────► Fit Scaler & Encoders ────► Transform Val (40,000)
   ├── Fold 2: Train (160,000) ────► Fit Scaler & Encoders ────► Transform Val (40,000)
   ├── Fold 3: Train (160,000) ────► Fit Scaler & Encoders ────► Transform Val (40,000)
   ├── Fold 4: Train (160,000) ────► Fit Scaler & Encoders ────► Transform Val (40,000)
   └── Fold 5: Train (160,000) ────► Fit Scaler & Encoders ────► Transform Val (40,000)
```

### Strict Isolation Rules
1. **Never fit on the full dataset:** Preprocessing classes (`TabNetPreprocessor`, `BaselinePreprocessor`) implement standard `fit(X_train)`, `transform(X_train)`, and `transform(X_val)`.
2. **Never apply SMOTE on validation data:** If SMOTE or ADASYN is evaluated in Phase 3, it must be applied **only** to the training fold features and labels (`X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)`).
3. **Artifact Serialization:** For production scoring in FastAPI (Phase 5), the preprocessor fit on the full development set will be saved as `ai/artifacts/preprocessor.joblib`.

---

## 7. Verification & Testing Criteria

The Phase 2 implementation will be verified through automated pytest tests in `tests/ai/test_preprocessing.py` and `tests/ai/test_cv_pipeline.py`:
- [ ] Shape invariance: Transformed datasets contain zero NaNs or infs.
- [ ] Cat dimensions check: `cat_dims` matches the number of unique classes per feature.
- [ ] Fold isolation test: Scaler parameters (`center_`, `scale_`) computed on Fold 1 differ from Fold 2.
- [ ] Target preservation: Validation target distribution across all 5 folds has exactly 5.044% fraud.
