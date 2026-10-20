# Phase 6 Report: Alternative Dataset Evaluation for Financial Fraud Detection

**Project:** AI-Based Financial Fraud Detection using TabNet  
**Phase:** 6 (Dataset Suitability Assessment & Candidate Evaluation)  
**Date:** October 2026  
**Status:** Completed Analysis & Architectural Roadmap  

---

## 1. Executive Summary & Context

In **Phase 5**, an exhaustive empirical and information-theoretic assessment of the current dataset (`Bank_Transaction_Fraud_Detection.csv`) established that:
1. **Unconditioned Synthetic Generation:** The target variable `Is_Fraud` was assigned via an independent random Bernoulli process ($\pi_1 \approx 0.05044$) completely decoupled from transactional attributes ($I(X; Y) < 0.0045$, mean $0.00090$; two-sample KS tests between fraud and non-fraud yield $p > 0.30$).
2. **Absence of Customer Sequences:** All 200,000 rows contain **100% unique `Customer_ID`s**. No customer has more than one transaction in the dataset. Consequently, personal behavioral baselines, rolling velocity checks (1h, 6h, 24h), inter-transaction intervals ($\Delta t$), device switching, and geospatial travel velocity are **structurally impossible** to compute.
3. **Mathematical Ceiling:** Cross-validation benchmarks across Logistic Regression, Random Forest, XGBoost, and TabNet—as well as experiments with SMOTENC oversampling and macro-behavioral engineering—consistently hover at $\text{PR-AUC} \approx 0.0504$ and $\text{ROC-AUC} \approx 0.5000$.

To transition this capstone research project from a synthetic sandbox to an academically defensible, high-discrimination fraud detection system, we conducted a rigorous evaluation of the two premier candidate benchmarks in financial fraud research:
- **IEEE-CIS Fraud Detection Dataset** (Vesta Corporation / IEEE Computational Intelligence Society)
- **PaySim Synthetic Financial Datasets For Fraud Detection** (NTNU / Edgar Lopez-Rojas et al.)

---

## 2. Comprehensive Dataset Comparison Matrix

| Evaluation Dimension | Current Dataset (`Bank_Transaction_Fraud_Detection`) | PaySim Simulator Dataset | IEEE-CIS Fraud Detection Dataset |
| :--- | :--- | :--- | :--- |
| **Data Nature & Origin** | Synthetic simulation for fictional "LOL Bank Pvt. Ltd." (Sagar Maru on Kaggle) | Multi-agent synthetic financial simulator based on mobile money logs (NTNU / E. A. Lopez-Rojas) | **Real-world commercial e-commerce transactions** provided by Vesta Corporation |
| **Total Rows & Tables** | 200,000 rows × 1 table | 6,362,620 rows × 1 table | **590,540 training rows** across 2 linked tables (`train_transaction.csv` + `train_identity.csv`) |
| **Total Features** | 24 columns (9 retained after dropping IDs/PII/redundancies) | 11 columns (only 2 categorical, 8 continuous/discrete) | **434 features** (card info, email domains, address, device info, network attributes, masked V-features) |
| **Target Variable & Base Rate** | `Is_Fraud` (5.044%, 10,088 positive instances) | `isFraud` (**0.129%**, 8,213 positive instances; extreme imbalance 774:1) | `isFraud` (**3.50%**, 20,663 positive instances; realistic industry imbalance 28:1) |
| **Entity Repetition & Customer Sequences** | **0% repeat entities** (200,000 unique `Customer_ID`s in 200,000 rows; 1 transaction per customer) | `nameDest` has repeats (~2.7M unique in 6.36M rows); `nameOrig` is **99.8% single-use** (6,353,307 unique IDs) | **Extensive entity sequences:** Customers share credit card hashes (`card1`-`card6`), address matches (`addr1`), and emails (`P_emaildomain`), enabling true multi-transaction tracking |
| **Temporal Granularity** | `Transaction_Date` (31 days) + `Transaction_Time` (HH:MM:SS) | `step` (integer hours from 1 to 744, covering 30 days) | `TransactionDT` (**continuous seconds** from reference timestamp covering 182 days / 6 months) |
| **Device & Identity Telemetry** | Generic labels (`POS`, `ATM`, `Mobile`, `Desktop`); no hardware identifiers | **None** (zero device, IP, or browser information) | **Extensive:** `DeviceType`, `DeviceInfo` (OS, hardware model), browser signatures, screen resolution (`id_01`-`id_38`) |
| **License & Access** | Kaggle Custom / Community Data License | Open Data (**CC0: Public Domain** / ODbL) | **Kaggle Competition License / Non-Commercial Research & Educational Use** |
| **CPU Training Feasibility (16 GB RAM)** | High (5-fold CV completes in 7–15 minutes) | Moderate-Low (6.36M rows require aggressive subsampling to fit in memory) | **High with curated feature subset** (~40–50 core features on 100k–200k rows; full 435 cols requires batching) |
| **Compatibility with TabNet & XAI** | Implemented, but attention masks diffuse due to zero mutual information | Low (only 2 categorical features; deterministic rules make deep attention redundant) | **Outstanding:** TabNet was originally designed for complex, high-dimensional tabular datasets with mixed embeddings |
| **Commercial & Academic Defensibility** | Low (synthetic labels with zero predictive correlation) | Moderate (good for agent-based cash-out modeling, but simplistic rules) | **Exceptional** (industry-standard benchmark for IEEE publications and master's/doctoral theses) |

---

## 3. Deep-Dive Candidate Assessment

### 3.1 PaySim Simulator Dataset
**Strengths:**
- **Open License (CC0):** Completely unrestricted public domain access; no competition or institutional sign-in barriers.
- **Compact Schema (11 columns):** Extremely simple tabular format with standard columns (`type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`).
- **Clear Fraud Patterns:** Fraud is simulated specifically around account draining via unauthorized `TRANSFER` followed by `CASH_OUT`.

**Critical Limitations for this Project:**
1. **Lack of Tabular Heterogeneity:** PaySim has only **one meaningful categorical feature** (`type` with 5 levels: `CASH-IN`, `CASH-OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`). The remaining columns are balances and amounts. TabNet's primary architectural innovation—**sequential sparse attention across diverse categorical embeddings and continuous variables**—has almost no meaningful role when there is only one categorical column.
2. **Deterministic Simulation Rules:** In PaySim, fraudulent transfers almost always set `newbalanceOrig == 0` and drain `oldbalanceOrg == amount`. Because these rules were hand-coded into the simulator, simple single-split decision trees achieve $\text{ROC-AUC} > 0.999$, making sophisticated neural networks like TabNet unnecessary.
3. **Absence of Device & Network Telemetry:** PaySim contains no device types, IP addresses, geographic locations, or browser information. It cannot support device mismatch or identity explainability.

### 3.2 IEEE-CIS Fraud Detection Dataset
**Strengths:**
- **Real-World Commercial Telemetry:** Provided by Vesta Corporation, a leading e-commerce payment protection company. Transaction patterns reflect genuine human card-not-present (CNP) purchases and adversarial cybercrime.
- **Rich Categorical & Embedding Space:** Features include multiple card dimensions (`card1`–`card6`), product types (`ProductCD`), billing/shipping regions (`addr1`, `addr2`), purchaser and recipient email domains (`P_emaildomain`, `R_emaildomain`), device operating systems, and browsers. This provides the exact environment where **TabNet’s categorical embeddings and attentive transformers excel**.
- **Authentic Behavioral Trajectories:** Individual user entities can be reliably reconstructed using composite card-address-email keys (`UID = card1 + addr1 + P_emaildomain`). Users make repeated purchases across hours, days, and months, enabling true customer velocity calculations (1h, 24h, 168h counts), spending ratio deviations, and time-delta ($\Delta t$) anomalies.
- **Genuine Device & Identity Table:** The linked `train_identity.csv` captures physical hardware signatures (`DeviceInfo`, `DeviceType`, `id_30` through `id_38`), enabling real device-change and IP-reputation modeling.
- **Research Prestige:** IEEE-CIS is the recognized standard in machine learning fraud literature. Results published on this benchmark carry definitive academic credibility.

**Limitations & Engineering Considerations:**
1. **High Dimensionality (434 features):** The full uncurated dataset contains 434 columns (including 339 Vesta masked features `V1`–`V339`). Training the full dataset across 5 folds would exceed our 16 GB CPU memory budget.
   * *Mitigation:* Apply a **principled feature selection strategy**: extract the core transaction features (amount, card, address, email, product, device, temporal) and top V-features (e.g., 40–50 salient features total), or utilize a stratified sample of 100,000–150,000 records for cross-validation.
2. **Masked Variable Names (`V1`-`V339`, `id_01`-`id_38`):** Some engineered features are anonymized for consumer privacy.
   * *Mitigation:* Focus the frontend dashboard and explainability narratives on the interpretable semantic features (`TransactionAmt`, `card4` [network], `card6` [debit/credit], `P_emaildomain`, `addr1`, `DeviceType`, `DeviceInfo`).

---

## 4. Architectural Impact Analysis & System Modularity

The current project architecture was designed with strong separation of concerns across the offline AI pipeline, FastAPI microservice, and Streamlit dashboard. 

Adopting the recommended alternative dataset requires targeted updates at specific layers without disrupting the underlying codebase:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   SYSTEM ARCHITECTURE IMPACT MATRIX                    │
└────────────────────────────────────────────────────────────────────────┘

1. AI / ML PIPELINE
   ├── ai/src/config.py               --> Update column schemas & dataset paths
   ├── ai/src/data_loader.py          --> Add IEEECISDataLoader (merges transaction + identity)
   ├── ai/src/preprocessing.py        --> Implement IEEECISPreprocessor (card/email/temporal encoding)
   ├── ai/src/cv_pipeline.py          --> 100% REUSABLE (Already generic, leak-free StratifiedKFold)
   ├── ai/src/models/tabnet_model.py  --> 100% REUSABLE (Dynamic categorical embedding generator)
   ├── ai/src/models/baseline_models  --> 100% REUSABLE (Logistic Regression, Random Forest)
   ├── ai/src/models/xgb_model.py     --> 100% REUSABLE (Hist-gradient boosting)
   ├── ai/src/evaluation.py           --> 100% REUSABLE (PR-AUC, ROC-AUC, Recall@FPR<=1%)
   └── ai/src/explain.py              --> 100% REUSABLE (Sparsemax attention mask extraction)

2. FASTAPI MICROSERVICE (backend/)
   ├── backend/app/schemas/schemas.py --> Add IEEE-CIS transaction payload schema (with fallback)
   ├── backend/app/ml/artifact_loader --> Loads newly trained IEEE-CIS TabNet artifacts
   └── backend/app/routers/           --> Endpoints remain identical (/predict, /explain, /benchmarks)

3. STREAMLIT AUDIT DASHBOARD (frontend/)
   ├── frontend/pages/1_Score_...py   --> Update transaction input form fields to match new schema
   ├── frontend/components/           --> Gauge, Heatmap, Contributors charts remain 100% identical
   └── frontend/services/api_client   --> Unaltered REST client
```

### Key Architectural Finding:
**Over 80% of the existing codebase is completely dataset-agnostic:**
- The cross-validation engine ([`ai/src/cv_pipeline.py`](file:///C:/Projects/FinancialFraudDetection/financial-fraud-tabnet/ai/src/cv_pipeline.py)), metric evaluation suite ([`ai/src/evaluation.py`](file:///C:/Projects/FinancialFraudDetection/financial-fraud-tabnet/ai/src/evaluation.py)), TabNet attentive wrapper ([`ai/src/models/tabnet_model.py`](file:///C:/Projects/FinancialFraudDetection/financial-fraud-tabnet/ai/src/models/tabnet_model.py)), baseline wrappers ([`ai/src/models/baseline_models.py`](file:///C:/Projects/FinancialFraudDetection/financial-fraud-tabnet/ai/src/models/baseline_models.py), [`ai/src/models/xgb_model.py`](file:///C:/Projects/FinancialFraudDetection/financial-fraud-tabnet/ai/src/models/xgb_model.py)), and TabNet explanation engine ([`ai/src/explain.py`](file:///C:/Projects/FinancialFraudDetection/financial-fraud-tabnet/ai/src/explain.py)) require **zero modification**.
- Only the **data loader**, **preprocessor**, and **UI input form schemas** need dataset-specific adapter updates.

---

## 5. Formal Recommendation

### Primary Recommendation: Adopt IEEE-CIS Fraud Detection Dataset (Curated 45-Feature Subset)

**Core Rationale:**
1. **Solves the Root Cause Identified in Phase 5:** Unlike PaySim or the current dataset, IEEE-CIS contains **genuine real-world credit card transaction sequences** with repeat card usage across time. This makes genuine behavioral feature engineering (spending deviations, rolling velocity, time-delta) analytically possible.
2. **Optimal for TabNet's Strengths:** TabNet was specifically developed for complex heterogeneous tabular data with high-cardinality categorical features (card networks, issuer banks, email domains, device models). PaySim's single categorical feature provides almost no utility for TabNet embeddings.
3. **Academic & Commercial Credibility:** B.Tech capstone reviewers and journal evaluators recognize IEEE-CIS as a rigorous, authentic fraud benchmark. Achieving genuine non-trivial discrimination (PR-AUC $> 0.50$, ROC-AUC $> 0.85$) on IEEE-CIS provides true research merit.
4. **Hardware Feasibility:** By curating a high-signal subset of **40–50 core features** (card metadata, transaction amounts, time-deltas, email domains, device signatures, and key V-features) on 100,000–150,000 transactions, the dataset will train comfortably on our 16 GB CPU within 10–15 minutes.

---

## 6. Phased Implementation Roadmap for Dataset Adoption

To adopt the IEEE-CIS dataset safely without deleting or overwriting any of the previous experimental milestones (Phases 1–5), the following exact sequence will be executed in subsequent phases:

### Phase 7: Dataset Acquisition & Exploratory Profiling
1. Download official IEEE-CIS files (`train_transaction.csv`, `train_identity.csv`) from the official Kaggle repository via the Kaggle CLI / official API into a dedicated directory: `ai/data/ieee_cis/raw/`.
2. Author `ai/src/data_loader_ieee.py` to merge transaction and identity tables via `TransactionID`.
3. Conduct EDA on target distribution (`isFraud`), card clustering, temporal spans, and missingness.
4. Curate an optimized 45-feature subset preserving maximum mutual information while fitting CPU memory constraints.
5. Create `ai/reports/ieee_eda_summary.md`.

### Phase 8: IEEE-CIS Preprocessing & Behavioral Engineering
1. Implement `ai/src/preprocessing_ieee.py` with:
   - Numerical scaling (`RobustScaler`) for transaction amounts, time-deltas, and balance ratios.
   - Categorical ordinal encoding with cardinality tracking for TabNet embeddings (`card4`, `card6`, `ProductCD`, `P_emaildomain`, `DeviceType`, `DeviceInfo`).
   - Entity-level behavioral features: rolling card transaction count (1h, 24h, 7d), spending deviation relative to card mean, and time elapsed ($\Delta t$).
2. Author unit tests `tests/ai/test_preprocessing_ieee.py` verifying fold isolation and zero leakage.

### Phase 9: Model Benchmarking & TabNet Training on IEEE-CIS
1. Run Stratified 5-Fold Cross-Validation across Logistic Regression, Random Forest, XGBoost, and TabNet.
2. Evaluate and persist PR-AUC, ROC-AUC, Recall@FPR$\le$1%, Precision, Recall, and F1 under `ai/results/ieee_*.json`.
3. Extract and visualize TabNet sequential attention masks ($M_b$) showing salient card and device feature choices.
4. Save trained production model weights to `ai/artifacts/tabnet_ieee_fraud_model.zip`.
5. Create `ai/reports/ieee_training_summary.md`.

### Phase 10: Service Adaptation & Final Delivery
1. Update `backend/app/schemas/schemas.py` to support IEEE-CIS transaction fields alongside existing schemas.
2. Update `frontend/pages/1_Score_Transaction.py` to allow compliance officers to inspect and score IEEE-CIS transactions with live TabNet attention masks.
3. Update project `README.md` documenting the verified transition and real-world results.
