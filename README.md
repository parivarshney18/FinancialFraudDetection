# AI-Based Financial Fraud Detection using TabNet

> **An Explainable-AI (XAI) Approach for Payment Fraud Detection**  
> **B.Tech Final Year Capstone Project — Session 2026-27**  
> **Department of Computer Science & Engineering, ABES Engineering College (ABESEC), Ghaziabad**  

[![Python Version](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20PyTorch--TabNet-EE4C2C.svg)](https://github.com/dreamquark-ai/tabnet)
[![Gradient Boosting](https://img.shields.io/badge/Model-XGBoost%203.4.1-orange.svg)](https://xgboost.readthedocs.io/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Pytest Status](https://img.shields.io/badge/Tests-135%20Passed%20(100%25)-brightgreen.svg)](tests/)
[![Status](https://img.shields.io/badge/Project%20Status-Phase%2010%20Verified-success.svg)]()

---

## Project Metadata

- **Institution:** ABES Engineering College (ABESEC), Ghaziabad
- **Department:** Computer Science & Engineering (CSE)
- **Academic Session:** 2026-2027
- **Project Type:** B.Tech Final Year Capstone Project
- **Project Guide:** Ms. Shipra Gautam
- **Team Members:**
  - **Pari Varshney**
  - **Neha Awasthi**
  - **Nandini Sharma**

---

## 1. Project Overview & Problem Statement

### 1.1 Problem Statement
The exponential growth of digital commerce and card-not-present (CNP) transactions has driven unprecedented transaction volumes alongside increasingly sophisticated fraud vectors. Financial fraud detection systems face four fundamental engineering and analytical challenges:
1. **Extreme Class Imbalance:** Fraudulent transactions constitute a small fraction of volume (~3.5% in real datasets), rendering naive accuracy misleading.
2. **Tabular Heterogeneity & Dense Velocity Signals:** Transaction logs combine continuous financial balances, cardholder velocity counters, elapsed timedeltas, categorical product channels, and hardware device signatures.
3. **Temporal Concept Drift:** Adversarial fraud patterns evolve rapidly over time, degrading models evaluated only on stationary cross-validation splits.
4. **The Regulatory Explainability Mandate:** Financial regulations (e.g., RBI guidelines, GDPR Art. 22) mandate auditable justifications when transactions are flagged or accounts frozen. Ensemble tree models lack native instance-level decision paths, requiring expensive post-hoc approximations.

### 1.2 Proposed Solution
This project implements an end-to-end, interpretable financial fraud detection system utilizing **TabNet** (*Arik & Pfister, Google Cloud AI / AAAI 2021*) benchmarked against **XGBoost**, **Random Forest**, and **Logistic Regression**. 

Key capabilities:
- **Real-World Ground Truth:** Benchmarked on the real-world **IEEE-CIS Fraud Detection dataset** (Vesta Corporation) across 590,540 transactions.
- **Intrinsic Interpretability:** TabNet generates instance-level feature importance masks via sequential Sparsemax attention without post-hoc perturbation.
- **Dual Evaluation Protocol:** Evaluates both stationary capacity via **Stratified 5-Fold Cross-Validation** and temporal generalization via **Forward Chronological 80/20 Holdout**.
- **Production Architecture:** Complemented by a high-throughput **FastAPI** inference microservice and an interactive **Streamlit** forensic dashboard for human compliance officers.
- **Legacy Compatibility:** Preserves full backward compatibility with the historical Phase 1–5 synthetic dataset baseline.

---

## 2. System Architecture

```
FinancialFraudDetection/financial-fraud-tabnet/
├── ai/                      # Offline ML Pipeline & Modeling Engine
│   ├── artifacts/           # Serialized preprocessors, encoders, and trained weights
│   │   ├── ieee_tabnet.zip                  # Production TabNet model weights
│   │   ├── ieee_xgboost.json                # Production XGBoost booster
│   │   ├── ieee_random_forest.joblib        # Scikit-learn Random Forest ensemble
│   │   ├── ieee_logistic_regression.joblib  # Scikit-learn Logistic Regression baseline
│   │   ├── ieee_preprocessor.joblib         # Fitted IEEEFraudPreprocessor (46 features)
│   │   ├── tabnet_fraud_model.zip           # Legacy synthetic TabNet model
│   │   └── preprocessor.joblib              # Legacy synthetic preprocessor
│   ├── data/
│   │   ├── ieee_cis/raw/    # IEEE-CIS transaction & identity CSVs (excluded from Git)
│   │   └── raw/             # Synthetic transaction CSV (Phase 1)
│   ├── reports/
│   │   ├── figures/         # Diagnostic EDA and evaluation figures (01 to 15)
│   │   ├── ieee_eda_summary.md              # Phase 7 full dataset profiling
│   │   ├── phase8_summary.md                # Phase 8 feature engineering specification
│   │   ├── phase9_training_summary.md       # Phase 9 benchmarking & training report
│   │   └── phase10_stage5_verification.md   # Final Phase 10 verification audit
│   ├── results/             # Benchmark JSONs and CSV comparisons
│   └── src/                 # Preprocessing, training, and explainability modules
│       ├── preprocessing_ieee.py            # Leak-free IEEE-CIS preprocessing (46 features)
│       ├── benchmark_ieee.py                # Dual protocol CV and chronological runner
│       ├── explain_ieee.py                  # TabNet Sparsemax attention extractor
│       └── models/                          # Model wrappers (TabNet, XGBoost, baselines)
├── backend/                 # FastAPI Real-Time Inference Microservice
│   ├── app/
│   │   ├── core/            # Configuration and security settings
│   │   ├── ml/              # Singleton artifact registry (artifact_loader.py)
│   │   ├── routers/         # REST API endpoints (predictions, benchmarks, models)
│   │   ├── schemas/         # Pydantic input/output schemas
│   │   └── services/        # Prediction, explanation, and registry services
│   └── main.py              # Application entrypoint
├── frontend/                # Streamlit Forensic Auditor Dashboard
│   ├── app.py               # Main navigation portal
│   ├── components/          # Reusable Plotly risk gauges, charts, heatmaps
│   ├── pages/               # 6-page interactive audit suite
│   │   ├── 1_Score_Transaction.py           # Real-time scoring & TabNet explainability
│   │   ├── 2_Review_Queue.py                # Compliance review & audit queue
│   │   ├── 3_Prediction_Detail.py           # Deep-dive forensic transaction inspector
│   │   ├── 4_Model_Comparison.py            # Phase 9 benchmark comparison & curves
│   │   ├── 5_EDA_Insights.py                # IEEE-CIS dataset profiling & topology
│   │   └── 6_Admin.py                       # System telemetry & model status
│   └── services/            # API client wrapper
├── tests/                   # Automated Pytest Suite (135 tests passing)
│   ├── ai/                  # AI preprocessing, CV, models, and explainability tests
│   ├── backend/             # FastAPI endpoint and artifact tests
│   └── frontend/            # Streamlit component and client tests
├── pytest.ini               # Pytest configuration
├── requirements.txt         # Pinned package dependencies
└── README.md                # Project documentation
```

---

## 3. Dataset Specifications

### 3.1 IEEE-CIS Fraud Detection Dataset (Production Benchmark)
Provided by **Vesta Corporation** and the **IEEE Computational Intelligence Society**:
- **Source Files:** `train_transaction.csv` (651.69 MB) and `train_identity.csv` (25.30 MB).
- **Total Transactions:** **590,540** rows.
- **Relational Structure:** Left-join on `TransactionID` (100% unique primary key, 0 duplicates).
- **Identity Telemetry Coverage:** 144,233 transactions (24.42%) have digital identity capture (`DeviceType`, `DeviceInfo`); 446,307 (75.58%) are guest transactions.
- **Ground-Truth Class Breakdown:**
  - **Legitimate (`isFraud = 0`):** 569,877 (96.501%)
  - **Fraudulent (`isFraud = 1`):** 20,663 (3.499%)
  - **Imbalance Ratio:** **27.58 : 1**

### 3.2 Curated 46-Feature Research Architecture
To prevent CPU memory exhaustion and enable feasible TabNet attention computation, Phase 8 established a **46-feature research subset** (11 Categorical + 35 Numerical) achieving an **89.4% dimensionality reduction** from the 434 raw columns:
1. **Categorical Entity & Channel (7):** `ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `DeviceType`, `DeviceInfo`.
2. **Verification Match Flags (4):** `M1`, `M4`, `M5`, `M6`.
3. **Core Financial & Spatial (4):** `TransactionAmt`, `card1`, `addr1`, `addr2`.
4. **Vesta Velocity Counters (7):** `C1`, `C2`, `C5`, `C6`, `C11`, `C13`, `C14` (**0.00% missing values!** Uninterrupted multi-window counters).
5. **Vesta Timedelta Telemetry (3):** `D1`, `D2`, `D15` (Days elapsed since prior card/account activity).
6. **Top Salient V-Features (15):** `V12`, `V13`, `V35`, `V36`, `V75`, `V76`, `V90`, `V91`, `V126`, `V127`, `V128`, `V257`, `V258`, `V282`, `V283`.
7. **Engineered Cyclical Temporal (2):** `Hour_Sin`, `Hour_Cos` ($\sin/\cos(2\pi \cdot h / 24)$ diurnal projection).
8. **Engineered Monetary Deviation (2):** `Log_TransactionAmt`, `Amt_to_Card1_Mean_Ratio` (Log1p scaling and ratio to historical mean).
9. **Engineered Cardholder Recency (2):** `card1_tx_count_prior`, `Time_Since_Prev_Card1_Tx` (Expanding prior transaction count and elapsed seconds $\Delta t$).

### 3.3 Historical Synthetic Dataset Baseline (Phase 1–5)
- **Source:** 200,000 synthetic banking records across 24 columns (5.044% fraud).
- **Documented Limitation:** In Phase 5, information-theoretic profiling proved that labels in this dataset were generated via an independent Bernoulli draw decoupled from transaction amount or customer history ($I(X; Y) < 0.0045$), which motivated acquiring the genuine IEEE-CIS dataset.

---

## 4. Multi-Model Benchmark Results (Phase 9)

All models were evaluated across both **Protocol A (Stratified 5-Fold Cross-Validation)** and **Protocol B (Chronological 80/20 Holdout Split)** on representative samples with exact 3.499% fraud prevalence:

| Model Architecture | Protocol | PR-AUC (Average Precision) | ROC-AUC | Recall @ FPR $\le$ 1% | F1-Score (Fraud Class) | Decision Threshold | Fit Time (s) | Latency (ms/1k) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | Stratified CV<br>Chronological | **0.4785 ± 0.0301**<br>**0.3554** | **0.8827 ± 0.0153**<br>**0.8838** | **0.3997 ± 0.0409**<br>**0.3377** | **0.4764 ± 0.0316**<br>**0.3692** | 0.81 (CV mean)<br>**0.79 (Holdout)** | **0.92s** | **1.2 ms** |
| **Random Forest** | Stratified CV<br>Chronological | 0.4720 ± 0.0202<br>0.3030 | 0.8750 ± 0.0118<br>0.8703 | 0.3951 ± 0.0314<br>0.3158 | 0.4761 ± 0.0317<br>0.3440 | 0.57 (CV mean)<br>**0.57 (Holdout)** | 2.81s | 6.0 ms |
| **TabNet (Deep Tabular)**| Stratified CV<br>Chronological | 0.2928 ± 0.0392<br>0.2554 | 0.8187 ± 0.0232<br>0.8447 | 0.2327 ± 0.0524<br>0.2018 | 0.3082 ± 0.0483<br>0.2624 | 0.84 (CV mean)<br>**0.89 (Holdout)** | 55.42s | 32.6 ms |
| **Logistic Regression** | Stratified CV<br>Chronological | 0.2684 ± 0.0198<br>0.1287 | 0.8074 ± 0.0174<br>0.6948 | 0.2327 ± 0.0215<br>0.1535 | 0.3109 ± 0.0315<br>0.1719 | 0.82 (CV mean)<br>**0.86 (Holdout)** | 4.21s | 0.1 ms |

### Key Analytical Takeaways:
1. **Cross-Validation vs Chronological Drop:** PR-AUC drops from ~0.48 in CV to ~0.35 in chronological holdout, reflecting real-world adversarial concept drift and proving the necessity of chronological testing.
2. **XGBoost Performance:** XGBoost delivers the highest PR-AUC (0.3554) and highest Recall at 1% bounded FPR (33.77%), with ultra-fast training (0.92s).
3. **TabNet Regulatory Compliance:** While TabNet scores PR-AUC 0.2554, it natively provides **Sparsemax attention masks** at each sequential decision step for human auditability without external SHAP sampling.
4. **Decision Threshold Calibration:** Thresholds were calibrated strictly on training partitions via $F_1$ optimization (`tune_threshold_on_train`, search range $[0.05, 0.90]$) with zero validation leakage.

---

## 5. Explainability Architecture (XAI)

### 5.1 TabNet Attentive Transformer
TabNet selects salient features sequentially using an Attentive Transformer equipped with **Sparsemax** activation:
$$\mathbf{M}[b] = \text{Sparsemax}(\mathbf{P}[b-1] \cdot h_a(\mathbf{a}[b-1]))$$
- At each decision step $b \in \{0, 1, 2\}$, the network assigns normalized attention weights across the 46 features.
- The global feature importance is the cumulative attention across all steps: $M_{\text{explain}} = \sum_{b=0}^{B-1} \mathbf{M}[b]$.

### 5.2 Regulatory Compliance & Non-Causal Notice
> **[Important Regulatory Notice]:** Feature attributions and attention weights indicate internal neural network focus and statistical correlation with historical fraud patterns. They **do not constitute legal liability or deterministic causal proof** that a transaction was fraudulent.

### 5.3 XGBoost Interpretability Limitation
In the current runtime environment, instance-level TreeSHAP explanations for XGBoost are unavailable because the optional `shap` C-extension is not installed. XGBoost predictions and calibrated threshold classifications remain fully active, while the system transparently documents this limitation in API responses and frontend notices.

---

## 6. REST API Reference

The FastAPI backend microservice provides standardized JSON endpoints under `/api`.

### 6.1 Core Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check returning service status and ML artifact readiness. |
| `GET` | `/api/models/ieee` | Returns active IEEE-CIS models, thresholds, and preprocessor metadata. |
| `GET` | `/api/benchmarks/ieee` | Returns unified Phase 9 benchmark comparison table. |
| `GET` | `/api/benchmarks/ieee/chronological` | Returns detailed chronological holdout evaluation metrics. |
| `GET` | `/api/benchmarks/ieee/metadata` | Returns experiment run configuration and sample sizes. |
| `POST` | `/api/predictions/score-ieee` | Scores a single transaction using the selected model. |
| `POST` | `/api/predictions/score-and-explain-ieee` | Scores transaction and returns TabNet Sparsemax attention attribution. |
| `POST` | `/api/predictions/batch-ieee` | High-throughput batch scoring on multiple transaction vectors. |

### 6.2 Request Example: Single Transaction Scoring
`POST /api/predictions/score-ieee?model=tabnet`
```json
{
  "transaction_id": "TX-PROD-001",
  "transaction_dt": 86400.0,
  "transaction_amt": 150.0,
  "product_cd": "W",
  "card1": 13500,
  "card4": "visa",
  "card6": "debit",
  "p_emaildomain": "gmail.com",
  "r_emaildomain": "gmail.com",
  "device_type": "mobile",
  "device_info": "iOS Device",
  "m1": "T",
  "m4": "M0",
  "m5": "F",
  "m6": "T",
  "addr1": 315.0,
  "addr2": 87.0,
  "c1": 1.0,
  "c2": 1.0,
  "c5": 0.0,
  "c6": 1.0,
  "c11": 1.0,
  "c13": 1.0,
  "c14": 1.0,
  "d1": 14.0,
  "d2": 14.0,
  "d15": 14.0,
  "v12": 1.0,
  "v13": 1.0,
  "v35": 1.0,
  "v36": 1.0,
  "v75": 1.0,
  "v76": 1.0,
  "v90": 0.0,
  "v91": 0.0,
  "v126": 0.0,
  "v127": 0.0,
  "v128": 0.0,
  "v257": 1.0,
  "v258": 1.0,
  "v282": 1.0,
  "v283": 1.0
}
```

### 6.3 Response Example
```json
{
  "transaction_id": "TX-PROD-001",
  "model_name": "tabnet",
  "fraud_probability": 0.6463,
  "is_fraud": 0,
  "decision_threshold": 0.89,
  "risk_tier": "HIGH",
  "latency_ms": 38.45,
  "metadata": {
    "dataset": "IEEE-CIS Fraud Detection",
    "n_processed_features": 46,
    "decision_rule": "is_fraud = 1 if prob >= 0.89 else 0",
    "calibrated_threshold": 0.89
  },
  "created_at": "2026-10-20T15:13:00.000Z"
}
```

---

## 7. Installation & Quickstart

### 7.1 Prerequisites
- Python 3.10 to 3.13 (Tested on Python 3.13.14 on Windows 11 / Linux)
- Git

### 7.2 Setup Environment
```bash
# Clone repository
git clone https://github.com/<your-username>/financial-fraud-tabnet.git
cd financial-fraud-tabnet

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 7.3 Run Full Automated Test Suite
```bash
pytest -v
```
*(Executes all 135 tests across AI algorithms, FastAPI endpoints, and Streamlit components).*

### 7.4 Start FastAPI Microservice
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive docs:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`
- **Health Check:** `http://127.0.0.1:8000/health`

### 7.5 Start Streamlit Forensic Dashboard
In a separate terminal:
```bash
streamlit run frontend/app.py
```
Open your browser at `http://localhost:8501`.

---

## 8. Verified Operational Limitations

1. **Hardware & Execution Environment:** All models were trained and benchmarked on commodity CPU hardware (16 GB RAM / 8-Core CPU). GPU training was not utilized.
2. **Benchmark Sample Scope:** Cross-validation experiments were conducted on a curated, leak-free 50,000-row representative slice (preserving the exact 3.499% fraud prevalence) rather than the uncompressed 1.94 GB full dataset, ensuring feasible training times (~8–10 minutes on CPU).
3. **Cardholder Proxy Limitations:** In the absence of real customer IDs (withheld for privacy), `card1` serves as an issuing bank/card cluster proxy. First-time transactions lack prior entity counts and fall back to neutral ratios ($1.0$).
4. **Academic Prototype Status:** This codebase represents an academic research project and engineering proof-of-concept. It has not undergone formal banking regulatory certification or PCI-DSS production audits.

---

## 9. References & Acknowledgments

### Academic Literature
1. Sercan O. Arik and Tomas Pfister. *"TabNet: Attentive Interpretable Tabular Learning."* AAAI Conference on Human Computation and Crowdsourcing (AAAI 2021). [arXiv:1908.07442](https://arxiv.org/abs/1908.07442).
2. Tianqi Chen and Carlos Guestrin. *"XGBoost: A Scalable Tree Boosting System."* ACM SIGKDD 2016.
3. IEEE Computational Intelligence Society & Vesta Corporation. *"IEEE-CIS Fraud Detection Benchmark Dataset."* Kaggle (2019).
4. Reserve Bank of India (RBI). *"Master Direction on Digital Payment Security Controls and Fraud Risk Management."*

### Acknowledgments
We express our gratitude to our project guide, **Ms. Shipra Gautam**, and the **Department of Computer Science & Engineering, ABESEC Ghaziabad**, for their continuous guidance and technical oversight throughout this capstone project.
