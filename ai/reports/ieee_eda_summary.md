# Phase 7 Report: IEEE-CIS Fraud Detection Dataset Acquisition & Exploratory Profiling

**Project:** AI-Based Financial Fraud Detection using TabNet  
**Phase:** 7 (Dataset Acquisition, Integrity Validation & Exploratory Profiling)  
**Status:** Verified Live Telemetry (Full Dataset & Sample Profile Analyzed)  

---

## 1. Dataset Provenance & File Verification

### 1.1 Dataset Identification
- **Official Title:** IEEE-CIS Fraud Detection
- **Organizers / Data Providers:** **Vesta Corporation** (leading payment fraud solution provider) in partnership with the **IEEE Computational Intelligence Society (IEEE-CIS)**.
- **Official Repository:** [Kaggle Competition: IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)
- **License & Terms:** Kaggle Competition Rules / Non-Commercial Research & Educational Use.
- **Data Integrity Guarantee:** Real-world e-commerce card-not-present (CNP) transaction logs with genuine adversarial fraud patterns.

### 1.2 Disk Verification Status
- **Raw Storage Directory:** `C:\Projects\FinancialFraudDetection\financial-fraud-tabnet\ai\data\ieee_cis\raw`
- **`train_transaction.csv` Present:** `True` (Size: **651.69 MB**)
- **`train_identity.csv` Present:** `True` (Size: **25.3 MB**)
- **Live Loading Readiness:** `True`

---

## 2. Verified Full-Dataset Ground Truth Statistics (590,540 Transactions)

The full-scale raw files were scanned using memory-efficient stream verification without loading the uncompressed multi-gigabyte dataset entirely into RAM.

### 2.1 Table Topology & Key Uniqueness

| Metric | `train_transaction.csv` | `train_identity.csv` | Merged Dataset (Left Join) |
| :--- | :---: | :---: | :---: |
| **Total Rows** | **590,540** | **144,233** | **590,540** |
| **Total Columns** | **394** | **41** | **434** |
| **Primary Key** | `TransactionID` | `TransactionID` | `TransactionID` |
| **Key Duplicates** | **0** (Strictly Unique) | **0** (Strictly Unique) | 0 |
| **Referential Integrity** | Target Primary | **100% IDs in Transaction Table** (`subset=True`) | Verified |
| **Identity Coverage** | N/A | N/A | **144,233 (24.42%)** |
| **Unmatched Transactions** | N/A | N/A | **446,307 (75.58%)** |

### 2.2 Full-Dataset Ground-Truth Class Distribution (`isFraud`)

- **Total Scanned Transactions:** 590,540
- **Legitimate Transactions (`isFraud = 0`):** 569,877 (96.501%)
- **Fraudulent Transactions (`isFraud = 1`):** 20,663 (3.499%)
- **True Commercial Imbalance Ratio:** **27.58:1**

> [!NOTE]
> **Defensible Real-World Imbalance:** The ground truth fraud prevalence is exactly **3.4990%** across all 590,540 transactions. This provides an authentic, high-signal, non-synthetic fraud benchmark for TabNet and gradient boosting baselines.

---

## 3. Sample Telemetry & Merging Confirmation (50,000-Row Sample)

### 3.1 Confirmation of Sample Merging Architecture
- **Sample Loading Method:** `load_ieee_cis_raw(nrows=50000, load_identity=True, validate=True)`
- **Loaded Transaction Rows:** 50,000
- **Loaded Merged Columns:** 434 (394 transaction cols + 41 identity cols - 1 shared `TransactionID` = **434 columns**)
- **Merge Architecture Confirmed:** **Left-join merge verified.** All 50,000 transaction rows are preserved.
- **Identity Matched in Sample:** **16,068 transactions (32.14%)** have active digital identity telemetry (e.g. `DeviceType`, `DeviceInfo`).

### 3.2 Distinguishing Sample Statistics from Full-Dataset Statistics

| Dimension | 50,000-Row Chronological Sample | Complete Dataset (590,540 Rows) | Analytical Observation |
| :--- | :---: | :---: | :--- |
| **Total Transactions** | 50,000 | 590,540 | Sample represents first ~15.4 days of the 182-day window. |
| **Legitimate Count** | 48,643 (97.2860%) | 569,877 (96.501%) | Nominal majority. |
| **Fraud Count** | 1,357 (2.7140%) | 20,663 (3.499%) | **Fraud Rate Delta:** 0.7850%. |
| **Imbalance Ratio** | 35.85:1 | 27.58:1 | Early chronological period exhibits lower fraud density than later adversarial spikes. |
| **Identity Match Rate** | 32.14% | 24.42% | Slightly higher identity capture rate in initial transaction cohort. |

---

## 4. Missing Value Profiling & Feature Group Architecture

Across the 50,000-row merged profile, the overall null rate across all cells is **45.24%** (9,816,044 / 21,700,000 cells).

| Feature Block | Column Count | Null Rate (%) | Representative Features | Analytical Interpretation |
| :--- | :---: | :---: | :--- | :--- |
| **Count Features (`C1`-`C14`)** | 14 | **0.0%** | `C1`, `C2`, `C5`, `C13`, `C14` | **Zero missingness!** Complete multi-window velocity features provided by Vesta. |
| **V-Features (`V1`-`V339`)** | 339 | **43.27%** | `V1`-`V339` | Dense clusters corresponding to payment product categories. |
| **Time Delta Features (`D1`-`D15`)** | 15 | **64.53%** | `D1`, `D2`, `D15` | Timedeltas since previous transaction; missing when user is first-time visitor. |
| **Match Features (`M1`-`M9`)** | 9 | **65.91%** | `M1`-`M9` | Name, address, and card match flags (`T`/`F`/`NaN`). |
| **Identity Features (`id_01`-`id_38`)** | 40 | **78.44%** | `DeviceType`, `DeviceInfo` | Missing for transactions without digital identity capture (~75% of total). |

---

## 5. Memory Requirements & Hardware Feasibility Strategy

### 5.1 Measured & Extrapolated Memory Footprint (16 GB RAM / 8-Core CPU)
- **50,000 Merged Rows in Memory:** **168.3196 MB**
- **Full Merged Dataset Extrapolated (590,540 rows):** **~1.94 GB** (in standard `float64`/`object` format)
- **Full Identity File Only:** **56.51 MB** in RAM

### 5.2 Curated 45-Feature Research Architecture for Phase 8
While loading 1.94 GB is feasible on 16 GB RAM, running 5-fold cross-validation with TabNet across 434 features on CPU would result in prolonged attention-mask computation. In Phase 8, we deploy a curated 45-feature research subset:
1. **Core Financial & Temporal (4):** `TransactionAmt`, `TransactionDT`, `Log_TransactionAmt`, `Amt_to_Mean_Ratio`
2. **Categorical Entity & Channel (7):** `ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `DeviceType`, `DeviceInfo`
3. **Card & Region Identifiers (3):** `card1`, `addr1`, `addr2`
4. **Causal Velocity & Recency (10):** `C1`, `C2`, `C5`, `C6`, `C11`, `C13`, `C14`, `D1`, `D2`, `D15`
5. **Verification Match Flags (4):** `M1`, `M4`, `M5`, `M6`
6. **Top Salient V-Features (15):** `V12`, `V13`, `V35`, `V36`, `V75`, `V76`, `V90`, `V91`, `V126`, `V127`, `V128`, `V257`, `V258`, `V282`, `V283`
- **Sample Size for CV Experiments:** Stratified sample of 100,000–150,000 transactions preserving exact 3.4990% fraud prevalence.
- **Runtime Expectation:** ~1.5 to 2.0 minutes per fold × 5 folds = **7.5 to 10 minutes total cross-validation runtime**.

---

## 6. Resolution of Phase 5 Limitations with Verified IEEE-CIS Data

| Phase 5 Structural Limitation | Status in Synthetic Dataset | Verified IEEE-CIS Reality | Analytical Advantage |
| :--- | :--- | :--- | :--- |
| **1. Customer Identity Tracking** | Impossible (100% unique `Customer_ID`; 1 tx per user) | **Verified:** Multi-transaction card clusters (`card1` + `addr1` + `P_emaildomain`) | Enables authentic customer-level historical profiles and repeat-entity tracking |
| **2. Rolling Velocity Checks** | Impossible (Historical counts identically 0) | **Verified:** Vesta's `C1`-`C14` have **0.00% missing values** and multi-window counters | Captures burst testing, bot attacks, and automated card testing |
| **3. Inter-Transaction Time (Delta t)** | Impossible (Undefined for all rows) | **Verified:** `D1`-`D15` timedeltas and `TransactionDT` continuous timestamps | Identifies high-frequency transaction bursts and dormancy wakeups |
| **4. Device Telemetry** | Infeasible (Generic simulated strings) | **Verified:** `DeviceInfo` and `DeviceType` record actual hardware models (`Samsung`, `iOS Device`, etc.) | Detects rapid hardware switches on identical payment cards |
| **5. Class Predictability** | Unconditioned random prior (MI < 0.005) | **Verified:** Genuine real-world adversarial fraud (3.4990% prevalence) | Replaces synthetic noise with authentic, learnable fraud signals |

---

## 7. Phase 7 Validation & Completion Sign-Off

- [x] Actual row counts verified: 590,540 transactions, 144,233 identities.
- [x] Required columns present in both tables.
- [x] `TransactionID` uniqueness verified (0 duplicates in either table).
- [x] Referential integrity confirmed (100% of identity IDs exist in transaction table).
- [x] Merging behavior verified: Left join preserves 100% of transactions; merges 41 identity features onto 394 transaction features.
- [x] Fraud prevalence calculated on full dataset without memory bloat: **3.4990%** (20,663 fraud transactions).
- [x] Sample vs full dataset distinctions explicitly documented (temporal drift accounted for).
- [x] Missingness profiled by feature group (C-features: 0% missing; V-features: 43.3% missing).
- [x] Memory requirements measured and CPU feasibility strategy established.
- [x] **Phase 7 is officially complete and verified. Ready to proceed to Phase 8.**
