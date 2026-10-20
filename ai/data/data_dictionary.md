# Data Dictionary: Indian Bank Transaction Fraud Detection

**Dataset Name:** `Bank_Transaction_Fraud_Detection.csv`  
**Location:** `ai/data/raw/Bank_Transaction_Fraud_Detection.csv`  
**Rows x Columns:** 200,000 rows × 24 columns  
**Nature:** Synthetic Indian banking transaction dataset with masked PII  
**Fraud Rate:** 5.044% (10,088 fraud transactions, 189,912 legitimate transactions)  
**Time Range:** 2025-01-01 to 2025-01-31 (January 2025)  
**Currency:** INR (100%)  

---

## Column Descriptions & Pipeline Roles

| Column Name | Inferred Type | Null Count | Unique Count | Description | Example Values | Preprocessing Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Customer_ID` | `str` (UUID) | 0 | 200,000 | Unique customer identifier | `d5f6ec07-d69e-4f47-b9b4-...` | **Drop** (Arbitrary UUID, high-cardinality leakage risk) |
| `Customer_Name` | `str` | 0 | 142,699 | Synthetic Indian customer full name | `Osha Tella`, `Krishna Nair` | **Drop** (Direct PII) |
| `Gender` | `str` | 0 | 2 | Customer gender (`Male`, `Female`) | `Male`, `Female` | **Retain** (Categorical encoding) |
| `Age` | `int64` | 0 | 53 | Customer age in years (Range: 18 – 70) | `18`, `44`, `70` | **Retain** (Numerical feature; RobustScaler) |
| `State` | `str` | 0 | 34 | Indian State / Union Territory of customer | `Kerala`, `Maharashtra`, `Bihar` | **Retain** (Categorical encoding) |
| `City` | `str` | 0 | 145 | City where customer/account is registered | `Thiruvananthapuram`, `Nashik` | **Retain** (Categorical encoding) |
| `Bank_Branch` | `str` | 0 | 145 | Bank branch name | `Nashik Branch` | **Drop** (100% redundant with `City + ' Branch'`) |
| `Account_Type` | `str` | 0 | 3 | Account tier (`Savings`, `Checking`, `Business`) | `Savings`, `Business`, `Checking` | **Retain** (Categorical encoding) |
| `Transaction_ID` | `str` (UUID) | 0 | 200,000 | Unique transaction identifier | `4fa3208f-9e23-42dc-b330-...` | **Drop** (Arbitrary UUID, memorization risk) |
| `Transaction_Date` | `str` (Date) | 0 | 31 | Date of transaction in `DD-MM-YYYY` format | `23-01-2025`, `11-01-2025` | **Engineer** (Extract Day, DayOfWeek, Is_Weekend) |
| `Transaction_Time` | `str` (Time) | 0 | 77,856 | Time of transaction in `HH:MM:SS` format | `16:04:07`, `03:09:52` | **Engineer** (Extract Hour, Is_Night) |
| `Transaction_Amount` | `float64` | 0 | 197,978 | Transaction value in INR (Range: ₹10.29 – ₹98,999.98) | `32415.45`, `63062.56` | **Retain** (Continuous numerical feature; RobustScaler) |
| `Merchant_ID` | `str` (UUID) | 0 | 200,000 | Merchant UUID identifier | `214e03c5-5c34-40d1-a66c-...` | **Drop** (100% unique per row, no repeat visits) |
| `Transaction_Type` | `str` | 0 | 5 | Transaction mechanism (Transfer, Bill Payment, Debit, Credit, Withdrawal) | `Transfer`, `Debit`, `Credit` | **Retain** (Categorical encoding) |
| `Merchant_Category` | `str` | 0 | 6 | Merchant commercial vertical | `Restaurant`, `Electronics`, `Clothing` | **Retain** (Categorical encoding) |
| `Account_Balance` | `float64` | 0 | 197,954 | Available account balance at time of transaction | `74557.27`, `52372.55` | **Retain** (Continuous numerical feature; RobustScaler) |
| `Transaction_Device` | `str` | 0 | 20 | Granular device or interface used | `ATM`, `POS Mobile App`, `Tablet` | **Retain** (Categorical encoding) |
| `Transaction_Location`| `str` | 0 | 148 | Location string (`City, State`) | `Nashik, Maharashtra` | **Drop** (100% redundant with `City + ', ' + State`) |
| `Device_Type` | `str` | 0 | 4 | High-level device category (`POS`, `ATM`, `Mobile`, `Desktop`) | `POS`, `ATM`, `Mobile`, `Desktop` | **Retain** (Categorical encoding) |
| `Is_Fraud` | `int64` | 0 | 2 | Fraud target label (0 = Legitimate, 1 = Fraud) | `0`, `1` | **Target Variable** (Ground truth) |
| `Transaction_Currency`| `str` | 0 | 1 | Currency ISO code (`INR` for all rows) | `INR` | **Drop** (Zero-variance constant feature) |
| `Customer_Contact` | `str` | 0 | 9,000 | Masked synthetic mobile phone number | `+9198579XXXXXX` | **Drop** (Masked synthetic PII) |
| `Transaction_Description`| `str` | 0 | 172 | Transaction narration text | `Bitcoin transaction`, `Grocery` | **Retain** (Categorical encoding / text categories) |
| `Customer_Email` | `str` | 0 | 4,779 | Masked synthetic email address | `oshaXXXXX@XXXXX.com` | **Drop** (Masked synthetic PII) |

---

## Engineered Features (Phase 2 & Phase 5)

| Feature Name | Type | Phase | Source Columns | Formula / Derivation | Description |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `Amount_to_Balance_Ratio` | Numerical | 2 | `Transaction_Amount`, `Account_Balance` | $\frac{\text{Amount}}{\text{Balance} + 1.0}$ | Financial burden relative to liquid balance |
| `Amount_Exceeds_Balance` | Binary | 2 | `Transaction_Amount`, `Account_Balance` | $\mathbb{I}(\text{Amount} > \text{Balance})$ | Overdraft indicator flag |
| `Tx_Hour` | Numerical | 2 | `Transaction_Time` | Hour from timestamp ($0 - 23$) | Diurnal temporal slot |
| `Tx_Day` | Numerical | 2 | `Transaction_Date` | Day of month ($1 - 31$) | Intra-month day |
| `Tx_DayOfWeek` | Numerical | 2 | `Transaction_Date` | Day of week ($0 = \text{Mon}, 6 = \text{Sun}$) | Weekly seasonality |
| `Is_Weekend` | Binary | 2 | `Tx_DayOfWeek` | $\mathbb{I}(\text{DayOfWeek} \ge 5)$ | Weekend flag |
| `Is_Night` | Binary | 2 | `Tx_Hour` | $\mathbb{I}(\text{Hour} \in \{23, 0, 1, 2, 3, 4, 5\})$ | High-risk night hours flag |
| `Hour_Sin` / `Hour_Cos` | Numerical | 2 | `Tx_Hour` | $\sin / \cos\left(\frac{2\pi \cdot \text{Hour}}{24}\right)$ | Cyclical trigonometric time representation |
| `Device_Mismatch_Flag` | Binary | 2 | `Device_Type`, `Transaction_Device` | $\mathbb{I}(\text{Transaction\_Device} \notin \text{Allowed}(\text{Device\_Type}))$ | Telemetry semantic mismatch indicator |
| `Tx_Day_Phase` | Numerical | 5 | `datetime` | $0$ (Days 1–10), $1$ (Days 11–20), $2$ (Days 21–31) | Paycheck / intra-month temporal phase |
| `Cat_Amount_Ratio` | Numerical | 5 | `Merchant_Category`, `Transaction_Amount` | $\frac{\text{Amount}}{\text{Expanding\_Cat\_Mean}_{< t} + 1.0}$ | Causal spending deviation relative to merchant vertical |
| `City_Amount_Ratio` | Numerical | 5 | `City`, `Transaction_Amount` | $\frac{\text{Amount}}{\text{Expanding\_City\_Mean}_{< t} + 1.0}$ | Causal spending deviation relative to municipal average |
| `Time_Since_Prev_Cat_Tx` | Numerical | 5 | `Merchant_Category`, `datetime` | $t_{\text{current}} - t_{\text{prev in category}}$ (seconds) | Inter-transaction interval within merchant vertical |
| `Time_Since_Prev_City_Tx` | Numerical | 5 | `City`, `datetime` | $t_{\text{current}} - t_{\text{prev in city}}$ (seconds) | Inter-transaction interval within city |
| `Cat_Tx_Density_1h` | Numerical | 5 | `Merchant_Category`, `datetime` | Rolling 1h count over $[t - 1\text{h}, t)$ | Causal 1-hour transaction volume in merchant vertical |
| `City_Tx_Density_1h` | Numerical | 5 | `City`, `datetime` | Rolling 1h count over $[t - 1\text{h}, t)$ | Causal 1-hour transaction volume in city |

