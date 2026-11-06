# Evaluation Procedures: Three Sample Datasets

No prior latency, GAN-fidelity, or embedding benchmark had reproducible logs, so no such benchmark result is claimed here. The sample dataset dimensions and simple statistics below were checked directly against the CSV files. Synthetic fidelity varies by run and dataset; do not treat it as a privacy measure.

---

## Dataset 1: Customer Segmentation & Spending Behavior (`sample_customers.csv`)

### 1. Dataset Profile
- **Domain:** Retail & Customer Analytics
- **Dimensions:** 10 Rows × 5 Columns
- **Features:** `CustomerID` (Identifier), `Age` (Numerical), `Gender` (Categorical), `AnnualIncome` (Numerical), `SpendingScore` (Numerical)
- **Missing Values:** 0 (100% complete)
- **Checked arithmetic:** Mean `Age` = 34; mean `AnnualIncome` = 45,400. Mean `AnnualIncome` by gender = 30,200 (Female) and 60,600 (Male).

### 2. Task Recommendation & Machine Learning Paradigm
- **Without Target:** Recommends **Clustering (Unsupervised Learning)**.
  - Algorithms: K-Means, DBSCAN, Hierarchical Clustering.
  - Usable Features: `Age`, `Gender`, `AnnualIncome`, `SpendingScore` (`CustomerID` successfully excluded).
- **With Target = SpendingScore:** Recommends **Regression (Supervised Learning)**.
  - Distinct Target Values: 10
  - Algorithms: Linear Regression, Random Forest Regressor.
  - Recommended Metrics: MAE, RMSE, R².

### 3. Chat and Synthetic Data
- Chat fast-path calculations were exercised locally; latency is environment-dependent and no historical speedup benchmark is claimed.

### 4. Synthetic Data Generation
- Use the app's lightweight conditional GAN and record the per-column numeric KS and correlation values produced for that run.
- This is CTGAN-inspired, not the canonical CTGAN implementation; the generator does not provide differential privacy.

---

## Dataset 2: Healthcare & Clinical Diabetes Risk (`sample_healthcare.csv`)

### 1. Dataset Profile
- **Domain:** Biomedical & Clinical Risk
- **Dimensions:** 15 Rows × 8 Columns
- **Features:** `PatientID`, `Age`, `Gender`, `BloodPressure`, `Cholesterol`, `GlucoseLevel`, `Outcome`, `Email`
- **Missing Values:** 0

### 2. Responsible AI Audit Findings
- **PII Detected:**
  - `Email`: Categorized as **PII: Contact / Location** via regex match (`\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b`).
  - Recommendation: Mask or exclude `Email` prior to model training to prevent identity leakage.
- **Protected Attributes Detected:**
  - `Gender`: Protected attribute under EEOC/GDPR.
  - `Age`: Age discrimination audit required.
- **Demographic Fairness Evaluation:**
  - Outcome Feature: `Outcome` (Diabetes risk flag).
  - Sensitive Group: `Gender`.
  - For `Outcome = 1`, the checked sample rates are 50.0% (Male) and 42.9% (Female).
  - **Disparate Impact Ratio (DIR):** 0.857. This is a small-sample screening statistic, not a legal compliance determination.

### 3. Class Imbalance Analysis
- **Target:** `Outcome`
- **Distribution:** Class 0 = 8 (53.3%), Class 1 = 7 (46.7%)
- **Imbalance Ratio:** 1.14:1 (Status: **Well Balanced**)
- **Normalized Shannon Entropy:** 0.997 / 1.000

---

## Dataset 3: Real Estate Housing Valuation (`sample_housing.csv`)

### 1. Dataset Profile
- **Domain:** Real Estate & Econometrics
- **Dimensions:** 15 Rows × 7 Columns
- **Features:** `HouseID`, `SquareFeet`, `Bedrooms`, `Bathrooms`, `AgeOfHome`, `DistanceToCityKm`, `PriceUSD`

### 2. Task Recommendation
- **Target = PriceUSD:** Recommends **Regression (Supervised Learning)**.
  - Algorithms: Linear Regression, Decision Tree Regressor, Random Forest Regressor.
  - Feature Importance Candidates: `SquareFeet`, `DistanceToCityKm`, `Bedrooms`.

### 3. Column Similarity Grouping
- The app supports BERT-tiny embeddings and an offline TF-IDF baseline. Cluster assignments are computed on demand; no fixed semantic grouping or redundancy result is asserted here.

---

---

## Known Limitations

### Synthetic data limitations
- This project uses a compact conditional GAN demonstration, not canonical CTGAN. Tiny datasets and high-cardinality columns can overfit or produce poor fidelity. Identifier columns are not automatically excluded from GAN training.
- Synthetic data has no differential privacy guarantee and must not be considered anonymized.

### Natural-language analysis limitations
- Unanchored questions report an error if no local fast-path applies and Gemini is missing, unavailable, or returns an unsupported plan. Live Gemini service behavior was not exercised in this audit.
