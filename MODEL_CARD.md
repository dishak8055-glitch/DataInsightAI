# Model & System Card: DataInsight AI (P10)

## Model Details
- **System Name:** DataInsight AI – CSV Dataset Insight & Task Recommender
- **Project Code:** P10 (Final-Year Generative AI Project)
- **Version:** 2.1.0
- **Architectures Involved:**
  - **Large Language Model (LLM):** Google Gemini 3.8 Flash (optional, via `google-genai`)
  - **Tabular Generative Model:** Lightweight educational PyTorch conditional GAN; CTGAN-inspired but not canonical CTGAN
  - **Embedding & Semantic Clustered Encoder:** Optional BERT-tiny contextual embeddings with Agglomerative Clustering; offline character n-gram TF-IDF baseline
  - **Speech Synthesis (Audio Modality):** Offline multi-platform text-to-speech engine (`pyttsx3`)
  - **Visual Synthesis (Image Modality):** Dark-theme infographic card generator via Matplotlib & Pillow
  - **Deterministic Engine:** Pandas and NumPy for reproducible statistical calculations

## Intended Use
- **Primary Use Cases:**
  - Exploratory Data Analysis (EDA) of tabular CSV datasets.
  - Automated Machine Learning (AutoML) task recommendation (Classification, Regression, Clustering).
  - Local natural-language question patterns for supported operations; latency depends on the environment.
  - Educational synthetic tabular data generation and comparison; no privacy guarantee.
  - Ethical, safety, and Responsible AI audits (PII detection, class imbalance, demographic fairness).
- **Primary Users:**
  - Data scientists, machine learning students, and viva examiners.
- **Out-of-Scope & Prohibited Use:**
  - Direct autonomous clinical decision-making or credit scoring without human auditing.
  - Processing highly classified or military intelligence data without air-gapped encryption.

## Data & Features
- **Inputs:** Tabular CSV files with arbitrary numerical, categorical, datetime, or identifier columns.
- **Built-in Benchmark Datasets:**
  - `sample_customers.csv`: 10 rows × 5 features (Customer segmentation & spending behavior).
  - `sample_healthcare.csv`: 15 rows × 8 features (Clinical health indicators with PII and binary outcome).
  - `sample_housing.csv`: 15 rows × 7 features (Real estate continuous price regression).

## Evaluation & Metrics
- Supported numeric synthetic-data comparisons report per-column KS statistics and, when at least two columns are evaluable, correlation similarity. No benchmark quality score is asserted in this card.
- **Demographic Fairness:**
  - An exploratory disparate-impact ratio can be computed for selected groups and outcomes. It is not a legal compliance determination.
- **Hallucination Detection:**
  - A heuristic matches extracted numeric claims against sample values and aggregate statistics. It may produce false positives and false negatives.

## Safety, Guardrails & Responsible AI
- **No Unrestricted Execution:** Zero usage of `eval()` or `exec()`. All pandas operations are mapped to a strictly validated whitelist.
- **Sensitive Feature Detection:** Header keywords and sampled value patterns flag possible PII; detection is incomplete and does not mask data.
- **Class Imbalance Warnings:** Shannon entropy metrics flag majority-to-minority ratios exceeding 1.75:1.
- **External Processing:** CSV metadata and questions are sent to Gemini when API features are used. Chart images are sent only after explicit consent. Do not upload information you are not authorized to share.
- **License/IP:** Users are responsible for confirming dataset rights, attribution, and third-party model/dependency terms before use or redistribution.

## Failure Modes & Limitations
1. **Synthetic data privacy and fidelity:** The compact conditional GAN may overfit, especially on small datasets. It does not use mode-specific normalization or differential privacy; generated rows may resemble training records.
2. **Ambiguous Multi-Hop Questions:** Complex questions with no identifiable column names (e.g. "What happened in the summer?") cannot be answered locally and require LLM interpretation.
3. **External and multimodal capabilities:** Gemini features require network access, a configured API key, and may incur quota limits. BERT-tiny downloads model weights on first use. Video and music generation are not implemented; the app provides an educational overview only. Local text-to-speech may fail on headless systems without audio support.

## Licensing
- **Application License:** MIT License (subject to third-party dependency and dataset licenses).
- **Dataset License:** No dataset license is assumed; verify each dataset's source terms.
