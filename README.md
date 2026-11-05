---
title: DataInsight AI - CSV Dataset Insight and Task Recommender
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# DataInsight AI – CSV Dataset Insight & Task Recommender (P10)

DataInsight AI is an advanced Generative AI and automated dataset intelligence system built for tabular CSV data. It bridges deterministic pandas statistical profiling with modern GenAI architectures: transformer natural-language querying, PyTorch Tabular CTGAN synthesis, BERT-family column semantic embeddings, multimodal audio/visual generation, and Responsible AI compliance audits.

---

## Architecture Overview

```mermaid
graph TD
    A["CSV Upload / Public Benchmark"] --> B["Data Profiling Engine"]
    B --> C["Type Identification & Missing Analysis"]
    B --> D["Plotly Interactive Visualizations"]
    B --> E["ML Task Recommender (Classification/Regression/Clustering)"]
    
    A --> F["High-Performance Chat Engine"]
    F -->|Instant Pattern Match| G["Fast-Path Pandas Execution (<10ms)"]
    F -->|Complex Query| H["Gemini Single-Call Structured Planner"]
    H --> G
    G --> I["Deterministic Natural-Language Explanation"]
    
    A --> J["U1 & U2: Foundations & Architectures"]
    J --> J1["Generative vs Discriminative Modeling"]
    J --> J2["Prompt Engineering Studio & Sampling Explorer"]
    J --> J3["LoRA PEFT Calculator & Attention Visualizer"]
    
    A --> K["U3: BERT & Multimodal Studio"]
    K --> K1["BERT-Family Column Semantic Clustering"]
    K --> K2["GPT Narrative Insight Generator"]
    K --> K3["Audio Narration (pyttsx3) & Dataset Cards"]
    
    A --> L["U4: PyTorch CTGAN Synthesis"]
    L --> L1["Tabular Generator & Discriminator Networks"]
    L --> L2["Kolmogorov-Smirnov & Correlation Validation"]
    
    A --> M["U5: Responsible AI & Model Card"]
    M --> M1["Sensitive-Column & PII Detection"]
    M --> M2["Class Imbalance Entropy & Demographic Fairness"]
    M --> M3["Hallucination Verification & Model Card Export"]
```

---

## Assignment Requirements Coverage (U1 to U5)

| Syllabus Unit | Requirement | Implementation Module |
| :--- | :--- | :--- |
| **U1 – Foundations of GenAI** | Dataset distributions & visualizations | `modules/visualization.py` (Histogram, Boxplot, Bar, Scatter, Violin, Heatmap) |
| | ML task recommendations | `modules/recommender.py` (Classification, Regression, Clustering) |
| | Learning-type identification & lifecycle | `modules/gen_vs_disc.py` (Supervised, Unsupervised, Self-Supervised, Generative) |
| | Generative vs Discriminative learning | `modules/gen_vs_disc.py` ($P(Y \mid X)$ vs $P(X, Y)$ mathematical comparison) |
| | Prompt engineering & latent-space sampling | `modules/prompt_engineering.py` (Zero-shot, Few-shot, CoT, Role, Schema, Temperature/Top-p) |
| **U2 – GANs and LLMs** | Transformer-based LLM | `modules/csv_chat.py` (Google Gemini 3.8 Flash integration) |
| | Natural-language CSV analysis using pandas | `modules/csv_chat.py` (Fast-Path Pandas + Structured JSON Execution) |
| | Fine-tuning vs prompt-based learning | `modules/finetune_vs_prompt.py` (Radar trade-offs + LoRA parameter calculator) |
| | GAN/VAE or attention explanation | `modules/attention_explainer.py` (Interactive attention weights + Minimax/ELBO comparison) |
| **U3 – GPT, BERT & Multimodal** | GPT-style generator | `modules/gpt_generator.py` (Autoregressive narrative storytelling) |
| | BERT-family embeddings to group columns | `modules/column_embeddings.py` (Subword TF-IDF + Cosine similarity clustering) |
| | Image or audio modality | `modules/multimodal.py` (pyttsx3 spoken speech + PIL Dataset Infographic Card) |
| **U4 – Tools & Generation** | Python with PyTorch | `modules/synthetic_data.py` (PyTorch 2.10.0 neural networks) |
| | CTGAN synthetic tabular data generation | `modules/synthetic_data.py` (Tabular Generator & Discriminator training) |
| | Compare real and synthetic datasets | `modules/synthetic_data.py` (Two-sample KS-test, correlation matrix Frobenius norm) |
| | Address image/video/music generation | `modules/multimodal.py` (Cross-modal survey: Diffusion, Sora, MusicLM) |
| **U5 – Responsible AI** | Sensitive-column & PII detection | `modules/responsible_ai.py` (Regex PII detection + Protected attribute audit) |
| | Class-imbalance detection | `modules/responsible_ai.py` (Shannon entropy, ratio metrics, SMOTE recommendations) |
| | Bias & hallucination evaluation | `modules/responsible_ai.py` (Disparate Impact 80% rule + Pandas factuality check) |
| | Safety guardrails | `modules/csv_chat.py` (Zero unrestricted `eval()`/`exec()`, strict operation whitelist) |
| | Licence/IP considerations | `modules/responsible_ai.py` (CC0, CC-BY, MIT, ODC-By, CC-BY-NC guidance) |
| | Model card | `MODEL_CARD.md` & `modules/responsible_ai.py` (Mitchell et al. 2019 standard) |

---

## Chat Performance Optimization

### Investigation Findings
1. **Double API Call Bottleneck:** The previous chat implementation invoked Gemini twice per question (once for JSON plan interpretation, and a second time to explain the pandas output). Each round-trip introduced 1.5 - 3.0 seconds of network latency.
2. **Missing Local Fast-Path:** Standard statistical questions (e.g. "average Age", "how many rows", "highest salary") were unnecessarily sent over the network to the LLM despite being deterministically resolvable locally in microseconds.
3. **Aggressive Retry Delays:** The API client configured 3 retry attempts with a 2-second initial delay and exponential base 2, blocking the UI thread for 15+ seconds during network drops.

### Solutions Implemented
- **Fast-Path Semantic Parser:** Matches column names using word-boundary regular expressions (`\bcol\b` preventing subword false-positives like 'Age' inside 'average') and resolves common statistical intents instantly (< 10 ms).
- **Single-Call Strategy:** When complex natural language requires Gemini, only a single structured call is made, executing the plan via pandas and generating clean explanations locally.
- **Resilient Fallback & Fast Retries:** Client caching and tight 1.0s retries ensure the Streamlit UI never freezes.
- **Clean Interface:** Removed cluttered technical JSON dumps from the default user view, providing an optional collapsed inspection expander for viva evaluation.

---

## Installation & Local Execution

### Prerequisites
- Windows 11, macOS, or Linux
- Python 3.11
- 16 GB RAM (runs on CPU / integrated graphics)

### Step 1: Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/DataInsightAI.git
cd DataInsightAI
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Create a `.env` file in the root folder:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(You can obtain a free API key from [Google AI Studio](https://aistudio.google.com/)).*

### Step 4: Run the Application
```bash
streamlit run app.py
```

### Step 5: Run Automated Tests
```bash
python -m pytest tests/
```
*(All 50 unit tests will run and pass).*

---

## Deployment to Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Choose **Streamlit** as the SDK.
3. Push your repository files (`app.py`, `modules/`, `requirements.txt`, `sample_*.csv`, `MODEL_CARD.md`).
4. In the Space settings, add your **Repository Secret**:
   - Variable name: `GEMINI_API_KEY`
   - Value: `your_gemini_api_key_here`
5. The application will build and launch publicly with free CPU tier hosting.

---

## Viva Voce Preparation Guide

### 1. What is the fundamental difference between Discriminative and Generative models?
- **Discriminative models** estimate the conditional probability $P(Y \mid X)$ directly. They learn the decision boundary separating classes (e.g. Logistic Regression, Random Forest, SVM) or predict continuous targets (Linear Regression).
- **Generative models** estimate the joint distribution $P(X, Y) = P(X \mid Y)P(Y)$ or data distribution $P(X)$. They model how the data was generated and can synthesize novel samples (e.g. Naive Bayes, CTGAN, VAEs, Diffusion).

### 2. How does CTGAN generate synthetic tabular data?
Tabular data presents mixed continuous columns with multimodal distributions and discrete categorical columns. CTGAN employs **Mode-Specific Normalization** (using a Variational Gaussian Mixture Model) to transform continuous features, and one-hot encodings for categories. A Generator network transforms random noise $z \sim \mathcal{N}(0, I)$ into synthetic rows, while a Discriminator network trained via Wasserstein loss penalizes deviations from real row distributions.

### 3. What is Low-Rank Adaptation (LoRA) and how does it reduce memory?
Instead of updating all weights $W_0 \in \mathbb{R}^{d \times k}$, LoRA freezes $W_0$ and injects trainable rank-decomposition matrices: $\Delta W = \frac{\alpha}{r} (B \cdot A)$ where $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ with rank $r \ll \min(d, k)$. For a 7B parameter LLM with rank $r=16$, LoRA trains less than 0.1% of weights, cutting GPU memory requirements by over 70%.

### 4. How does Scaled Dot-Product Attention work?
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
Queries ($Q$) and Keys ($K$) measure pairwise compatibility scores. Dividing by $\sqrt{d_k}$ prevents dot products from growing excessively large in high dimensions (which would cause vanishing gradients in softmax). Softmax yields non-negative attention weights summing to 1.0, which weight the Values ($V$).

### 5. What are the key Responsible AI considerations implemented in DataInsight AI?
- **Sensitive Feature Detection:** Identifies PII (email, phone, national IDs) and protected classes (gender, age, race).
- **Class Imbalance Mitigation:** Computes Shannon entropy and majority-to-minority ratios, guiding the user toward SMOTE, class-weighted loss, or focal loss.
- **Demographic Fairness:** Evaluates Disparate Impact Ratio (DIR) across sensitive demographic slices using the EEOC 80% (4/5ths) rule.
- **Hallucination Auditing:** Verifies that AI claims are grounded in deterministic pandas facts.
- **Standardized Model Card:** Exports a complete Mitchell et al. (2019) compliance document.

---

## Deliverables & Artifacts Included
- `app.py`: Streamlit application with tabbed interface.
- `modules/`: 9 modular engines covering profiling, visualization, recommender, chat, GenAI, BERT, CTGAN, and Responsible AI.
- `tests/`: 11 test suites comprising 50 unit tests (100% passing).
- `MODEL_CARD.md`: Mitchell et al. compliant documentation.
- `REPORTS.md`: Experimental benchmark reports on 3 public datasets with failure case analysis.
- `DEMO_SCRIPT.md`: Timed 2-3 minute demonstration video script.
- `requirements.txt`: Clean cross-platform dependency specifications.

---
*Developed as part of the Final-Year Generative AI Curriculum (P10).*
