# Project Demonstration Video Script (2-3 Minutes)

This script provides a timed, step-by-step walkthrough to record a high-impact 2 to 3 minute project demonstration video for your final-year viva presentation.

---

## Video Outline & Timing

| Segment | Duration | Screen Action | Narration Voiceover Script |
| :--- | :--- | :--- | :--- |
| **0:00 - 0:25** | 25s | Launch app, show header, select `sample_customers.csv` | *"Hello examiners. Today I am presenting DataInsight AI, a Generative AI platform for CSV dataset profiling, natural-language questioning, and synthetic data generation. We start by selecting our customer dataset. The system instantly profiles total rows, columns, data types, and checks for missing or duplicate values."* |
| **0:25 - 0:50** | 25s | Click **Tab 2: Visualizations** and **Tab 3: ML Task Recommender** | *"In Tab 2, we have interactive Plotly charts, including distributions, box plots, and correlation heatmaps. In Tab 3, DataInsight AI analyzes target cardinality. If we predict SpendingScore, it recommends Regression with Linear Regression and Random Forest, providing a 7-step machine learning lifecycle."* |
| **0:50 - 1:20** | 30s | Click **Tab 4: Dataset Chat**, type: `average AnnualIncome by Gender` | *"One major performance improvement is our Chat feature. Previously, two Gemini calls took over 4 seconds. Now, our Fast-Path Pandas engine detects the intent and runs deterministically in just 8 milliseconds with zero API calls. For complex queries, Gemini generates structured execution plans with zero unrestricted eval()."* |
| **1:20 - 1:50** | 30s | Click **Tab 5: GenAI Foundations** & **Tab 6: Multimodal** | *"Tab 5 covers generative versus discriminative learning, prompt engineering, latent-space sampling, and parameter-efficient fine-tuning concepts. Tab 6 offers optional BERT-tiny contextual embeddings, GPT-style narrative output, a dataset infographic, speech synthesis, and consent-gated Gemini chart understanding. The lexical TF-IDF option works without downloading BERT weights."* |
| **1:50 - 2:20** | 30s | Click **Tab 7: PyTorch Conditional GAN**, click **Train** | *"For Unit 4, the project demonstrates a lightweight PyTorch conditional GAN with one-hot categorical features. It is CTGAN-inspired, not the canonical CTGAN implementation, and does not provide differential privacy. The app compares numeric distributions and correlations where measurable; reported scores must be interpreted as sample fidelity metrics, not privacy guarantees."* |
| **2:20 - 2:45** | 25s | Click **Tab 8: Responsible AI & Model Card**, download card | *"Finally, Unit 5 includes sensitive-column screening, class-imbalance analysis, selectable-outcome fairness screening, heuristic numeric claim checks, license guidance, and a downloadable system card. These are exploratory checks, not guarantees of legal compliance, privacy, or factual accuracy. Video and music generation are discussed but not implemented."* |

---

## Screen Recording Tips
1. **Resolution:** 1080p (1920 × 1080) at 30 or 60 fps.
2. **Tools:** OBS Studio, Windows Xbox Game Bar (`Win + Alt + R`), or Loom.
3. **Pacing:** Speak clearly and let the Streamlit charts render for 2 seconds before switching tabs.
4. **Do not quote performance or quality scores unless you have rerun and recorded the benchmark on the displayed dataset and environment.**
