# modules/gpt_generator.py

import os
import json
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


def generate_dataset_story(df, target_column=None, focus_mode="Executive Summary", use_api=True):
    """
    GPT-style autoregressive narrative generator.
    Produces comprehensive, viva-ready analytical reports and data stories.
    Uses Gemini when API key is configured and use_api=True, with an offline structured generator fallback.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    col_names = list(df.columns)
    missing_sum = int(df.isnull().sum().sum())

    # Compile data facts
    facts = {
        "rows": total_rows,
        "columns": total_cols,
        "feature_names": col_names,
        "total_missing": missing_sum,
        "target": target_column or "None specified"
    }

    # Attempt LLM generation if requested and available
    api_key = os.getenv("GEMINI_API_KEY")
    api_error = None
    if use_api and HAS_GENAI and api_key:
        try:
            client = genai.Client(api_key=api_key)
            prompt = f"""
You are an expert GPT-style narrative data generator.
Analyze the following tabular dataset metadata and generate a coherent 3-paragraph {focus_mode}:

Dataset Facts:
{json.dumps(facts, indent=2)}

Guidelines:
- Paragraph 1: Dataset architecture, scale, and data hygiene observations.
- Paragraph 2: Core analytical hypotheses, feature interplay, and target implications.
- Paragraph 3: Recommended machine learning roadmap and generative AI opportunities.
- Format with clear professional prose. Avoid emojis. Keep explanations simple and precise for viva examination.
"""
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )
            return {
                "narrative": response.text.strip(),
                "generator_type": "Transformer Autoregressive (Gemini Flash)",
                "facts": facts
            }
        except Exception as exc:
            api_error = f"{type(exc).__name__}: {exc}"

    # Offline deterministic generator fallback
    p1 = (
        f"The active dataset comprises {total_rows} recorded observations distributed across {total_cols} distinct features: "
        f"{', '.join(col_names[:6])}{' and others' if total_cols > 6 else ''}. "
        f"Data integrity inspection reveals {missing_sum} missing values across the schema. "
        f"{'The absence of missing values establishes strong baseline data readiness.' if missing_sum == 0 else 'Handling these missing values via median or mode imputation is recommended prior to model fitting.'}"
    )

    p2 = (
        f"Focusing on the analytical objective with target variable '{facts['target']}', "
        f"the feature space presents opportunities for both predictive discrimination and generative modeling. "
        f"Numerical columns require standard normalization to prevent scale dominance, while categorical variables "
        f"should be encoded using one-hot or ordinal representations depending on category cardinality."
    )

    p3 = (
        "From an engineering perspective, establishing a discriminative baseline (such as a Random Forest or Logistic Classifier) "
        "provides an objective benchmark. Subsequently, generative approaches such as CTGAN or Variational Autoencoders can be applied "
        "to augment rare cohorts or create test data. Synthetic output is not guaranteed to protect privacy."
    )

    return {
        "narrative": f"{p1}\n\n{p2}\n\n{p3}",
        "generator_type": (
            "Deterministic fallback (Gemini request failed)"
            if api_error else "Deterministic Autoregressive Fallback"
        ),
        "facts": facts,
        "api_error": api_error
    }


def get_transformer_architectures_comparison():
    """
    Returns a conceptual comparison between GPT (Decoder-only), BERT (Encoder-only),
    and T5/BART (Encoder-Decoder) for viva examination.
    """
    return [
        {
            "Family": "GPT Family (Decoder-Only)",
            "Attention Type": "Causal / Masked Self-Attention (Left-to-Right only)",
            "Pre-training Objective": "Autoregressive Next-Token Prediction: max log P(w_t | w_<t)",
            "Key Models": "GPT-2, GPT-3, GPT-4, Llama, Gemini (Decoder variant)",
            "Primary Use Cases": "Text generation, chat reasoning, code completion, narrative synthesis",
            "Why it fits our project": "Powers the natural language CSV chat and data storytelling generator."
        },
        {
            "Family": "BERT Family (Encoder-Only)",
            "Attention Type": "Bidirectional Self-Attention (Attends to left and right simultaneously)",
            "Pre-training Objective": "Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)",
            "Key Models": "BERT, RoBERTa, DeBERTa, DistilBERT",
            "Primary Use Cases": "Feature extraction, sentence embeddings, classification, semantic column clustering",
            "Why it fits our project": "Computes semantic representations to cluster similar dataset columns."
        },
        {
            "Family": "T5 / BART (Encoder-Decoder)",
            "Attention Type": "Bidirectional in Encoder, Causal Cross-Attention in Decoder",
            "Pre-training Objective": "Span Corruption & Text-to-Text Denoising",
            "Key Models": "T5, BART, FLAN-T5",
            "Primary Use Cases": "Translation, document summarization, question answering over text",
            "Why it fits our project": "Standard reference architecture for sequence-to-sequence transformation."
        }
    ]
