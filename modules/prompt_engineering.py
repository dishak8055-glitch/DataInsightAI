# modules/prompt_engineering.py

import json
import numpy as np
import pandas as pd


def generate_prompt_templates(df, target_column=None):
    """
    Generate dataset-specific prompt engineering templates demonstrating:
    1. Zero-Shot Prompting
    2. Few-Shot In-Context Prompting
    3. Chain-of-Thought (CoT) Prompting
    4. System Role Prompting
    5. Structured Output (JSON Schema) Prompting
    """
    cols = list(df.columns)
    sample_records = df.head(3).to_dict(orient="records")
    target_str = target_column if target_column else (cols[-1] if cols else "Outcome")

    # 1. Zero-Shot
    zero_shot = f"""Analyze the following dataset columns: {', '.join(cols[:6])}.
Question: What is the primary operational risk or trend observed in records where {cols[0]} is above average?
Answer:"""

    # 2. Few-Shot
    few_shot_examples = []
    for i, rec in enumerate(sample_records[:2], start=1):
        few_shot_examples.append(
            f"Example {i}:\n"
            f"Data: {json.dumps(rec)}\n"
            f"Insight: The record demonstrates typical profile values with {target_str} = {rec.get(target_str, 'N/A')}."
        )

    few_shot = "\n\n".join(few_shot_examples) + f"""\n\nExample 3 (Predict/Analyze):
Data: {json.dumps(sample_records[-1] if sample_records else {})}
Insight:"""

    # 3. Chain-of-Thought (CoT)
    cot = f"""You are analyzing a dataset with {len(df)} records and columns {', '.join(cols[:5])}.
Task: Determine whether a record with values {json.dumps(sample_records[0] if sample_records else {})} is an anomaly.

Step-by-step reasoning:
Step 1: Identify baseline distributions for numerical features.
Step 2: Compare the given record's values against standard deviations (Z-score).
Step 3: Evaluate categorical coherence with the rest of the cohort.
Step 4: Synthesize confidence score and final conclusion.

Conclusion:"""

    # 4. System Role Prompting
    system_role = f"""[System Role: Senior Principal Data Scientist & Biostatistician]
[Context: Active tabular dataset profiling with {len(df)} rows and {len(cols)} features]
[Instruction: Deliver a rigorous 3-point executive summary highlighting statistical dispersion, potential target collinearity, and actionable data hygiene steps. Avoid conversational filler.]"""

    # 5. Structured Output
    structured_json = f"""Extract high-level metadata from the dataset into the following strict JSON schema:
{{
  "dataset_summary": {{
    "total_records": {len(df)},
    "key_target": "{target_str}",
    "recommended_task": "Supervised/Unsupervised",
    "top_correlations": ["feature_a", "feature_b"],
    "data_readiness_score": 85
  }}
}}
Return ONLY valid JSON matching this schema."""

    return {
        "zero_shot": {
            "title": "Zero-Shot Prompting",
            "prompt": zero_shot,
            "description": "Direct task instruction with no prerequisite examples. Tests the model's pre-trained parametric knowledge.",
            "viva_tip": "Works well with instruction-tuned models such as Groq-served GPT-OSS for general reasoning."
        },
        "few_shot": {
            "title": "Few-Shot In-Context Prompting",
            "prompt": few_shot,
            "description": "Provides 2-5 explicit input-output pairs inside the prompt before asking for the target inference.",
            "viva_tip": "In-context learning guides output format and reduces variance without updating any model weights."
        },
        "chain_of_thought": {
            "title": "Chain-of-Thought (CoT) Prompting",
            "prompt": cot,
            "description": "Forces the model to generate intermediate reasoning steps ('Let's think step by step') before outputting an answer.",
            "viva_tip": "Dramatically improves accuracy on multi-step mathematical, tabular, and logical deductions."
        },
        "system_role": {
            "title": "System Role / Persona Prompting",
            "prompt": system_role,
            "description": "Sets the behavioral persona, tone, depth of domain expertise, and operational boundaries in the system prefix.",
            "viva_tip": "Activates relevant clusters in the model's representation space for higher domain rigor."
        },
        "structured_output": {
            "title": "Structured Output (JSON Schema) Prompting",
            "prompt": structured_json,
            "description": "Constrains LLM generation to deterministic structured schemas (JSON/Pydantic) for reliable API parsing.",
            "viva_tip": "Essential for production agentic pipelines to prevent syntax parsing errors."
        }
    }


def simulate_sampling_distribution(logits=None, temperature=1.0, top_k=5, top_p=0.9):
    """
    Simulates token probability distribution transformation under:
    - Temperature scaling: P(w_i) = exp(z_i / T) / sum(exp(z_j / T))
    - Top-k filtering: Keep top k logits, set rest to -inf
    - Top-p (Nucleus) filtering: Keep cumulative probability mass up to p
    """
    if logits is None:
        # Default representative token candidates with vocabulary logits
        candidates = ["Cluster", "Predict", "Outlier", "Regress", "Normalize", "Impute", "Validate", "Classify"]
        raw_logits = np.array([4.2, 3.8, 2.9, 2.7, 2.1, 1.8, 1.2, 0.6])
    else:
        candidates = [f"Token_{i+1}" for i in range(len(logits))]
        raw_logits = np.array(logits, dtype=float)

    # 1. Temperature Scaling
    temp = max(float(temperature), 1e-4)
    scaled_logits = raw_logits / temp

    # 2. Top-K filtering
    k = max(1, min(int(top_k), len(scaled_logits)))
    top_k_indices = np.argsort(scaled_logits)[::-1][:k]
    
    filtered_logits = np.full_like(scaled_logits, -1e9)
    filtered_logits[top_k_indices] = scaled_logits[top_k_indices]

    # Softmax on top-k
    exp_logits = np.exp(filtered_logits - np.max(filtered_logits))
    probs = exp_logits / np.sum(exp_logits)

    # 3. Top-P (Nucleus) filtering
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)

    # Cut off where cumulative probability exceeds p
    cutoff_index = np.searchsorted(cumulative_probs, top_p)
    valid_sorted_indices = sorted_indices[:cutoff_index + 1]

    nucleus_probs = np.zeros_like(probs)
    nucleus_probs[valid_sorted_indices] = probs[valid_sorted_indices]
    if np.sum(nucleus_probs) > 0:
        nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
    else:
        nucleus_probs = probs

    results = []
    for idx, token in enumerate(candidates):
        results.append({
            "Token": token,
            "Raw Logit": round(float(raw_logits[idx]), 2),
            "Raw Softmax (%)": round(float(np.exp(raw_logits[idx] - np.max(raw_logits)) / np.sum(np.exp(raw_logits - np.max(raw_logits))) * 100), 2),
            "After Temperature (%)": round(float(np.exp(scaled_logits[idx] - np.max(scaled_logits)) / np.sum(np.exp(scaled_logits - np.max(scaled_logits))) * 100), 2),
            "Final Sample Prob (%)": round(float(nucleus_probs[idx] * 100), 2)
        })

    return pd.DataFrame(results).sort_values(by="Final Sample Prob (%)", ascending=False)
