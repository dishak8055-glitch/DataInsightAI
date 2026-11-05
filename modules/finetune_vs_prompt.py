# modules/finetune_vs_prompt.py

import pandas as pd


def get_comparison_matrix():
    """
    Returns a structured comparison across Prompting, RAG, PEFT/LoRA, and Full Fine-Tuning.
    """
    data = [
        {
            "Dimension": "Primary Objective",
            "Prompt Engineering": "In-context steering of general pre-trained LLM",
            "RAG (Retrieval)": "Grounding generation in external dynamic knowledge",
            "PEFT / LoRA": "Domain/style adaptation by training small low-rank adapter matrices",
            "Full Fine-Tuning": "Complete domain overhaul by updating all model weights"
        },
        {
            "Dimension": "Training Data Needed",
            "Prompt Engineering": "0 to 10 prompt examples",
            "RAG (Retrieval)": "Unstructured knowledge corpus (PDFs, docs, CSVs)",
            "PEFT / LoRA": "500 to 5,000 high-quality task pairs",
            "Full Fine-Tuning": "50,000+ domain-specific curated examples"
        },
        {
            "Dimension": "Compute / GPU Requirement",
            "Prompt Engineering": "None (runs on API / CPU)",
            "RAG (Retrieval)": "Low (vector embedding generation + API inference)",
            "PEFT / LoRA": "Moderate (1 consumer GPU e.g. RTX 3060/4090 with 4-bit QLoRA)",
            "Full Fine-Tuning": "Very High (multi-node A100 / H100 clusters)"
        },
        {
            "Dimension": "Financial Cost",
            "Prompt Engineering": "Free to low (API token charges only)",
            "RAG (Retrieval)": "Low (vector DB storage + token charges)",
            "PEFT / LoRA": "$5 to $50 cloud compute time",
            "Full Fine-Tuning": "$1,000 to $50,000+ infrastructure"
        },
        {
            "Dimension": "Risk of Catastrophic Forgetting",
            "Prompt Engineering": "Zero (original weights untouched)",
            "RAG (Retrieval)": "Zero (original weights untouched)",
            "PEFT / LoRA": "Extremely Low (frozen base model + adapter)",
            "Full Fine-Tuning": "High (can degrade general reasoning capabilities)"
        },
        {
            "Dimension": "Inference Latency Impact",
            "Prompt Engineering": "Slightly higher prompt processing latency (longer context)",
            "RAG (Retrieval)": "Retrieval step + prompt token latency",
            "PEFT / LoRA": "Zero additional latency (adapters can be fused: W = W0 + BA)",
            "Full Fine-Tuning": "Zero additional latency"
        },
        {
            "Dimension": "Best Suited For",
            "Prompt Engineering": "Ad-hoc CSV Q&A, prototyping, interactive analysis",
            "RAG (Retrieval)": "Dynamic documentation, querying large CSV databases",
            "PEFT / LoRA": "Specialized domain formats, enterprise tabular translation",
            "Full Fine-Tuning": "Novel languages, entirely bespoke vocabulary/industry"
        }
    ]
    return pd.DataFrame(data)


def calculate_lora_savings(base_params_billions=7.0, rank=16, alpha=32, target_modules_count=4):
    """
    Calculates parameter and memory savings achieved by LoRA (Low-Rank Adaptation).
    For a linear layer W in R^(d x k), LoRA decomposes delta W = (alpha/r) * B * A
    where A in R^(r x k), B in R^(d x r), r << min(d, k).
    """
    # Typical hidden dimensions for 7B LLM (e.g. Llama / Mistral)
    hidden_dim = 4096
    intermediate_dim = 14336
    num_layers = 32

    # Standard linear layer parameters: d * k
    layer_params = hidden_dim * hidden_dim
    # LoRA parameters per adapted layer: r * hidden_dim + hidden_dim * r = 2 * r * hidden_dim
    lora_params_per_matrix = 2 * rank * hidden_dim
    
    total_lora_params = lora_params_per_matrix * target_modules_count * num_layers
    base_params = int(base_params_billions * 1e9)
    trainable_percentage = (total_lora_params / base_params) * 100

    memory_base_fp16_gb = round(base_params * 2 / (1024**3), 2)
    memory_lora_fp16_mb = round(total_lora_params * 2 / (1024**2), 2)

    return {
        "base_params_billions": base_params_billions,
        "rank": rank,
        "alpha": alpha,
        "target_modules_count": target_modules_count,
        "total_trainable_lora_params": total_lora_params,
        "trainable_percentage": round(trainable_percentage, 4),
        "memory_base_fp16_gb": memory_base_fp16_gb,
        "memory_lora_fp16_mb": memory_lora_fp16_mb,
        "formula": "W = W_0 + \\frac{\\alpha}{r} (B \\cdot A), \\quad A \\in \\mathbb{R}^{r \\times d}, B \\in \\mathbb{R}^{k \\times r}"
    }


def recommend_learning_strategy(num_samples, budget_dollars, has_gpu, latency_critical):
    """
    Deterministic decision advisor for Viva examination and practical engineering.
    """
    if num_samples < 100 or budget_dollars == 0 or not has_gpu:
        return {
            "strategy": "Prompt Engineering + In-Context Few-Shot Learning",
            "justification": (
                "With limited labeled data or zero GPU budget, prompt engineering provides "
                "immediate utility with zero capital outlay and no risk of overfitting."
            ),
            "suggested_pipeline": "Use structured system prompts + few-shot CSV examples via Gemini API."
        }
    elif num_samples >= 500 and has_gpu and budget_dollars <= 50:
        return {
            "strategy": "Parameter-Efficient Fine-Tuning (PEFT / QLoRA)",
            "justification": (
                "You have sufficient samples and a GPU. LoRA trains less than 0.1% of parameters, "
                "drastically lowering VRAM usage while adapting the model to your exact CSV schema."
            ),
            "suggested_pipeline": "Freeze base model weights, attach rank-16 LoRA adapters to Q & V projection layers."
        }
    elif num_samples >= 50000 and budget_dollars > 1000:
        return {
            "strategy": "Full Domain Fine-Tuning",
            "justification": (
                "Enterprise-scale data volume and compute budget allow continuous pre-training "
                "or full supervised fine-tuning across all layers."
            ),
            "suggested_pipeline": "Multi-GPU distributed training with deepspeed/FSDP."
        }
    else:
        return {
            "strategy": "Retrieval-Augmented Generation (RAG) + Prompting",
            "justification": (
                "Balances accuracy and compute by dynamically retrieving relevant rows/metadata "
                "from a vector store before prompting the LLM."
            ),
            "suggested_pipeline": "Index dataset embeddings and retrieve top-k context rows at query time."
        }
