# tests/test_finetune_vs_prompt.py

from modules.finetune_vs_prompt import (
    get_comparison_matrix,
    calculate_lora_savings,
    recommend_learning_strategy
)


def test_get_comparison_matrix():
    matrix = get_comparison_matrix()
    assert not matrix.empty
    assert "Prompt Engineering" in matrix.columns
    assert "PEFT / LoRA" in matrix.columns


def test_calculate_lora_savings():
    savings = calculate_lora_savings(base_params_billions=7.0, rank=16)
    assert savings["total_trainable_lora_params"] > 0
    assert savings["trainable_percentage"] < 1.0  # LoRA trains < 1% of base weights


def test_recommend_learning_strategy_prompting():
    rec = recommend_learning_strategy(num_samples=50, budget_dollars=0, has_gpu=False, latency_critical=False)
    assert "Prompt Engineering" in rec["strategy"]


def test_recommend_learning_strategy_lora():
    rec = recommend_learning_strategy(num_samples=1000, budget_dollars=30, has_gpu=True, latency_critical=False)
    assert "LoRA" in rec["strategy"] or "PEFT" in rec["strategy"]
