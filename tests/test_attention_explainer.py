# tests/test_attention_explainer.py

from modules.attention_explainer import (
    compute_scaled_dot_product_attention,
    get_gan_vs_vae_breakdown
)


def test_scaled_dot_product_attention():
    tokens = ["Age", "Income", "SpendingScore"]
    res = compute_scaled_dot_product_attention(tokens, d_k=8)
    assert res is not None
    assert res["attention_weights"].shape == (3, 3)
    # Each row in attention weights must sum to 1.0 (softmax property)
    row_sums = res["attention_weights"].sum(axis=1)
    for s in row_sums:
        assert round(float(s), 2) == 1.0
    assert res["figure"] is not None


def test_gan_vs_vae_breakdown():
    breakdown = get_gan_vs_vae_breakdown()
    assert "gan" in breakdown
    assert "vae" in breakdown
    assert "minimax_formula" in breakdown["gan"]
    assert "elbo_formula" in breakdown["vae"]
