# modules/attention_explainer.py

import numpy as np
import pandas as pd
import plotly.express as px


def compute_scaled_dot_product_attention(tokens, d_k=8, seed=42):
    """
    Simulates and computes the exact Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax((Q @ K.T) / sqrt(d_k)) @ V

    Returns attention weight matrix, Q, K, V matrices and summary.
    """
    np.random.seed(seed)
    n = len(tokens)
    if n == 0:
        return None, None

    # Deterministic pseudo-embedding projection for columns/tokens
    # Each token receives an embedding based on character codes
    embeddings = np.zeros((n, d_k))
    for i, tok in enumerate(tokens):
        chars = [ord(c) for c in str(tok)[:d_k]]
        while len(chars) < d_k:
            chars.append(32)
        embeddings[i] = np.array(chars[:d_k], dtype=float)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    # Simulated Projection Matrices W_Q, W_K, W_V
    W_Q = np.random.randn(d_k, d_k) * 0.5
    W_K = np.random.randn(d_k, d_k) * 0.5
    W_V = np.random.randn(d_k, d_k) * 0.5

    Q = embeddings @ W_Q
    K = embeddings @ W_K
    V = embeddings @ W_V

    # Raw attention scores: (Q @ K.T) / sqrt(d_k)
    scores = (Q @ K.T) / np.sqrt(d_k)

    # Softmax over last dimension (rows)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Context vectors
    context_vectors = attention_weights @ V

    # Build Plotly heatmap
    fig = px.imshow(
        attention_weights,
        x=tokens,
        y=tokens,
        labels=dict(x="Key (Referenced Column)", y="Query (Focusing Column)", color="Attention Weight"),
        color_continuous_scale="Viridis",
        zmin=0.0,
        zmax=1.0,
        title="Interactive Self-Attention Weight Matrix Softmax(Q Kᵀ / √dₖ)",
        template="plotly_dark",
        text_auto=".2f"
    )

    fig.update_layout(
        xaxis_title="Key Columns (K)",
        yaxis_title="Query Columns (Q)",
        height=450
    )

    return {
        "attention_weights": np.round(attention_weights, 4),
        "figure": fig,
        "tokens": tokens,
        "d_k": d_k,
        "formula": "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V"
    }


def get_gan_vs_vae_breakdown():
    """
    Returns mathematical and architectural comparison between GANs and VAEs.
    """
    gan_info = {
        "architecture": "Generative Adversarial Network (GAN)",
        "objective": "Two-player zero-sum minimax game between Generator G and Discriminator D.",
        "minimax_formula": "\\min_G \\max_D V(D, G) = \\mathbb{E}_{x \\sim p_{data}}[\\log D(x)] + \\mathbb{E}_{z \\sim p_z}[\\log(1 - D(G(z)))]",
        "latent_space": "Unstructured prior z ~ N(0, I) or Uniform(-1, 1). No explicit likelihood estimation.",
        "training_dynamics": "Adversarial gradients. Equilibrium at D(x) = 0.5 (Nash equilibrium).",
        "strengths": [
            "Produces sharp, high-fidelity synthetic samples.",
            "Applicable to tabular data via conditional generator (CTGAN).",
            "Does not require explicit tractability of likelihood P(x)."
        ],
        "challenges": [
            "Susceptible to Mode Collapse (generator outputs only a few repeated rows).",
            "Training instability (vanishing/exploding gradients without Wasserstein WGAN-GP).",
            "No direct encoder: cannot easily map an existing record x back to its latent code z."
        ],
        "viva_summary": "GANs learn implicitly through competition. Generator invents data; Discriminator spots fakes."
    }

    vae_info = {
        "architecture": "Variational Autoencoder (VAE)",
        "objective": "Maximize the Evidence Lower Bound (ELBO) on data log-likelihood log P(x).",
        "elbo_formula": "\\mathcal{L}_{ELBO}(\\theta, \\phi; x) = \\mathbb{E}_{q_\\phi(z|x)}[\\log p_\\theta(x|z)] - D_{KL}(q_\\phi(z|x) \\parallel p(z))",
        "latent_space": "Smooth, continuous probabilistic manifold z ~ N(mu, sigma^2). Allows semantic interpolation.",
        "training_dynamics": "Gradient descent via the Reparameterization Trick: z = mu + sigma * epsilon, epsilon ~ N(0, I).",
        "strengths": [
            "Stable training via standard backpropagation (no adversarial instability).",
            "Principled probabilistic framework with closed-form KL divergence.",
            "Bidirectional: Encoder maps x -> z, Decoder reconstructs z -> x."
        ],
        "challenges": [
            "Reconstructions can be blurry or over-smoothed due to the Gaussian prior assumption.",
            "Requires balancing reconstruction loss against KL divergence (beta-VAE)."
        ],
        "viva_summary": "VAEs learn explicitly via density approximation. Encoder compresses to (mu, sigma); Decoder reconstructs."
    }

    return {"gan": gan_info, "vae": vae_info}
