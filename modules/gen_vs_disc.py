# modules/gen_vs_disc.py

import pandas as pd
import numpy as np


def analyze_learning_paradigms(df, target_column=None):
    """
    Analyzes the uploaded dataset from the perspective of Generative vs Discriminative learning,
    and identifies the learning paradigm (Supervised, Unsupervised, Self-Supervised, Generative).
    """
    total_rows = len(df)
    total_cols = len(df.columns)

    if target_column and target_column in df.columns:
        target_series = df[target_column]
        is_numeric = pd.api.types.is_numeric_dtype(target_series)
        unique_vals = target_series.nunique(dropna=True)

        if is_numeric and unique_vals > 10:
            task_type = "Regression"
            sub_type = "Continuous target prediction"
        else:
            task_type = "Classification"
            sub_type = f"Discrete class prediction ({unique_vals} classes)"

        primary_paradigm = "Supervised Learning"
        paradigm_details = (
            f"The dataset has a designated target feature '{target_column}'. "
            f"Supervised learning algorithms infer a mapping function from input features X to output Y."
        )

        discriminative_approach = {
            "name": "Discriminative Modeling",
            "objective": "Model the conditional probability P(Y | X) directly.",
            "goal": "Focus on the decision boundary separating classes or regression line.",
            "algorithms": (
                ["Logistic Regression", "Random Forest Classifier", "XGBoost", "Support Vector Machines (SVM)"]
                if task_type == "Classification"
                else ["Linear Regression", "Ridge/Lasso", "Random Forest Regressor", "Gradient Boosting Regressor"]
            ),
            "data_efficiency": "High sample efficiency for prediction. Does not model how X was generated.",
            "key_formula": "P(Y|X) = \\frac{P(X, Y)}{P(X)} \\quad \\text{or directly via } f(X; \\theta)",
            "when_to_use": (
                f"Use when the primary goal is high-accuracy prediction of '{target_column}' "
                "given observed features."
            )
        }

        generative_approach = {
            "name": "Generative Modeling",
            "objective": "Model the joint probability P(X, Y) = P(X | Y) P(Y).",
            "goal": "Capture the complete data generation mechanism and feature distributions.",
            "algorithms": [
                "Gaussian Naive Bayes (Class-conditional density P(X|Y))",
                "Linear Discriminant Analysis (LDA)",
                "Conditional GAN / CTGAN (Generates synthetic (X, Y) pairs)",
                "Variational Autoencoder (VAE) with label conditioning"
            ],
            "data_efficiency": "Requires more data to accurately estimate joint distribution, but can generate synthetic samples.",
            "key_formula": "P(X, Y) = P(X|Y)P(Y) \\implies P(Y|X) = \\frac{P(X|Y)P(Y)}{\\sum_{y'} P(X|y')P(y')}",
            "when_to_use": (
                "Use when you want to handle missing values naturally, detect outliers, "
                "or generate synthetic patient/customer records preserving class relationships."
            )
        }

    else:
        primary_paradigm = "Unsupervised / Generative Learning"
        task_type = "Clustering & Density Estimation"
        sub_type = "Pattern discovery without target labels"
        paradigm_details = (
            "No target column is selected. The objective is to discover underlying structures, "
            "clusters, manifold projections, or model the data distribution P(X)."
        )

        discriminative_approach = {
            "name": "Discriminative Clustering / Representation",
            "objective": "Partition feature space into discrete partitions.",
            "goal": "Maximize inter-cluster distance while minimizing intra-cluster variance.",
            "algorithms": ["K-Means Clustering", "DBSCAN", "Spectral Clustering", "PCA (Principal Component Analysis)"],
            "data_efficiency": "Directly minimizes geometric clustering objectives.",
            "key_formula": "\\min_S \\sum_{i=1}^k \\sum_{x \\in S_i} ||x - \\mu_i||^2",
            "when_to_use": "Use when grouping similar rows into customer segments or cohorts."
        }

        generative_approach = {
            "name": "Unconditional Generative Modeling",
            "objective": "Model the true data distribution P(X).",
            "goal": "Generate novel realistic rows or estimate data likelihood.",
            "algorithms": [
                "Gaussian Mixture Models (GMM)",
                "CTGAN (Tabular Generative Adversarial Network)",
                "Variational Autoencoders (Tabular VAE)",
                "Kernel Density Estimation (KDE)"
            ],
            "data_efficiency": "Learns the continuous multi-modal distribution of all tabular features.",
            "key_formula": "\\max_\\theta \\mathbb{E}_{x \\sim P_{data}}[\\log P_\\theta(x)]",
            "when_to_use": "Use when generating synthetic data for privacy protection or data augmentation."
        }

    lifecycle_stages = [
        {
            "step": "1. Problem Formulation & Data Scope",
            "description": "Define business objective, identify target variable (or generative goal), and audit data constraints.",
            "viva_insight": "Clarifies whether the task requires predictive modeling P(Y|X) or synthetic augmentation P(X)."
        },
        {
            "step": "2. Data Profiling & Quality Assessment",
            "description": "Inspect missing values, duplicates, outliers, class balance, and PII/sensitive attributes.",
            "viva_insight": "Garbage in, garbage out: generative models will replicate data errors and biases if not cleaned."
        },
        {
            "step": "3. Representation & Feature Engineering",
            "description": "Encode categorical variables, normalize continuous features, and handle datetime/ID columns.",
            "viva_insight": "Tabular GANs require specific transformations (e.g. mode-specific normalization for multimodal distributions)."
        },
        {
            "step": "4. Model Paradigm Selection",
            "description": "Select Discriminative baseline (e.g. Random Forest) or Generative pipeline (CTGAN / VAE).",
            "viva_insight": "Always build a discriminative baseline first to benchmark predictive utility before applying generative models."
        },
        {
            "step": "5. Training & Convergence Monitoring",
            "description": "Train models with cross-validation. For GANs, monitor generator vs discriminator minimax loss to avoid mode collapse.",
            "viva_insight": "GAN equilibrium occurs when the discriminator cannot distinguish real from fake (D(x) = 0.5)."
        },
        {
            "step": "6. Evaluation & Responsible AI Audit",
            "description": "Compute statistical fidelity (KS test, correlation preservation), fairness across sensitive slices, and hallucination checks.",
            "viva_insight": "Evaluates three pillars: Fidelity (accuracy), Diversity (no mode collapse), and Privacy (no memorization)."
        },
        {
            "step": "7. Deployment, Model Card & Monitoring",
            "description": "Package into an interactive interface (Streamlit), export Hugging Face Model Card, and monitor for data drift.",
            "viva_insight": "Model Cards communicate intended use, limitations, and ethical guardrails to stakeholders."
        }
    ]

    return {
        "primary_paradigm": primary_paradigm,
        "task_type": task_type,
        "sub_type": sub_type,
        "paradigm_details": paradigm_details,
        "discriminative": discriminative_approach,
        "generative": generative_approach,
        "lifecycle_stages": lifecycle_stages
    }
