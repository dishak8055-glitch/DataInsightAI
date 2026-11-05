# modules/synthetic_data.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ks_2samp
import plotly.express as px
import plotly.graph_objects as go


class TabularGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim),
            nn.Tanh()  # Outputs bounded in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


class TabularDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class TabularCTGAN:
    """
    Lightweight, fast Tabular Generative Adversarial Network in PyTorch.
    Learns continuous and categorical distributions directly on CPU.
    """
    def __init__(self, latent_dim=16, epochs=60, batch_size=16, lr=0.001):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_cols = []
        self.cat_cols = []
        self.cat_mappings = {}
        self.num_mins = {}
        self.num_maxs = {}
        self.generator = None
        self.discriminator = None
        self.is_fitted = False

    def _preprocess(self, df):
        processed = df.copy()
        self.num_cols = []
        self.cat_cols = []
        feature_columns = []

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.num_cols.append(col)
                series = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median() if not df[col].empty else 0)
                col_min = float(series.min())
                col_max = float(series.max())
                if col_max == col_min:
                    col_max += 1.0
                self.num_mins[col] = col_min
                self.num_maxs[col] = col_max
                # Scale to [-1, 1]
                scaled = 2.0 * (series - col_min) / (col_max - col_min) - 1.0
                processed[col] = scaled
                feature_columns.append(col)
            else:
                self.cat_cols.append(col)
                series = df[col].astype(str).fillna("Missing")
                unique_cats = list(series.unique())
                self.cat_mappings[col] = unique_cats
                # Simple integer encoding scaled to [-1, 1]
                mapping = {cat: i for i, cat in enumerate(unique_cats)}
                n_cats = max(1, len(unique_cats) - 1)
                encoded = series.map(mapping)
                scaled = 2.0 * (encoded / n_cats) - 1.0 if n_cats > 0 else np.zeros(len(series))
                processed[col] = scaled
                feature_columns.append(col)

        return processed[feature_columns].values.astype(np.float32)

    def fit(self, df):
        """Train the Generator and Discriminator on the input dataframe."""
        X_train = self._preprocess(df)
        n_samples, n_features = X_train.shape

        torch.manual_seed(42)
        self.generator = TabularGenerator(self.latent_dim, n_features)
        self.discriminator = TabularDiscriminator(n_features)

        criterion = nn.BCELoss()
        optimizer_g = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

        tensor_data = torch.tensor(X_train)
        dataset = torch.utils.data.TensorDataset(tensor_data)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, n_samples),
            shuffle=True
        )

        for _ in range(self.epochs):
            for real_batch, in dataloader:
                b_size = real_batch.size(0)
                real_labels = torch.ones(b_size, 1)
                fake_labels = torch.zeros(b_size, 1)

                # 1. Train Discriminator
                optimizer_d.zero_grad()
                d_real_loss = criterion(self.discriminator(real_batch), real_labels)

                z = torch.randn(b_size, self.latent_dim)
                fake_data = self.generator(z)
                d_fake_loss = criterion(self.discriminator(fake_data.detach()), fake_labels)

                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                optimizer_d.step()

                # 2. Train Generator
                optimizer_g.zero_grad()
                g_loss = criterion(self.discriminator(fake_data), real_labels)
                g_loss.backward()
                optimizer_g.step()

        self.is_fitted = True
        return self

    def sample(self, n_samples=20):
        """Generate n_samples synthetic rows matching the original dataset schema."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before sampling.")

        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim)
            fake_tensor = self.generator(z).numpy()

        synthetic_dict = {}
        feature_idx = 0

        # Unscale numerical columns
        for col in self.num_cols:
            scaled_vals = fake_tensor[:, feature_idx]
            # Clip to [-1, 1]
            scaled_vals = np.clip(scaled_vals, -1.0, 1.0)
            c_min = self.num_mins[col]
            c_max = self.num_maxs[col]
            unscaled = (scaled_vals + 1.0) / 2.0 * (c_max - c_min) + c_min
            synthetic_dict[col] = np.round(unscaled, 2)
            feature_idx += 1

        # Unscale categorical columns
        for col in self.cat_cols:
            scaled_vals = fake_tensor[:, feature_idx]
            scaled_vals = np.clip(scaled_vals, -1.0, 1.0)
            cats = self.cat_mappings[col]
            n_cats = max(1, len(cats) - 1)
            cat_indices = np.round((scaled_vals + 1.0) / 2.0 * n_cats).astype(int)
            cat_indices = np.clip(cat_indices, 0, len(cats) - 1)
            synthetic_dict[col] = [cats[i] for i in cat_indices]
            feature_idx += 1

        return pd.DataFrame(synthetic_dict)


def compare_real_and_synthetic(real_df, synth_df):
    """
    Evaluates fidelity and statistical similarity between real and synthetic data.
    Computes:
    - Kolmogorov-Smirnov (KS) test per numerical column
    - Mean and Standard Deviation divergence
    - Correlation matrix similarity (Frobenius distance)
    - Aggregate Synthetic Quality Score (0 - 100%)
    """
    metrics = []
    ks_scores = []
    common_num_cols = [
        c for c in real_df.columns
        if c in synth_df.columns and pd.api.types.is_numeric_dtype(real_df[c])
    ]

    for col in common_num_cols:
        r_series = pd.to_numeric(real_df[col], errors="coerce").dropna()
        s_series = pd.to_numeric(synth_df[col], errors="coerce").dropna()

        if len(r_series) < 2 or len(s_series) < 2:
            continue

        ks_stat, p_val = ks_2samp(r_series, s_series)
        ks_scores.append(1.0 - ks_stat)  # Higher is better: 1 - KS distance

        r_mean, s_mean = float(r_series.mean()), float(s_series.mean())
        r_std, s_std = float(r_series.std()), float(s_series.std())

        metrics.append({
            "Feature": col,
            "Real Mean": round(r_mean, 2),
            "Synth Mean": round(s_mean, 2),
            "Real Std": round(r_std, 2),
            "Synth Std": round(s_std, 2),
            "KS Statistic": round(float(ks_stat), 3),
            "Distribution Match (%)": round((1.0 - float(ks_stat)) * 100, 1)
        })

    # Correlation matrix difference
    corr_similarity = 85.0  # Baseline
    if len(common_num_cols) >= 2:
        r_corr = real_df[common_num_cols].corr().fillna(0).values
        s_corr = synth_df[common_num_cols].corr().fillna(0).values
        frob_diff = np.linalg.norm(r_corr - s_corr)
        max_frob = np.sqrt(len(common_num_cols)**2 * 4)  # Maximum possible distance
        corr_similarity = max(0.0, min(100.0, (1.0 - frob_diff / max_frob) * 100))

    avg_ks_score = (np.mean(ks_scores) * 100) if ks_scores else 80.0
    aggregate_score = round(0.7 * avg_ks_score + 0.3 * corr_similarity, 1)

    return {
        "metrics_table": pd.DataFrame(metrics),
        "aggregate_quality_score": aggregate_score,
        "correlation_similarity_score": round(corr_similarity, 1),
        "num_features_evaluated": len(common_num_cols)
    }


def create_side_by_side_distribution(real_df, synth_df, column):
    """Creates a side-by-side Plotly histogram comparing real vs synthetic distributions."""
    fig = go.Figure()

    r_data = real_df[column].dropna()
    s_data = synth_df[column].dropna()

    fig.add_trace(go.Histogram(
        x=r_data,
        name="Real Data",
        opacity=0.75,
        marker_color="#1f6feb"
    ))

    fig.add_trace(go.Histogram(
        x=s_data,
        name="CTGAN Synthetic Data",
        opacity=0.75,
        marker_color="#3fb950"
    ))

    fig.update_layout(
        barmode="overlay",
        title=f"Real vs CTGAN Synthetic Distribution: {column}",
        xaxis_title=column,
        yaxis_title="Count",
        template="plotly_dark",
        height=400
    )

    return fig
