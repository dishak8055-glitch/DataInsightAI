# modules/synthetic_data.py

import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ks_2samp
import plotly.express as px
import plotly.graph_objects as go


def is_sdv_backend_enabled():
    """Require explicit opt-in before exposing SDV synthesis in the app."""
    return os.getenv("DATAINSIGHT_ENABLE_SDV", "").strip().lower() in {
        "1", "true", "yes"
    }


def is_identifier_column(column):
    """Return whether a column name clearly identifies an identifier field."""
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(column))
    tokens = re.findall(r"[A-Za-z0-9]+", name.lower())
    return any(token in {"id", "identifier", "uuid", "guid"} for token in tokens)


def sanitize_spreadsheet_formulas(df):
    """Prefix risky text cells so spreadsheet apps do not evaluate them as formulas."""
    sanitized = df.copy()
    for column in sanitized.columns:
        dtype = sanitized[column].dtype
        if (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            sanitized[column] = sanitized[column].map(
                lambda value: (
                    "'" + value
                    if isinstance(value, str)
                    and (
                        value.startswith(("\t", "\r", "\n"))
                        or value.lstrip(" \t\r\n")[:1] in {"=", "+", "-", "@"}
                    )
                    else value
                )
            )
    return sanitized


def _get_sdv_ctgan_components():
    try:
        from sdv.metadata import Metadata
        from sdv.single_table import CTGANSynthesizer
    except ImportError as exc:
        raise ImportError(
            "SDV is required for the SDV CTGAN backend. Install the project requirements."
        ) from exc
    return Metadata, CTGANSynthesizer


class SDVCTGANSynthesizer:
    """Wrapper around SDV's actual CTGANSynthesizer with explicit validation."""

    def __init__(self, epochs=50, batch_size=100, pac=10):
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("Training epochs must be a positive integer.")
        if not isinstance(pac, int) or pac < 1:
            raise ValueError("pac must be a positive integer.")
        if not isinstance(batch_size, int) or batch_size < 1 or batch_size % pac:
            raise ValueError("Batch size must be a positive multiple of pac.")
        self.epochs = epochs
        self.batch_size = batch_size
        self.pac = pac
        self.synthesizer = None
        self.columns = None
        self.is_fitted = False

    def fit(self, df):
        """Fit SDV CTGAN without dropping rows or columns from the training data."""
        self.is_fitted = False
        if not isinstance(df, pd.DataFrame) or df.empty or len(df.columns) == 0:
            raise ValueError(
                "SDV CTGAN training requires at least one row and one column."
            )
        if len(df) < 2:
            raise ValueError("SDV CTGAN training requires at least two rows.")
        if df.columns.has_duplicates:
            raise ValueError("SDV CTGAN requires unique column names.")
        if df.isna().all(axis=0).any():
            raise ValueError(
                "SDV CTGAN cannot infer columns that contain only missing values. "
                "Resolve those columns explicitly before training."
            )

        training_data = df.copy(deep=True)
        numeric_columns = training_data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            training_data[column] = training_data[column].replace(
                [np.inf, -np.inf], np.nan
            )

        try:
            Metadata, CTGANSynthesizer = _get_sdv_ctgan_components()
            metadata = Metadata.detect_from_dataframe(
                training_data,
                table_name="dataset"
            )
            table_columns = metadata.to_dict()["tables"]["dataset"]["columns"]
            for column in training_data.columns:
                dtype = training_data[column].dtype
                if (
                    pd.api.types.is_object_dtype(dtype)
                    or pd.api.types.is_string_dtype(dtype)
                    or isinstance(dtype, pd.CategoricalDtype)
                    or pd.api.types.is_bool_dtype(dtype)
                ) and table_columns[column]["sdtype"] != "id":
                    metadata.update_column(
                        column,
                        table_name="dataset",
                        sdtype="categorical"
                    )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="We strongly recommend saving the metadata"
                )
                warnings.filterwarnings(
                    "ignore",
                    message=".*cuda.*deprecated.*",
                    category=FutureWarning
                )
                self.synthesizer = CTGANSynthesizer(
                    metadata,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    pac=self.pac,
                    cuda=False,
                    verbose=False,
                    enforce_min_max_values=True,
                    enforce_rounding=True
                )
                self.synthesizer.fit(training_data)
        except ImportError as exc:
            self.synthesizer = None
            raise RuntimeError(str(exc)) from exc
        except Exception:
            self.synthesizer = None
            raise RuntimeError(
                "SDV CTGAN training failed. Check the input schema and values; "
                "no alternate backend was used."
            ) from None

        self.columns = list(df.columns)
        self.is_fitted = True
        return self

    def sample(self, n_samples=20):
        """Generate rows with the original input schema using SDV CTGAN."""
        if not self.is_fitted or self.synthesizer is None:
            raise RuntimeError("SDV CTGAN must be fitted before sampling.")
        if not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("Number of synthetic rows must be a positive integer.")

        try:
            synthetic = self.synthesizer.sample(num_rows=n_samples)
        except Exception:
            raise RuntimeError(
                "SDV CTGAN sampling failed; no alternate backend was used."
            ) from None

        if synthetic.columns.tolist() != self.columns:
            raise RuntimeError(
                "SDV CTGAN returned an output schema that differs from the training data."
            )
        if len(synthetic) != n_samples:
            raise RuntimeError(
                "SDV CTGAN returned a different number of rows than requested."
            )
        return synthetic


class TabularGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, condition_dim=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, z, condition=None):
        inputs = torch.cat((z, condition), dim=1) if condition is not None else z
        return self.net(inputs)


class TabularDiscriminator(nn.Module):
    def __init__(self, input_dim, condition_dim=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, condition=None):
        inputs = torch.cat((x, condition), dim=1) if condition is not None else x
        return self.net(inputs)


class TabularCTGAN:
    """
    Lightweight conditional tabular GAN demonstration implemented in PyTorch.
    It uses one-hot discrete columns and category conditioning; it is not the
    canonical CTGAN implementation with mode-specific continuous normalization.
    """
    def __init__(self, latent_dim=16, epochs=60, batch_size=16, lr=0.001):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_cols = []
        self.integer_cols = []
        self.cat_cols = []
        self.cat_mappings = {}
        self.num_mins = {}
        self.num_maxs = {}
        self.columns = []
        self.feature_slices = {}
        self.condition_slices = {}
        self.category_counts = {}
        self.condition_dim = 0
        self.output_columns = []
        self.id_cols = []
        self.generator = None
        self.discriminator = None
        self.is_fitted = False

    def _preprocess(self, df):
        if df.empty or len(df.columns) == 0:
            raise ValueError("Synthetic training requires at least one row and one column.")

        self.output_columns = list(df.columns)
        self.id_cols = [col for col in df.columns if is_identifier_column(col)]
        self.columns = [col for col in df.columns if col not in self.id_cols]
        if not self.columns:
            raise ValueError("Synthetic training requires at least one non-identifier column.")
        self.num_cols = []
        self.integer_cols = []
        self.cat_cols = []
        self.cat_mappings = {}
        self.num_mins = {}
        self.num_maxs = {}
        self.feature_slices = {}
        self.condition_slices = {}
        self.category_counts = {}
        arrays = []
        feature_offset = 0
        condition_offset = 0

        for col in self.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.num_cols.append(col)
                original_values = pd.to_numeric(df[col], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                if (
                    not original_values.empty
                    and np.isclose(original_values, np.round(original_values)).all()
                ):
                    self.integer_cols.append(col)
                series = pd.to_numeric(df[col], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                median = series.median()
                series = series.fillna(0.0 if pd.isna(median) else median)
                col_min = float(series.min())
                col_max = float(series.max())
                self.num_mins[col] = col_min
                self.num_maxs[col] = col_max
                if col_max == col_min:
                    scaled = np.zeros(len(series), dtype=np.float32)
                else:
                    scaled = 2.0 * (series - col_min) / (col_max - col_min) - 1.0
                arrays.append(np.asarray(scaled, dtype=np.float32).reshape(-1, 1))
                self.feature_slices[col] = slice(feature_offset, feature_offset + 1)
                feature_offset += 1
            else:
                self.cat_cols.append(col)
                series = df[col].astype("string").fillna("Missing").astype(str)
                unique_cats = list(series.unique())
                self.cat_mappings[col] = unique_cats
                mapping = {category: index for index, category in enumerate(unique_cats)}
                encoded = series.map(mapping).to_numpy()
                one_hot = np.eye(len(unique_cats), dtype=np.float32)[encoded]
                arrays.append(one_hot)
                self.feature_slices[col] = slice(
                    feature_offset, feature_offset + len(unique_cats)
                )
                self.condition_slices[col] = slice(
                    condition_offset, condition_offset + len(unique_cats)
                )
                self.category_counts[col] = series.value_counts(normalize=True).reindex(
                    unique_cats
                ).to_numpy(dtype=np.float64)
                feature_offset += len(unique_cats)
                condition_offset += len(unique_cats)

        self.condition_dim = condition_offset
        return np.concatenate(arrays, axis=1).astype(np.float32)

    def _activate(self, logits):
        parts = []
        for col in self.columns:
            segment = logits[:, self.feature_slices[col]]
            if col in self.num_cols:
                parts.append(torch.tanh(segment))
            else:
                parts.append(torch.softmax(segment, dim=1))
        return torch.cat(parts, dim=1)

    def _sample_conditions(self, real_batch):
        batch_size = real_batch.size(0)
        conditions = torch.zeros(batch_size, self.condition_dim)
        selected_columns = torch.randint(len(self.cat_cols), (batch_size,))
        selected_categories = torch.zeros(batch_size, dtype=torch.long)

        for column_index, col in enumerate(self.cat_cols):
            rows = torch.nonzero(selected_columns == column_index, as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            feature_slice = self.feature_slices[col]
            condition_slice = self.condition_slices[col]
            categories = real_batch[rows, feature_slice].argmax(dim=1)
            conditions[rows, condition_slice.start + categories] = 1.0
            selected_categories[rows] = categories
        return conditions, selected_columns, selected_categories

    def fit(self, df):
        """Train the Generator and Discriminator on the input dataframe."""
        self.is_fitted = False
        X_train = self._preprocess(df)
        n_samples, n_features = X_train.shape

        torch.manual_seed(42)
        self.generator = TabularGenerator(
            self.latent_dim, n_features, self.condition_dim
        )
        self.discriminator = TabularDiscriminator(n_features, self.condition_dim)

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
                conditions = None
                selected_columns = selected_categories = None
                if self.cat_cols:
                    (
                        conditions,
                        selected_columns,
                        selected_categories
                    ) = self._sample_conditions(real_batch)

                # 1. Train Discriminator
                optimizer_d.zero_grad()
                d_real_loss = criterion(
                    self.discriminator(real_batch, conditions), real_labels
                )

                z = torch.randn(b_size, self.latent_dim)
                fake_logits = self.generator(z, conditions)
                fake_data = self._activate(fake_logits)
                d_fake_loss = criterion(
                    self.discriminator(fake_data.detach(), conditions), fake_labels
                )

                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                optimizer_d.step()

                # 2. Train Generator
                optimizer_g.zero_grad()
                g_loss = criterion(
                    self.discriminator(fake_data, conditions), real_labels
                )
                if self.cat_cols:
                    category_logits = torch.zeros(
                        b_size, max(len(self.cat_mappings[col]) for col in self.cat_cols)
                    )
                    for column_index, col in enumerate(self.cat_cols):
                        rows = torch.nonzero(
                            selected_columns == column_index, as_tuple=True
                        )[0]
                        if rows.numel():
                            segment = self.feature_slices[col]
                            category_logits[rows, :segment.stop - segment.start] = (
                                fake_logits[rows, segment]
                            )
                    g_loss = g_loss + nn.functional.cross_entropy(
                        category_logits, selected_categories
                    )
                g_loss.backward()
                optimizer_g.step()

        self.is_fitted = True
        return self

    def sample(self, n_samples=20):
        """Generate n_samples synthetic rows matching the original dataset schema."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before sampling.")
        if n_samples < 1:
            raise ValueError("n_samples must be at least one.")

        self.generator.eval()
        conditions = None
        if self.cat_cols:
            conditions = torch.zeros(n_samples, self.condition_dim)
            selected_columns = torch.randint(len(self.cat_cols), (n_samples,))
            for column_index, col in enumerate(self.cat_cols):
                rows = torch.nonzero(selected_columns == column_index, as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                category_indices = np.random.choice(
                    len(self.cat_mappings[col]),
                    size=rows.numel(),
                    p=self.category_counts[col]
                )
                condition_slice = self.condition_slices[col]
                conditions[rows, condition_slice.start + torch.as_tensor(category_indices)] = 1.0
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim)
            fake_logits = self.generator(z, conditions)
            fake_tensor = self._activate(fake_logits).numpy()

        synthetic_dict = {}
        for col in self.columns:
            values = fake_tensor[:, self.feature_slices[col]]
            if col in self.num_cols:
                c_min = self.num_mins[col]
                c_max = self.num_maxs[col]
                if c_max == c_min:
                    generated = np.full(n_samples, c_min)
                else:
                    unscaled = (np.clip(values[:, 0], -1, 1) + 1) / 2 * (c_max - c_min) + c_min
                    generated = np.clip(unscaled, c_min, c_max)
                if col in self.integer_cols:
                    generated = np.rint(np.clip(generated, c_min, c_max)).astype(np.int64)
                else:
                    generated = np.round(generated, 2)
                synthetic_dict[col] = generated
            else:
                categories = self.cat_mappings[col]
                indices = values.argmax(axis=1)
                synthetic_dict[col] = [categories[index] for index in indices]
        for col in self.id_cols:
            synthetic_dict[col] = [
                f"SYNTHETIC_{col}_{index + 1}" for index in range(n_samples)
            ]
        return pd.DataFrame(synthetic_dict, columns=self.output_columns)


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
    evaluated_cols = []
    common_num_cols = [
        c for c in real_df.columns
        if (
            c in synth_df.columns
            and not is_identifier_column(c)
            and pd.api.types.is_numeric_dtype(real_df[c])
        )
    ]

    for col in common_num_cols:
        r_series = pd.to_numeric(real_df[col], errors="coerce").dropna()
        s_series = pd.to_numeric(synth_df[col], errors="coerce").dropna()

        if len(r_series) < 2 or len(s_series) < 2:
            continue

        ks_stat, _ = ks_2samp(r_series, s_series)
        ks_scores.append(1.0 - ks_stat)  # Higher is better: 1 - KS distance
        evaluated_cols.append(col)

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
    corr_similarity = None
    if len(evaluated_cols) >= 2:
        r_corr = real_df[evaluated_cols].corr().fillna(0).values
        s_corr = synth_df[evaluated_cols].corr().fillna(0).values
        frob_diff = np.linalg.norm(r_corr - s_corr)
        max_frob = np.sqrt(len(evaluated_cols) ** 2 * 4)
        corr_similarity = max(0.0, min(100.0, (1.0 - frob_diff / max_frob) * 100))

    avg_ks_score = float(np.mean(ks_scores) * 100) if ks_scores else None
    if avg_ks_score is None:
        aggregate_score = None
    elif corr_similarity is None:
        aggregate_score = round(avg_ks_score, 1)
    else:
        aggregate_score = round(0.7 * avg_ks_score + 0.3 * corr_similarity, 1)

    return {
        "metrics_table": pd.DataFrame(metrics),
        "aggregate_quality_score": aggregate_score,
        "correlation_similarity_score": (
            round(corr_similarity, 1) if corr_similarity is not None else None
        ),
        "num_features_evaluated": len(evaluated_cols)
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
        name="Conditional GAN Synthetic Data",
        opacity=0.75,
        marker_color="#3fb950"
    ))

    fig.update_layout(
        barmode="overlay",
        title=f"Real vs Conditional GAN Synthetic Distribution: {column}",
        xaxis_title=column,
        yaxis_title="Count",
        template="plotly_dark",
        height=400
    )

    return fig
