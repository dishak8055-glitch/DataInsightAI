# modules/multimodal.py

import os
import io
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False


def generate_audio_summary(text, output_path=None):
    """
    Synthesizes speech from dataset insights using pyttsx3 (Audio Modality).
    Returns the file path or bytes of the generated .wav audio file.
    """
    if not HAS_PYTTSX3:
        raise RuntimeError("pyttsx3 is not installed or available.")

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    engine = pyttsx3.init()
    engine.setProperty("rate", 160)  # Moderate speaking rate
    engine.setProperty("volume", 0.95)

    # Clean text of markdown formatting symbols before synthesis
    clean_text = text.replace("**", "").replace("*", "").replace("#", "").replace("-", " ")
    clean_text = " ".join(clean_text.split())

    engine.save_to_file(clean_text, output_path)
    engine.runAndWait()
    engine.stop()

    with open(output_path, "rb") as f:
        audio_bytes = f.read()

    return {
        "file_path": output_path,
        "audio_bytes": audio_bytes,
        "format": "audio/wav"
    }


def generate_dataset_infographic_card(df, target_column=None, recommended_task="Supervised Learning"):
    """
    Generates a visual Dataset Infographic Card (Image Modality)
    summarizing key dataset metrics and distributions using Matplotlib and Pillow.
    Returns PNG image bytes.
    """
    # Create dark-theme high-resolution card
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor("#0e1117")

    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    # 1. Top metrics card (Text)
    ax_meta = axes[0, 0]
    ax_meta.axis("off")
    total_missing = int(df.isnull().sum().sum())
    meta_text = (
        f"DATASET CARD: DataInsight AI\n\n"
        f"• Total Rows: {len(df):,}\n"
        f"• Total Columns: {len(df.columns)}\n"
        f"• Missing Values: {total_missing:,}\n"
        f"• Target Column: {target_column or 'None (Clustering)'}\n"
        f"• Recommended Task: {recommended_task}"
    )
    ax_meta.text(
        0.05, 0.5, meta_text,
        color="#58a6ff", fontsize=10, family="sans-serif",
        verticalalignment="center", linespacing=1.6
    )

    # 2. Missing Value bar chart
    ax_miss = axes[0, 1]
    missing_series = df.isnull().sum()
    top_missing = missing_series[missing_series > 0].head(5)
    if not top_missing.empty:
        top_missing.plot(kind="barh", ax=ax_miss, color="#f85149")
        ax_miss.set_title("Missing Values by Feature", color="#c9d1d9", fontsize=9)
    else:
        ax_miss.text(0.5, 0.5, "Zero Missing Values\nDataset is 100% Complete",
                     color="#3fb950", fontsize=10, ha="center", va="center")
        ax_miss.set_title("Data Completeness", color="#c9d1d9", fontsize=9)

    # 3. First numerical column distribution
    ax_num = axes[1, 0]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        col = num_cols[0]
        data = df[col].dropna()
        ax_num.hist(data, bins=15, color="#1f6feb", edgecolor="#0e1117", alpha=0.85)
        ax_num.set_title(f"Distribution: {col}", color="#c9d1d9", fontsize=9)
    else:
        ax_num.text(0.5, 0.5, "No Numerical Columns", color="#8b949e", ha="center", va="center")

    # 4. First categorical column distribution
    ax_cat = axes[1, 1]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if cat_cols:
        col = cat_cols[0]
        counts = df[col].dropna().astype(str).value_counts().head(5)
        counts.plot(kind="bar", ax=ax_cat, color="#a371f7", edgecolor="#0e1117")
        ax_cat.set_title(f"Categories: {col}", color="#c9d1d9", fontsize=9)
        ax_cat.tick_params(axis="x", rotation=30)
    elif len(num_cols) > 1:
        col2 = num_cols[1]
        data2 = df[col2].dropna()
        ax_cat.hist(data2, bins=15, color="#238636", edgecolor="#0e1117", alpha=0.85)
        ax_cat.set_title(f"Distribution: {col2}", color="#c9d1d9", fontsize=9)
    else:
        ax_cat.text(0.5, 0.5, "Single Feature Dataset", color="#8b949e", ha="center", va="center")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def get_multimodal_genai_survey():
    """
    Returns comparative survey across Image, Video, Music, and Tabular Generative AI
    to satisfy the cross-modal curriculum requirement for Viva.
    """
    return [
        {
            "Modality": "Tabular Data (Our Primary Focus)",
            "Core Architecture": "Conditional Tabular GAN (CTGAN), TVAE",
            "Data Representation": "Continuous mixed distributions + One-hot discrete categorical columns",
            "Key Challenge": "Multimodal numerical distributions and mixed categorical column dependencies",
            "Evaluation Metric": "Kolmogorov-Smirnov (KS) test, Wasserstein distance, Machine Learning Efficacy (TSTR)"
        },
        {
            "Modality": "Image Generation",
            "Core Architecture": "Denoising Diffusion Probabilistic Models (DDPM), Latent Diffusion (Stable Diffusion)",
            "Data Representation": "Pixel grid tensors in continuous spatial domain (C x H x W)",
            "Key Challenge": "High computational cost of iterative denoising reverse Markov chain",
            "Evaluation Metric": "Fréchet Inception Distance (FID), Inception Score (IS)"
        },
        {
            "Modality": "Video Generation",
            "Core Architecture": "Diffusion Transformers (DiT), Spatio-temporal Attention (e.g. OpenAI Sora, CogVideo)",
            "Data Representation": "Sequences of video frame latents (T x C x H x W) with 3D patchify",
            "Key Challenge": "Temporal coherence, physical consistency, and dynamic motion drift",
            "Evaluation Metric": "FVD (Fréchet Video Distance), Subject Consistency Score"
        },
        {
            "Modality": "Music / Audio Generation",
            "Core Architecture": "Neural Audio Codecs (EnCodec) + Autoregressive / Diffusion (MusicLM, AudioCraft)",
            "Data Representation": "1D raw waveforms or 2D Mel-spectrograms converted to discrete acoustic tokens",
            "Key Challenge": "Long-range harmonic structure, rhythmic timing, and polyphonic instrument balance",
            "Evaluation Metric": "FAD (Fréchet Audio Distance), Mel-Spectral Distortion"
        }
    ]
