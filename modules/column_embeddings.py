# modules/column_embeddings.py

import re
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering


DEFAULT_BERT_MODEL = "prajjwal1/bert-tiny"


def generate_column_context(df, col):
    """
    Constructs a rich semantic descriptive profile for a column,
    including decomposed name tokens, dtype, cardinality, and sample values.
    """
    col_str = str(col)
    # Split camelCase and snake_case into space-separated natural words
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", col_str)
    words = words.replace("_", " ").replace("-", " ").lower()

    series = df[col]
    dtype_str = str(series.dtype)
    n_unique = series.nunique(dropna=True)
    is_num = pd.api.types.is_numeric_dtype(series)

    # Generate summary tokens
    if is_num:
        stat_tokens = f"numerical continuous mean {round(float(series.mean()), 2) if not series.empty else 0} min {round(float(series.min()), 2) if not series.empty else 0} max {round(float(series.max()), 2) if not series.empty else 0}"
    else:
        sample_cats = " ".join([str(x) for x in series.dropna().unique()[:5]])
        stat_tokens = f"categorical discrete categories {sample_cats}"

    # Semantic description string for embedding model
    return f"{words} {words} column feature {dtype_str} unique {n_unique} {stat_tokens}"


@lru_cache(maxsize=4)
def _load_bert_encoder(model_name):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "BERT embeddings require the 'transformers' package. Install requirements.txt."
        ) from exc

    return AutoTokenizer.from_pretrained(model_name), AutoModel.from_pretrained(model_name)


@lru_cache(maxsize=32)
def _encode_contexts_with_bert(contexts, model_name):
    import torch

    tokenizer, model = _load_bert_encoder(model_name)
    model.eval()
    encoded = tokenizer(
        list(contexts),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


def group_similar_columns_bert(
    df,
    similarity_threshold=0.35,
    embedding_backend="tfidf",
    model_name=DEFAULT_BERT_MODEL
):
    """
    Groups columns using contextual BERT embeddings or a lightweight TF-IDF baseline.
    BERT weights are downloaded from Hugging Face on first use and cached locally.
    """
    columns = list(df.columns)
    if len(columns) < 2:
        return {
            "clusters": {0: columns},
            "similarity_matrix": pd.DataFrame(
                np.eye(len(columns)), index=columns, columns=columns
            ),
            "figure": None,
            "redundancy_alerts": [],
            "total_clusters": len(columns),
            "embedding_backend": embedding_backend
        }

    contexts = [generate_column_context(df, c) for c in columns]
    if embedding_backend == "bert":
        embeddings = _encode_contexts_with_bert(tuple(contexts), model_name)
        sim_matrix = cosine_similarity(embeddings)
        backend_label = f"BERT embeddings ({model_name})"
    elif embedding_backend == "tfidf":
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            analyzer="char_wb",
            min_df=1,
            sublinear_tf=True
        )
        tfidf_matrix = vectorizer.fit_transform(contexts)
        sim_matrix = cosine_similarity(tfidf_matrix)
        backend_label = "Character n-gram TF-IDF baseline (not a BERT model)"
    else:
        raise ValueError("embedding_backend must be either 'bert' or 'tfidf'.")

    sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
    np.fill_diagonal(sim_matrix, 1.0)

    sim_df = pd.DataFrame(sim_matrix, index=columns, columns=columns).round(3)

    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0.0)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - similarity_threshold
    )
    labels = clustering.fit_predict(dist_matrix)

    clusters = {}
    for col, label in zip(columns, labels):
        clusters.setdefault(int(label), []).append(col)

    redundancy_alerts = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            sim_score = sim_matrix[i, j]
            if sim_score >= 0.65:
                redundancy_alerts.append({
                    "col1": columns[i],
                    "col2": columns[j],
                    "similarity": round(float(sim_score), 3),
                    "note": f"High semantic similarity between '{columns[i]}' and '{columns[j]}'. Check for multi-collinearity or redundant encoding."
                })

    fig = px.imshow(
        sim_df,
        text_auto=True,
        color_continuous_scale="Tealrose",
        zmin=0.0,
        zmax=1.0,
        title=f"Column Similarity Matrix — {backend_label}",
        template="plotly_dark"
    )
    fig.update_layout(
        xaxis_title="Columns",
        yaxis_title="Columns",
        height=450
    )

    return {
        "clusters": clusters,
        "similarity_matrix": sim_df,
        "figure": fig,
        "redundancy_alerts": redundancy_alerts,
        "total_clusters": len(clusters),
        "embedding_backend": backend_label
    }
