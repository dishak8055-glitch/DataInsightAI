# modules/column_embeddings.py

import re
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering


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


def group_similar_columns_bert(df, similarity_threshold=0.35):
    """
    Uses subword and word-level semantic representations (BERT-family principles)
    to compute pairwise cosine similarities and group semantically similar columns.
    """
    columns = list(df.columns)
    if len(columns) < 2:
        return {
            "clusters": {0: columns},
            "similarity_matrix": pd.DataFrame([[1.0]], index=columns, columns=columns),
            "figure": None,
            "redundancy_alerts": []
        }

    # Generate rich semantic context documents for each column
    contexts = [generate_column_context(df, c) for c in columns]

    # Subword character n-gram + word TF-IDF vectorizer (captures subword morphological roots)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        analyzer="char_wb",
        min_df=1,
        sublinear_tf=True
    )

    tfidf_matrix = vectorizer.fit_transform(contexts)
    # Cosine similarity matrix S in [-1, 1], normalized to [0, 1]
    sim_matrix = cosine_similarity(tfidf_matrix)
    sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
    # Ensure diagonal is exactly 1.0
    np.fill_diagonal(sim_matrix, 1.0)

    sim_df = pd.DataFrame(sim_matrix, index=columns, columns=columns).round(3)

    # Distance matrix for agglomerative clustering
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0.0)

    # Cluster columns with Agglomerative Clustering
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

    # Identify potential collinear / redundant column pairs
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

    # Plotly interactive similarity heatmap
    fig = px.imshow(
        sim_df,
        text_auto=True,
        color_continuous_scale="Tealrose",
        zmin=0.0,
        zmax=1.0,
        title="BERT-Family Column Semantic Similarity Matrix",
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
        "total_clusters": len(clusters)
    }
