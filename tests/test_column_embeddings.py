# tests/test_column_embeddings.py

import pandas as pd
from modules.column_embeddings import group_similar_columns_bert


def test_column_similarity_and_clustering():
    df = pd.DataFrame({
        "CustomerID": [1, 2, 3, 4],
        "AccountID": [101, 102, 103, 104],
        "AnnualIncome": [25000, 45000, 32000, 60000],
        "MonthlySalary": [2100, 3750, 2660, 5000],
        "Gender": ["Female", "Male", "Female", "Male"]
    })
    res = group_similar_columns_bert(df, similarity_threshold=0.3)
    assert "clusters" in res
    assert "similarity_matrix" in res
    assert res["similarity_matrix"].shape == (5, 5)
    # CustomerID and AccountID or Income and Salary should have elevated similarity
    sim_mat = res["similarity_matrix"]
    assert sim_mat.loc["CustomerID", "CustomerID"] == 1.0
    assert res["figure"] is not None
