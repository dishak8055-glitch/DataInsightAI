# tests/test_gen_vs_disc.py

import pandas as pd
from modules.gen_vs_disc import analyze_learning_paradigms


def test_supervised_classification_paradigm():
    df = pd.DataFrame({
        "Age": [22, 35, 28, 42],
        "Gender": ["Female", "Male", "Female", "Male"]
    })
    res = analyze_learning_paradigms(df, target_column="Gender")
    assert res["primary_paradigm"] == "Supervised Learning"
    assert res["task_type"] == "Classification"
    assert "Logistic Regression" in res["discriminative"]["algorithms"]
    assert "Gaussian Naive Bayes" in res["generative"]["algorithms"][0]
    assert len(res["lifecycle_stages"]) == 7


def test_supervised_regression_paradigm():
    df = pd.DataFrame({
        "Age": [22, 35, 28, 42, 50, 60, 25, 30, 45, 55, 65, 70],
        "Salary": [25000, 45000, 32000, 60000, 75000, 90000, 28000, 36000, 62000, 78000, 95000, 110000]
    })
    res = analyze_learning_paradigms(df, target_column="Salary")
    assert res["primary_paradigm"] == "Supervised Learning"
    assert res["task_type"] == "Regression"
    assert "Linear Regression" in res["discriminative"]["algorithms"]


def test_unsupervised_clustering_paradigm():
    df = pd.DataFrame({
        "Age": [22, 35, 28, 42],
        "Income": [25000, 45000, 32000, 60000]
    })
    res = analyze_learning_paradigms(df, target_column=None)
    assert "Unsupervised" in res["primary_paradigm"]
    assert "K-Means Clustering" in res["discriminative"]["algorithms"]
    assert any("CTGAN" in a for a in res["generative"]["algorithms"])
