import pandas as pd

from modules.recommender import recommend_task


def test_classification_recommendation():
    df = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40, 45],
        "Gender": ["M", "F", "M", "F", "M", "F"]
    })

    result = recommend_task(df, "Gender")

    assert "Classification" in result["task"]
    assert "Random Forest Classifier" in result["algorithms"]


def test_regression_recommendation():
    df = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40, 45],
        "Salary": [20000, 25000, 30000, 35000, 40000, 45000]
    })

    result = recommend_task(df, "Salary")

    assert "Regression" in result["task"]
    assert "Linear Regression" in result["algorithms"]


def test_clustering_without_target():
    df = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40],
        "Income": [20000, 25000, 30000, 35000, 40000],
        "Spending": [80, 60, 70, 40, 50]
    })

    result = recommend_task(df)

    assert "Clustering" in result["task"]
    assert "K-Means" in result["algorithms"]


def test_identifier_is_not_used_as_feature():
    df = pd.DataFrame({
        "CustomerID": [1, 2, 3, 4, 5],
        "Age": [20, 25, 30, 35, 40],
        "Gender": ["M", "F", "M", "F", "M"]
    })

    result = recommend_task(df, "Gender")

    assert "CustomerID" not in result["features"]
    assert "Age" in result["features"]


def test_constant_target_is_rejected():
    df = pd.DataFrame({
        "Age": [20, 25, 30, 35],
        "Status": ["Active", "Active", "Active", "Active"]
    })

    result = recommend_task(df, "Status")

    assert result["task"] == "Insufficient target data"