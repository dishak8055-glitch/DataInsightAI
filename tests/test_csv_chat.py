# tests/test_csv_chat.py

import pandas as pd
import pytest
from modules.csv_chat import (
    fast_local_plan,
    execute_plan,
    generate_local_explanation,
    answer_question,
    find_matching_columns
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "CustomerID": [1, 2, 3, 4, 5],
        "Age": [22, 35, 28, 42, 25],
        "Gender": ["Female", "Male", "Female", "Male", "Female"],
        "AnnualIncome": [25000, 45000, 32000, 60000, 28000],
        "SpendingScore": [80, 45, 90, 30, 75]
    })


def test_find_matching_columns(sample_df):
    cols = find_matching_columns(sample_df, "What is the average AnnualIncome and Age?")
    assert "AnnualIncome" in cols
    assert "Age" in cols


def test_fast_local_plan_row_count(sample_df):
    plan = fast_local_plan(sample_df, "How many rows are in this dataset?")
    assert plan is not None
    assert plan["operation"] == "count_rows"


def test_fast_local_plan_mean(sample_df):
    plan = fast_local_plan(sample_df, "What is the average Age?")
    assert plan is not None
    assert plan["operation"] == "mean"
    assert plan["column"] == "Age"


def test_fast_local_plan_max(sample_df):
    plan = fast_local_plan(sample_df, "highest AnnualIncome")
    assert plan is not None
    assert plan["operation"] == "maximum"
    assert plan["column"] == "AnnualIncome"


def test_fast_local_plan_correlation(sample_df):
    plan = fast_local_plan(sample_df, "What is the correlation between Age and SpendingScore?")
    assert plan is not None
    assert plan["operation"] == "correlation"
    assert "Age" in [plan["column"], plan["second_column"]]
    assert "SpendingScore" in [plan["column"], plan["second_column"]]


def test_fast_local_plan_groupby(sample_df):
    plan = fast_local_plan(sample_df, "average AnnualIncome by Gender")
    assert plan is not None
    assert plan["operation"] == "groupby_mean"
    assert plan["column"] == "AnnualIncome"
    assert plan["group_column"] == "Gender"


def test_execute_plan_mean(sample_df):
    plan = {"operation": "mean", "column": "Age", "group_column": None, "second_column": None}
    res = execute_plan(sample_df, plan)
    assert "Age" in res["result"]
    assert res["result"]["Age"] == round(sample_df["Age"].mean(), 4)


def test_execute_plan_correlation(sample_df):
    plan = {"operation": "correlation", "column": "Age", "group_column": None, "second_column": "SpendingScore"}
    res = execute_plan(sample_df, plan)
    assert "correlation" in res["result"]
    expected_corr = round(sample_df["Age"].corr(sample_df["SpendingScore"]), 4)
    assert res["result"]["correlation"] == expected_corr


def test_execute_plan_invalid_column(sample_df):
    plan = {"operation": "mean", "column": "NonExistentColumn", "group_column": None, "second_column": None}
    with pytest.raises(ValueError):
        execute_plan(sample_df, plan)


def test_execute_plan_unsupported_op(sample_df):
    plan = {"operation": "drop_table", "column": "Age", "group_column": None, "second_column": None}
    with pytest.raises(ValueError):
        execute_plan(sample_df, plan)


def test_answer_question_fast_path(sample_df):
    res = answer_question(sample_df, "What is the average Age?")
    assert res["source"] == "fast_pandas"
    assert res["used_fallback"] is False
    assert "Age" in res["explanation"]
    assert res["latency_ms"] < 1000  # fast-path should execute in < 50ms locally
