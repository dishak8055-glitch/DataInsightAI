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
        "SpendingScore": [80, 45, 90, 30, 75],
        "Department": ["Sales", "Engineering", "Sales", "HR", "Sales"],
        "Location": ["West", "East", "West", "North", "East"]
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


@pytest.mark.parametrize(
    ("question", "expected_operation", "group_column"),
    [
        ("How many employees are in each department?", "groupby_count", "Department"),
        (
            "Which department has the highest number of employees?",
            "groupby_max_count",
            "Department"
        ),
        (
            "Show department-wise employee distribution.",
            "groupby_count",
            "Department"
        ),
        ("Which location has the most employees?", "groupby_max_count", "Location")
    ]
)
def test_employee_category_questions_use_safe_group_counts(
    sample_df, question, expected_operation, group_column
):
    result = answer_question(sample_df, question)

    assert result["source"] == "fast_pandas"
    assert result["plan"]["operation"] == expected_operation
    assert result["plan"]["group_column"] == group_column
    assert group_column in result["explanation"]
    assert "maximum" not in result["explanation"].lower()
    if expected_operation == "groupby_count":
        assert result["execution"]["result"] == (
            {"Sales": 3, "Engineering": 1, "HR": 1}
        )
        assert "employee counts" in result["explanation"].lower()
    else:
        expected_groups = {"Sales"} if group_column == "Department" else {"West", "East"}
        assert set(result["execution"]["result"]["groups"]) == expected_groups
        assert result["execution"]["result"]["count"] == (
            3 if group_column == "Department" else 2
        )
        assert "employees" in result["explanation"].lower()


def test_group_counts_include_missing_values_and_top_group_ties():
    df = pd.DataFrame({
        "Team": ["A", "A", "B", "B", None],
        "Score": [1, 2, 3, 4, 5]
    })

    counts = execute_plan(df, {
        "operation": "groupby_count",
        "group_column": "Team"
    })
    top_group = execute_plan(df, {
        "operation": "groupby_max_count",
        "group_column": "Team"
    })

    assert counts["result"] == {"A": 2, "B": 2, "Missing": 1}
    assert top_group["result"]["groups"] == ["A", "B"]
    assert top_group["result"]["count"] == 2
    assert "maximum" not in counts["pandas_logic"]


def test_group_count_rejects_unknown_group_column(sample_df):
    with pytest.raises(ValueError, match="Group column"):
        execute_plan(sample_df, {
            "operation": "groupby_count",
            "group_column": "Unknown"
        })


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


def test_unsupported_question_does_not_return_a_misleading_row_count(
    sample_df, monkeypatch
):
    def fail_gemini(*args, **kwargs):
        raise RuntimeError("Gemini unavailable")

    monkeypatch.setattr("modules.csv_chat.interpret_question", fail_gemini)
    with pytest.raises(RuntimeError, match="could not be resolved locally"):
        answer_question(sample_df, "Who won the match yesterday?")


def test_gemini_failure_uses_clear_local_fast_path(sample_df, monkeypatch):
    def fail_gemini(*args, **kwargs):
        raise TimeoutError("simulated provider timeout")

    monkeypatch.setattr("modules.csv_chat.interpret_question", fail_gemini)
    result = answer_question(
        sample_df,
        "average Age",
        use_fast_path=False
    )
    assert result["source"] == "local_fallback"
    assert result["used_fallback"] is True
    assert result["execution"]["result"]["Age"] == 30.4
