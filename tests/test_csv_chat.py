# tests/test_csv_chat.py

from types import SimpleNamespace

import pandas as pd
import pytest
from modules.csv_chat import (
    execute_restricted_pandas_code,
    fast_local_plan,
    execute_plan,
    generate_local_explanation,
    answer_question,
    find_matching_columns,
    interpret_pandas_code,
    validate_pandas_code,
)
from modules.dataset_catalog import load_curated_dataset
from modules.groq_client import get_groq_api_key, summarize_groq_error


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
    "question",
    [
        "Group by species and calculate mean body_mass_g.",
        "Calculate the average body_mass_g for each species.",
        "Calculate mean body_mass_g grouped by species.",
    ],
)
def test_grouped_mean_keeps_categorical_group_and_numeric_value(
    question,
):
    penguins = pd.DataFrame({
        "species": ["Adelie", "Adelie", "Gentoo", "Gentoo"],
        "body_mass_g": [3500, 3600, 5000, 5200],
    })

    plan = fast_local_plan(penguins, question)
    result = answer_question(penguins, question)

    assert plan["operation"] == "groupby_mean"
    assert plan["group_column"] == "species"
    assert plan["column"] == "body_mass_g"
    assert result["execution"]["pandas_logic"] == (
        "df.groupby('species')['body_mass_g'].mean()"
    )
    assert result["execution"]["result"] == {
        "Adelie": 3550.0,
        "Gentoo": 5100.0,
    }


@pytest.fixture
def penguins_metrics_df():
    return pd.DataFrame({
        "species": ["Adelie", "Adelie", "Gentoo", "Gentoo"],
        "body_mass_g": [3500, 3600, 5000, 5200],
        "flipper_length_mm": [180, 185, 210, 215],
    })


def _mock_csv_chat_completion(monkeypatch, content, calls):
    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content)
                    )
                ]
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr("modules.csv_chat.get_groq_client", lambda: client)


def test_fast_path_routes_two_measure_groupby_to_selected_llm_mode(
    penguins_metrics_df, monkeypatch
):
    question = (
        "Calculate the average of body_mass_g and flipper_length_mm for "
        "every species using groupby. Return both averages in one table."
    )
    assert fast_local_plan(penguins_metrics_df, question) is None

    calls = []
    _mock_csv_chat_completion(
        monkeypatch,
        '{"operation":"groupby_mean","column":"body_mass_g",'
        '"group_column":"species","second_column":null}',
        calls,
    )

    with pytest.raises(RuntimeError, match="multiple measures"):
        answer_question(penguins_metrics_df, question, execution_mode="json_plan")

    assert len(calls) == 1
    assert calls[0]["model"] == "openai/gpt-oss-20b"


def test_highest_average_for_both_measures_routes_to_json_plan(
    penguins_metrics_df, monkeypatch
):
    question = (
        "Compare the average body mass and average flipper length for each "
        "penguin species. Identify which species has the highest average in both measurements."
    )
    assert fast_local_plan(penguins_metrics_df, question) is None

    calls = []
    _mock_csv_chat_completion(
        monkeypatch,
        '{"operation":"groupby_rank","column":"body_mass_g",'
        '"group_column":"species","aggregation":"mean","ranking":"max"}',
        calls,
    )

    with pytest.raises(RuntimeError, match="multiple measures"):
        answer_question(penguins_metrics_df, question, execution_mode="json_plan")

    assert len(calls) == 1


def test_highest_average_for_both_measures_routes_to_pandas_code_mode(
    penguins_metrics_df, monkeypatch
):
    question = (
        "Compare the average body_mass_g and flipper_length_mm for each "
        "species and identify the species with the highest averages in both."
    )
    assert fast_local_plan(penguins_metrics_df, question) is None

    calls = []
    _mock_csv_chat_completion(
        monkeypatch,
        'result = df.groupby("species")["body_mass_g"].mean()',
        calls,
    )

    with pytest.raises(RuntimeError, match="multiple measures"):
        answer_question(
            penguins_metrics_df,
            question,
            use_fast_path=False,
            execution_mode="pandas_code",
        )

    assert len(calls) == 1


def test_grouped_request_rejects_mismatched_json_plan(
    penguins_metrics_df, monkeypatch
):
    question = "Calculate average body_mass_g by species."
    calls = []
    _mock_csv_chat_completion(
        monkeypatch,
        '{"operation":"value_counts","column":"species"}',
        calls,
    )

    with pytest.raises(RuntimeError, match="does not match the requested grouped"):
        answer_question(penguins_metrics_df, question, use_fast_path=False)

    assert len(calls) == 1


def test_grouped_request_rejects_scalar_pandas_code(
    penguins_metrics_df, monkeypatch
):
    calls = []
    _mock_csv_chat_completion(
        monkeypatch, 'result = df["body_mass_g"].mean()', calls
    )

    with pytest.raises(RuntimeError, match="does not group rows as requested"):
        answer_question(
            penguins_metrics_df,
            "Calculate average body_mass_g by species.",
            use_fast_path=False,
            execution_mode="pandas_code",
        )

    assert len(calls) == 1


def test_simple_frequency_and_overall_mean_stay_on_fast_path(
    penguins_metrics_df, monkeypatch
):
    def unexpected_groq_call(*args, **kwargs):
        raise AssertionError("deterministic questions must stay local")

    monkeypatch.setattr(
        "modules.csv_chat.get_groq_client", unexpected_groq_call
    )
    frequency = answer_question(
        penguins_metrics_df, "Show frequency distribution for species."
    )
    mean = answer_question(
        penguins_metrics_df, "What is the overall mean of body_mass_g?"
    )

    assert frequency["source"] == "fast_pandas"
    assert frequency["plan"]["operation"] == "value_counts"
    assert mean["source"] == "fast_pandas"
    assert mean["plan"]["operation"] == "mean"


@pytest.mark.parametrize(
    ("question", "aggregation", "ranking", "expected", "pandas_logic"),
    [
        (
            "Which penguin species has the highest average body_mass_g?",
            "mean",
            "max",
            "Gentoo",
            "df.groupby('species')['body_mass_g'].mean().idxmax()",
        ),
        (
            "Which penguin species has the lowest average body_mass_g?",
            "mean",
            "min",
            "Adelie",
            "df.groupby('species')['body_mass_g'].mean().idxmin()",
        ),
        (
            "Which species has the highest total body_mass_g?",
            "sum",
            "max",
            "Gentoo",
            "df.groupby('species')['body_mass_g'].sum().idxmax()",
        ),
        (
            "Which species has the lowest count?",
            "count",
            "min",
            "Adelie",
            "df.groupby('species').size().idxmin()",
        ),
    ],
)
def test_fast_path_ranks_groups_by_aggregate(
    question, aggregation, ranking, expected, pandas_logic
):
    penguins = pd.DataFrame({
        "species": ["Adelie", "Adelie", "Gentoo", "Gentoo", "Gentoo"],
        "body_mass_g": [3500, 3600, 5000, 5200, 5400],
    })

    result = answer_question(penguins, question)

    assert result["source"] == "fast_pandas"
    assert result["plan"]["operation"] == "groupby_rank"
    assert result["plan"]["aggregation"] == aggregation
    assert result["plan"]["ranking"] == ranking
    assert result["plan"]["group_column"] == "species"
    assert result["execution"]["result"] == expected
    assert result["execution"]["pandas_logic"] == pandas_logic


def test_grouped_mean_rejects_categorical_measure():
    df = pd.DataFrame({
        "species": ["Adelie", "Gentoo"],
        "island": ["Dream", "Biscoe"],
    })

    with pytest.raises(ValueError, match="must be numeric"):
        execute_plan(df, {
            "operation": "groupby_mean",
            "column": "island",
            "group_column": "species",
        })


@pytest.mark.parametrize(
    ("question", "ascending", "limit", "expected_values", "pandas_logic"),
    [
        (
            "Sort the dataset by body_mass_g in descending order and show the top 5 rows.",
            False,
            5,
            [5400, 5200, 5000, 3600, 3500],
            "df.sort_values('body_mass_g', ascending=False).head(5)",
        ),
        (
            "Order rows by body_mass_g ascending and show top 2 rows.",
            True,
            2,
            [3500, 3600],
            "df.sort_values('body_mass_g', ascending=True).head(2)",
        ),
        (
            "Sort by body_mass_g descending.",
            False,
            None,
            [5400, 5200, 5000, 3600, 3500],
            "df.sort_values('body_mass_g', ascending=False)",
        ),
    ],
)
def test_fast_path_sorts_rows_in_requested_order(
    question, ascending, limit, expected_values, pandas_logic
):
    penguins = pd.DataFrame({
        "species": ["Adelie", "Adelie", "Gentoo", "Gentoo", "Gentoo"],
        "body_mass_g": [3500, 3600, 5000, 5200, 5400],
    })

    result = answer_question(penguins, question)

    assert result["source"] == "fast_pandas"
    assert result["plan"]["operation"] == "sort_values"
    assert result["plan"]["column"] == "body_mass_g"
    assert result["plan"]["ascending"] is ascending
    assert result["plan"]["limit"] == limit
    assert [row["body_mass_g"] for row in result["execution"]["result"]] == (
        expected_values
    )
    assert result["execution"]["pandas_logic"] == pandas_logic


def test_explicit_frequency_distribution_is_not_treated_as_sorting():
    df = pd.DataFrame({"body_mass_g": [3500, 3500, 5000]})

    result = answer_question(df, "Show the frequency distribution of body_mass_g.")

    assert result["plan"]["operation"] == "value_counts"
    assert result["execution"]["result"] == {"3500": 2, "5000": 1}


@pytest.mark.parametrize(
    ("phrase", "operator", "threshold", "expected"),
    [
        ("greater than", ">", 5000, [5200, 5400]),
        ("less than", "<", 4000, [3500, 3600]),
        ("equal to", "==", 5000, [5000]),
        ("greater than or equal to", ">=", 5200, [5200, 5400]),
        ("less than or equal to", "<=", 3600, [3500, 3600]),
    ],
)
def test_fast_path_filters_numeric_rows(
    phrase, operator, threshold, expected
):
    penguins = pd.DataFrame({
        "species": ["Adelie", "Adelie", "Gentoo", "Gentoo", "Gentoo"],
        "body_mass_g": [3500, 3600, 5000, 5200, 5400],
    })
    question = (
        f"Show only penguins whose body_mass_g is {phrase} {threshold}."
    )

    result = answer_question(penguins, question)

    assert result["source"] == "fast_pandas"
    assert result["plan"]["operation"] == "filter_rows"
    assert result["plan"]["column"] == "body_mass_g"
    assert result["plan"]["comparison"] == operator
    assert result["execution"]["result"]["count"] == len(expected)
    assert [
        row["body_mass_g"] for row in result["execution"]["result"]["rows"]
    ] == expected


def test_fast_path_filters_categorical_equality_case_insensitively():
    df = pd.DataFrame({
        "species": ["Adelie", "Gentoo", "Chinstrap", "Gentoo"],
        "body_mass_g": [3500, 5000, 4000, 5200],
    })

    result = answer_question(df, "Show only penguins whose species is Gentoo.")

    assert result["plan"]["operation"] == "filter_rows"
    assert result["execution"]["result"]["count"] == 2
    assert {
        row["species"] for row in result["execution"]["result"]["rows"]
    } == {"Gentoo"}


def test_filter_rows_validates_column_and_numeric_value():
    df = pd.DataFrame({"body_mass_g": [3500, 5000]})

    with pytest.raises(ValueError, match="does not exist"):
        execute_plan(df, {
            "operation": "filter_rows",
            "column": "unknown_column",
            "comparison": ">",
            "value": 5000,
        })
    with pytest.raises(ValueError, match="must be numeric"):
        execute_plan(df, {
            "operation": "filter_rows",
            "column": "body_mass_g",
            "comparison": ">",
            "value": "heavy",
        })


def test_filter_rows_rejects_ordering_categorical_values():
    df = pd.DataFrame({"species": ["Adelie", "Gentoo"]})

    with pytest.raises(ValueError, match="requires a numeric column"):
        execute_plan(df, {
            "operation": "filter_rows",
            "column": "species",
            "comparison": ">",
            "value": "Adelie",
        })


def test_missing_value_counts_cover_each_column_and_empty_strings():
    df = pd.DataFrame({
        "Name": ["Ada", " ", None, "Lin"],
        "Score": [10, 20, None, 40],
    })

    result = answer_question(df, "How many missing values are there in each column?")

    assert result["source"] == "fast_pandas"
    assert result["plan"]["operation"] == "missing_values"
    assert result["plan"]["scope"] == "all_columns"
    assert result["execution"]["result"]["by_column"] == {
        "Name": 2,
        "Score": 1,
    }
    assert result["execution"]["result"]["total"] == 3


@pytest.mark.parametrize(
    ("question", "expected_scope", "expected"),
    [
        ("How many NaN values are in this dataset?", "total", {"total": 2}),
        ("Show null counts per column.", "all_columns", None),
        (
            "What percentage of missing values are in Score?",
            "column",
            {"column": "Score", "count": 1, "percentage": 25.0},
        ),
    ],
)
def test_missing_value_query_scope_and_percentages(
    question, expected_scope, expected
):
    df = pd.DataFrame({
        "Name": ["Ada", None, "Lin", "Grace"],
        "Score": [10, 20, None, 40],
    })

    result = answer_question(df, question)

    assert result["source"] == "fast_pandas"
    assert result["plan"]["operation"] == "missing_values"
    assert result["plan"]["scope"] == expected_scope
    if expected is not None:
        assert result["execution"]["result"] == expected
    else:
        assert result["execution"]["result"]["by_column"] == {
            "Name": 1,
            "Score": 1,
        }


def test_missing_value_percentages_include_blank_cells():
    df = pd.DataFrame({
        "Name": ["Ada", "", "Lin", "Grace"],
        "Score": [10, 20, None, 40],
    })

    result = answer_question(df, "Show missing value percentages by column.")

    assert result["plan"]["scope"] == "all_columns"
    assert result["execution"]["result"]["by_column"] == {
        "Name": 1,
        "Score": 1,
    }
    assert result["execution"]["result"]["percentages"] == {
        "Name": 25.0,
        "Score": 25.0,
    }


@pytest.mark.parametrize(
    "dataset_name",
    [
        "Adult Census Income",
        "Palmer Penguins",
        "Wine Quality Red",
        "Wine Quality White",
    ],
)
def test_fast_path_missing_counts_match_curated_dataset(dataset_name):
    df = load_curated_dataset(dataset_name)

    result = answer_question(df, "Show missing values in each column.")

    expected = (
        df.replace(r"^\s*$", pd.NA, regex=True)
        .isna()
        .sum()
        .astype(int)
        .to_dict()
    )
    assert result["plan"]["operation"] == "missing_values"
    assert result["execution"]["result"]["by_column"] == expected


def test_missing_value_query_does_not_replace_frequency_distribution():
    df = pd.DataFrame({"species": ["Adelie", "Adelie", "Gentoo"]})

    result = answer_question(df, "Show frequency distribution for species.")

    assert result["plan"]["operation"] == "value_counts"
    assert result["execution"]["result"] == {"Adelie": 2, "Gentoo": 1}


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
    def fail_groq(*args, **kwargs):
        raise RuntimeError("Groq unavailable")

    monkeypatch.setattr("modules.csv_chat.interpret_question", fail_groq)
    with pytest.raises(RuntimeError, match="could not be resolved locally"):
        answer_question(sample_df, "Who won the match yesterday?")


def test_groq_failure_uses_clear_local_fast_path(sample_df, monkeypatch):
    def fail_groq(*args, **kwargs):
        raise TimeoutError("simulated provider timeout")

    monkeypatch.setattr("modules.csv_chat.interpret_question", fail_groq)
    result = answer_question(
        sample_df,
        "average Age",
        use_fast_path=False
    )
    assert result["source"] == "local_fallback"
    assert result["used_fallback"] is True
    assert result["execution"]["result"]["Age"] == 30.4


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ('result = df["Age"].mean()', 30.4),
        ('result = df.groupby("Department").size()', {
            "Sales": 3,
            "Engineering": 1,
            "HR": 1,
        }),
        ('result = df.groupby("Department")["AnnualIncome"].mean()', {
            "Engineering": 45000.0,
            "HR": 60000.0,
            "Sales": 28333.333333333332,
        }),
        ('result = df.groupby("Department")["AnnualIncome"].median()', {
            "Engineering": 45000.0,
            "HR": 60000.0,
            "Sales": 28000.0,
        }),
        ('result = df["Gender"].value_counts()', {
            "Female": 3,
            "Male": 2,
        }),
        ("result = len(df)", 5),
    ],
)
def test_restricted_pandas_code_executes_approved_aggregations(
    sample_df, code, expected
):
    result = execute_restricted_pandas_code(sample_df, code)

    assert result["result"] == expected
    assert result["pandas_logic"] == code


@pytest.mark.parametrize(
    "code",
    [
        "import os\nresult = 1",
        "result = __import__('os').getcwd()",
        "result = df.__class__",
        "result = df['Gender'].mean()",
        "result = df['Department'].sum()",
        "result = df",
        "result = df['Missing'].mean()",
        "result = [row for row in df]",
        "result = df['Age'].mean(); print('no')",
    ],
)
def test_restricted_pandas_code_rejects_unsafe_or_invalid_syntax(
    sample_df, code
):
    with pytest.raises(ValueError):
        validate_pandas_code(code, sample_df)


def test_pandas_code_mode_uses_groq_code_and_restricted_interpreter(
    sample_df, monkeypatch
):
    monkeypatch.setattr(
        "modules.csv_chat.interpret_pandas_code",
        lambda df, question: 'result = df["Age"].mean()',
    )

    result = answer_question(
        sample_df,
        "Is the Age field normally distributed?",
        execution_mode="pandas_code",
    )

    assert result["source"] == "groq_pandas_ast"
    assert result["execution"]["result"] == 30.4
    assert result["execution"]["pandas_logic"] == 'result = df["Age"].mean()'
    assert "30.4" in result["explanation"]


def test_groq_pandas_code_prompt_uses_schema_not_dataset_rows(
    sample_df, monkeypatch
):
    calls = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='result = df["Age"].mean()'
                        )
                    )
                ]
            )

    fake_client = type(
        "FakeClient",
        (),
        {"chat": type(
            "Chat",
            (),
            {"completions": FakeCompletions()}
        )()}
    )()

    monkeypatch.setattr("modules.csv_chat.get_groq_client", lambda: fake_client)

    code = interpret_pandas_code(sample_df, "average Age")

    assert code == 'result = df["Age"].mean()'
    assert calls["model"] == "openai/gpt-oss-20b"
    prompt = str(calls["messages"])
    assert "CustomerID" in prompt
    assert "25000" not in prompt


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        ("mean", {"Adelie": 2.0, "Gentoo": 11.0}),
        ("median", {"Adelie": 2.0, "Gentoo": 11.0}),
        ("sum", {"Adelie": 4.0, "Gentoo": 22.0}),
    ],
)
def test_groq_generated_grouped_numeric_aggregations(
    monkeypatch, aggregation, expected
):
    df = pd.DataFrame({
        "species": ["Adelie", "Adelie", "Gentoo", "Gentoo"],
        "flipper_length_mm": [1.0, 3.0, 10.0, 12.0],
    })
    question = (
        f"For each species, calculate the {aggregation} of flipper_length_mm."
    )
    code = (
        'result = df.groupby("species")["flipper_length_mm"].'
        f"{aggregation}()"
    )
    calls = []
    _mock_csv_chat_completion(monkeypatch, code, calls)

    result = answer_question(
        df, question, use_fast_path=False, execution_mode="pandas_code"
    )

    assert result["source"] == "groq_pandas_ast"
    assert result["execution"]["result"] == expected
    assert len(calls) == 1
    assert f".{aggregation}()" in str(calls[0]["messages"])


def test_groq_prompt_explicitly_allows_grouped_median(monkeypatch):
    df = pd.DataFrame({
        "species": ["Adelie", "Gentoo"],
        "flipper_length_mm": [180, 210],
    })
    calls = []
    code = 'result = df.groupby("species")["flipper_length_mm"].median()'
    _mock_csv_chat_completion(monkeypatch, code, calls)

    from modules.csv_chat import interpret_pandas_code

    assert interpret_pandas_code(
        df, "For each species, calculate the median flipper_length_mm."
    ) == code
    assert ".median()" in str(calls[0]["messages"])


@pytest.mark.parametrize(
    ("code", "expected_message"),
    [
        (
            'result = df.groupby("species")["flipper_length_mm"].mode()',
            "Attribute 'mode' is not allowed",
        ),
        (
            'result = df.groupby("missing")["flipper_length_mm"].median()',
            "groupby requires one existing column name",
        ),
        (
            'result = df.groupby("species")["missing"].median()',
            "Column 'missing' does not exist",
        ),
        (
            'result = df.groupby("species")["Gender"].median()',
            "numeric value column",
        ),
        (
            'result = df.groupby("species")["flipper_length_mm"].apply(__import__("os"))',
            "Attribute 'apply' is not allowed",
        ),
    ],
)
def test_restricted_grouped_code_rejects_invalid_or_unsafe_requests(
    code, expected_message
):
    df = pd.DataFrame({
        "species": ["Adelie", "Gentoo"],
        "flipper_length_mm": [180, 210],
        "Gender": ["F", "M"],
    })
    with pytest.raises(ValueError, match=expected_message):
        execute_restricted_pandas_code(df, code)


def test_invalid_generated_pandas_code_is_not_reported_as_api_failure(
    sample_df, monkeypatch
):
    monkeypatch.setattr(
        "modules.csv_chat.interpret_pandas_code",
        lambda df, question: "result = df['Missing'].median()",
    )

    with pytest.raises(
        RuntimeError,
        match="Generated pandas code execution failed: Column 'Missing' does not exist",
    ):
        answer_question(
            sample_df,
            "What is the median value in the data?",
            use_fast_path=False,
            execution_mode="pandas_code",
        )


def test_unsafe_generated_syntax_is_reported_as_validation_not_api_failure(
    sample_df, monkeypatch
):
    monkeypatch.setattr(
        "modules.csv_chat.get_groq_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content='result = __import__("os")'
                                )
                            )
                        ]
                    )
                )
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Generated pandas code validation failed: Name '__import__' is not allowed",
    ):
        answer_question(
            sample_df,
            "What is an unusual transformation?",
            use_fast_path=False,
            execution_mode="pandas_code",
        )


def test_json_plan_mode_uses_groq_and_validates_the_plan(sample_df, monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"operation":"mean","column":"Age",'
                                '"group_column":null,"second_column":null}'
                            )
                        )
                    )
                ]
            )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr("modules.csv_chat.get_groq_client", lambda: fake_client)

    result = answer_question(
        sample_df,
        "What is the mean age among customers?",
        use_fast_path=False,
    )

    assert result["source"] == "groq"
    assert result["execution"]["result"]["Age"] == pytest.approx(30.4)


def test_json_plan_mode_rejects_unknown_model_generated_column(
    sample_df, monkeypatch
):
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"operation":"mean","column":"Secret"}'
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr("modules.csv_chat.get_groq_client", lambda: fake_client)
    with pytest.raises(RuntimeError, match="could not be resolved locally"):
        answer_question(
            sample_df,
            "Give me an unusual analysis of the customer data",
            use_fast_path=False,
        )


def test_pandas_code_mode_uses_fast_path_without_groq_call(sample_df, monkeypatch):
    def unexpected_groq_call(*args, **kwargs):
        raise AssertionError("deterministic questions must stay local")

    monkeypatch.setattr(
        "modules.csv_chat.interpret_pandas_code", unexpected_groq_call
    )
    result = answer_question(
        sample_df,
        "What is the average Age?",
        execution_mode="pandas_code",
    )
    assert result["source"] == "fast_pandas"
    assert result["execution"]["result"]["Age"] == pytest.approx(30.4)


def test_pandas_code_mode_does_not_fall_back_to_json_plan(
    sample_df, monkeypatch
):
    def fail_generation(*args, **kwargs):
        raise RuntimeError("API unavailable")

    monkeypatch.setattr("modules.csv_chat.interpret_pandas_code", fail_generation)
    with pytest.raises(RuntimeError, match="Groq pandas-code request failed"):
        answer_question(
            sample_df,
            "Is the Age field normally distributed?",
            execution_mode="pandas_code",
        )


def test_groq_key_loader_prefers_environment_without_exposing_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-a-real-secret")

    assert get_groq_api_key() == "test-key-not-a-real-secret"


def test_groq_key_loader_handles_missing_environment_and_secrets(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert get_groq_api_key() is None


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (type("ProviderError", (Exception,), {"status_code": 429})(), "rate limit"),
        (type("ProviderError", (Exception,), {"status_code": 503})(), "temporarily unavailable"),
        (TimeoutError("secret details"), "timed out"),
    ],
)
def test_groq_errors_are_actionable_without_provider_payload(error, expected):
    assert expected in summarize_groq_error(error)


def test_pandas_code_mode_surfaces_rate_limit(sample_df, monkeypatch):
    quota_error = type("QuotaError", (Exception,), {"status_code": 429})()

    def fail_generation(*args, **kwargs):
        raise quota_error

    monkeypatch.setattr("modules.csv_chat.interpret_pandas_code", fail_generation)

    with pytest.raises(RuntimeError, match="rate limit"):
        answer_question(
            sample_df,
            "Is the Age field normally distributed?",
            execution_mode="pandas_code",
        )
