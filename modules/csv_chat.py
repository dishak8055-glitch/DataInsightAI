# modules/csv_chat.py

import ast
import json
import re
import time
import pandas as pd
from modules.groq_client import (
    TEXT_MODEL,
    completion_text,
    get_groq_client,
    summarize_groq_error,
)

ALLOWED_OPERATIONS = {
    "count_rows",
    "describe",
    "mean",
    "median",
    "sum",
    "minimum",
    "maximum",
    "std",
    "value_counts",
    "sort_values",
    "filter_rows",
    "missing_values",
    "unique_count",
    "groupby_mean",
    "groupby_sum",
    "groupby_count",
    "groupby_max_count",
    "groupby_rank",
    "correlation"
}

def normalize_string(text):
    """Normalize text for insensitive fuzzy matching."""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def find_matching_columns(df, question):
    """
    Find columns from the dataframe that are referenced in the question.
    Uses word-boundary regex to prevent sub-word false positives (e.g., 'Age' inside 'average').
    Preserves order of appearance in the question text.
    """
    q_lower = question.lower()
    matches = []

    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower()
        col_norm = normalize_string(col_str)

        # 1. Word boundary regex match (e.g. 'age' matches ' age ', not 'average')
        pattern = r"\b" + re.escape(col_lower) + r"\b"
        match = re.search(pattern, q_lower)
        if match:
            matches.append((match.start(), col))
            continue

        # 2. De-camelcased match: 'AnnualIncome' -> 'annual income'
        de_camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", col_str).lower()
        if de_camel != col_lower:
            pattern2 = r"\b" + re.escape(de_camel) + r"\b"
            match2 = re.search(pattern2, q_lower)
            if match2:
                matches.append((match2.start(), col))
                continue

        # 3. Normalized compound match (e.g. 'annualincome' in question)
        if len(col_norm) >= 4:
            q_norm = normalize_string(question)
            if col_norm in q_norm:
                pos = q_norm.find(col_norm)
                matches.append((pos, col))

    # Sort by order of appearance in the question
    matches.sort(key=lambda x: x[0])

    seen = set()
    ordered_columns = []
    for _, col in matches:
        if col not in seen:
            seen.add(col)
            ordered_columns.append(col)

    return ordered_columns


def _grouped_aggregation_intent(question):
    """Return the requested grouped aggregate and ranking, if any."""
    q = question.lower()
    if not re.search(
        r"\b(?:group\s*by|grouped\s+by|for each|per|by)\b", q
    ):
        return None

    aggregation = (
        "mean" if re.search(r"\b(?:average|avg|mean)\b", q)
        else "median" if re.search(r"\bmedian\b", q)
        else "sum" if re.search(r"\b(?:sum|total)\b", q)
        else "count" if re.search(r"\b(?:count|counts|how many|number)\b", q)
        else None
    )
    if aggregation is None:
        return None

    ranking = (
        "max" if re.search(r"\b(?:highest|maximum|largest|greatest|most)\b", q)
        else "min" if re.search(r"\b(?:lowest|minimum|smallest|least)\b", q)
        else None
    )
    return aggregation, ranking


def _requests_multiple_grouped_measures(df, question):
    """Detect grouped requests that the current single-measure schemas cannot express."""
    intent = _grouped_aggregation_intent(question)
    if intent is None or intent[0] not in {"mean", "median", "sum"}:
        return False

    numeric_columns = [
        column for column in find_matching_columns(df, question)
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    return len(numeric_columns) > 1 or bool(re.search(r"\bboth\b", question, re.I))


def _validate_grouped_plan_intent(df, question, plan):
    """Reject model plans that do not represent the grouped operation asked for."""
    intent = _grouped_aggregation_intent(question)
    if intent is None:
        return
    aggregation, ranking = intent
    operation = plan.get("operation")
    if ranking:
        matches_intent = (
            operation == "groupby_rank"
            and plan.get("aggregation") == aggregation
            and plan.get("ranking") == ranking
        )
    else:
        expected_operation = {
            "mean": "groupby_mean",
            "median": "groupby_median",
            "sum": "groupby_sum",
            "count": "groupby_count",
        }.get(aggregation)
        matches_intent = (
            operation == expected_operation if expected_operation else False
        )
    if not matches_intent:
        raise ValueError(
            "Groq returned a plan that does not match the requested grouped "
            f"{aggregation} operation."
        )
    if _requests_multiple_grouped_measures(df, question):
        raise ValueError(
            "This grouped request asks for multiple measures, which the "
            "restricted JSON operation-plan mode cannot represent."
        )


def _validate_pandas_code_intent(df, question, code):
    """Ensure generated code expresses the requested grouped/scalar aggregate."""
    intent = _grouped_aggregation_intent(question)
    if intent is not None:
        aggregation, ranking = intent
        if _requests_multiple_grouped_measures(df, question):
            raise ValueError(
                "This grouped request asks for multiple measures, which the "
                "restricted Pandas Code mode cannot represent."
            )
        if not re.search(r"\.groupby\s*\(", code):
            raise ValueError(
                "Generated pandas code does not group rows as requested."
            )
        method = {
            "mean": "mean",
            "median": "median",
            "sum": "sum",
            "count": "size",
        }[aggregation]
        if not re.search(rf"\.{method}\s*\(", code):
            raise ValueError(
                f"Generated pandas code does not calculate the requested "
                f"grouped {aggregation}."
            )
        if ranking:
            ranking_method = "idxmax" if ranking == "max" else "idxmin"
            if not re.search(rf"\.{ranking_method}\s*\(", code):
                raise ValueError(
                    "Generated pandas code does not identify the requested "
                    "highest or lowest group."
                )
        return

    q = question.lower()
    matched_numeric = [
        column for column in find_matching_columns(df, question)
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    if (
        matched_numeric
        and re.search(r"\b(?:average|avg|mean)\b", q)
        and not re.search(r"\b(?:group\s*by|grouped\s+by|for each|per|by)\b", q)
        and not re.search(r"\.mean\s*\(", code)
    ):
        raise ValueError(
            "Generated pandas code does not calculate the requested mean."
        )


def fast_local_plan(df, question):
    """
    Fast-path semantic parser.
    Interprets common CSV questions locally with zero LLM latency and 100% precision.
    Returns an analysis plan dict or None if no clear pattern matches.
    """
    q = question.lower().strip()
    matched_cols = find_matching_columns(df, question)

    row_filter_intent = bool(
        re.search(r"\b(?:show )?rows? (?:with|containing|that have)\b", q)
        or re.search(r"\b(?:filter|show only)\b", q)
    )
    missing_terms = re.search(r"\b(?:missing|null|nan|blank|empty)\b", q)
    missing_report_intent = bool(
        missing_terms
        and not row_filter_intent
        and not any(
            term in q for term in [
                "sort", "ascending", "descending", "frequency distribution",
                "filter", "show only", "rows where", "records where"
            ]
        )
        and (
            re.search(
                r"\b(?:missing|null|nan|blank|empty)\s+values?\b", q
            )
            or re.search(
                r"\b(?:how many|count|number|percentages?|percents?|share|"
                r"each|every|per|total|overall|which columns)\b",
                q
            )
        )
        and not (
            re.search(r"\b(?:average|avg|mean|sum|aggregate)\b", q)
            and re.search(r"\b(?:by|per|each|group(?:ed)? by)\b", q)
        )
    )
    if missing_report_intent:
        include_percentage = bool(
            re.search(r"\b(?:percentages?|percents?|share|rate)\b", q)
        )
        has_column_scope = bool(
            re.search(r"\b(?:each|every|per|by) columns?\b", q)
            or re.search(r"\bwhich columns\b", q)
        )
        is_total = bool(
            re.search(
                r"\b(?:total|overall|across (?:the )?dataset|in (?:the )?dataset)\b",
                q
            )
            or (
                re.search(r"\b(?:how many|count|number)\b", q)
                and not has_column_scope
                and not re.search(r"\bcolumns?\b", q)
                and not matched_cols
            )
        ) and not has_column_scope
        if is_total:
            scope = "total"
            column = None
        elif has_column_scope or (not matched_cols and "column" not in q):
            scope = "all_columns"
            column = None
        elif len(matched_cols) == 1:
            scope = "column"
            column = matched_cols[0]
        else:
            scope = "all_columns"
            column = None
        return {
            "operation": "missing_values",
            "column": column,
            "group_column": None,
            "second_column": None,
            "scope": scope,
            "include_percentage": include_percentage,
            "source": "fast_pandas"
        }

    # Parse a single safe comparison before other column operations.
    comparison_patterns = [
        (r"\b(?:greater than or equal to|at least)\b|>=", ">="),
        (r"\b(?:less than or equal to|at most|no more than)\b|<=", "<="),
        (r"\b(?:greater than|more than|above)\b|(?<![<])>", ">"),
        (r"\b(?:less than|below)\b|(?<![>])<", "<"),
        (r"\b(?:equal to|equals|is|are)\b|==|=", "=="),
    ]
    explicit_filter_intent = any(
        phrase in q for phrase in ["show only", "filter", "where", "whose"]
    )
    sort_intent = any(
        term in q for term in ["sort", "order", "ascending", "descending"]
    )
    filter_intent = explicit_filter_intent or (
        not sort_intent
        and any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern, _ in comparison_patterns
        )
    )
    if filter_intent and matched_cols:
        for column_name in matched_cols:
            column_spellings = [str(column_name)]
            de_camel = re.sub(
                r"([a-z])([A-Z])", r"\1 \2", str(column_name)
            )
            if de_camel.lower() != str(column_name).lower():
                column_spellings.append(de_camel)
            column_match = next(
                (
                    match for spelling in column_spellings
                    if (match := re.search(
                        r"\b" + re.escape(spelling) + r"\b",
                        question,
                        re.IGNORECASE
                    ))
                ),
                None
            )
            if not column_match:
                continue

            remainder = question[column_match.end():]
            for pattern, operator in comparison_patterns:
                operator_match = re.match(
                    r"\s*(?:is\s+)?(?:" + pattern + r")\s*(.*)",
                    remainder,
                    re.IGNORECASE
                )
                if not operator_match:
                    continue
                raw_value = operator_match.group(1).strip()
                quoted_value = re.match(r"""^(['"])(.*?)\1""", raw_value)
                if quoted_value:
                    value = quoted_value.group(2)
                else:
                    numeric_value = re.match(
                        r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)", raw_value
                    )
                    if numeric_value:
                        value = numeric_value.group(0)
                    else:
                        value = re.split(
                            r"\s+(?:and\s+)?(?:show|display|return)\b|[,.!?]",
                            raw_value,
                            maxsplit=1,
                            flags=re.IGNORECASE
                        )[0].strip()
                if not value:
                    continue

                if pd.api.types.is_numeric_dtype(df[column_name]):
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if value.is_integer():
                        value = int(value)
                elif operator not in {"=="}:
                    continue

                return {
                    "operation": "filter_rows",
                    "column": column_name,
                    "group_column": None,
                    "second_column": None,
                    "comparison": operator,
                    "value": value,
                    "source": "fast_pandas"
                }

    # 1. Dataset-level row count
    row_count_patterns = [
        "how many rows", "row count", "count rows", "total records",
        "total rows", "number of rows", "total customers", "dataset size",
        "length of dataset", "how many records", "how many samples", "shape of data"
    ]
    if any(pat in q for pat in row_count_patterns):
        return {
            "operation": "count_rows",
            "column": None,
            "group_column": None,
            "second_column": None,
            "source": "fast_pandas"
        }

    sort_requested = (
        any(term in q for term in ["sort", "order", "ascending", "descending"])
        and not any(
            term in q
            for term in ["distribution", "frequency", "counts", "count of"]
        )
    )
    if sort_requested and matched_cols:
        top_match = re.search(r"\btop\s+(\d+)\b", q)
        limit = int(top_match.group(1)) if top_match else None
        if limit is None or limit > 0:
            return {
                "operation": "sort_values",
                "column": matched_cols[0],
                "group_column": None,
                "second_column": None,
                "ascending": not any(
                    term in q for term in ["descending", "desc"]
                ),
                "limit": limit,
                "source": "fast_pandas"
            }

    employee_count_query = any(term in q for term in [
        "employee", "employees", "staff", "people"
    ])
    asks_for_top_group = (
        any(term in q for term in ["most", "highest", "largest", "greatest"])
        and employee_count_query
    )
    asks_for_group_counts = (
        employee_count_query
        or any(term in q for term in ["distribution", "counts", "how many"])
    ) and (
        any(term in q for term in ["each", "every", "per", " by ", "grouped by"])
        or re.search(r"\b[\w ]+-wise\b", q) is not None
    )
    if asks_for_top_group or asks_for_group_counts:
        if matched_cols:
            group_col = matched_cols[-1]
            return {
                "operation": (
                    "groupby_max_count" if asks_for_top_group else "groupby_count"
                ),
                "column": None,
                "group_column": group_col,
                "second_column": None,
                "source": "fast_pandas"
            }

    # Rank groups by an aggregate, e.g. "highest average body_mass_g by species".
    rank_terms = {
        "max": ["highest", "maximum", "largest", "greatest", "most"],
        "min": ["lowest", "minimum", "smallest", "least"]
    }
    ranking = next(
        (
            direction for direction, terms in rank_terms.items()
            if any(term in q for term in terms)
        ),
        None
    )
    aggregation = (
        "mean" if any(term in q for term in ["average", "avg", "mean"])
        else "sum" if any(term in q for term in ["sum", "total"])
        else "count" if re.search(r"\b(count|counts|number|how many)\b", q)
        else None
    )
    if ranking and aggregation:
        categorical_cols = [
            col for col in matched_cols
            if not pd.api.types.is_numeric_dtype(df[col])
        ]
        numeric_cols = [
            col for col in matched_cols
            if pd.api.types.is_numeric_dtype(df[col])
        ]
        if aggregation == "count" and categorical_cols:
            return {
                "operation": "groupby_rank",
                "column": None,
                "group_column": categorical_cols[0],
                "second_column": None,
                "aggregation": aggregation,
                "ranking": ranking,
                "source": "fast_pandas"
            }
        if (
            aggregation in {"mean", "sum"}
            and len(categorical_cols) == 1
            and len(numeric_cols) == 1
        ):
            return {
                "operation": "groupby_rank",
                "column": numeric_cols[0],
                "group_column": categorical_cols[0],
                "second_column": None,
                "aggregation": aggregation,
                "ranking": ranking,
                "source": "fast_pandas"
            }

    if ranking and aggregation and _grouped_aggregation_intent(question):
        return None

    # 2. Dataset describe / summary
    if any(pat in q for pat in ["describe dataset", "statistical summary", "overall summary", "summarize data", "data summary"]):
        return {
            "operation": "describe",
            "column": None,
            "group_column": None,
            "second_column": None,
            "source": "fast_pandas"
        }

    # 3. Correlation between two columns
    if any(pat in q for pat in ["correlation", "relationship between", "how does"]):
        if len(matched_cols) >= 2:
            return {
                "operation": "correlation",
                "column": matched_cols[0],
                "group_column": None,
                "second_column": matched_cols[1],
                "source": "fast_pandas"
            }

    # 4. Grouped aggregation (also recognize the single token "groupby").
    grouped_aggregation = _grouped_aggregation_intent(question)
    if grouped_aggregation and grouped_aggregation[1] is None:
        if len(matched_cols) >= 2:
            numeric_cols = [
                col for col in matched_cols
                if pd.api.types.is_numeric_dtype(df[col])
            ]
            categorical_cols = [
                col for col in matched_cols
                if not pd.api.types.is_numeric_dtype(df[col])
            ]
            if len(numeric_cols) > 1:
                return None
            if numeric_cols and len(categorical_cols) == 1:
                if grouped_aggregation[0] == "median":
                    return None
                val_col = numeric_cols[0]
                grp_col = categorical_cols[0]
                op = (
                    "groupby_sum"
                    if grouped_aggregation[0] == "sum"
                    else "groupby_mean"
                )
                return {
                    "operation": op,
                    "column": val_col,
                    "group_column": grp_col,
                    "second_column": None,
                    "source": "fast_pandas"
                }
            return None
        # A lone category is a frequency query only when frequency/count
        # intent was explicit; never infer it from "average ... for each".
        if len(matched_cols) == 1 and grouped_aggregation[0] == "count":
            return {
                "operation": "value_counts",
                "column": matched_cols[0],
                "group_column": None,
                "second_column": None,
                "source": "fast_pandas"
            }
        return None

    # 5. Column-specific aggregations
    if matched_cols:
        col = matched_cols[0]

        if any(term in q for term in ["average", "avg", "mean"]):
            return {"operation": "mean", "column": col, "group_column": None, "second_column": None, "source": "fast_pandas"}

        if any(term in q for term in ["median", "middle value", "50th percentile"]):
            return {"operation": "median", "column": col, "group_column": None, "second_column": None, "source": "fast_pandas"}

        if any(term in q for term in ["highest", "maximum", "max", "top value", "greatest", "peak", "most"]):
            return {"operation": "maximum", "column": col, "group_column": None, "second_column": None, "source": "fast_pandas"}

        if any(term in q for term in ["lowest", "minimum", "min", "bottom value", "least", "smallest"]):
            return {"operation": "minimum", "column": col, "group_column": None, "second_column": None, "source": "fast_pandas"}

        if any(term in q for term in ["total", "sum", "aggregate"]):
            return {"operation": "sum", "column": col, "group_column": None, "second_column": None, "source": "fast_pandas"}

        if any(term in q for term in ["standard deviation", "std dev", "std", "variance", "spread"]):
            return {"operation": "std", "column": col, "group_column": None, "second_column": None, "source": "fast_pandas"}

        if any(term in q for term in ["unique", "distinct", "how many different"]):
            return {"operation": "unique_count", "column": col, "group_column": None, "second_column": None, "source": "fast_pandas"}

        if any(term in q for term in ["distribution", "frequency", "categories", "breakdown", "counts", "values of"]):
            return {"operation": "value_counts", "column": col, "group_column": None, "second_column": None, "source": "fast_pandas"}

    return None


def interpret_question(df, question):
    """
    Ask Groq to convert a complex user question into a restricted,
    structured JSON analysis plan.
    """
    client = get_groq_client()

    columns = []
    for column in df.columns:
        columns.append({
            "name": str(column),
            "dtype": str(df[column].dtype),
            "unique_values": int(df[column].nunique(dropna=True))
        })

    prompt = f"""
You are a CSV data analysis planner.
Convert the user's natural language question into ONE restricted JSON analysis plan.

Dataset information:
- Total rows: {len(df)}
- Columns: {json.dumps(columns)}

User question:
{question}

Allowed operations:
- count_rows (column: null)
- describe (column: null)
- mean (column: <col>)
- median (column: <col>)
- sum (column: <col>)
- minimum (column: <col>)
- maximum (column: <col>)
- std (column: <col>)
- value_counts (column: <col>)
- filter_rows (column: <col>, comparison: >, <, >=, <=, or ==, value: <value>)
- missing_values (scope: total, all_columns, or column; optional column)
- unique_count (column: <col>)
- groupby_mean (column: <numeric_col>, group_column: <category_col>)
- groupby_sum (column: <numeric_col>, group_column: <category_col>)
- groupby_count (group_column: <category_col>)
- groupby_max_count (group_column: <category_col>)
- groupby_rank (column: <numeric_col> or null for count,
  group_column: <category_col>, aggregation: mean/sum/count,
  ranking: max/min)
- correlation (column: <col1>, second_column: <col2>)

Return ONLY valid JSON in this exact structure with no Markdown wrapping:
{{
    "operation": "mean",
    "column": "AnnualIncome",
    "group_column": null,
    "second_column": null,
    "scope": null,
    "include_percentage": false,
    "aggregation": null,
    "ranking": null
}}

Rules:
- Never invent column names. Use exact column names from the dataset.
- If unsupported, return {{"operation": "unsupported"}}.
- Do not output explanations or python code.
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Return one JSON analysis plan using only the allowed operations."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    raw_response = completion_text(response)

    # Clean markdown fences if any
    if "```" in raw_response:
        raw_response = re.sub(r"^```(?:json)?\s*", "", raw_response)
        raw_response = re.sub(r"\s*```$", "", raw_response)

    try:
        plan = json.loads(raw_response.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("Groq returned invalid JSON for the analysis plan.") from exc
    validate_operation_plan(df, plan)
    plan["source"] = "groq"
    return plan


class _PandasCodeRequestError(Exception):
    """Sanitized failure while requesting generated pandas code."""


def interpret_pandas_code(df, question):
    """Ask Groq for a single pandas expression; execution is AST-interpreted."""
    columns = [
        {"name": str(column), "dtype": str(df[column].dtype)}
        for column in df.columns
    ]
    prompt = f"""
You generate one concise pandas expression to answer a CSV question.
Dataset schema only (no row values): {json.dumps(columns)}
Question: {question}

Return exactly one Python statement assigning its result to `result`.
The only available object is `df`.
Allowed forms:
- len(df)
- df["column"].mean(), .median(), .sum(), .min(), .max(), .std()
- df["column"].value_counts(), .nunique(), or .describe()
- df.groupby("group_column").size()
- df.groupby("group_column")["numeric_column"].mean(), .median(), or .sum()
- df["numeric_column"].corr(df["other_numeric_column"])

Use exact column names. Do not import modules, define functions, use loops,
access attributes other than the listed pandas methods, or include explanations.
Example: result = df["Age"].mean()
"""
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Generate only the requested restricted pandas expression."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
    except Exception as error:
        raise _PandasCodeRequestError(
            summarize_groq_error(error)
        ) from None

    try:
        code = completion_text(response)
    except ValueError as error:
        raise ValueError(f"Groq returned an invalid pandas-code response: {error}") from None
    if code.startswith("```"):
        code = re.sub(r"^```(?:python)?\s*", "", code)
        code = re.sub(r"\s*```$", "", code)
    _validate_pandas_code_tree(code, df)
    return code


def validate_operation_plan(df, plan):
    """Validate a provider-generated plan against the local operation schema."""
    if not isinstance(plan, dict):
        raise ValueError("The analysis plan must be a JSON object.")

    operation = plan.get("operation")
    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(
            f"Operation '{operation}' is not allowed or unsupported."
        )

    column = plan.get("column")
    group_column = plan.get("group_column")
    second_column = plan.get("second_column")
    column_optional = operation in {
        "count_rows", "describe", "groupby_count", "groupby_max_count",
        "missing_values"
    } or (operation == "groupby_rank" and plan.get("aggregation") == "count")
    if not column_optional and (
        not isinstance(column, str) or column not in df.columns
    ):
        raise ValueError("The analysis plan must reference an existing column.")
    if column is not None and (
        not isinstance(column, str) or column not in df.columns
    ):
        raise ValueError("The analysis plan references an unknown column.")

    grouped_operations = {
        "groupby_mean", "groupby_sum", "groupby_count",
        "groupby_max_count", "groupby_rank"
    }
    if operation in grouped_operations and (
        not isinstance(group_column, str) or group_column not in df.columns
    ):
        raise ValueError("The analysis plan must reference an existing group column.")
    if operation == "correlation" and (
        not isinstance(second_column, str) or second_column not in df.columns
    ):
        raise ValueError("The correlation plan must reference an existing second column.")

    numeric_operations = {
        "mean", "median", "sum", "minimum", "maximum", "std",
        "groupby_mean", "groupby_sum"
    }
    if operation in numeric_operations and not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Operation '{operation}' requires a numeric column.")
    if operation == "groupby_rank":
        if plan.get("aggregation") not in {"mean", "sum", "count"}:
            raise ValueError("Grouped ranking requires mean, sum, or count.")
        if plan.get("ranking") not in {"max", "min"}:
            raise ValueError("Grouped ranking requires max or min.")
        if (
            plan["aggregation"] in {"mean", "sum"}
            and not pd.api.types.is_numeric_dtype(df[column])
        ):
            raise ValueError("Grouped mean and sum require a numeric column.")
    if operation == "correlation" and (
        not pd.api.types.is_numeric_dtype(df[column])
        or not pd.api.types.is_numeric_dtype(df[second_column])
    ):
        raise ValueError("Correlation requires numeric columns.")
    if operation == "filter_rows":
        if plan.get("comparison") not in {">", "<", ">=", "<=", "=="}:
            raise ValueError("Unsupported row-filter comparison operator.")
        if plan.get("value") is None:
            raise ValueError("Row-filter comparison value must not be empty.")
        if (
            not pd.api.types.is_numeric_dtype(df[column])
            and plan["comparison"] != "=="
        ):
            raise ValueError("Categorical columns support equality filters only.")
    if operation == "missing_values":
        if plan.get("scope") not in {"total", "all_columns", "column"}:
            raise ValueError("Missing-value scope must be total, all_columns, or column.")
        if plan["scope"] == "column" and (
            not isinstance(column, str) or column not in df.columns
        ):
            raise ValueError("Missing-value plan must reference an existing column.")
    if operation == "sort_values":
        if not isinstance(plan.get("ascending"), bool):
            raise ValueError("Sort direction must be a boolean.")
        limit = plan.get("limit")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
        ):
            raise ValueError("Sort row limit must be a positive integer.")
    return plan


_ALLOWED_PANDAS_METHODS = {
    "corr",
    "describe",
    "dropna",
    "groupby",
    "max",
    "mean",
    "median",
    "min",
    "nunique",
    "size",
    "std",
    "sum",
    "value_counts",
}
_ALLOWED_AST_NODES = (
    ast.Module,
    ast.Assign,
    ast.Expr,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Subscript,
    ast.Call,
    ast.Attribute,
)


class _PandasCodeValidator(ast.NodeVisitor):
    def generic_visit(self, node):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"Unsupported pandas code syntax: {type(node).__name__}.")
        super().generic_visit(node)

    def visit_Name(self, node):
        if node.id not in {"df", "result", "len"}:
            raise ValueError(f"Name '{node.id}' is not allowed.")

    def visit_Attribute(self, node):
        if node.attr.startswith("_") or node.attr not in _ALLOWED_PANDAS_METHODS:
            raise ValueError(f"Attribute '{node.attr}' is not allowed.")
        self.generic_visit(node)


class _SafeGroupBy:
    def __init__(self, grouped, numeric_series=None):
        self.grouped = grouped
        self.numeric_series = numeric_series


def _validate_pandas_code_tree(code, df):
    if not isinstance(code, str) or not code.strip() or len(code) > 2000:
        raise ValueError("Pandas code must be a non-empty statement under 2000 characters.")
    try:
        tree = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Generated pandas code is not valid Python syntax.") from exc

    if len(tree.body) != 1:
        raise ValueError("Exactly one pandas expression statement is allowed.")
    statement = tree.body[0]
    if isinstance(statement, ast.Assign):
        if (
            len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
            or statement.targets[0].id != "result"
        ):
            raise ValueError("The only permitted assignment target is 'result'.")
        expression = statement.value
    elif isinstance(statement, ast.Expr):
        expression = statement.value
    else:
        raise ValueError("Only one expression or result assignment is allowed.")

    nodes = list(ast.walk(tree))
    if len(nodes) > 50:
        raise ValueError("Generated pandas expression is too complex.")
    _PandasCodeValidator().visit(tree)
    return expression


def _evaluate_pandas_expression(node, df):
    if isinstance(node, ast.Name):
        if node.id == "df":
            return df
        raise ValueError(f"Name '{node.id}' cannot be read in an expression.")

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (str, int, float, bool, type(None))):
            raise ValueError("Unsupported constant value.")
        return node.value

    if isinstance(node, ast.Subscript):
        value = _evaluate_pandas_expression(node.value, df)
        key = _evaluate_pandas_expression(node.slice, df)
        if not isinstance(key, str):
            raise ValueError("Only literal column names can be selected.")
        if isinstance(value, pd.DataFrame):
            if key not in value.columns:
                raise ValueError(f"Column '{key}' does not exist in the dataset.")
            return value[key]
        if isinstance(value, _SafeGroupBy):
            if key not in df.columns:
                raise ValueError(f"Column '{key}' does not exist in the dataset.")
            if not pd.api.types.is_numeric_dtype(df[key]):
                raise ValueError(
                    "Grouped aggregations require a numeric value column."
                )
            return _SafeGroupBy(value.grouped, value.grouped[key])
        raise ValueError("Subscript access is only allowed on df or a grouped dataframe.")

    if isinstance(node, ast.Call):
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed in pandas code.")
        if isinstance(node.func, ast.Name) and node.func.id == "len":
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Name) or node.args[0].id != "df":
                raise ValueError("Only len(df) is allowed.")
            return len(df)
        if not isinstance(node.func, ast.Attribute):
            raise ValueError("Only listed pandas methods can be called.")
        target = _evaluate_pandas_expression(node.func.value, df)
        method = node.func.attr
        args = [_evaluate_pandas_expression(arg, df) for arg in node.args]

        if isinstance(target, pd.DataFrame) and method == "groupby":
            if len(args) != 1 or not isinstance(args[0], str) or args[0] not in df.columns:
                raise ValueError("groupby requires one existing column name.")
            return _SafeGroupBy(df.groupby(args[0]))
        if isinstance(target, pd.DataFrame) and method == "describe":
            if args:
                raise ValueError("DataFrame describe does not accept arguments here.")
            return target.describe()

        if isinstance(target, pd.Series):
            if method == "corr":
                if len(args) != 1 or not isinstance(args[0], pd.Series):
                    raise ValueError("corr requires one selected numeric column.")
                if not pd.api.types.is_numeric_dtype(target) or not pd.api.types.is_numeric_dtype(args[0]):
                    raise ValueError("Correlation requires numeric columns.")
            elif args:
                raise ValueError(f"{method} does not accept arguments in this workflow.")
            if method in {"mean", "median", "min", "max", "std", "sum", "corr"} and not pd.api.types.is_numeric_dtype(target):
                raise ValueError(f"{method} requires a numeric column.")
            if method in {"mean", "median", "min", "max", "std", "sum", "value_counts", "nunique", "describe", "dropna", "corr"}:
                return getattr(target, method)(*args)

        if isinstance(target, _SafeGroupBy):
            if args:
                raise ValueError(f"Grouped {method} does not accept arguments.")
            if method == "size" and target.numeric_series is None:
                return target.grouped.size()
            if method in {"mean", "median", "sum"} and target.numeric_series is not None:
                return getattr(target.numeric_series, method)()

        raise ValueError(f"Method '{method}' is not allowed for this value.")

    raise ValueError(f"Unsupported pandas expression: {type(node).__name__}.")


def validate_pandas_code(code, df):
    """Validate generated syntax and evaluate it without exec/eval or Python globals."""
    expression = _validate_pandas_code_tree(code, df)
    result = _evaluate_pandas_expression(expression, df)
    aggregate_methods = {
        "corr",
        "describe",
        "max",
        "mean",
        "median",
        "min",
        "nunique",
        "size",
        "std",
        "sum",
        "value_counts",
    }
    is_aggregate_call = (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and expression.func.attr in aggregate_methods
    )
    if isinstance(result, _SafeGroupBy) or (
        isinstance(result, (pd.DataFrame, pd.Series)) and not is_aggregate_call
    ):
        raise ValueError("The expression must return an aggregate, not raw dataset rows.")
    return result


def _normalize_pandas_code_result(value):
    if isinstance(value, pd.DataFrame):
        return _normalize_pandas_code_result(value.to_dict())
    if isinstance(value, pd.Series):
        return {str(key): _normalize_pandas_code_result(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {str(key): _normalize_pandas_code_result(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_pandas_code_result(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if value is pd.NA or (not isinstance(value, (dict, list, tuple)) and pd.isna(value)):
        return None
    return value


def execute_restricted_pandas_code(df, code):
    """Run a validated pandas expression through the restricted AST interpreter."""
    result = validate_pandas_code(code, df)
    return {
        "result": _normalize_pandas_code_result(result),
        "pandas_logic": code,
    }


def _explain_restricted_result(result):
    if isinstance(result, dict):
        items = list(result.items())[:15]
        formatted = ", ".join(f"{key}: {value}" for key, value in items)
        suffix = " (showing the first 15 entries)" if len(result) > 15 else ""
        return f"Computed the requested result from the dataset: {formatted}{suffix}."
    return f"Computed the requested result from the dataset: **{result}**."


def execute_plan(df, plan):
    """
    Validate and execute the approved analysis plan using safe pandas operations.
    Strictly avoids eval() or exec().
    """
    operation = plan.get("operation")
    column = plan.get("column")
    group_column = plan.get("group_column")
    second_column = plan.get("second_column")

    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(
            f"Operation '{operation}' is not allowed or unsupported."
        )

    # 1. Row count
    if operation == "count_rows":
        total = int(len(df))
        return {
            "result": {"total_rows": total},
            "pandas_logic": "len(df)"
        }

    # 2. Describe
    if operation == "describe":
        result = df.describe(include="all").fillna("").to_dict()
        return {
            "result": result,
            "pandas_logic": "df.describe(include='all')"
        }

    # Column existence check for operations that use a value column.
    if (
        operation not in {
            "groupby_count", "groupby_max_count", "missing_values"
        }
        and not (
            operation == "groupby_rank"
            and plan.get("aggregation") == "count"
        )
        and column not in df.columns
    ):
        raise ValueError(f"Column '{column}' does not exist in dataset.")

    if operation == "sort_values":
        ascending = plan.get("ascending", True)
        if not isinstance(ascending, bool):
            raise ValueError("Sort direction must be ascending or descending.")
        limit = plan.get("limit")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
        ):
            raise ValueError("Sort row limit must be a positive integer.")

        sorted_df = df.sort_values(column, ascending=ascending)
        pandas_logic = (
            f"df.sort_values('{column}', ascending="
            f"{'True' if ascending else 'False'})"
        )
        if limit is not None:
            sorted_df = sorted_df.head(limit)
            pandas_logic += f".head({limit})"
        return {
            "result": _normalize_pandas_code_result(
                sorted_df.to_dict(orient="records")
            ),
            "pandas_logic": pandas_logic
        }

    if operation == "filter_rows":
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in dataset.")
        comparison = plan.get("comparison")
        if comparison not in {">", "<", ">=", "<=", "=="}:
            raise ValueError("Unsupported row-filter comparison operator.")
        value = plan.get("value")
        if value is None or (
            isinstance(value, str) and not value.strip()
        ):
            raise ValueError("Row-filter comparison value must not be empty.")

        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            try:
                value = pd.to_numeric(value, errors="raise")
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Comparison value for numeric column '{column}' must be numeric."
                ) from error
        elif comparison != "==":
            raise ValueError(
                f"Comparison '{comparison}' requires a numeric column."
            )

        if pd.api.types.is_numeric_dtype(series):
            equality = series == value
        else:
            equality = (
                series.astype("string").str.casefold()
                == str(value).casefold()
            )
        comparisons = {
            ">": lambda: series > value,
            "<": lambda: series < value,
            ">=": lambda: series >= value,
            "<=": lambda: series <= value,
            "==": lambda: equality
        }
        try:
            filtered = df.loc[comparisons[comparison]()]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Comparison value is incompatible with column '{column}'."
            ) from error

        value_repr = repr(value)
        pandas_logic = f"df[df['{column}'] {comparison} {value_repr}]"
        return {
            "result": {
                "rows": _normalize_pandas_code_result(
                    filtered.to_dict(orient="records")
                ),
                "count": int(len(filtered))
            },
            "pandas_logic": pandas_logic
        }

    if operation == "missing_values":
        scope = plan.get("scope")
        if scope not in {"total", "all_columns", "column"}:
            raise ValueError("Missing-value scope must be total, all_columns, or column.")
        if scope == "column" and column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in dataset.")

        normalized = df.replace(r"^\s*$", pd.NA, regex=True)
        missing_counts = normalized.isna().sum()
        missing_total = int(missing_counts.sum())
        include_percentage = plan.get("include_percentage", False)
        pandas_logic = (
            "df.replace(r'^\\s*$', pd.NA, regex=True).isna().sum()"
        )

        if scope == "total":
            result = {"total": missing_total}
            pandas_logic += ".sum()"
            if include_percentage:
                total_cells = normalized.size
                result["percentage"] = (
                    round(missing_total / total_cells * 100, 2)
                    if total_cells else 0.0
                )
                pandas_logic += " / df.size * 100"
        elif scope == "column":
            count = int(missing_counts[column])
            result = {"column": str(column), "count": count}
            if include_percentage:
                result["percentage"] = (
                    round(float(normalized[column].isna().mean() * 100), 2)
                    if len(normalized) else 0.0
                )
                pandas_logic = (
                    f"df['{column}'].replace(r'^\\s*$', pd.NA, "
                    "regex=True).isna().sum()"
                )
        else:
            result = {
                "by_column": {
                    str(name): int(count)
                    for name, count in missing_counts.items()
                },
                "total": missing_total
            }
            if include_percentage:
                result["percentages"] = {
                    str(name): (
                        round(float(normalized[name].isna().mean() * 100), 2)
                        if len(normalized) else 0.0
                    )
                    for name in normalized.columns
                }
                pandas_logic += " and per-column missing percentage"

        return {"result": result, "pandas_logic": pandas_logic}

    # 3. Frequency distribution
    if operation == "value_counts":
        counts = df[column].value_counts(dropna=False).head(15).to_dict()
        result = {str(k): int(v) for k, v in counts.items()}
        return {
            "result": result,
            "pandas_logic": f"df['{column}'].value_counts().head(15)"
        }

    # 3a. Counts by category and highest-count category
    if operation in {"groupby_count", "groupby_max_count"}:
        if group_column not in df.columns:
            raise ValueError(f"Group column '{group_column}' not found.")

        grouped = df.groupby(group_column, dropna=False).size()
        counts = {
            ("Missing" if pd.isna(key) else str(key)): int(value)
            for key, value in grouped.items()
        }
        pandas_logic = f"df.groupby('{group_column}', dropna=False).size()"
        if operation == "groupby_count":
            return {
                "result": counts,
                "pandas_logic": pandas_logic
            }

        if not counts:
            raise ValueError(f"Column '{group_column}' has no groups to count.")
        highest_count = max(counts.values())
        highest_groups = [
            group for group, count in counts.items() if count == highest_count
        ]
        return {
            "result": {
                "groups": highest_groups,
                "count": highest_count,
                "group_column": str(group_column)
            },
            "pandas_logic": f"{pandas_logic}.idxmax()"
        }

    if operation == "groupby_rank":
        aggregation = plan.get("aggregation")
        ranking = plan.get("ranking")
        if aggregation not in {"mean", "sum", "count"}:
            raise ValueError("Grouped ranking requires mean, sum, or count.")
        if ranking not in {"max", "min"}:
            raise ValueError("Grouped ranking requires max or min.")
        if group_column not in df.columns:
            raise ValueError(f"Group column '{group_column}' not found.")

        if aggregation == "count":
            grouped = df.groupby(group_column).size()
            pandas_logic = f"df.groupby('{group_column}').size()"
        else:
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(
                    f"Column '{column}' must be numeric for grouped {aggregation}."
                )
            grouped = getattr(
                df.groupby(group_column)[column], aggregation
            )().dropna()
            pandas_logic = (
                f"df.groupby('{group_column}')['{column}'].{aggregation}()"
            )

        if grouped.empty:
            raise ValueError("No groups contain values that can be ranked.")
        winner = grouped.idxmax() if ranking == "max" else grouped.idxmin()
        return {
            "result": str(winner),
            "pandas_logic": f"{pandas_logic}.idx{ranking}()"
        }

    # 4. Unique count
    if operation == "unique_count":
        uniq = int(df[column].nunique(dropna=True))
        return {
            "result": {column: uniq},
            "pandas_logic": f"df['{column}'].nunique(dropna=True)"
        }

    # 5. Numerical scalar aggregations
    if operation in {"mean", "median", "sum", "minimum", "maximum", "std"}:
        numeric_series = pd.to_numeric(df[column], errors="coerce").dropna()
        if numeric_series.empty:
            raise ValueError(
                f"Column '{column}' does not contain usable numerical values."
            )

        op_funcs = {
            "mean": numeric_series.mean,
            "median": numeric_series.median,
            "sum": numeric_series.sum,
            "minimum": numeric_series.min,
            "maximum": numeric_series.max,
            "std": numeric_series.std
        }

        val = op_funcs[operation]()
        val = 0.0 if pd.isna(val) else float(val)

        return {
            "result": {column: round(val, 4)},
            "pandas_logic": f"pd.to_numeric(df['{column}'], errors='coerce').dropna().{operation}()"
        }

    # 6. Grouped aggregations
    if operation in {"groupby_mean", "groupby_sum"}:
        if group_column not in df.columns:
            raise ValueError(f"Group column '{group_column}' not found.")
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' must be numeric for grouped aggregation."
            )

        numeric_series = pd.to_numeric(df[column], errors="coerce")
        temp = pd.DataFrame({
            group_column: df[group_column],
            column: numeric_series
        })

        if operation == "groupby_mean":
            grouped = temp.groupby(group_column)[column].mean().dropna()
            pandas_logic = f"df.groupby('{group_column}')['{column}'].mean()"
        else:
            grouped = temp.groupby(group_column)[column].sum().dropna()
            pandas_logic = f"df.groupby('{group_column}')['{column}'].sum()"

        result = {str(k): round(float(v), 4) for k, v in grouped.items()}
        return {
            "result": result,
            "pandas_logic": pandas_logic
        }

    # 7. Correlation
    if operation == "correlation":
        if second_column not in df.columns:
            raise ValueError(f"Second column '{second_column}' not found.")

        s1 = pd.to_numeric(df[column], errors="coerce")
        s2 = pd.to_numeric(df[second_column], errors="coerce")
        valid = pd.DataFrame({column: s1, second_column: s2}).dropna()

        if len(valid) < 2:
            raise ValueError("At least 2 complete numeric rows are needed for correlation.")

        corr_val = valid[column].corr(valid[second_column])
        if pd.isna(corr_val):
            raise ValueError("Correlation could not be computed (constant or invalid values).")

        return {
            "result": {"correlation": round(float(corr_val), 4)},
            "pandas_logic": f"df['{column}'].corr(df['{second_column}'])"
        }

    raise ValueError(f"Execution not implemented for '{operation}'.")


def generate_local_explanation(question, plan, execution):
    """
    Create a clean, natural-language explanation deterministically.
    Instant latency, zero API calls, 100% mathematically correct.
    """
    op = plan["operation"]
    res = execution["result"]
    col = plan.get("column")

    if op == "count_rows":
        return f"The dataset contains a total of **{res['total_rows']:,}** records."

    if op == "describe":
        return "Statistical summary generated successfully across all features."

    if op == "mean":
        return f"The average (mean) of **{col}** is **{res[col]:,}**."

    if op == "median":
        return f"The median value of **{col}** is **{res[col]:,}**."

    if op == "maximum":
        return f"The maximum (highest) value in **{col}** is **{res[col]:,}**."

    if op == "minimum":
        return f"The minimum (lowest) value in **{col}** is **{res[col]:,}**."

    if op == "sum":
        return f"The total sum of **{col}** is **{res[col]:,}**."

    if op == "std":
        return f"The standard deviation of **{col}** is **{res[col]:,}**, representing the degree of dispersion."

    if op == "unique_count":
        return f"Column **{col}** has **{res[col]}** unique distinct values."

    if op == "value_counts":
        lines = [f"- **{k}**: {v:,}" for k, v in res.items()]
        return f"Frequency distribution for **{col}**:\n\n" + "\n".join(lines)

    if op == "sort_values":
        direction = "ascending" if plan.get("ascending", True) else "descending"
        return (
            f"Rows sorted by **{col}** in {direction} order"
            + (
                f" (showing the top **{plan['limit']}** rows)."
                if plan.get("limit") is not None
                else "."
            )
        )

    if op == "filter_rows":
        count = res["count"]
        return (
            f"Filtered **{count}** rows where **{col}** "
            f"{plan['comparison']} **{plan['value']}**."
        )

    if op == "missing_values":
        if plan["scope"] == "total":
            result = res["total"]
            percentage = (
                f" ({res['percentage']}% of all cells)"
                if "percentage" in res else ""
            )
            return f"The dataset contains **{result:,}** missing values{percentage}."
        if plan["scope"] == "column":
            percentage = (
                f" (**{res['percentage']}%**)"
                if "percentage" in res else ""
            )
            return (
                f"Column **{col}** contains **{res['count']:,}** missing values"
                f"{percentage}."
            )
        lines = [
            f"- **{name}**: {count:,}"
            + (
                f" ({res['percentages'][name]}%)"
                if "percentages" in res else ""
            )
            for name, count in res["by_column"].items()
        ]
        return (
            f"Missing values by column (**{res['total']:,}** total):\n\n"
            + "\n".join(lines)
        )

    if op == "groupby_count":
        lines = [f"- **{k}**: {v:,}" for k, v in res.items()]
        count_label = (
            "employee counts"
            if any(term in question.lower() for term in ["employee", "staff", "people"])
            else "record counts"
        )
        return (
            f"{count_label.capitalize()} for each **{plan['group_column']}**:\n\n"
            + "\n".join(lines)
        )

    if op == "groupby_max_count":
        group_names = ", ".join(f"**{group}**" for group in res["groups"])
        count_label = (
            "employees"
            if any(term in question.lower() for term in ["employee", "staff", "people"])
            else "records"
        )
        return (
            f"**{res['group_column']}** with the most {count_label}: {group_names} "
            f"({res['count']:,} {count_label})."
        )

    if op == "groupby_rank":
        ranking_label = "highest" if plan["ranking"] == "max" else "lowest"
        aggregate_label = {
            "mean": "average",
            "sum": "sum",
            "count": "count"
        }[plan["aggregation"]]
        measure = (
            f" **{col}**" if plan["aggregation"] in {"mean", "sum"} else ""
        )
        return (
            f"**{res}** has the {ranking_label} {aggregate_label}{measure} "
            f"among **{plan['group_column']}** groups."
        )

    if op in {"groupby_mean", "groupby_sum"}:
        agg_label = "average" if op == "groupby_mean" else "total sum"
        lines = [f"- **{k}**: {v:,}" for k, v in res.items()]
        return (
            f"Breakdown of {agg_label} **{col}** by **{plan['group_column']}**:\n\n"
            + "\n".join(lines)
        )

    if op == "correlation":
        c_val = res["correlation"]
        strength = (
            "strong positive" if c_val >= 0.7 else
            "moderate positive" if c_val >= 0.3 else
            "weak/negligible" if c_val > -0.3 else
            "moderate negative" if c_val > -0.7 else
            "strong negative"
        )
        return (
            f"The Pearson correlation coefficient between **{col}** and **{plan['second_column']}** "
            f"is **{c_val}**, indicating a **{strength}** linear relationship."
        )

    return f"Computed result: {res}"


def explain_result(question, plan, execution):
    """
    Optional Groq explanation when detailed narrative reasoning is requested.
    """
    client = get_groq_client()

    prompt = f"""
You are an expert data science analyst.
The user asked: "{question}"
Analysis operation performed: {execution['pandas_logic']}
Calculation result: {json.dumps(execution['result'], default=str)}

Explain this result to the user in 1-2 clear, concise sentences.
Do not hallucinate or invent numbers. Use only the provided result.
"""
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Explain analysis results briefly without inventing facts."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return completion_text(response)


def answer_question(
    df,
    question,
    use_fast_path=True,
    explain_with_llm=False,
    execution_mode="json_plan",
):
    """
    High-performance question answering pipeline:
    1. Fast-Path Pandas: Instantly matches common questions locally (<5ms latency, 0 API calls).
    2. Single-Call LLM: If complex, asks Groq to plan once, executes via pandas, and formats locally.
    3. Robust Fallback: Recovers cleanly from API errors, quota exhaustion, or missing keys.
    """
    if not question or not str(question).strip():
        raise ValueError("Please enter a question.")
    if execution_mode not in {"json_plan", "pandas_code"}:
        raise ValueError("execution_mode must be 'json_plan' or 'pandas_code'.")

    start_time = time.perf_counter()
    used_fallback = False
    plan = None
    source = "unknown"

    # Step 1: Attempt Fast-Path local resolution
    if use_fast_path:
        local_plan = fast_local_plan(df, question)
        if local_plan:
            try:
                execution = execute_plan(df, local_plan)
                explanation = generate_local_explanation(question, local_plan, execution)
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return {
                    "plan": local_plan,
                    "execution": execution,
                    "explanation": explanation,
                    "used_fallback": False,
                    "source": "fast_pandas",
                    "latency_ms": elapsed_ms
                }
            except Exception:
                # If local execution hit unexpected data error, continue to Groq/fallback.
                pass

    if execution_mode == "pandas_code":
        try:
            code = interpret_pandas_code(df, question)
        except _PandasCodeRequestError as error:
            raise RuntimeError(
                f"Groq pandas-code request failed: {error}"
            ) from None
        except ValueError as error:
            raise RuntimeError(
                f"Generated pandas code validation failed: {error}"
            ) from None
        except Exception as error:
            raise RuntimeError(
                "Groq pandas-code request failed: "
                f"{summarize_groq_error(error)}"
            ) from None
        try:
            _validate_pandas_code_intent(df, question, code)
        except ValueError as error:
            raise RuntimeError(
                f"Generated pandas code validation failed: {error}"
            ) from None
        try:
            execution = execute_restricted_pandas_code(df, code)
        except Exception as error:
            detail = (
                str(error)
                if isinstance(error, ValueError)
                else f"{type(error).__name__}"
            )
            raise RuntimeError(
                f"Generated pandas code execution failed: {detail}"
            ) from None
        return {
            "plan": {"operation": "restricted_pandas_code", "source": "groq"},
            "execution": execution,
            "explanation": _explain_restricted_result(execution["result"]),
            "used_fallback": False,
            "source": "groq_pandas_ast",
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }

    # Step 2: Use Groq for complex or ambiguous natural language questions.
    try:
        plan = interpret_question(df, question)
        source = "groq"
        if plan.get("operation") == "unsupported":
            raise ValueError("Unsupported operation requested.")
        _validate_grouped_plan_intent(df, question, plan)
        execution = execute_plan(df, plan)

        # Explain remotely only when the caller explicitly requests it.
        if explain_with_llm:
            try:
                explanation = explain_result(question, plan, execution)
            except Exception:
                explanation = generate_local_explanation(question, plan, execution)
        else:
            explanation = generate_local_explanation(question, plan, execution)

    except Exception as llm_error:
        if (
            isinstance(llm_error, ValueError)
            and (
                str(llm_error).startswith("Groq returned a plan")
                or "asks for multiple measures" in str(llm_error)
            )
        ):
            raise RuntimeError(str(llm_error)) from None

        # Step 3: Resilient fallback
        used_fallback = True
        source = "local_fallback"
        plan = fast_local_plan(df, question)
        if not plan:
            raise RuntimeError(
                "The question could not be resolved locally and Groq did not return "
                "a usable analysis plan. "
                f"{summarize_groq_error(llm_error)}"
            ) from llm_error

        execution = execute_plan(df, plan)
        explanation = generate_local_explanation(question, plan, execution)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "plan": plan,
        "execution": execution,
        "explanation": explanation,
        "used_fallback": used_fallback,
        "source": source,
        "latency_ms": elapsed_ms
    }
