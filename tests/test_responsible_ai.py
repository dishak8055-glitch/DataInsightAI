# tests/test_responsible_ai.py

import pandas as pd
from modules.responsible_ai import (
    detect_sensitive_columns,
    detect_class_imbalance,
    evaluate_demographic_fairness,
    verify_hallucination,
    audit_dataset_license,
    generate_model_card
)


def test_detect_sensitive_columns():
    df = pd.DataFrame({
        "CustomerID": [1, 2],
        "Gender": ["Female", "Male"],
        "UserEmail": ["alice@example.com", "bob@test.org"],
        "Age": [25, 40],
        "Income": [50000, 60000]
    })
    res = detect_sensitive_columns(df)
    assert not res["is_clean"]
    assert res["total_sensitive_found"] >= 3  # Gender, Email, Age
    detected_cols = [x["Column"] for x in res["sensitive_columns"]]
    assert "Gender" in detected_cols
    assert "UserEmail" in detected_cols
    assert "Age" in detected_cols


def test_detect_class_imbalance():
    df = pd.DataFrame({
        "Target": ["Yes"] * 20 + ["No"] * 4
    })
    res = detect_class_imbalance(df, "Target")
    assert res is not None
    assert res["is_imbalanced"] is True
    assert res["ratio"] == 5.0
    assert "Severe Imbalance" in res["severity"]


def test_evaluate_demographic_fairness():
    df = pd.DataFrame({
        "Gender": ["Male", "Male", "Male", "Male", "Female", "Female", "Female", "Female"],
        "Approved": [1, 1, 1, 0, 1, 0, 0, 0]  # Male rate 75%, Female rate 25% -> DIR = 25/75 = 0.33
    })
    fairness = evaluate_demographic_fairness(df, "Gender", "Approved")
    assert fairness is not None
    assert fairness["violates_80_rule"] is True
    assert fairness["disparate_impact_ratio"] < 0.80


def test_verify_hallucination():
    df = pd.DataFrame({
        "Age": [20, 30, 40],
        "Salary": [50000, 60000, 70000]
    })
    # Text stating the true mean salary (60000) and max salary (70000)
    grounded_text = "The mean salary of the cohort is 60000, reaching a maximum of 70000."
    res1 = verify_hallucination(grounded_text, df)
    assert res1["grounding_score"] >= 80.0
    assert res1["hallucination_risk"] == "Low"

    # Text making up ungrounded numbers
    hallucinated_text = "The average salary was 999999 with 888888 bonus."
    res2 = verify_hallucination(hallucinated_text, df)
    assert res2["grounding_score"] < 50.0
    assert res2["hallucination_risk"] == "High"


def test_audit_dataset_license():
    lic = audit_dataset_license("CC-BY 4.0")
    assert lic["commercial_use"] is True
    assert lic["attribution_required"] is True
    assert lic["ai_training_safe"] is True


def test_generate_model_card():
    df = pd.DataFrame({
        "Age": [25, 30],
        "Salary": [50000, 60000]
    })
    card = generate_model_card(df, target_column="Salary", recommended_task="Regression")
    assert "# Model Card" in card
    assert "DataInsight AI" in card
    assert "Salary" in card
