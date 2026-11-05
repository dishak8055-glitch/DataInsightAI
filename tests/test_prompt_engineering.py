# tests/test_prompt_engineering.py

import pandas as pd
from modules.prompt_engineering import (
    generate_prompt_templates,
    simulate_sampling_distribution
)


def test_generate_prompt_templates():
    df = pd.DataFrame({
        "CustomerID": [1, 2, 3],
        "Age": [22, 35, 28],
        "Gender": ["Female", "Male", "Female"],
        "SpendingScore": [80, 45, 90]
    })
    templates = generate_prompt_templates(df, target_column="SpendingScore")
    assert "zero_shot" in templates
    assert "few_shot" in templates
    assert "chain_of_thought" in templates
    assert "system_role" in templates
    assert "structured_output" in templates
    assert "SpendingScore" in templates["zero_shot"]["prompt"] or "Age" in templates["zero_shot"]["prompt"]


def test_simulate_sampling_distribution():
    res = simulate_sampling_distribution(temperature=0.8, top_k=4, top_p=0.9)
    assert not res.empty
    assert "Final Sample Prob (%)" in res.columns
    total_prob = res["Final Sample Prob (%)"].sum()
    assert round(total_prob, 0) == 100.0
