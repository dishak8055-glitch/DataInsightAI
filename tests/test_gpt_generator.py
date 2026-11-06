# tests/test_gpt_generator.py

import pandas as pd
from modules.gpt_generator import (
    generate_dataset_story,
    get_transformer_architectures_comparison
)


def test_generate_dataset_story():
    df = pd.DataFrame({
        "Age": [22, 35, 28],
        "SpendingScore": [80, 45, 90]
    })
    res = generate_dataset_story(df, target_column="SpendingScore", use_api=False)
    assert "narrative" in res
    assert len(res["narrative"]) > 50
    assert "facts" in res


def test_transformer_architectures_comparison():
    comp = get_transformer_architectures_comparison()
    assert len(comp) == 3
    assert comp[0]["Family"].startswith("GPT")
    assert comp[1]["Family"].startswith("BERT")
    assert comp[2]["Family"].startswith("T5")
