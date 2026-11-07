# tests/test_gpt_generator.py

from types import SimpleNamespace

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
    assert res["generator_type"] == "Deterministic structured fallback (not an LLM)"


def test_generate_dataset_story_uses_groq_with_aggregate_facts_only(monkeypatch):
    df = pd.DataFrame(
        {"Age": [22, 35, 28], "CustomerName": ["Private A", "Private B", "Private C"]}
    )
    calls = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="The dataset has three records and two columns."
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(
        "modules.gpt_generator.get_groq_api_key", lambda: "test-key"
    )
    monkeypatch.setattr(
        "modules.gpt_generator.get_groq_client", lambda: fake_client
    )

    result = generate_dataset_story(df, use_api=True)

    assert result["generator_type"] == (
        "Groq chat completion (openai/gpt-oss-20b)"
    )
    prompt = str(calls["messages"])
    assert "Private A" not in prompt
    assert "CustomerName" in prompt


def test_generate_dataset_story_falls_back_when_groq_fails(monkeypatch):
    monkeypatch.setattr(
        "modules.gpt_generator.get_groq_api_key", lambda: "test-key"
    )

    def fail_request():
        raise TimeoutError("private provider response")

    monkeypatch.setattr(
        "modules.gpt_generator.get_groq_client", fail_request
    )

    result = generate_dataset_story(
        pd.DataFrame({"value": [1, 2]}), use_api=True
    )

    assert result["generator_type"] == (
        "Deterministic fallback (Groq request failed)"
    )
    assert "timed out" in result["api_error"]
    assert "private provider response" not in result["api_error"]


def test_transformer_architectures_comparison():
    comp = get_transformer_architectures_comparison()
    assert len(comp) == 3
    assert comp[0]["Family"].startswith("GPT")
    assert comp[1]["Family"].startswith("BERT")
    assert comp[2]["Family"].startswith("T5")
