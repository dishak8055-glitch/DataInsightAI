# tests/test_column_embeddings.py

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from modules.column_embeddings import (
    _encode_contexts_with_bert,
    group_similar_columns_bert,
)


def test_column_similarity_and_clustering():
    df = pd.DataFrame({
        "CustomerID": [1, 2, 3, 4],
        "AccountID": [101, 102, 103, 104],
        "AnnualIncome": [25000, 45000, 32000, 60000],
        "MonthlySalary": [2100, 3750, 2660, 5000],
        "Gender": ["Female", "Male", "Female", "Male"]
    })
    res = group_similar_columns_bert(df, similarity_threshold=0.3)
    assert "clusters" in res
    assert "similarity_matrix" in res
    assert res["similarity_matrix"].shape == (5, 5)
    # CustomerID and AccountID or Income and Salary should have elevated similarity
    sim_mat = res["similarity_matrix"]
    assert sim_mat.loc["CustomerID", "CustomerID"] == 1.0
    assert res["figure"] is not None
    assert "TF-IDF baseline" in res["embedding_backend"]


def test_column_similarity_rejects_unknown_embedding_backend():
    df = pd.DataFrame({"first": [1], "second": [2]})
    try:
        group_similar_columns_bert(df, embedding_backend="unknown")
    except ValueError as exc:
        assert "embedding_backend" in str(exc)
    else:
        raise AssertionError("An unknown embedding backend should be rejected.")


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), True])
def test_column_similarity_rejects_invalid_threshold(threshold):
    df = pd.DataFrame({"first": [1], "second": [2]})

    with pytest.raises(ValueError, match="between 0 and 1"):
        group_similar_columns_bert(df, similarity_threshold=threshold)


def test_column_similarity_rejects_duplicate_column_names():
    df = pd.DataFrame([[1, 2]], columns=["same", "same"])

    with pytest.raises(ValueError, match="unique column names"):
        group_similar_columns_bert(df)


def test_bert_encoder_uses_attention_mask_and_normalizes_embeddings(monkeypatch):
    import torch

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            assert texts == ["column context", "another context"]
            return {
                "input_ids": torch.tensor([[1, 2], [3, 4]]),
                "attention_mask": torch.tensor([[1, 1], [1, 0]]),
            }

    class FakeModel:
        def eval(self):
            self.evaluated = True

        def __call__(self, **kwargs):
            return SimpleNamespace(
                last_hidden_state=torch.tensor(
                    [
                        [[1.0, 0.0], [0.0, 1.0]],
                        [[3.0, 4.0], [100.0, 100.0]],
                    ]
                )
            )

    model = FakeModel()
    _encode_contexts_with_bert.cache_clear()
    monkeypatch.setattr(
        "modules.column_embeddings._load_bert_encoder",
        lambda model_name: (FakeTokenizer(), model),
    )

    embeddings = _encode_contexts_with_bert(
        ("column context", "another context"), "test-bert"
    )

    np.testing.assert_allclose(
        embeddings,
        np.array([[2**-0.5, 2**-0.5], [0.6, 0.8]]),
        atol=1e-6,
    )
    assert model.evaluated
    _encode_contexts_with_bert.cache_clear()


def test_bert_backend_groups_using_mocked_embeddings(monkeypatch):
    df = pd.DataFrame({
        "annual_income": [1, 2],
        "yearly_salary": [3, 4],
        "region": ["north", "south"],
    })
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.1, 0.0],
        [0.0, 0.0, 1.0],
    ])
    monkeypatch.setattr(
        "modules.column_embeddings._encode_contexts_with_bert",
        lambda contexts, model_name: embeddings,
    )

    result = group_similar_columns_bert(
        df, similarity_threshold=0.8, embedding_backend="bert"
    )

    assert result["embedding_backend"] == (
        "BERT embeddings (prajjwal1/bert-tiny)"
    )
    income_group = next(
        group for group in result["clusters"].values()
        if "annual_income" in group
    )
    assert income_group == ["annual_income", "yearly_salary"]
