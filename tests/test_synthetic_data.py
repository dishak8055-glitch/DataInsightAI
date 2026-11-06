# tests/test_synthetic_data.py

import pandas as pd
import torch
from modules.synthetic_data import (
    TabularCTGAN,
    compare_real_and_synthetic,
    create_side_by_side_distribution,
    sanitize_spreadsheet_formulas
)


def test_tabular_ctgan_training_and_sampling():
    df = pd.DataFrame({
        "Age": [22, 35, 28, 42, 25, 50, 31, 39, 23, 45],
        "AnnualIncome": [25000, 45000, 32000, 60000, 28000, 75000, 40000, 55000, 26000, 68000],
        "Gender": ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"]
    })

    # Train CTGAN for a few epochs
    gan = TabularCTGAN(latent_dim=8, epochs=15, batch_size=8)
    gan.fit(df)
    assert gan.is_fitted

    # Sample synthetic records
    synth = gan.sample(n_samples=10)
    assert len(synth) == 10
    assert "Age" in synth.columns
    assert "AnnualIncome" in synth.columns
    assert "Gender" in synth.columns
    assert synth["Gender"].isin(df["Gender"]).all()
    assert list(synth.columns) == list(df.columns)


def test_conditional_gan_handles_missing_and_constant_values():
    df = pd.DataFrame({
        "Category": ["A", None, "A", "B"],
        "Constant": [7, 7, 7, 7],
        "Value": [1.0, None, 3.0, 4.0]
    })
    gan = TabularCTGAN(latent_dim=4, epochs=1, batch_size=4).fit(df)
    synthetic = gan.sample(4)
    assert list(synthetic.columns) == list(df.columns)
    assert synthetic["Category"].isin(["A", "B", "Missing"]).all()
    assert (synthetic["Constant"] == 7).all()


def test_conditional_gan_excludes_identifiers_and_recreates_labeled_ids():
    df = pd.DataFrame({
        "CustomerID": [101, 102, 103, 104],
        "user_id": ["u1", "u2", "u3", "u4"],
        "Age": [22, 35, 28, 42],
        "Gender": ["Female", "Male", "Female", "Male"]
    })

    gan = TabularCTGAN(latent_dim=4, epochs=1, batch_size=4).fit(df)
    synthetic = gan.sample(4)

    assert gan.id_cols == ["CustomerID", "user_id"]
    assert gan.columns == ["Age", "Gender"]
    assert list(synthetic.columns) == list(df.columns)
    assert synthetic["CustomerID"].tolist() == [
        "SYNTHETIC_CustomerID_1",
        "SYNTHETIC_CustomerID_2",
        "SYNTHETIC_CustomerID_3",
        "SYNTHETIC_CustomerID_4"
    ]
    assert synthetic["user_id"].tolist() == [
        "SYNTHETIC_user_id_1",
        "SYNTHETIC_user_id_2",
        "SYNTHETIC_user_id_3",
        "SYNTHETIC_user_id_4"
    ]
    assert synthetic["Gender"].isin(df["Gender"]).all()


def test_integer_columns_are_rounded_and_all_numeric_columns_stay_in_range(monkeypatch):
    df = pd.DataFrame({
        "Bedrooms": [1.0, 2.0, 3.0, 4.0],
        "PriceUSD": [100.123, 200.789, 300.456, 400.987],
        "DistanceToCityKm": [0.123, 0.456, 0.789, 0.987],
        "Gender": ["A", "B", "A", "B"]
    })
    gan = TabularCTGAN(latent_dim=4, epochs=1, batch_size=4).fit(df)
    monkeypatch.setattr(
        gan,
        "_activate",
        lambda _: torch.tensor([
            [0.2, -0.5, 0.25, 0.0, 1.0],
            [-0.4, 0.0, -0.25, 1.0, 0.0]
        ])
    )

    synthetic = gan.sample(2)

    assert gan.integer_cols == ["Bedrooms"]
    assert pd.api.types.is_integer_dtype(synthetic["Bedrooms"])
    assert synthetic["Bedrooms"].between(1, 4).all()
    assert synthetic["PriceUSD"].between(df["PriceUSD"].min(), df["PriceUSD"].max()).all()
    assert synthetic["DistanceToCityKm"].between(
        df["DistanceToCityKm"].min(), df["DistanceToCityKm"].max()
    ).all()
    assert (synthetic["PriceUSD"] % 1 != 0).all()
    assert (synthetic["DistanceToCityKm"] % 1 != 0).all()
    assert synthetic["Gender"].isin(df["Gender"]).all()


def test_comparison_excludes_numeric_identifier_columns():
    real_df = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "CustomerID": [10, 20, 30, 40],
        "Value": [1, 2, 3, 4]
    })
    synth_df = pd.DataFrame({
        "ID": [100, 200, 300, 400],
        "CustomerID": [1000, 2000, 3000, 4000],
        "Value": [1, 2, 3, 4]
    })

    comparison = compare_real_and_synthetic(real_df, synth_df)

    assert comparison["num_features_evaluated"] == 1
    assert comparison["metrics_table"]["Feature"].tolist() == ["Value"]


def test_conditional_gan_rejects_empty_training_data():
    gan = TabularCTGAN(epochs=1)
    try:
        gan.fit(pd.DataFrame())
    except ValueError as exc:
        assert "at least one row and one column" in str(exc)
    else:
        raise AssertionError("Training on an empty dataframe should fail.")


def test_compare_real_and_synthetic():
    real_df = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40, 45],
        "Salary": [20000, 25000, 30000, 35000, 40000, 45000]
    })
    synth_df = pd.DataFrame({
        "Age": [22, 26, 31, 36, 41, 46],
        "Salary": [21000, 26000, 31000, 36000, 41000, 46000]
    })
    eval_res = compare_real_and_synthetic(real_df, synth_df)
    assert "metrics_table" in eval_res
    assert "aggregate_quality_score" in eval_res
    assert eval_res["aggregate_quality_score"] > 50.0


def test_compare_categorical_only_data_does_not_invent_scores():
    real_df = pd.DataFrame({"group": ["A", "B", "A"]})
    synth_df = pd.DataFrame({"group": ["A", "A", "B"]})
    eval_res = compare_real_and_synthetic(real_df, synth_df)
    assert eval_res["aggregate_quality_score"] is None
    assert eval_res["correlation_similarity_score"] is None
    assert eval_res["num_features_evaluated"] == 0


def test_create_side_by_side_distribution():
    real_df = pd.DataFrame({"Age": [20, 30, 40]})
    synth_df = pd.DataFrame({"Age": [22, 32, 42]})
    fig = create_side_by_side_distribution(real_df, synth_df, "Age")
    assert fig is not None


def test_sanitize_spreadsheet_formulas_only_changes_risky_text():
    df = pd.DataFrame({
        "category": ["=1+1", "  @SUM(A1:A2)", "+cmd", "ordinary", None],
        "number": [-1, 2, 3, 4, 5]
    })

    sanitized = sanitize_spreadsheet_formulas(df)

    assert sanitized["category"].tolist()[:4] == [
        "'=1+1", "'  @SUM(A1:A2)", "'+cmd", "ordinary"
    ]
    assert pd.isna(sanitized.loc[4, "category"])
    assert sanitized["number"].tolist() == [-1, 2, 3, 4, 5]
    assert df.loc[0, "category"] == "=1+1"
