# tests/test_synthetic_data.py

import pandas as pd
from modules.synthetic_data import (
    TabularCTGAN,
    compare_real_and_synthetic,
    create_side_by_side_distribution
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


def test_create_side_by_side_distribution():
    real_df = pd.DataFrame({"Age": [20, 30, 40]})
    synth_df = pd.DataFrame({"Age": [22, 32, 42]})
    fig = create_side_by_side_distribution(real_df, synth_df, "Age")
    assert fig is not None
