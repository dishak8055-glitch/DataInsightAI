
import pandas as pd

from modules.visualization import (
    create_histogram,
    create_boxplot,
    create_bar_chart,
    create_correlation_heatmap
)


def test_histogram():

    df = pd.DataFrame({"Age": [20, 25, 30]})

    fig = create_histogram(df, "Age")

    assert fig is not None
    assert fig.data[0].type == "histogram"
    assert list(fig.data[0].x) == [20, 25, 30]


def test_histogram_sample_housing_dataset():
    df = pd.read_csv("sample_housing.csv")

    fig = create_histogram(df, "PriceUSD")

    assert fig is not None
    assert fig.data[0].type == "histogram"
    assert len(fig.data[0].x) == len(df)


def test_histogram_returns_none_for_all_missing_values():
    df = pd.DataFrame({"Value": [None, float("nan")]})

    assert create_histogram(df, "Value") is None


def test_boxplot():

    df = pd.DataFrame({"Income": [20000, 30000, 40000]})

    fig = create_boxplot(df, "Income")

    assert fig is not None


def test_bar_chart():

    df = pd.DataFrame({
        "Gender": ["Male", "Female", "Male"]
    })

    fig = create_bar_chart(df, "Gender")

    assert fig is not None


def test_correlation_heatmap():

    df = pd.DataFrame({
        "Age": [20, 30, 40],
        "Income": [20000, 40000, 60000]
    })

    fig = create_correlation_heatmap(
        df, ["Age", "Income"]
    )

    assert fig is not None



def test_scatter_plot():
    df = pd.DataFrame({
        "Age": [20, 25, 30],
        "Income": [20000, 30000, 40000]
    })
    from modules.visualization import create_scatter_plot
    fig = create_scatter_plot(df, "Age", "Income")
    assert fig is not None


def test_violin_plot():
    df = pd.DataFrame({
        "Age": [20, 25, 30],
        "Gender": ["M", "F", "M"]
    })
    from modules.visualization import create_violin_plot
    fig = create_violin_plot(df, "Age", "Gender")
    assert fig is not None
