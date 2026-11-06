import pandas as pd
import pytest

from modules.dataset_catalog import (
    ADULT_COLUMNS,
    CURATED_DATASETS,
    get_dataset_metadata,
    load_curated_dataset,
    normalize_adult_dataset,
)


@pytest.mark.parametrize(
    ("name", "rows", "columns", "missing_values"),
    [
        ("Adult Census Income", 48842, ADULT_COLUMNS, 6465),
        (
            "Palmer Penguins",
            344,
            [
                "species",
                "island",
                "bill_length_mm",
                "bill_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
                "sex",
                "year",
            ],
            19,
        ),
        (
            "Wine Quality Red",
            1599,
            [
                "fixed acidity",
                "volatile acidity",
                "citric acid",
                "residual sugar",
                "chlorides",
                "free sulfur dioxide",
                "total sulfur dioxide",
                "density",
                "pH",
                "sulphates",
                "alcohol",
                "quality",
            ],
            0,
        ),
        (
            "Wine Quality White",
            4898,
            [
                "fixed acidity",
                "volatile acidity",
                "citric acid",
                "residual sugar",
                "chlorides",
                "free sulfur dioxide",
                "total sulfur dioxide",
                "density",
                "pH",
                "sulphates",
                "alcohol",
                "quality",
            ],
            0,
        ),
    ],
)
def test_curated_dataset_row_count_schema_and_missing_values(
    name, rows, columns, missing_values
):
    df = load_curated_dataset(name)

    assert len(df) == rows
    assert df.columns.tolist() == columns
    assert int(df.isna().sum().sum()) == missing_values


def test_adult_normalization_handles_whitespace_missing_markers_and_test_labels():
    df = pd.DataFrame({
        " workclass ": [" Private ", " ? ", " State-gov "],
        " income ": [" <=50K. ", " >50K. ", " >50K "],
    })

    normalized = normalize_adult_dataset(df)

    assert normalized.columns.tolist() == ["workclass", "income"]
    assert normalized.loc[0, "workclass"] == "Private"
    assert pd.isna(normalized.loc[1, "workclass"])
    assert normalized["income"].tolist() == ["<=50K", ">50K", ">50K"]


def test_curated_categorical_values_and_target_labels_are_preserved():
    adult = load_curated_dataset("Adult Census Income")
    penguins = load_curated_dataset("Palmer Penguins")
    red_wine = load_curated_dataset("Wine Quality Red")

    assert set(adult["income"].dropna().unique()) == {"<=50K", ">50K"}
    assert set(adult["sex"].dropna().unique()) == {"Female", "Male"}
    assert set(penguins["species"].dropna().unique()) == {
        "Adelie",
        "Chinstrap",
        "Gentoo",
    }
    assert set(red_wine["quality"].dropna().unique()).issubset(set(range(0, 11)))


@pytest.mark.parametrize("name", list(CURATED_DATASETS))
def test_curated_dataset_metadata_has_source_citation_and_license(name):
    metadata = get_dataset_metadata(name)

    assert metadata["source"]
    assert metadata["source_url"].startswith("https://")
    assert metadata["citation"]
    assert metadata["license"] in {"CC BY 4.0", "CC0"}
    assert metadata["license_url"].startswith("https://")


def test_unknown_curated_dataset_is_reported():
    with pytest.raises(ValueError, match="Unknown curated dataset"):
        load_curated_dataset("Unknown dataset")
