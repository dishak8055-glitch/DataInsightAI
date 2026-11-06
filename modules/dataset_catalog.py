"""Metadata and local loading helpers for bundled public datasets."""

from pathlib import Path

import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATASET_DIRECTORY = _PROJECT_ROOT / "datasets"

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

CURATED_DATASETS = {
    "Adult Census Income": {
        "filename": "adult_census_income.csv",
        "delimiter": ",",
        "source": "UCI Machine Learning Repository",
        "source_url": "https://archive.ics.uci.edu/dataset/2/adult",
        "citation": (
            "Becker, B. & Kohavi, R. (1996). Adult [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20."
        ),
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "expected_rows": 48842,
        "expected_columns": ADULT_COLUMNS,
    },
    "Palmer Penguins": {
        "filename": "palmer_penguins.csv",
        "delimiter": ",",
        "source": "palmerpenguins project; data collected by Kristen Gorman and Palmer Station LTER",
        "source_url": "https://github.com/allisonhorst/palmerpenguins",
        "citation": (
            "Horst, A. M., Hill, A. P., & Gorman, K. B. (2020). "
            "palmerpenguins: Palmer Archipelago (Antarctica) penguin data "
            "(Version 0.1.0). Zenodo. https://doi.org/10.5281/zenodo.3960218."
        ),
        "license": "CC0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "expected_rows": 344,
        "expected_columns": [
            "species",
            "island",
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
            "year",
        ],
    },
    "Wine Quality Red": {
        "filename": "wine_quality_red.csv",
        "delimiter": ";",
        "source": "UCI Machine Learning Repository",
        "source_url": "https://archive.ics.uci.edu/dataset/186/wine+quality",
        "citation": (
            "Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). "
            "Wine Quality [Dataset]. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C56S3T."
        ),
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "expected_rows": 1599,
        "expected_columns": [
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
    },
    "Wine Quality White": {
        "filename": "wine_quality_white.csv",
        "delimiter": ";",
        "source": "UCI Machine Learning Repository",
        "source_url": "https://archive.ics.uci.edu/dataset/186/wine+quality",
        "citation": (
            "Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). "
            "Wine Quality [Dataset]. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C56S3T."
        ),
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "expected_rows": 4898,
        "expected_columns": [
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
    },
}


def get_dataset_metadata(dataset_name):
    """Return a copy of metadata for a bundled curated dataset."""
    try:
        return CURATED_DATASETS[dataset_name].copy()
    except KeyError as exc:
        raise ValueError(f"Unknown curated dataset: {dataset_name}") from exc


def normalize_adult_dataset(df):
    """Normalize whitespace, missing markers, and income labels for Adult CSV data."""
    normalized = df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    if normalized.shape[1] == len(ADULT_COLUMNS) and normalized.columns.tolist() != ADULT_COLUMNS:
        normalized.columns = ADULT_COLUMNS

    for column in normalized.select_dtypes(include=["object", "string"]).columns:
        values = normalized[column].astype("string").str.strip()
        normalized[column] = values.mask(values.eq("?"), pd.NA)

    if "income" in normalized.columns:
        income = normalized["income"].astype("string").str.strip()
        normalized["income"] = income.str.replace(r"\.$", "", regex=True)

    return normalized


def load_curated_dataset(dataset_name):
    """Load a bundled curated dataset without dropping or imputing missing values."""
    metadata = get_dataset_metadata(dataset_name)
    path = _DATASET_DIRECTORY / metadata["filename"]
    if not path.is_file():
        raise FileNotFoundError(f"Bundled dataset file not found: {path}")

    df = pd.read_csv(
        path,
        sep=metadata["delimiter"],
        skipinitialspace=True,
        na_values=["?"],
        keep_default_na=True,
    )
    if dataset_name == "Adult Census Income":
        df = normalize_adult_dataset(df)
    return df
