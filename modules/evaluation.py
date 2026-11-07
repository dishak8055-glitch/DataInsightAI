"""Reproducible evaluation helpers for curated dataset column-type detection."""

from modules.dataset_catalog import ADULT_COLUMNS, CURATED_DATASETS
from modules.profiling import identify_column_types


EXPECTED_COLUMN_TYPES = {
    "Adult Census Income": {
        "age": "numerical",
        "workclass": "categorical",
        "fnlwgt": "numerical",
        "education": "categorical",
        "education-num": "numerical",
        "marital-status": "categorical",
        "occupation": "categorical",
        "relationship": "categorical",
        "race": "categorical",
        "sex": "categorical",
        "capital-gain": "numerical",
        "capital-loss": "numerical",
        "hours-per-week": "numerical",
        "native-country": "categorical",
        "income": "categorical",
    },
    "Palmer Penguins": {
        "species": "categorical",
        "island": "categorical",
        "bill_length_mm": "numerical",
        "bill_depth_mm": "numerical",
        "flipper_length_mm": "numerical",
        "body_mass_g": "numerical",
        "sex": "categorical",
        "year": "numerical",
    },
    "Wine Quality Red": {
        "fixed acidity": "numerical",
        "volatile acidity": "numerical",
        "citric acid": "numerical",
        "residual sugar": "numerical",
        "chlorides": "numerical",
        "free sulfur dioxide": "numerical",
        "total sulfur dioxide": "numerical",
        "density": "numerical",
        "pH": "numerical",
        "sulphates": "numerical",
        "alcohol": "numerical",
        "quality": "numerical",
    },
    "Wine Quality White": {
        "fixed acidity": "numerical",
        "volatile acidity": "numerical",
        "citric acid": "numerical",
        "residual sugar": "numerical",
        "chlorides": "numerical",
        "free sulfur dioxide": "numerical",
        "total sulfur dioxide": "numerical",
        "density": "numerical",
        "pH": "numerical",
        "sulphates": "numerical",
        "alcohol": "numerical",
        "quality": "numerical",
    },
}


def _column_type_map(df):
    identified = identify_column_types(df)
    result = {}
    for label in ("numerical", "categorical", "identifiers", "datetime"):
        result.update({column: label for column in identified[label]})
    return result


def evaluate_column_types(dataset_name, df):
    """Compare profiler outputs with manually reviewed expected column types."""
    if dataset_name not in CURATED_DATASETS:
        raise ValueError(f"Unknown curated dataset: {dataset_name}")

    expected = EXPECTED_COLUMN_TYPES[dataset_name]
    predicted = _column_type_map(df)
    if set(expected) != set(df.columns):
        raise ValueError(
            f"Expected-type schema for {dataset_name} does not match the loaded data."
        )

    comparisons = [
        {
            "column": column,
            "expected": expected[column],
            "predicted": predicted[column],
            "correct": expected[column] == predicted[column],
        }
        for column in df.columns
    ]
    correct = sum(item["correct"] for item in comparisons)
    total = len(comparisons)
    return {
        "dataset": dataset_name,
        "correct": correct,
        "total": total,
        "accuracy_pct": round(correct / total * 100, 2) if total else None,
        "columns": comparisons,
        "notes": (
            "Types describe stored analytical roles; integer-valued ordinal "
            "quality and education-num are treated as numerical."
            if dataset_name == "Adult Census Income"
            or dataset_name.startswith("Wine Quality")
            else "Stored numeric measurements and year are numerical."
        ),
    }


def validate_expected_type_schemas():
    """Fail fast if the manual labels drift from the curated catalog schemas."""
    for dataset_name, metadata in CURATED_DATASETS.items():
        expected = EXPECTED_COLUMN_TYPES[dataset_name]
        if set(expected) != set(metadata["expected_columns"]):
            raise ValueError(
                f"Manual expected-type labels do not match {dataset_name} schema."
            )
        if dataset_name == "Adult Census Income":
            if list(expected) != ADULT_COLUMNS:
                raise ValueError("Adult expected types do not match the canonical schema.")
