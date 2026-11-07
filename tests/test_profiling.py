
import pandas as pd

from modules.profiling import (
    profile_dataset,
    identify_column_types
)


def test_profile():

    df = pd.DataFrame({
        "CustomerID": [1, 2, 3],
        "Age": [20, 30, 40],
        "Gender": ["F", "M", "F"]
    })

    result = profile_dataset(df)

    assert result["total_rows"] == 3
    assert result["total_columns"] == 3
    assert result["missing_values"] == 0


def test_column_detection():

    df = pd.DataFrame({
        "CustomerID": [1, 2, 3],
        "Age": [20, 30, 40],
        "Gender": ["F", "M", "F"]
    })

    result = identify_column_types(df)

    assert "CustomerID" in result["identifiers"]
    assert "Age" in result["numerical"]
    assert "Gender" in result["categorical"]


def test_identifier_detection_uses_name_tokens_not_substrings():
    df = pd.DataFrame({
        "CustomerID": [1, 2],
        "user_id": [3, 4],
        "IDNumber": [5, 6],
        "citric acid": [0.1, 0.2],
    })

    result = identify_column_types(df)

    assert result["identifiers"] == ["CustomerID", "user_id", "IDNumber"]
    assert result["numerical"] == ["citric acid"]
