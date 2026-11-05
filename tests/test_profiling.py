
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
