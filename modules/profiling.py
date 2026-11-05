
import pandas as pd



def identify_column_types(df):

    numerical = []
    categorical = []
    identifiers = []
    datetime_columns = []

    for column in df.columns:

        name = str(column).lower().strip()
        series = df[column]

        # Identify ID columns (case-insensitive)
        if (
            name == "id"
            or name.endswith("id")
            or name.startswith("id_")
        ):
            identifiers.append(column)

        # Identify datetime columns
        elif (
            pd.api.types.is_datetime64_any_dtype(series)
            or any(word in name for word in ["date", "time"])
        ):
            datetime_columns.append(column)

        # Identify numerical columns
        elif pd.api.types.is_numeric_dtype(series):
            numerical.append(column)

        # Identify categorical columns
        else:
            categorical.append(column)

    return {
        "numerical": numerical,
        "categorical": categorical,
        "identifiers": identifiers,
        "datetime": datetime_columns
    }



def profile_dataset(df):

    column_types = identify_column_types(df)

    profile = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isnull().sum().sum()),
        "numerical_columns": column_types["numerical"],
        "categorical_columns": column_types["categorical"],
        "identifier_columns": column_types["identifiers"],
        "datetime_columns": column_types["datetime"]
    }

    return profile


def get_statistical_summary(df):

    numerical = identify_column_types(df)["numerical"]

    if not numerical:
        return pd.DataFrame()

    return df[numerical].describe().round(2)


def get_missing_value_report(df):

    report = pd.DataFrame({
        "Missing Values": df.isnull().sum(),
        "Missing Percentage": (
            df.isnull().mean() * 100
        ).round(2)
    })

    return report
