# modules/recommender.py

import pandas as pd


def identify_ids(df):
    """Identify columns that are likely to contain record IDs."""
    identifiers = []

    for column in df.columns:
        name = str(column).lower().strip()

        if (
            name == "id"
            or name.endswith("_id")
            or name.endswith("id")
            or name.startswith("id_")
        ):
            identifiers.append(column)

    return identifiers


def recommend_task(df, target_column=None):
    """
    Recommend a possible ML task based on the dataset
    and the target column selected by the user.
    """

    if df is None or df.empty:
        return {
            "task": "Unable to recommend",
            "reason": "The uploaded dataset is empty.",
            "algorithms": [],
            "features": [],
            "next_steps": []
        }

    identifiers = identify_ids(df)

    usable_columns = [
        col for col in df.columns
        if col not in identifiers
        and df[col].notna().any()
        and df[col].nunique(dropna=True) > 1
    ]

    warnings = []

    if len(df) < 30:
        warnings.append(
            "The dataset has fewer than 30 rows. "
            "Model results may not be reliable."
        )

    if target_column is None:
        features = usable_columns

        if len(features) < 2:
            return {
                "task": "Insufficient data",
                "reason": "At least two useful columns are needed for clustering.",
                "algorithms": [],
                "features": features,
                "warnings": warnings,
                "next_steps": [
                    "Upload a dataset with more useful columns.",
                    "Remove constant or irrelevant columns."
                ]
            }

        return {
            "task": "Clustering (Unsupervised Learning)",
            "reason": (
                "No target column was selected. "
                "Clustering can help discover groups or patterns "
                "in the available data."
            ),
            "algorithms": [
                "K-Means",
                "DBSCAN",
                "Hierarchical Clustering"
            ],
            "features": features,
            "warnings": warnings,
            "next_steps": [
                "Remove identifier columns.",
                "Handle missing values.",
                "Encode categorical features.",
                "Scale numerical features.",
                "Apply a clustering algorithm.",
                "Evaluate clusters using a suitable metric, such as silhouette score."
            ]
        }

    if target_column not in df.columns:
        return {
            "task": "Invalid target",
            "reason": "The selected target column does not exist.",
            "algorithms": [],
            "features": [],
            "warnings": warnings,
            "next_steps": ["Select a valid target column."]
        }

    target = df[target_column]
    unique_count = target.nunique(dropna=True)
    missing_percentage = (target.isna().mean() * 100)

    features = [
        col for col in usable_columns
        if col != target_column
    ]

    if target_column in identifiers:
        warnings.append(
            "The selected target looks like an identifier. "
            "It may not be a meaningful prediction target."
        )

    if missing_percentage > 0:
        warnings.append(
            f"The target column has {missing_percentage:.1f}% missing values."
        )

    if unique_count < 2:
        return {
            "task": "Insufficient target data",
            "reason": "The target column must contain at least two distinct values.",
            "algorithms": [],
            "features": features,
            "warnings": warnings,
            "next_steps": [
                "Choose another target column.",
                "Check whether the selected column contains enough variation."
            ]
        }

    if pd.api.types.is_numeric_dtype(target) and (
    unique_count > 10 or unique_count >= len(df) * 0.5
    ):
        task = "Regression (Supervised Learning)"
        reason = (
            f"'{target_column}' contains numerical values with "
            f"{unique_count} distinct values. Regression may be suitable "
            "for predicting a continuous outcome."
        )
        algorithms = [
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor"
        ]
        evaluation = "MAE, RMSE and R²"

    else:
        task = "Classification (Supervised Learning)"
        reason = (
            f"'{target_column}' contains {unique_count} distinct values. "
            "Classification may be suitable for predicting categories or classes."
        )
        algorithms = [
            "Logistic Regression",
            "Decision Tree Classifier",
            "Random Forest Classifier"
        ]
        evaluation = "Accuracy, Precision, Recall and F1-score"

        if unique_count > 20:
            warnings.append(
                "The target has many distinct categories. "
                "Check whether classification is appropriate."
            )

    if not features:
        warnings.append(
            "No usable feature columns remain after excluding the target, "
            "identifiers and constant columns."
        )

    next_steps = [
        "Check the target column and remove irrelevant features.",
        "Handle missing values.",
        "Encode categorical features.",
        "Split the data into training and testing sets.",
        "Train a simple baseline model.",
        f"Evaluate the model using {evaluation}.",
        "Compare multiple algorithms and improve the best model."
    ]

    return {
        "task": task,
        "reason": reason,
        "algorithms": algorithms,
        "features": features,
        "warnings": warnings,
        "next_steps": next_steps
    }