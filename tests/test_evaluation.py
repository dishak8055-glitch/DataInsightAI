from modules.dataset_catalog import CURATED_DATASETS, load_curated_dataset
from modules.evaluation import (
    EXPECTED_COLUMN_TYPES,
    evaluate_column_types,
    validate_expected_type_schemas,
)
from scripts.evaluate_project import evaluate_groq_latency


def test_manual_type_schemas_match_all_curated_datasets():
    validate_expected_type_schemas()
    assert set(EXPECTED_COLUMN_TYPES) == set(CURATED_DATASETS)


def test_column_type_accuracy_is_measured_for_every_curated_dataset():
    evaluations = [
        evaluate_column_types(name, load_curated_dataset(name))
        for name in CURATED_DATASETS
    ]

    assert sum(result["total"] for result in evaluations) == 47
    assert all(result["total"] == len(result["columns"]) for result in evaluations)
    assert all(
        result["accuracy_pct"]
        == round(result["correct"] / result["total"] * 100, 2)
        for result in evaluations
    )
    assert all(
        result["dataset"] in CURATED_DATASETS for result in evaluations
    )


def test_groq_latency_reports_each_run_as_failed_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    runs = evaluate_groq_latency(5)

    assert len(runs) == 5
    assert all(run["status"] == "failed" for run in runs)
    assert all(run["latency_ms"] is None for run in runs)
    assert all("not configured" in run["error"] for run in runs)
