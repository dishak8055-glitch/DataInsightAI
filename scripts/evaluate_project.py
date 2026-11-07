"""Run the DataInsight AI dataset, GAN-fidelity, and optional Groq latency evaluations."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly
import sdv
import streamlit
import torch

from modules.dataset_catalog import CURATED_DATASETS, load_curated_dataset
from modules.evaluation import evaluate_column_types, validate_expected_type_schemas
from modules.groq_client import TEXT_MODEL, get_groq_api_key, summarize_groq_error
from modules.csv_chat import interpret_question
from modules.synthetic_data import (
    SDVCTGANSynthesizer,
    TabularCTGAN,
    compare_real_and_synthetic,
)


HISTORICAL_PENGUINS_RESULTS = {
    "dataset": "Palmer Penguins",
    "status": "User-provided prior results; not independently reproduced",
    "custom_gan": {
        "distribution_similarity_pct": 56.4,
        "correlation_similarity_pct": 79.9,
    },
    "sdv_ctgan": {
        "distribution_similarity_pct": 68.0,
        "correlation_similarity_pct": 78.7,
    },
    "training_time_seconds": None,
    "missing_run_configuration": [
        "epoch counts",
        "random seed",
        "generated row count",
        "hardware and software versions",
    ],
}


def _versions():
    return {
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "plotly": plotly.__version__,
        "sdv": sdv.__version__,
        "streamlit": streamlit.__version__,
        "torch": torch.__version__,
    }


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_gan_backend(name, synthesizer, dataset, epochs):
    """Fit one backend and compare its output with the full source dataset."""
    result = {
        "backend": name,
        "epochs": epochs,
        "training_time_seconds": None,
        "generated_rows": None,
        "distribution_similarity_pct": None,
        "correlation_similarity_pct": None,
        "numeric_features_evaluated": 0,
        "status": "failed",
        "error": None,
    }
    try:
        started = time.perf_counter()
        synthesizer.fit(dataset)
        result["training_time_seconds"] = round(time.perf_counter() - started, 3)
        synthetic = synthesizer.sample(n_samples=len(dataset))
        metrics = compare_real_and_synthetic(dataset, synthetic)
        distribution_scores = metrics["metrics_table"].get(
            "Distribution Match (%)", pd.Series(dtype=float)
        )
        result.update(
            {
                "generated_rows": len(synthetic),
                "distribution_similarity_pct": (
                    round(float(distribution_scores.mean()), 1)
                    if not distribution_scores.empty
                    else None
                ),
                "correlation_similarity_pct": metrics[
                    "correlation_similarity_score"
                ],
                "numeric_features_evaluated": metrics["num_features_evaluated"],
                "status": "success",
            }
        )
    except Exception as exc:
        result["error"] = (
            f"{type(exc).__name__}: {str(exc)[:300]}"
            if isinstance(exc, (ValueError, RuntimeError, ImportError))
            else f"{type(exc).__name__}: error details omitted"
        )
    return result


def evaluate_gans(penguins, custom_epochs, sdv_epochs):
    numeric = penguins.select_dtypes(include=[np.number])
    results = []
    results.append(
        evaluate_gan_backend(
            "Custom PyTorch GAN (CTGAN-inspired; not canonical CTGAN)",
            TabularCTGAN(epochs=custom_epochs, batch_size=16),
            penguins,
            custom_epochs,
        )
    )
    results.append(
        evaluate_gan_backend(
            "SDV CTGAN (actual CTGANSynthesizer)",
            SDVCTGANSynthesizer(epochs=sdv_epochs, batch_size=100),
            penguins,
            sdv_epochs,
        )
    )
    for result in results:
        result["dataset"] = "Palmer Penguins"
        result["metric_definition"] = (
            "Distribution similarity is the mean of per-column Distribution "
            "Match percentages from compare_real_and_synthetic (100*(1-KS), "
            "rounded to one decimal place) over non-ID numeric columns. "
            "Correlation similarity is the "
            "existing compare_real_and_synthetic normalized Frobenius score."
        )
        result["features_used_for_similarity"] = list(numeric.columns)
    return results


def evaluate_groq_latency(runs):
    """Measure explicitly requested Groq planning calls and preserve failures."""
    if not get_groq_api_key():
        return [
            {
                "run": index + 1,
                "status": "failed",
                "latency_ms": None,
                "error": "GROQ_API_KEY is not configured.",
            }
            for index in range(runs)
        ]

    sample = pd.DataFrame(
        {"value": [10.0, 20.0, 30.0], "group": ["A", "A", "B"]}
    )
    question = (
        "Calculate the arithmetic mean for the numeric value column and return "
        "the supported analysis plan."
    )
    results = []
    for index in range(runs):
        started = time.perf_counter()
        try:
            plan = interpret_question(sample, question)
            results.append(
                {
                    "run": index + 1,
                    "status": "success",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "operation": plan.get("operation"),
                    "model": TEXT_MODEL,
                    "error": None,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "run": index + 1,
                    "status": "failed",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "model": TEXT_MODEL,
                    "error": summarize_groq_error(exc),
                }
            )
    return results


def run_evaluation(args):
    validate_expected_type_schemas()
    type_results = []
    datasets = {
        name: load_curated_dataset(name) for name in CURATED_DATASETS
    }
    for name, dataset in datasets.items():
        type_results.append(evaluate_column_types(name, dataset))

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "seed": args.seed,
            "custom_gan_epochs": args.custom_epochs,
            "sdv_ctgan_epochs": args.sdv_epochs,
            "groq_latency_runs": args.groq_runs,
            "groq_text_model": TEXT_MODEL,
        },
        "software_versions": _versions(),
        "column_type_detection": {
            "method": "Per-column comparison with manually reviewed expected types.",
            "datasets": type_results,
            "overall_correct": sum(item["correct"] for item in type_results),
            "overall_total": sum(item["total"] for item in type_results),
            "overall_accuracy_pct": round(
                sum(item["correct"] for item in type_results)
                / sum(item["total"] for item in type_results)
                * 100,
                2,
            ),
        },
        "gan_evaluation": {
            "historical_user_provided_reference": HISTORICAL_PENGUINS_RESULTS,
            "fresh_run": None,
        },
        "groq_latency": {
            "requested_runs": args.groq_runs,
            "runs": None,
            "successful_runs": 0,
            "failed_runs": 0,
            "mean_success_latency_ms": None,
        },
        "deployment_check": {
            "python_version": _versions()["python"],
            "sdv_version": _versions()["sdv"],
            "sdv_import": "success",
            "note": (
                "This local dependency/import check does not prove deployment "
                "on Streamlit Community Cloud or Hugging Face Spaces."
            ),
        },
    }

    if not args.skip_gan:
        _seed_everything(args.seed)
        output["gan_evaluation"]["fresh_run"] = evaluate_gans(
            datasets["Palmer Penguins"],
            args.custom_epochs,
            args.sdv_epochs,
        )

    if not args.skip_groq:
        runs = evaluate_groq_latency(args.groq_runs)
        successful = [item for item in runs if item["status"] == "success"]
        output["groq_latency"].update(
            {
                "runs": runs,
                "successful_runs": len(successful),
                "failed_runs": len(runs) - len(successful),
                "mean_success_latency_ms": (
                    round(
                        sum(item["latency_ms"] for item in successful)
                        / len(successful),
                        2,
                    )
                    if successful
                    else None
                ),
            }
        )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Evaluation JSON written to: {output_path}")
    print(
        "Column type detection: "
        f"{output['column_type_detection']['overall_correct']}/"
        f"{output['column_type_detection']['overall_total']} "
        f"({output['column_type_detection']['overall_accuracy_pct']}%)"
    )
    if output["gan_evaluation"]["fresh_run"] is not None:
        for item in output["gan_evaluation"]["fresh_run"]:
            print(
                f"{item['backend']}: {item['status']}; "
                f"training={item['training_time_seconds']}s; "
                f"distribution={item['distribution_similarity_pct']}%; "
                f"correlation={item['correlation_similarity_pct']}%"
            )
    if output["groq_latency"]["runs"] is not None:
        print(
            "Groq latency: "
            f"{output['groq_latency']['successful_runs']} successful, "
            f"{output['groq_latency']['failed_runs']} failed; "
            f"mean successful latency="
            f"{output['groq_latency']['mean_success_latency_ms']} ms"
        )
        for item in output["groq_latency"]["runs"]:
            print(
                f"Groq run {item['run']}: {item['status']}; "
                f"{item['latency_ms']} ms; {item.get('error') or 'OK'}"
            )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--custom-epochs", type=int, default=30)
    parser.add_argument("--sdv-epochs", type=int, default=50)
    parser.add_argument("--groq-runs", type=int, default=5)
    parser.add_argument("--output", default="evaluation_results.json")
    parser.add_argument("--skip-gan", action="store_true")
    parser.add_argument("--skip-groq", action="store_true")
    args = parser.parse_args()
    if args.groq_runs < 1:
        parser.error("--groq-runs must be at least 1")
    if args.custom_epochs < 1 or args.sdv_epochs < 1:
        parser.error("GAN epoch counts must be at least 1")
    run_evaluation(args)


if __name__ == "__main__":
    main()
