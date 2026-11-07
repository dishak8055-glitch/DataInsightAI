# Model & System Card: DataInsight AI (P10)

## System overview

DataInsight AI is an educational Streamlit application for CSV profiling,
visualization, task recommendations, natural-language analysis, synthetic
tabular data, and introductory Responsible AI checks. It is not a validated
decision-support product.

### Models and algorithms

- **Groq:** Optional remote text and vision API used for structured CSV-query
  plans, generated narratives, and consent-gated chart-image understanding.
  Text defaults to `openai/gpt-oss-20b`; vision defaults to
  `qwen/qwen3.8-27b`. API access, latency, model availability and quotas are
  external dependencies. Live Groq availability has not been verified here.
- **Custom GAN:** Small PyTorch conditional tabular GAN using one-hot discrete
  data. It is an educational CTGAN-inspired implementation, not canonical
  CTGAN.
- **SDV CTGAN:** SDV 1.28.0 `CTGANSynthesizer`, the separate actual CTGAN
  backend. Neither backend implements differential privacy.
- **Column grouping:** Optional `prajjwal1/bert-tiny` embeddings; offline
  character n-gram TF-IDF is a lexical baseline, not a BERT model.
- **Other modalities:** Generated dataset infographic PNGs and local
  text-to-speech are implemented. Groq can analyze a user-selected image
  after consent. No text-to-image, video, or music generator is implemented.

## Intended use and prohibited interpretation

Use for education, exploratory data analysis, and demonstrations on data the
operator is authorized to process. Do not use recommendations, fairness
screens, PII flags, or synthetic data as the sole basis for clinical, credit,
employment, legal, or other high-impact decisions. The app does not certify
fairness, privacy, anonymity, factuality, or regulatory compliance.

## Data

User-provided CSVs and the following bundled public datasets are supported.
Loaders preserve missing values, categorical values, and target labels.

| Dataset | Rows | Source, citation and identifier | License |
|---|---:|---|---|
| Adult Census Income | 48,842 | Becker & Kohavi (1996), [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), DOI [10.24432/C5XW20](https://doi.org/10.24432/C5XW20) | CC BY 4.0 |
| Palmer Penguins | 344 | Horst, Hill & Gorman (2020), [palmerpenguins v0.1.0](https://doi.org/10.5281/zenodo.3960218) | CC0 |
| Wine Quality Red | 1,599 | Cortez et al. (2009), [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), DOI [10.24432/C56S3T](https://doi.org/10.24432/C56S3T) | CC BY 4.0 |
| Wine Quality White | 4,898 | Cortez et al. (2009), [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), DOI [10.24432/C56S3T](https://doi.org/10.24432/C56S3T) | CC BY 4.0 |

Adult source data includes sensitive demographic attributes and an income
label. Wine `quality` is an integer-valued ordinal label stored numerically.
The manually reviewed type evaluation follows stored dtype/analysis role:
`quality` and Adult `education-num` are expected to be numerical.

## Evaluation

Reproduce the latest measured results with:

```bash
python -m pip install -r requirements.txt
python scripts/evaluate_project.py
python -m pytest -q
```

The generated [evaluation_results.json](./evaluation_results.json) records
per-column expected/predicted types for all four bundled datasets, fresh
Palmer Penguins GAN metrics and training times, historical external-provider
latency outcomes, and package versions. The detailed dated run, metric
definitions, known
failure case and validation results are in [REPORTS.md](./REPORTS.md).
The prior Palmer Penguins numbers (Custom GAN 56.4% distribution / 79.9%
correlation; SDV CTGAN 68.0% / 78.7%) were provided without the original
run configuration and are retained as unverified historical references, not
reproducible benchmark claims.

Verified run, 2026-11-07 (Python 3.11.0, CPU torch 2.15.0, SDV 1.28.0):

| Evaluation | Result |
|---|---|
| Column-type labels | 47/47 correct overall: Adult 15/15, Penguins 8/8, red wine 12/12, white wine 12/12 |
| Custom GAN on Palmer Penguins | 30 epochs; 26.302 s fit; 88.6% distribution; 96.4% correlation |
| SDV CTGAN on Palmer Penguins | 50 epochs; 14.390 s fit; 76.0% distribution; 80.9% correlation |
| Historical external-provider planner latency (pre-migration) | 0/5 successful; three server errors, two client errors; no valid response latency mean |

Both GANs generated 344 records and the similarity scores use five numeric
non-identifier fields. The measured timings and scores are one seeded run, not
stability or privacy guarantees. The type benchmark is a fixed curated
schema check, not an uploaded-data generalization result.

### Metrics and caveats

- Column-type accuracy is exact per-column agreement with the manual expected
  labels. It does not evaluate whether those labels are semantically optimal.
- Synthetic distribution similarity is the mean per-column `Distribution Match
  (%)` emitted by the comparison helper (`100 * (1 - KS statistic)`, rounded to
  one decimal place) over common non-identifier numeric columns; correlation
  similarity is the application's normalized Frobenius-distance score.
  Results depend on dataset, run, seed, epoch count, library versions, and
  sample size.
- Small-sample GAN metrics are unstable and do not establish utility,
  non-membership, or privacy.
- Fairness ratios and imbalance warnings are descriptive screening only.
- LLM numerical grounding checks are heuristic and can miss errors.

## Safety, privacy, and security

- The CSV planner uses validated operations, and Pandas Code mode interprets
  expressions through a strict AST allowlist without unrestricted `eval()` or
  `exec()`.
- Groq text features send the user's question and dataset schema/aggregate
  facts. Image understanding sends a selected image only after explicit
  consent. Do not submit confidential or unauthorized data.
- The sensitive-column detector can miss PII and does not automatically
  redact it. Users remain responsible for review and data minimization.
- Synthetic output may memorize or closely resemble source records. No
  differential-privacy mechanism or privacy evaluation is provided.
- API keys belong in `.env` locally or the deployment platform's secret
  manager. `.env` is ignored by git and is not tracked in the inspected
  worktree; the tracked-file scan found no embedded API token/private-key
  patterns.

## Licensing and deployment constraints

Dataset terms are listed above; attribution is required for the CC BY 4.0
datasets. SDV 1.28.0 reports **Business Source License 1.1**, not an open
source license. Its installed license's Additional Use Grant excludes a
"Synthetic Data Service" that exposes SDV functionality to third parties.
Obtain licensing guidance before public operation of the SDV generation
feature; otherwise disable it in the public service. The app has not been
deployed to Streamlit Community Cloud or Hugging Face Spaces. Local Python
3.11 dependency/import checks do not prove cloud build success or license
eligibility.

## Known failure and limitations

The earlier deployed histogram failure was reported as `ValueError: Invalid
value` from Plotly `px.histogram(...)`. Plotly is pinned at 5.24.1; the focused
histogram tests pass against that installed version. This is locally verified,
not re-tested on the previous remote deployment.

Video and music generation remain incomplete assignment requirements. The
image output is a deterministic infographic, not a generative image model;
audio output is speech synthesis, not music generation. See [README.md](./README.md)
and [REPORTS.md](./REPORTS.md) for the full implementation checklist and
deployment setup.
