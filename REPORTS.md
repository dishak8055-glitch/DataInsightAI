# DataInsight AI Evaluation and Quality Report

**Evaluation date:** 2026-11-07
**Reproduction command:** `python scripts/evaluate_project.py`
**Machine/runtime:** Python 3.11 on Windows; exact package versions are recorded
in `evaluation_results.json`.

The JSON file is the machine-readable output of the evaluation script. The
reported local run is one measured run, not a population-level benchmark.

## 1. Column-type detection

The evaluation compares each output from `modules.profiling.identify_column_types`
with a manually reviewed, per-column expected type in `modules/evaluation.py`.
Accuracy is the exact number of matches divided by the number of columns.
`education-num` and wine `quality` are expected to be numerical because the
evaluation is about stored analytical type, not target/ordinal semantics.
Missing values are retained by the dataset loader and do not change these
expected labels.

| Curated dataset | Rows | Columns | Correct | Accuracy |
|---|---:|---:|---:|---:|
| Adult Census Income | 48,842 | 15 | 15/15 | 100% |
| Palmer Penguins | 344 | 8 | 8/8 | 100% |
| Wine Quality Red | 1,599 | 12 | 12/12 | 100% |
| Wine Quality White | 4,898 | 12 | 12/12 | 100% |
| **Overall** | — | **47** | **47/47** | **100%** |

This is a fixed curated-dataset check, not a generalization estimate for
arbitrary uploaded CSV files. The initial evaluation exposed a false positive
for `citric acid`: the old identifier check considered any name ending in
`id` an identifier. The profiler now checks identifier tokens; the regression
test covers `CustomerID`, `user_id`, `IDNumber`, and numeric `citric acid`.

## 2. Synthetic-data comparison

Both backends are evaluated on all Palmer Penguins rows. Their training time
covers `fit()` only. Each then generates the same number of rows as the source.
Identifier fields are excluded from the fidelity metrics.

- **Distribution similarity:** mean of the per-column `Distribution Match (%)`
  values from the existing comparison helper over common numeric non-identifier
  fields. Each per-column value is `100 * (1 - KS statistic)`, rounded to one
  decimal place by that helper.
- **Correlation similarity:** the existing comparison's normalized
  Frobenius-distance match over the same fields.
- Epoch counts, seed, versions, output rows, individual feature scores, and
  status/error are recorded in `evaluation_results.json`. No failed backend is
  silently substituted.

| Backend | Epochs | Training seconds | Distribution similarity | Correlation similarity | Status |
|---|---:|---:|---:|---:|---|
| Custom PyTorch GAN (CTGAN-inspired; not canonical) | 30 | 26.302 | 88.6% | 96.4% | Success |
| SDV `CTGANSynthesizer` (actual CTGAN) | 50 | 14.390 | 76.0% | 80.9% | Success |

Both backends generated 344 rows. Numeric fields evaluated were
`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, and
`year`. Runtime and similarity are specific to this seeded run and environment
(Python 3.11.0, CPU torch 2.15.0, SDV 1.28.0); values can vary in other runs.

The provided earlier Palmer Penguins experiment is retained separately:

| Prior result supplied by user | Distribution similarity | Correlation similarity |
|---|---:|---:|
| Custom GAN | 56.4% | 79.9% |
| SDV CTGAN | 68.0% | 78.7% |

Its epoch counts, seed, generated row count, hardware, and library versions were
not available; it is therefore **not independently verified or reproducible**
and must not be conflated with the fresh run above. Neither quality score is a
privacy measurement. Neither model includes differential privacy.

## 3. Historical external-provider response latency

The earlier evaluation made five requests directly to the configured remote
planner, avoiding reporting the local pandas fast path as provider latency.
Requests used a small in-memory, non-sensitive example. Each run is reported independently in
`evaluation_results.json`; failures are counted and are not replaced with
local success responses. The summary records successful/failed count and mean
latency over successes only. If all five calls fail, the mean is `null`.

Observed result: **0/5 successful; 5/5 failed**, so
there is no successful-response mean latency. The request wall times below are
failure latencies only, not response latency measurements:

| Run | Outcome | Failure latency |
|---:|---|---:|
| 1 | ServerError | 3,946.95 ms |
| 2 | ServerError | 11,775.82 ms |
| 3 | ServerError | 3,629.54 ms |
| 4 | ClientError | 3,440.11 ms |
| 5 | ClientError | 2,270.10 ms |

The API key was configured locally, so these failures were not treated as
missing-key skips. The exact provider response detail is omitted from artifacts
to avoid storing sensitive provider internals; the API/model availability or
request is a blocker for a successful latency benchmark. The key is never
written to the JSON.

## 4. Verified Plotly failure case

- **Observed failure:** The deployed app log supplied for this project showed
  `app.py` calling `create_histogram(df, sel_col)`, followed by
  `px.histogram(...)` raising `ValueError: Invalid value`.
- **Fix present:** `requirements.txt` pins Plotly to `5.24.1` rather than
  allowing an incompatible deployed Plotly release.
- **Verification:** The current virtual environment reports Plotly 5.24.1.
  Focused histogram tests cover generated Plotly figures, a bundled housing
  dataset, and all-missing input; the complete pytest suite result is reported
  below after execution.
- **Scope:** This validates local figure creation and tests. The original
  remote deployment was not re-run, so remote resolution is not claimed.

## 5. Multimodal assignment status

| Requirement | Status | Evidence / limitation |
|---|---|---|
| Image output | Partial | Deterministic Matplotlib/Pillow dataset infographic; not text-to-image generation. |
| Chart/image understanding | Implemented with external API | Current implementation uses Groq vision; image analysis is opt-in and consent-gated and requires model access. |
| Audio output | Partial | `pyttsx3` text-to-speech narrates text; no audio/music content generator. |
| Video generation | Missing | Only a conceptual survey row; no model or generation path. |
| Music generation | Missing | Only a conceptual survey row; no model or generation path. |

Thus image/video/music generation as a set is **not complete**. The project
must not present the educational architecture survey as generation
functionality.

## 6. Security, licensing and deployment checks

- The tracked-worktree scan found no API-token, private-key, or long-secret
  patterns. `.env` exists locally, is listed in `.gitignore`, and is not
  tracked. No secret value was printed or copied into evaluation output.
- Current Groq keys must be set as local environment variables/`.env` or platform
  secrets. No key should be committed.
- The installed SDV 1.28.0 package reports **BSL-1.1**. The included license
  states that the license is not an open-source license and its Additional Use
  Grant excludes a commercial "Synthetic Data Service" offering SDV
  functionality to third parties. A public app that exposes synthesis needs
  DataCebo licensing review; alternatively disable SDV synthesis in that
  service. This is a material release blocker until resolved.
- Streamlit's dependency documentation supports root `requirements.txt` and
  Linux external packages via `packages.txt`. The project has those files;
  its SDV, Streamlit, and torch imports and Python 3.11 environment are locally
  validated. A local check does not prove a Cloud build, available memory,
  runtime behavior, or BSL eligibility.
- No Streamlit Community Cloud or Hugging Face Spaces deployment was attempted.
  Do not claim deployment success.
- The README metadata advertises MIT for Hugging Face, but no project-root
  `LICENSE` file was present in the inspected worktree. Confirm the intended
  application license and add its license text before public release.
- The requirements allow broad version ranges for most dependencies. Re-run
  the evaluation and test suite in the target cloud's exact Python/runtime
  before release.

## 7. Dataset citations and licenses

| Dataset | Citation | License |
|---|---|---|
| Adult Census Income | Becker, B. & Kohavi, R. (1996). *Adult*. UCI Machine Learning Repository. [DOI 10.24432/C5XW20](https://doi.org/10.24432/C5XW20). | CC BY 4.0 |
| Palmer Penguins | Horst, A. M., Hill, A. P. & Gorman, K. B. (2020). *palmerpenguins* v0.1.0. [DOI 10.5281/zenodo.3960218](https://doi.org/10.5281/zenodo.3960218). | CC0 |
| Wine Quality Red and White | Cortez, P. et al. (2009). *Wine Quality*. UCI Machine Learning Repository. [DOI 10.24432/C56S3T](https://doi.org/10.24432/C56S3T). | CC BY 4.0 |

The bundled CSVs are loaded locally. Adult whitespace, `?` markers and test-set
income punctuation are normalized; wine uses a semicolon delimiter. Missing
values and categorical labels are preserved.

## 8. Reproduction and tests

From the repository root with Python 3.11:

```powershell
python -m pip install -r requirements.txt
python scripts/evaluate_project.py
python -m pytest -q
```

The evaluation defaults to seed 42, 30 custom-GAN epochs, 50 SDV CTGAN epochs,
and five Groq API calls. Use `--skip-groq` when no API request should be
made, `--skip-gan` for the data-type-only run, or adjust the epoch options.
No API call is necessary for the pytest suite.

**Current full-suite result:** 99 passed.

Final local checks: `pip check` reported no broken requirements; Python
compilation and `git diff --check` passed. A temporary local Streamlit process
returned HTTP `200 ok` from `/_stcore/health`. This confirms local startup
health only, not a full browser exercise or cloud deployment.
