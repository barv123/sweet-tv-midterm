# SWEET.TV Midterm

Data audit, reproducible preprocessing, and first baselines for the SWEET.TV recommendation project.

## Project structure

```text
data/
  raw/       Original datasets (local only; ignored by Git)
  src/       Audit, matching, baseline, advanced-model, and verification scripts
  reports/   Preserved original placeholder directory
  results/   Preserved original placeholder directory
  sessions/  OpenCode session exports (local only; ignored by Git)
reports/     Measured findings and validation design
results/     Audit outputs, evaluation evidence, and local candidate submissions
```

Existing files and directories are preserved. Raw data and user-level derived tables
remain local; a configurable disk-backed cache holds the working database.

## Getting started

```sh
git clone https://github.com/barv123/sweet-tv-midterm.git
cd sweet-tv-midterm
```

Create `data/raw/` locally and place the original datasets there. Create
`data/sessions/` if needed for local session exports. Neither directory is
included in the repository.

## Run the audit and preprocessing

Python 3.12 was used for the verified run. From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:SWEET_TV_CACHE = "D:\sweet-tv-midterm-cache"
.\.venv\Scripts\python.exe data/src/run_pipeline.py
```

Use a cache drive with several GB free; omit `SWEET_TV_CACHE` to use `.cache/audit/`.
The completed local run uses D: because C: lacked spill space. DuckDB has a 768 MB
managed-memory budget and processes viewing intervals in weekly batches.

The pipeline audits all five CSVs, tests channel/time compatibility, performs
DST-aware interval matching, independently verifies real examples, rechecks raw-file
SHA-256 hashes, and writes [the audit report](reports/data_audit.md) and `results/`.
It does not train models. Individual stages can also be run in order:
`audit.py`, `match.py`, `verify.py`, `report.py` under `data/src/`.

The local cache database exposes `interactions`, `train_interactions`, and
`validation_interactions`. Default interactions require an identified positive item,
at least 60 seconds of overlap, and documented quality gates. Counts are conditional
on the channel mapping and clock interpretation described in the report.
Never use `tv_show_id = 0` as a recommendation item.

## Run the first recommendation experiment

After the audited cache has been prepared:

```powershell
$env:SWEET_TV_CACHE = "D:\sweet-tv-midterm-cache"
.\.venv\Scripts\python.exe data/src/test_metrics.py
.\.venv\Scripts\python.exe data/src/baseline_experiment.py
.\.venv\Scripts\python.exe data/src/validate_submission.py
```

The experiment evaluates distinct-viewer popularity and a separate fixed repeat/recency
heuristic on the same 1,260 users, using the assignment's MAP@5 denominator convention.
It reconciles cutoff-crossing interactions and checks temporal isolation, independent
viewer counts, Python/SQL metric agreement, and source-file integrity.

Outputs include [the experiment report](reports/baseline_experiment.md),
`results/baseline_experiment.json`, per-user evaluation evidence, ranked candidate lists,
and **`results/submission_baseline.csv`**. The submission uses popularity refitted on
weeks 11-30, with five eligible positive IDs per user in the original template order.
The validator can also check another file: `python data/src/validate_submission.py PATH`.
No Kaggle upload is performed.

## Run the advanced validation experiment

After the baseline cache and first critical audit have been prepared:

```powershell
$env:SWEET_TV_CACHE = "D:\sweet-tv-midterm-cache"
.\.venv\Scripts\python.exe data/src/test_recommendation_methods.py
.\.venv\Scripts\python.exe data/src/advanced_experiment.py
.\.venv\Scripts\python.exe data/src/advanced_submission.py
.\.venv\Scripts\python.exe data/src/report_advanced.py
```

`advanced_experiment.py` compares content-based filtering, directional association
rules, the preserved baselines, and an RRF hybrid on the same chronological validation
cohort. `advanced_submission.py` refits the selected validation strategy before the
final period and writes the separate, locally validated
`results/advanced_candidate_submission.csv`; it does not modify the baseline submission
or upload anything. [Advanced results](reports/advanced_recommendation_experiment.md)
and numerical verification files are in `results/`. Sparse matrices and caches remain
in the ignored `SWEET_TV_CACHE` directory.

## Pre-Kaggle reliability check

The additional fixed weeks 15-16 holdout and final candidate checks reuse the existing
cache; they do not rerun the original audit or advanced-validation experiment:

```powershell
$env:SWEET_TV_CACHE = "D:\sweet-tv-midterm-cache"
.\.venv\Scripts\python.exe data/src/pre_kaggle_reliability.py
.\.venv\Scripts\python.exe data/src/final_protocol_check.py
.\.venv\Scripts\python.exe data/src/report_pre_kaggle.py
```

See [the pre-Kaggle reliability report](reports/pre_kaggle_reliability.md) for the
new holdout results, user case studies, locally verified submission constraints, and
remaining competition-rule uncertainties. Per-user evaluation files and submission
CSVs are intentionally kept local and ignored by Git.

## Strict single-airing relevance reassessment

The participant-described >80% airing ground truth is evaluated separately from the
>50% single-airing training signal. Re-run from the existing interval cache with:

```powershell
$env:SWEET_TV_CACHE = "D:\sweet-tv-midterm-cache"
.\.venv\Scripts\python.exe data/src/test_strict_labels.py
.\.venv\Scripts\python.exe data/src/strict_relevance_experiment.py
.\.venv\Scripts\python.exe data/src/report_strict80.py
```

This reuses cached viewing-to-schedule interval edges, independently checks selected
labels against original CSV rows, evaluates both chronological holdouts, and creates
separate repeat/recency and robust-alternative candidates. See
[the strict relevance assessment](reports/strict_relevance_assessment.md). The >80%
definition remains participant-reported rather than confirmed by accessible official
Kaggle rule text; neither candidate is uploaded automatically.
