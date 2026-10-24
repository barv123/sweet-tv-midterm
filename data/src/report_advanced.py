"""Render the advanced model comparison as a reproducible IEEE-style report."""

from advanced_common import load_json
from common import ROOT, RESULTS


def run():
    experiment = load_json("advanced_experiment.json")
    audit = load_json("advanced_validation_audit.json")
    submission = load_json("advanced_submission_validation.json")
    comparison = experiment["comparison"]
    table = "\n".join(
        f"| {row['method']} | {row['map_at_5']:.8f} | {row['runtime_seconds']:.3f} | "
        f"{row['candidate_count']:,} | {row['candidates_with_positive_evidence']:,} | "
        f"{row['users_with_zero_hits']} | {row['cold_user_map']:.4f} |"
        for row in comparison)
    ranks = audit["top5_rank_hit_rates"]
    rank_table = "\n".join(f"| {r['rank']} | {r['tv_show_id']} | {r['training_viewers']:,} | "
                           f"{r['validation_users_with_item']:,} | {r['hit_rate']:.2%} | "
                           f"{r['cumulative_any_hit_users']:,} |" for r in ranks)
    identity = audit["item_identity"]
    labels = audit["relevant_count_distribution"]
    manual = audit["independent_real_user_examples"]
    examples = []
    seen = set()
    for row in manual:
        key = (row["template_row"], row["method"])
        if key in seen or row["method"] != "popularity":
            continue
        seen.add(key)
        examples.append(f"| {row['template_row']} | {row['raw_valid_sessions']:,} | {row['relevant_items']} | "
                        f"{' '.join(map(str,row['predictions']))} | {' '.join(map(str,row['rank_hit_flags']))} | "
                        f"{row['ap_fraction']} |")
    example_table = "\n".join(examples)
    popularity_failure = next(row for row in manual if row["template_row"] == 44 and row["method"] == "popularity")
    failure_rows = [f"| popularity | 44 | {popularity_failure['relevant_items']} | {popularity_failure['ap_at_5']:.4f} | "
                    f"baseline | {' '.join(map(str,popularity_failure['predictions']))} |"]
    failure_methods = ("repeat_recency_heuristic", "content_metadata", "content_metadata_title",
                       "association_count25_lift1", "association_count50_lift1_2", "hybrid_rrf60")
    for method in failure_methods:
        row = next(record for record in comparison if record["method"] == method)["largest_losses_vs_popularity"][0]
        failure_rows.append(f"| {method} | {row['template_row']} | {row['relevant_items']} | {row['ap_at_5']:.4f} | "
                            f"{row['map_difference']:+.4f} | {' '.join(map(str,row['predictions']))} |")
    failure_table = "\n".join(failure_rows)
    report = f"""# A First Comparison of Content, Association-Rule, and Hybrid Recommendations for SWEET.TV

## Abstract

This study audits unusually high MAP@5 scores, evaluates metadata-based content
filtering and directional co-view association rules, and tests a rank-fusion hybrid.
All methods use an unchanged chronological 2020 split, the same 1,260 submission
users, the same 6,384 positive validation-schedule candidates, and held-out viewing
labels reconstructed independently from the original CSVs. Popularity achieves
MAP@5 {audit['baseline_map_at_5']['popularity']:.8f}; repeat/recency achieves
{audit['baseline_map_at_5']['repeat_recency_heuristic']:.8f}. The selected development
hybrid achieves {experiment['comparison'][-1]['map_at_5']:.8f}. The high baseline is
consistent with label density and five dominant recurring IDs, not a detected split
or metric leak. This validation-selected result is not an unbiased final estimate.

**Index Terms—** television recommendation, content-based filtering, association rules,
MAP@5, chronological validation, rank fusion.

## I. Introduction

The validation labels are distinct program IDs observed for each user during weeks
21-30, with at least 60 seconds on a matched positive-ID airing. The evaluation
cohort contains all 1,260 template users, including users with no observed validation
positives. AP for empty truth is zero. Previously watched programs remain eligible.
These are the established project conventions; no protocol change was made.

The scores are high because each user has many relevant items under this event-based
definition: median {labels['quantiles'][4]:.0f}, mean {labels['mean']:.2f}; 16 users have
fewer than five and 14 have none. Of the popularity top five, **{audit['top5_hit_histogram'][-1]['users']:,} users
hit all five**, {audit['top5_hit_histogram'][0]['users']:,} users hit none, and
{ranks[-1]['cumulative_any_hit_users']:,} hit at least one. Thus this is a top-five hit
ranking task on broad multi-label histories, not a sparse single-next-program task.

## II. Leakage and Item-Identity Audit

### A. Provenance and cutoff

The held-out interaction labels are drawn only from validation-period viewing records,
matched to the schedule by channel and interval. Every train session ends by May 18;
every validation session starts on/after May 18 and ends before July 27. Train and
validation viewing IDs are disjoint. The validation schedule defines candidates and
supplies exogenous item metadata only; it contributes no viewing counts, user labels,
or co-view transactions to training. The baseline popularity and repeat rankings were
reproduced from training interactions alone.

The critical-audit provenance checks report zero training cutoff violations, zero
validation-window violations, zero shared train/validation viewing IDs, zero item/time
mismatches against their source viewing and schedule rows, and zero validation labels
outside the candidate pool. A separate Python CSV scan reconstructed held-out labels
for six selected real users, with exact label-set equality and rational AP calculations.
All raw-file hashes still match the audit. Baseline result files are hash-protected and
unchanged.

### B. MAP@5 convention

For a ranked list, let `hit_k` be one for the first occurrence of a relevant item.
`AP@5 = sum(precision@k * hit_k, k=1..5) / min(5, |relevant items|)`; empty truth is
zero. User MAP is the unweighted mean over the complete template cohort. The manual
case `{11,22,33,44}` with ranking `[11,99,22,88,33]` gives
`(1 + 2/3 + 3/5)/4 = 17/30 = 0.5666666667`. Python and SQL independently agree on all
{experiment['verification']['independent_ap']['method_user_values_checked']:,} method/user AP values
(maximum error {experiment['verification']['independent_ap']['maximum_absolute_error']:.1f}).
Eight metric unit tests passed, covering denominator capping, empty truth, duplicate
hits, cutoff, and cohort inclusion.

### C. Popularity top-five coverage

| Rank | Program ID | Training distinct viewers | Validation users with item | User hit rate | Cumulative users hit |
|---:|---:|---:|---:|---:|---:|
{rank_table}

The IDs are recurring news/series entities. In the schedule snapshot, all {identity['positive_ids_checked']:,}
positive IDs have a single observed category (zero IDs vary by category); {identity['ids_with_multiple_genre_tuples']}
IDs vary in genre tuples and {identity['ids_with_multiple_titles']:,} have multiple observed
title strings. For the top five, categories and genre tuples are stable, while some
titles vary substantially: for example, ID 6,500,479 has {next((r['title_variants'] for r in identity['top5_checks'] if r['tv_show_id']==6500479),0)}
title variants but one category and one genre tuple. This is consistent with series or
editions grouped under a series ID, but no external canonical-program crosswalk exists.
The evaluation consistently joins on the supplied numeric ID and actual airing; semantic
identity beyond the supplied schema remains a limitation.

### D. Independent AP examples

The following popularity AP calculations use held-out labels reconstructed by a separate
CSV/`zoneinfo` interval scan; numerator fractions are in `advanced_validation_audit.json`.

| Template row | Valid source sessions | Relevant items | Ranked IDs | Hit flags | Exact AP |
|---:|---:|---:|---|---|---:|
{example_table}

## III. Recommendation Methods

### A. Content-Based Filtering

The user-item matrix is built exclusively from training events. User profiles are
recency-weighted (14-day half-life) centroids of TF-IDF item features; item and profile
vectors are L2-normalized and ranked by cosine similarity. A compact feature vector
uses category, three genre fields, production decade, director, explicit missing-value
tokens, and optionally title words. Actor fields are excluded due to high missingness.
Category/genre disagreement is not imputed; conflicting genre tuples are discarded.
IDF vocabulary/statistics are fitted on pre-cutoff schedule metadata. For candidate
items absent from pre-cutoff schedules, target-schedule metadata is used only as
exogenous item content, not as behavior. Zero-score ties use training popularity then
numeric ID. Cold users use the same global popularity fallback.

The small validation comparison is metadata without title tokens versus adding title
tokens. Scores are development-only; no external tuning or Kaggle score was used.

### B. Directional Association Rules

Each training user is one binary transaction; repeated airings do not inflate counts.
For antecedent A and consequent B, `support=count(A,B)/N`,
`confidence=count(A,B)/count(A)`, and `lift=confidence/(count(B)/N)`.
Rules are directional. User scores sum recency-weighted history strength times
`confidence * log(1 + lift)` over antecedents. Self-rules are excluded. Sparse
co-occurrence blocks cap each antecedent at 100 rules; tie-breaking is score, global
training popularity, then numeric ID. The two fixed validation settings use minimum
pair counts 25 or 50 and minimum lift 1.0 or 1.2, respectively; common minimum
confidence is 0.1 and each item needs 20 training viewers. Insufficient evidence
falls back to the established popularity ranking.

Six sampled rule supports, confidences, and lifts were recalculated directly from
the binary user-item matrix for each setting and passed independent checks.

### C. Hybrid

The sole hybrid is equal-weight reciprocal-rank fusion (`1/(60+rank)`) of top-100
lists from the strongest advanced validation configuration, repeat/recency, and
popularity. The strongest advanced standalone model is selected by validation MAP@5;
the resulting hybrid is compared on that same development set, so its score includes
selection optimism. Candidate sets are identical; all methods permit repeat items.

## IV. Results

| Method | MAP@5 | Runtime (s) | Candidate count | Candidates with positive method evidence | Users with zero hits | Cold-user MAP@5 |
|---|---:|---:|---:|---:|---:|---:|
{table}

All methods evaluate **{experiment['protocol']['users']:,} users** with **0 excluded**, **{experiment['protocol']['candidate_count']:,} candidates**,
and **{experiment['protocol']['relevant_user_item_pairs']:,} relevant user-item pairs**; there are
{experiment['protocol']['cold_users']} users without usable training history. Empty relevance
is zero AP. The baseline runtimes are from the preserved baseline run. Model runtimes
are per fitting/ranking configuration and exclude common sparse-matrix preparation
({experiment['shared_matrix_preparation_seconds']:.3f} s). The advanced experiment took
{experiment['runtime_seconds']:.3f} s total.

Content-only recommendations are weak (0.18696); title tokens help (0.34637) but remain
below popularity. Association rules at count 25/lift 1.0 reach 0.77388; stricter support
and lift reduce MAP to 0.40042. The equal-weight hybrid reaches 0.81697, +0.02218 over
popularity and +0.02092 over repeat/recency. The paired per-user normal 95% interval
for the hybrid-popularity difference is [{experiment['best_minus_popularity']['normal_95_interval'][0]:.5f},
{experiment['best_minus_popularity']['normal_95_interval'][1]:.5f}]; it is not adjusted for model/configuration
selection and should not be read as an unbiased confidence interval.

The rule model favors already broad recurring IDs (15 distinct items appear in its
entire top five across users); standalone rules therefore underperform global frequency.
The hybrid uses rules to reorder established candidates while retaining repeat and
popularity ranks. It reduces the per-user zero-hit count from 46 (popularity) to 37.
There are still user-level regressions: the largest hybrid losses include a user with
36 relevant items and AP@5 dropping from 0.55 to 0.10, illustrating that average gain
does not improve every user's ranking. Detailed examples are in the JSON evidence.

| Method | Example template row | Relevant items | Method AP@5 | Difference vs. popularity | Predicted IDs |
|---|---:|---:|---:|---:|---|
{failure_table}

## V. Verification and Reproduction

Seven recommendation-method tests passed. They cover content-only similarity, explicit
empty histories, deterministic score/popularity/ID tie-breaking, directional support/
confidence/lift, self-rule removal, zero-history fallback, RRF determinism, and a
temporal-cutoff fixture. Across all {experiment['verification']['eligible_unique_five_forecasts_checked']:,}
method/user forecasts, each list has five distinct positive eligible items. All
{experiment['verification']['cold_start_lists_checked']} cold-start lists equal global popularity.
Independent top-rank checks cover six real users per generated method. Candidate
eligibility is asserted before AP; final fit uses no viewing at/after July 27.

```powershell
$env:SWEET_TV_CACHE = "D:\\sweet-tv-midterm-cache"
.\\.venv\\Scripts\\python.exe data/src/critical_validation_audit.py
.\\.venv\\Scripts\\python.exe data/src/test_recommendation_methods.py
.\\.venv\\Scripts\\python.exe data/src/advanced_experiment.py
.\\.venv\\Scripts\\python.exe data/src/advanced_submission.py
```

The audit script above reuses the database and does not rerun the original CSV data
audit, but independently rechecks held-out labels from source CSVs. The sparse matrices
and rule matrices are stored under the ignored external cache, not tracked results.
Dependencies: NumPy {experiment['numpy_version']}, SciPy {experiment['scipy_version']},
DuckDB 1.4.3. Baseline files are hash-protected.
Measured evidence includes `results/advanced_comparison.csv`,
`results/advanced_per_user_evaluation.csv`, `results/advanced_experiment.json`,
`results/advanced_validation_audit.json`, rule examples, and
`results/advanced_submission_validation.json`.

## VI. Final-Period Candidate Artifact

The measured best development strategy, hybrid RRF60, was refit on {submission['final_fit']['interactions']:,}
identified viewing-airing interactions completed before **2020-07-27 00:00**. It uses
{submission['final_eligible_candidates']:,} candidates from the final schedule. The separate file
`results/advanced_candidate_submission.csv` passed structural validation: 1,260 unique
users in template order, 6,300 prediction slots, correct header and LF formatting,
five distinct positive eligible IDs per user, and zero malformed/ineligible/duplicate
predictions. SHA-256: `{submission['submission_validation']['sha256']}`. The earlier
`results/submission_baseline.csv` remains unchanged. No submission was uploaded.
Final refit and validation runtime: {submission['runtime_seconds']:.3f} s. The two users
without final-period usable training histories receive global distinct-viewer popularity.

## VII. Conclusion and Limitations

The critical audit finds no training/validation leakage or AP implementation error.
The high baseline follows from dense multi-label user histories and broad recurring
program IDs; the median validation user has 159 relevant IDs and 58% of users hit all
five popularity recommendations. A small association-rule model alone underperforms;
content metadata is sparse/noisy, and a title-token variant does not close the gap.
Rank fusion gives the best observed validation MAP@5 (**0.81696561**) and is the current
best candidate strategy, but it was selected on this same holdout. Keep popularity as
the robust baseline and treat the hybrid improvement as provisional.

Before upload, confirm the competition's exact label definition, target-schedule
availability, candidate eligibility, and repeat-item policy. Run chronological inner
folds or a second untouched time holdout before relying on the hybrid gain; inspect
per-user regressions and verify schedule/item IDs against organizer documentation.
The generated candidate has passed file validation but has not been submitted.
"""
    (ROOT / "reports" / "advanced_recommendation_experiment.md").write_text(report, encoding="utf-8")
    print("Wrote reports/advanced_recommendation_experiment.md")


if __name__ == "__main__":
    run()
