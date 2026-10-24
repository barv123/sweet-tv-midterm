# A First Comparison of Content, Association-Rule, and Hybrid Recommendations for SWEET.TV

## Abstract

This study audits unusually high MAP@5 scores, evaluates metadata-based content
filtering and directional co-view association rules, and tests a rank-fusion hybrid.
All methods use an unchanged chronological 2020 split, the same 1,260 submission
users, the same 6,384 positive validation-schedule candidates, and held-out viewing
labels reconstructed independently from the original CSVs. Popularity achieves
MAP@5 0.79478307; repeat/recency achieves
0.79604233. The selected development
hybrid achieves 0.81696561. The high baseline is
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
definition: median 159, mean 182.28; 16 users have
fewer than five and 14 have none. Of the popularity top five, **734 users
hit all five**, 46 users hit none, and
1,214 hit at least one. Thus this is a top-five hit
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
case `(11, 22, 33, 44)` with ranking `[11,99,22,88,33]` gives
`(1 + 2/3 + 3/5)/4 = 17/30 = 0.5666666667`. Python and SQL independently agree on all
8,820 method/user AP values
(maximum error 0.0).
Eight metric unit tests passed, covering denominator capping, empty truth, duplicate
hits, cutoff, and cohort inclusion.

### C. Popularity top-five coverage

| Rank | Program ID | Training distinct viewers | Validation users with item | User hit rate | Cumulative users hit |
|---:|---:|---:|---:|---:|---:|
| 1 | 2400480 | 4,482 | 1,138 | 90.32% | 1,138 |
| 2 | 240081 | 4,243 | 1,078 | 85.56% | 1,190 |
| 3 | 2400508 | 3,961 | 974 | 77.30% | 1,200 |
| 4 | 6500479 | 3,944 | 1,035 | 82.14% | 1,211 |
| 5 | 2400467 | 3,809 | 985 | 78.17% | 1,214 |

The IDs are recurring news/series entities. In the schedule snapshot, all 8,234
positive IDs have a single observed category (zero IDs vary by category); 108
IDs vary in genre tuples and 3,498 have multiple observed
title strings. For the top five, categories and genre tuples are stable, while some
titles vary substantially: for example, ID 6,500,479 has 100
title variants but one category and one genre tuple. This is consistent with series or
editions grouped under a series ID, but no external canonical-program crosswalk exists.
The evaluation consistently joins on the supplied numeric ID and actual airing; semantic
identity beyond the supplied schema remains a limitation.

### D. Independent AP examples

The following popularity AP calculations use held-out labels reconstructed by a separate
CSV/`zoneinfo` interval scan; numerator fractions are in `advanced_validation_audit.json`.

| Template row | Valid source sessions | Relevant items | Ranked IDs | Hit flags | Exact AP |
|---:|---:|---:|---|---|---:|
| 1 | 3,347 | 171 | 2400480 240081 2400508 6500479 2400467 | 1 1 1 1 1 | 1 |
| 2 | 255 | 19 | 2400480 240081 2400508 6500479 2400467 | 1 1 0 1 0 | 11/20 |
| 12 | 182 | 109 | 2400480 240081 2400508 6500479 2400467 | 1 1 1 1 0 | 4/5 |
| 44 | 102 | 87 | 2400480 240081 2400508 6500479 2400467 | 0 0 0 0 0 | 0 |
| 87 | 0 | 0 | 2400480 240081 2400508 6500479 2400467 | 0 0 0 0 0 | 0 |
| 114 | 479 | 102 | 2400480 240081 2400508 6500479 2400467 | 1 1 0 1 0 | 11/20 |

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
| popularity | 0.79478307 | 13.314 | 6,384 | 2,206 | 46 | 0.4410 |
| repeat_recency_heuristic | 0.79604233 | 26.574 | 6,384 | 2,170 | 37 | 0.4410 |
| content_metadata | 0.18695767 | 2.105 | 6,384 | 6,384 | 661 | 0.4410 |
| content_metadata_title | 0.34637037 | 2.370 | 6,384 | 6,384 | 422 | 0.4410 |
| association_count25_lift1 | 0.77388360 | 4.158 | 6,384 | 1,197 | 48 | 0.4410 |
| association_count50_lift1_2 | 0.40041534 | 3.946 | 6,384 | 953 | 153 | 0.4410 |
| hybrid_rrf60 | 0.81696561 | 4.492 | 6,384 | 2,040 | 37 | 0.4410 |

All methods evaluate **1,260 users** with **0 excluded**, **6,384 candidates**,
and **229,679 relevant user-item pairs**; there are
10 users without usable training history. Empty relevance
is zero AP. The baseline runtimes are from the preserved baseline run. Model runtimes
are per fitting/ranking configuration and exclude common sparse-matrix preparation
(4.565 s). The advanced experiment took
27.032 s total.

Content-only recommendations are weak (0.18696); title tokens help (0.34637) but remain
below popularity. Association rules at count 25/lift 1.0 reach 0.77388; stricter support
and lift reduce MAP to 0.40042. The equal-weight hybrid reaches 0.81697, +0.02218 over
popularity and +0.02092 over repeat/recency. The paired per-user normal 95% interval
for the hybrid-popularity difference is [0.01469,
0.02967]; it is not adjusted for model/configuration
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
| popularity | 44 | 87 | 0.0000 | baseline | 2400480 240081 2400508 6500479 2400467 |
| repeat_recency_heuristic | 363 | 293 | 0.0667 | -0.9333 | 90072034 120078072 1000376 90063178 90056307 |
| content_metadata | 7 | 284 | 0.0000 | -1.0000 | 120069194 90077525 90082810 90031726 90054169 |
| content_metadata_title | 8 | 193 | 0.0000 | -1.0000 | 90077525 90082810 90054169 90081609 6000473 |
| association_count25_lift1 | 1221 | 75 | 0.3833 | -0.4167 | 2400467 240081 2400480 6500479 10002514 |
| association_count50_lift1_2 | 11 | 104 | 0.0000 | -1.0000 | 6000483 200506 400426 10002893 10002451 |
| hybrid_rrf60 | 532 | 36 | 0.1000 | -0.4500 | 200352 240081 700389 2400508 10002514 |

## V. Verification and Reproduction

Seven recommendation-method tests passed. They cover content-only similarity, explicit
empty histories, deterministic score/popularity/ID tie-breaking, directional support/
confidence/lift, self-rule removal, zero-history fallback, RRF determinism, and a
temporal-cutoff fixture. Across all 8,820
method/user forecasts, each list has five distinct positive eligible items. All
70 cold-start lists equal global popularity.
Independent top-rank checks cover six real users per generated method. Candidate
eligibility is asserted before AP; final fit uses no viewing at/after July 27.

```powershell
$env:SWEET_TV_CACHE = "D:\sweet-tv-midterm-cache"
.\.venv\Scripts\python.exe data/src/critical_validation_audit.py
.\.venv\Scripts\python.exe data/src/test_recommendation_methods.py
.\.venv\Scripts\python.exe data/src/advanced_experiment.py
.\.venv\Scripts\python.exe data/src/advanced_submission.py
```

The audit script above reuses the database and does not rerun the original CSV data
audit, but independently rechecks held-out labels from source CSVs. The sparse matrices
and rule matrices are stored under the ignored external cache, not tracked results.
Dependencies: NumPy 2.4.4, SciPy 1.17.1,
DuckDB 1.4.3. Baseline files are hash-protected.
Measured evidence includes `results/advanced_comparison.csv`,
`results/advanced_per_user_evaluation.csv`, `results/advanced_experiment.json`,
`results/advanced_validation_audit.json`, rule examples, and
`results/advanced_submission_validation.json`.

## VI. Final-Period Candidate Artifact

The measured best development strategy, hybrid RRF60, was refit on 3,741,530
identified viewing-airing interactions completed before **2020-07-27 00:00**. It uses
6,649 candidates from the final schedule. The separate file
`results/advanced_candidate_submission.csv` passed structural validation: 1,260 unique
users in template order, 6,300 prediction slots, correct header and LF formatting,
five distinct positive eligible IDs per user, and zero malformed/ineligible/duplicate
predictions. SHA-256: `4abb82f4dd24b0c8f199ea33262f2c2ec474e42cd161f8b8083f3d92d85248ee`. The earlier
`results/submission_baseline.csv` remains unchanged. No submission was uploaded.
Final refit and validation runtime: 33.137 s. The two users
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


## VIII. Pre-Kaggle Reliability Follow-up

One fixed weeks 15-16 chronological inner holdout and a local candidate-file rules
check were completed after this report. See [the reliability addendum](pre_kaggle_reliability.md)
for measured scores, per-user cases, boundary validation, and the upload recommendation.


## IX. Strict Relevance Reassessment

The later strict single-airing >80% ground-truth reassessment supersedes the 60-second
proxy MAP@5 values above for Kaggle-alignment. See [the strict relevance report](strict_relevance_assessment.md).
