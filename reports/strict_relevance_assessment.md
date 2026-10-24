# Reassessment with Single-Airing >80% Viewing Labels

## Protocol provenance and status

The two-layered solution article describes a TV program as watched when more than 80%
of one airing is watched in one sitting without changing channels. The same author
separately describes a >50% per-sitting training feature. This provides strong
competition-specific guidance, but is a participant's account, **not independently
confirmed official rule text**. The author is the article's second-place participant,
not the winner; the post reports that author's private MAP@5 as 0.42905.

Official Kaggle page: [sweet.tv - TV program recommender](https://www.kaggle.com/c/sweettv-tv-program-recommender/overview).
The official title/closed status is accessible; the detailed evaluation/rules page
returns only its JavaScript shell here. An indexed Kaggle leaderboard under the
[`sweet-tv-channel` alias](https://www.kaggle.com/competitions/sweet-tv-channel/leaderboard)
shows a public leader score of 0.28663. This cannot be reconciled
with the participant's stated private 0.42905 from the accessible official page. Neither
score was used for model selection. Thus, use the following as a **strict participant-rule
reconstruction**, not an officially certified evaluator.

### Operational label definition

For each usable raw viewing row and one scheduled airing on the matching numeric channel,
`coverage = overlap_seconds / airing_duration_seconds`. A user-program validation label
is positive if **any single viewing-row/airing pair has coverage > 0.80**. Exactly 80%
is negative. We do not add multiple sessions, airing fragments, or channels together.
`vsetv_id` is treated as the source channel ID mapped to `channel_id`; an input log row
contains only one channel ID, but there is no separate intra-row channel-change telemetry.
An event is therefore assumed uninterrupted within that row. Existing quality gates
(positive intervals, correct DST-aware elapsed duration, <=24-hour viewing record,
first duplicate only, and no overlapping schedule ambiguity) remain applied.

Training features use a separate **strictly >50% coverage per individual airing/session**
signal. No event meeting only 50% qualifies; training signals do not define validation
relevance. Repeat and association histories are deduplicated by user-program for their
binary features. All previously used models retain their fixed parameters, including
the 14-day repeat half-life, association min-pair-count 25/lift 1.0, and RRF60. No new
hyperparameter search was done.

## Label audit and held-out results

Both folds evaluate all 1,260 template users, with empty truth receiving AP=0, and
identical per-fold schedule-intersection candidates for every method.

| Holdout | Method | Old 60-sec proxy MAP@5 | Strict >80% MAP@5 | Users | Excluded | Candidates | Proxy relevant pairs | Strict relevant pairs | Empty users |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inner_weeks15_16 | popularity | 0.63760317 | 0.32256239 | 1260 | 0 | 2143 | 75493 | 8326 | 104 |
| inner_weeks15_16 | repeat_recency | 0.72389418 | 0.40682760 | 1260 | 0 | 2143 | 75493 | 8326 | 104 |
| inner_weeks15_16 | content_based | n/a | 0.30822950 | 1260 | 0 | 2143 | 75493 | 8326 | 104 |
| inner_weeks15_16 | association_rules | n/a | 0.31323567 | 1260 | 0 | 2143 | 75493 | 8326 | 104 |
| inner_weeks15_16 | hybrid_rrf60 | 0.67445966 | 0.37190498 | 1260 | 0 | 2143 | 75493 | 8326 | 104 |
| outer_weeks21_30 | popularity | 0.79478307 | 0.45134193 | 1260 | 0 | 6384 | 229679 | 39698 | 24 |
| outer_weeks21_30 | repeat_recency | 0.79604233 | 0.49195503 | 1260 | 0 | 6384 | 229679 | 39698 | 24 |
| outer_weeks21_30 | content_based | 0.34637037 | 0.34429740 | 1260 | 0 | 6384 | 229679 | 39698 | 24 |
| outer_weeks21_30 | association_rules | 0.77388360 | 0.44183708 | 1260 | 0 | 6384 | 229679 | 39698 | 24 |
| outer_weeks21_30 | hybrid_rrf60 | 0.81696561 | 0.50994400 | 1260 | 0 | 6384 | 229679 | 39698 | 24 |

The old results in the proxy column used a >=60-second matching rule and are **not
Kaggle-equivalent ground truth**. Training signals for the strict rows above use >50%
per-airing coverage; the proxy training configuration had its previous 60-second event
threshold. The strict criteria sharply reduce user-program relevance:

| Holdout | 60-sec proxy pairs | Strict >80% pairs | Proxy-only pairs | Strict >80% positive event/airing pairs |
| --- | --- | --- | --- | --- |
| inner_weeks15_16 | 75493 | 8326 | 67167 | 50932 |
| outer_weeks21_30 | 229679 | 39698 | 189981 | 282231 |

Inner strict labels are 8,326 pairs versus
75,493 proxy pairs; outer strict labels are
39,698 versus
229,679. That is an 89.0%
proxy-pair reduction in the inner fold and 82.7% in the outer fold. In both folds all strict
pairs are also proxy pairs under the cached >=60-second definition; the proxy adds
many short program views.

Across the two strict folds, mean MAP@5 is: popularity 0.386952, repeat_recency 0.449391, content_based 0.326263, association_rules 0.377536, hybrid_rrf60 0.440924.
Repeat/recency has the best mean and worst-fold strict MAP, **0.449391**
and **0.406828**. Hybrid RRF60 is the best non-repeat alternative
by maximin, at mean 0.440924 and worst-fold 0.371905.
Hybrid remains the strongest outer-fold method but trails repeat/recency in the inner
fold. This is a clearer, stricter signal than the proxy results, but only two temporal
holdouts and the participant-reported definition are available.

## Independent source-row verification

The existing interval cache was reused. For six positive and six negative event examples,
a separate Python `csv.DictReader` plus `zoneinfo` scan re-read the raw viewing and
schedule rows, recomputed UTC interval overlaps and airing fractions, and reconstructed
the complete strict label sets for 12 sampled users. All 12 user-level sets matched the
cached SQL labels. Positive examples include full-airing views; negatives include exact
80% and 79.982% coverage. See `results/strict80_label_examples.json` for source row
numbers, overlap seconds, airing seconds, and direct fractions.

The >80% strict labels were independently compared with the <=80% event labels and
the separate >50% training features. The split checks show zero future training rows,
zero validation rows outside the target windows, and zero strict labels outside the
schedule candidate pools. Python MAP@5 was checked against independent DuckDB window
calculation for 12,600 method-user-fold scores; maximum difference was zero.

## Final-period submissions

The models were refit on all usable pre-final viewing with **>50% coverage of one
airing per viewing row**; final viewing behavior was never read. Final candidates are
positive IDs from schedule airings intersecting `[2020-07-27, 2020-10-19)`, total
6,649 programs. Two separate, validator-passed files were created:

- `results/submission_strict80_repeat_recency.csv` — recommended first candidate based
  on the best mean and worst-fold strict MAP@5.
- `results/submission_strict80_alternative_hybrid_rrf60.csv` — best maximin non-repeat
  alternative under the two folds.

Each has the exact template header/order, 1,260 users, 6,300 slots, five unique positive
eligible items per user, zero malformed rows, and no item 0. Hashes and validation are
in `results/strict80_summary.json`. Previously watched programs were **not** excluded;
the participant's baseline itself recommends past programs if scheduled again. The
official repeat policy remains unverified. Neither file was uploaded.

## Reproducibility and limitations

Run `data/src/strict_relevance_experiment.py` after the existing audit cache is available;
it reuses `match_edges` and does not rebuild the original audit. Five threshold unit tests
cover >80%, >50%, exactly 80%, nonpositive airing duration, and nonaggregation of two
separate 51% sessions. The original CSV SHA-256 values remained unchanged.

Unresolved: official Kaggle scoring/rules content is inaccessible from the official
JavaScript page; the participant post is not organizer documentation; the participant's
0.42905 private score conflicts with the accessible indexed public leaderboard value;
the source cannot detect channel changes hidden within a single viewing row; and the
official policy on recommending previously viewed items is not confirmed. These limits
mean the strict labels are well-verified from supplied records but still provisional as
an exact reconstruction of the organizer evaluator.

## Academic Integrity and Implementation Provenance

The cited Medium article was inspected as **related-work prose only**. It motivated
testing the reported >80% single-airing relevance definition and keeping the >50%
training signal separate. Its linked repository and Kaggle notebook code were **not
opened, cloned, downloaded, executed, copied, translated, ported, or adapted**. Search
results surfaced repository/notebook titles and snippets only; no external source code
was accessed or reused.

The required content-based and association-rule methods are the project's own
independent implementations in `data/src/recommendation_methods.py`: sparse TF-IDF
metadata profiles/cosine ranking, and directional user-transaction rules computed from
sparse co-occurrence. The strict reassessment uses the project's cached SQL interval
joins and a separate source-CSV verification scan. NumPy, SciPy, and DuckDB are the
implementation libraries. The article is cited only for its participant-reported
methodology; its code is not a source of this implementation. No session history or raw
CSV was modified.
