# First recommendation experiment

## Resolved audit discrepancy

**3,741,530 = 2,310,865 training + 1,428,512 validation + 2,153 cutoff-crossing interactions.**

The 1,007 raw sessions starting before May 18 00:00 and ending after it contain
765 sessions with default usable interactions and
242 with none. One session can overlap several airings;
session counts and session-airing interaction counts are different units.
Of the 2,153 edges, 896 lie entirely before the cutoff,
530 entirely after it, and
727 straddle it.
The conservative split excludes the **entire crossing session** from both partitions,
even edges before the cutoff, so no future session endpoint/duration is needed at training time.
All of these interactions return for final refitting, when they are historical.
This is an intentional exclusion, not lost or duplicated data. Detailed reasons and
zero-violation leakage assertions are in `results/baseline_experiment.json`.

## Fixed experiment design

- Train: completed sessions in `[2020-03-09, 2020-05-18]`, with start at/after March 9.
  Validation: sessions starting at/after May 18 and completed by July 27.
  Viewing intervals are half-open; a session ending exactly at a cutoff is historical.
- Both methods predict exactly the same **1,260 template users**, using the same
  **6,384** eligible validation items; **0 users are excluded from the primary metric**.
- Candidates are positive IDs from positive-duration scheduled airings intersecting
  `[May 18, July 27)`, including the previous schedule file's boundary spillover.
  No candidate is selected using future viewing counts; never use item 0.
- Labels are distinct user/item pairs from the audit's >=60-second, quality-approved
  validation interactions. Previously watched items are allowed.
- The primary cohort has **229,679 relevant user-item pairs**,
  **3,169 distinct relevant programs**, and
  **14 empty-truth users**. All positives are in the candidate pool.
  Nonempty-truth and warm/cold scores below are additional diagnostics, not different method cohorts.
- This retains the audit's supported equal-channel-ID and Kyiv-compatible clock interpretation.
  Full target-period schedule access is treated as the supplied competition scenario;
  archives do not establish publication-time availability for an operational deployment.
- The 60-second label threshold is inherited, not claimed to be the hidden Kaggle truth rule.
  No Kaggle score was queried, and no hyperparameter search was performed.

## Methods and deterministic scoring

**Popularity:** `score(item) = count(distinct historical user_id)` among all usable training
interactions, including training users outside the evaluation template. Restrict to target
candidates; candidates with no historical viewers receive zero. Sort by score descending,
then numeric program ID ascending. Each user receives the same first five IDs.
Streaming Python sets independently verified every candidate's SQL viewer count.

**Separate repeat/recency heuristic:** one vote per distinct `(user, item, local overlap-start date)`.
Sum `2^(-age_in_calendar_days / 14)` relative to the prediction cutoff date; round the sum
to 12 decimal places. Rank historical eligible items by personal score descending, global
distinct-viewer popularity descending, then numeric item ID ascending. Fill unused slots
from the same popularity ordering, skipping repeats. Repeated viewing dates increase the
score; recent dates receive more weight; fragmented sessions on one date add no extra votes.
The 14-day half-life was fixed before evaluation. This heuristic is an additional diagnostic,
**not one of the two required advanced recommendation methods**.

There are 10 template users without usable training history.
The heuristic uses a complete popularity fallback for 10 users
with no eligible personal candidates and popularity padding for
14 users with fewer than five personal candidates.
Cold-user predictions are checked to equal the popularity baseline exactly.

## Observed MAP@5

| Method | Cohort | Evaluated | Excluded template users | MAP@5 |
| --- | --- | --- | --- | --- |
| popularity | all_template_users | 1260 | 0 | 0.79478307 |
| popularity | nonempty_truth | 1246 | 14 | 0.80371322 |
| popularity | warm_history | 1250 | 10 | 0.79761333 |
| popularity | cold_history | 10 | 1250 | 0.44100000 |
| repeat_recency_heuristic | all_template_users | 1260 | 0 | 0.79604233 |
| repeat_recency_heuristic | nonempty_truth | 1246 | 14 | 0.80498662 |
| repeat_recency_heuristic | warm_history | 1250 | 10 | 0.79888267 |
| repeat_recency_heuristic | cold_history | 10 | 1250 | 0.44100000 |

`AP@5 = sum(precision@rank for first-time relevant hits) / min(5, number of distinct relevant items)`.
Empty ground truth receives AP=0. User means are unweighted. Duplicate predictions never
earn additional hits and are rejected in actual ranked outputs.

Independent manual example: relevance `{11,22,33,44}`, ranking `[11,99,22,88,33]`.
Hits at ranks 1,3,5 give `(1 + 2/3 + 3/5) / 4 = 17/30 = 0.5666666667`.
Python and an independent SQL cumulative-hit implementation agreed on all
**2,520** method/user AP values
(maximum difference 0).
`test_metrics.py` also checks capped denominators, empty labels, duplicate handling,
rank truncation, cohort inclusion, and missing predictions.

## First Kaggle submission

**`results/submission_baseline.csv` uses popularity**, preselected before comparing scores.
Refit on all 3,741,530 usable interactions completed by July 27;
no final-period viewing is available or accessed. The target is `[July 27, October 19)`.
The final schedule candidate pool contains **6,649** positive items,
including previous-file boundary spillover. All 1,260 users receive the final popularity
top five: `2400480 240081 6500479 2400508 2400467`.

Programmatic validation passed:
- Exact original template order, **1260 rows / 1260 unique users**.
- Exact header `user_id,tv_show_id`, LF line endings, no BOM/index/missing fields.
- **6,300 prediction slots**, five distinct positive eligible IDs per user.
- Invalid formatting rows: 0; ineligible-item rows: 0;
  repeated-item rows: 0.
- CSV SHA-256: `42525454d7611e9d88d3267bebadd51d112665e1683425e85f799afa9a6b4fe1`.
- All five raw CSV checksums match the completed audit.

This establishes local structural and candidate validity. The file has **not** been uploaded
to Kaggle; no leaderboard or private-score result is claimed.

## Runtime and reproduction

- Validation popularity fit/predict including independent count verification: 13.314s.
- Personalized heuristic additional fit/predict: 13.260s.
- Evaluation plus independent SQL checks: 44.575s.
- Final popularity refit/predict including verification: 17.944s.
- Total measured experiment including reconciliation, input hashing, and submission checks:
  **94.617s**, excluding report rendering and the prerequisite audit.

```powershell
$env:SWEET_TV_CACHE = "D:\sweet-tv-midterm-cache"
.\.venv\Scripts\python.exe data/src/test_metrics.py
.\.venv\Scripts\python.exe data/src/baseline_experiment.py
.\.venv\Scripts\python.exe data/src/validate_submission.py
```

Run `data/src/run_pipeline.py` first if the audited cache is absent. Dependencies are in
`requirements.txt`. Large history tables remain in the local cache; results contain only
small rankings, evaluation summaries/per-template-row scores, checks, and the submission.
No advanced recommendation method was started.

## Recommended next improvement

Use chronological inner folds within weeks 11-20 to compare repeat/recency settings
and training-only channel/category preferences, especially for cold items. Keep the
weeks 21-30 holdout protocol and candidate rules fixed; compare against this recorded
popularity benchmark. Confirm the assignment's exact relevance and schedule-access rules
before interpreting offline MAP@5 as an estimate of Kaggle performance.

## Follow-on experiment

The subsequent comparison of content-based filtering, association rules, and rank
fusion is documented in [the advanced recommendation experiment report](advanced_recommendation_experiment.md).
The original baseline results and submission remain preserved.
