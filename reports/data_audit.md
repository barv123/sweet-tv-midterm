# SWEET.TV data audit and recommendation data pipeline

## 1. Scope and reproducibility

All five local CSVs were parsed in full, with strict CSV validation and no skipped bad rows.
Counts below are data records excluding headers, not physical newline estimates.
Inputs were read-only; all five SHA-256 checksums matched before and after processing.
No recommendation model was trained and no predictions were generated.

Reproduce from the repository root (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:SWEET_TV_CACHE = "D:\sweet-tv-midterm-cache"
.\.venv\Scripts\python.exe data/src/run_pipeline.py
```

The cache path is configurable; without `SWEET_TV_CACHE` it defaults to `.cache/audit/`.
This run used D: with permission after C: ran out of temporary spill space. The initial
C: attempt and an interrupted matching attempt did not change source CSVs. Measured
runtimes below refer to completed runs, excluding setup and failed attempts.

DuckDB uses a 768 MB managed-memory budget, two workers, disk-backed tables, and
weekly interval-join batches. CSVs are imported as text one at a time, then converted
with explicit formats. Python holds only summaries and small verification samples.
The managed-memory budget is not a limit on total process resident memory.

## 2. Dataset inventory

| File | Rows | Columns | Users | Channels | Positive program IDs |
| --- | --- | --- | --- | --- | --- |
| dataset11-30.csv | 5,821,987 | 5 | 4,914 | 202 | N/A |
| export_arh_11-20-final.csv | 1,074,235 | 13 | N/A | 400 | 5847 |
| export_arh_21-30-final.csv | 1,074,394 | 13 | N/A | 403 | 6363 |
| export_arh_31-42-final.csv | 1,295,378 | 13 | N/A | 404 | 6636 |
| submission.csv | 1,260 | 2 | 1,260 | N/A | N/A |

Across all schedules: **405 channel IDs** and
**9,671 positive program IDs**. Program titles are
not item identifiers. Schedule-only tables have no user dimension. Viewing rows have
no direct program ID; submission item lists are all placeholders.

### Columns and logical types

Viewing: `user_id` is an exact decimal identifier stored as VARCHAR; `vsetv_id` is BIGINT;
`start_time` and `stop_time` parse as timestamps; **`duraton`** is the actual misspelled
source column, interpreted as integer elapsed seconds. All five columns have zero
missing values. There are no invalid numeric IDs or unparsable timestamps.
2,918,899 rows have user IDs above the
signed-int64 maximum. Never load IDs through floating point or signed int64.

All three schedule files share exactly these columns/types:

| Column | Validated logical type |
| --- | --- |
| channel_id | BIGINT |
| channel_title | VARCHAR |
| start_time | TIMESTAMP |
| duration | BIGINT |
| tv_show_title | VARCHAR |
| tv_show_id | BIGINT |
| tv_show_category | VARCHAR |
| tv_show_genre_1 | VARCHAR |
| tv_show_genre_2 | VARCHAR |
| tv_show_genre_3 | VARCHAR |
| year_of_production | VARCHAR |
| director | VARCHAR |
| actors | VARCHAR |

`year_of_production` remains text because values include ranges/open-ended years.
`duration` is integer elapsed seconds. No schedule identifier or timestamp failed
conversion. Native CSV fields are text; logical types above describe explicit parsing.

Submission columns are exactly `user_id,tv_show_id`; both are strings during parsing.
The second is a five-token list, not a scalar numeric column. Neither is missing.

### Schedule missingness

Missing means an empty, null, or whitespace-only field. Zero is not counted as missing.

| Column | Weeks 11-20 | Weeks 21-30 | Weeks 31-42 |
| --- | --- | --- | --- |
| channel_id | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| channel_title | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| start_time | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| duration | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| tv_show_title | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| tv_show_id | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| tv_show_category | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| tv_show_genre_1 | 620,089 (57.72%) | 616,370 (57.37%) | 737,871 (56.96%) |
| tv_show_genre_2 | 928,009 (86.39%) | 924,020 (86.00%) | 1,113,598 (85.97%) |
| tv_show_genre_3 | 1,054,059 (98.12%) | 1,050,668 (97.79%) | 1,266,621 (97.78%) |
| year_of_production | 959,500 (89.32%) | 954,382 (88.83%) | 1,151,706 (88.91%) |
| director | 974,050 (90.67%) | 969,487 (90.24%) | 1,170,189 (90.34%) |
| actors | 973,861 (90.66%) | 968,834 (90.17%) | 1,169,519 (90.28%) |

### Actual timestamps and coverage

| File | Earliest start | Latest start | Latest stop/end |
| --- | --- | --- | --- |
| dataset11-30.csv | 2020-03-09 00:00:02 | 2020-07-26 23:58:42 | 2020-07-26 23:59:58 |
| export_arh_11-20-final.csv | 2020-03-08 23:00:00 | 2020-05-18 11:45:00 | 2020-05-18 12:00:00 |
| export_arh_21-30-final.csv | 2020-05-18 00:00:00 | 2020-07-27 11:55:00 | 2020-07-27 12:00:00 |
| export_arh_31-42-final.csv | 2020-07-27 00:00:00 | 2020-10-19 11:55:00 | 2020-10-19 12:00:00 |

Viewing format: `%Y-%m-%d %H:%M:%S`; schedule format: `%d.%m.%Y %H:%M:%S`.
Both omit timezone offsets. Schedule end is derived, not a source column.
The minimum/maximum schedule ends above are unchanged by the DST correction below.
Viewing covers ISO weeks 11-30 of **2020**; there are no final-period viewing labels.
Schedules contain broadcast-day spillover beyond their named periods; never split by
filename alone. Schedule rows outside named periods are counted in `matching.json`.

`viewing_by_week.csv` and `schedule_by_week.csv` give all weekly counts and bounds.
Viewing activity ranges from 105,223 sessions in week 27
to 465,318 in week 13. User/session activity drops
substantially during June and rebounds in July. These files alone cannot establish
whether this is user behavior, sampling, or logging coverage; it creates distribution shift.

## 3. Data quality and identifier integrity

### Viewing durations

All 5,821,987 rows have positive reported durations; minimum
61 seconds, median 396 seconds,
mean 1480.46 seconds, p95 6344 seconds,
p99 14079.14 seconds, maximum
342,145 seconds (95.04 hours).
There are 52 sessions longer than 24 hours and no sessions
at or below 60 seconds. This is consistent with upstream filtering, but its policy is unknown.
Exact p1/p5/p25/p50/p75/p90/p95/p99/p99.9, standard deviations, and schedule-duration
distributions are in `audit.json`. The schedules contain one zero-duration row, excluded
from matching, and no negative durations.

### Duplicates, overlaps, and inconsistent keys

- Exact duplicate viewing rows (excess copies): **17**; retain the first source record.
- Exact duplicate rows in each schedule and submission: **0**. Semantic duplicate schedule rows: **0**.
- Same `(user, channel, start)` with different end/duration: **105 keys**,
  affecting 210 records and 35 users.
- 694,730 viewing rows overlap a previous
  running interval for the same user; 3,243 users are affected.
  After exact deduplication, 96,555 rows still overlap on
  the same user/channel. These may be multiple devices, restarts, or logging duplication;
  the data do not identify which. They remain separate source events, not independent watch-time evidence.
  Downstream weights must union overlapping intervals per user/item before summing time;
  binary user-item labels are already deduplicated for evaluation.
- 1 conflicting schedule `(channel,start)` key;
  13 overlapping schedule rows across all periods;
  9 historical viewing sessions overlap simultaneous airings and are quarantined.
  Consecutive programs in a long session are valid multiple interactions, not ambiguity.
- 9 channel IDs have multiple titles;
  1 channel title maps to multiple IDs.
  4,351 positive program IDs have multiple titles and
  284 exact titles map to multiple positive IDs.
  Episode/season/prefix changes can explain some variation; no title-based ID merging was performed.

### Unknown program IDs

| Schedule | Rows with ID 0 | Percentage |
| --- | --- | --- |
| export_arh_11-20-final.csv | 958,622 | 89.2376% |
| export_arh_21-30-final.csv | 952,889 | 88.6908% |
| export_arh_31-42-final.csv | 1,150,604 | 88.8238% |

Overall: 3,062,115 / 3,444,007
(88.9114%). **ID 0 is never a recommendation item.**
Unknown airings contribute only to coverage diagnostics; their titles are not used to
invent IDs. The missing metadata and item IDs make identifiable viewing a biased subset.

## 4. Channel mapping and timezone investigation

### Evidence for channel identity

197 of 202 viewing IDs occur as schedule
channel IDs, covering 5,708,408 viewing records
(98.0491%). Missing IDs are
`149, 1174, 1508, 1509, 1510`.
`channel_coverage.csv` preserves each observed numeric ID, schedule title, and temporal
coverage. Names in numerical outputs are original source values, not translations.

The viewing file contains no channel names or external crosswalk. Numeric agreement
therefore supports, but does not independently certify, semantic channel identity.
A fixed cyclic permutation of shared channel IDs is used as a negative control below;
equal IDs show stronger alignment with program boundaries. This is empirical evidence,
not a proof that every channel ID is correct. A public VseTV lookup was attempted but
was unavailable; no external identity verification is claimed. No remapping was invented.

### DST: measured correction

Naive `stop-start` differs from reported viewing duration in exactly
117 rows, always by +3,600 seconds, starting March 28-29.
All **117** resolve exactly when interpreted as
`Europe/Kyiv` local timestamps and subtracted in UTC. Neither viewing endpoints nor
schedule starts occur in the missing local hour 03:00-03:59 on March 29.
For **394** schedule intervals spanning that transition, naive `start+duration` is
one hour earlier than the next airing. Adding elapsed seconds in UTC removes that gap.
This supports a Kyiv-compatible DST rule; geographic timezone identity is not encoded
and cannot be distinguished from other zones with the same offsets during this window.

The matcher localizes both sources to this rule, adds schedule duration on the UTC
timeline, and measures overlap seconds in UTC. Corrected local boundaries are stored
for readability and calendar splits. There is no autumn repeated hour within the
observed March 9-October 19 horizon. A general extension beyond this horizon must
explicitly resolve ambiguous/nonexistent local timestamps, rather than reuse this assumption.
After correction, **0 viewing-duration mismatches remain**.

### Offset sensitivity and channel negative control

Deterministic sample: every 100th source row in training, restricted to March 10-May 17;
38,800 sessions. Only training sessions enter the overlap
and boundary-alignment measurements. The negative-control permutation uses the shared-ID
inventory from the full audit; it is a diagnostic, not a fitted prediction feature.
Shifts are applied to viewing timestamps relative to schedules.
Stop-boundary alignment means within 120 seconds of a scheduled program start.

| Mapping | Shift (hours) | Sample | Identified matches | Stop alignment | Before DST | After DST |
| --- | --- | --- | --- | --- | --- | --- |
| equal_numeric_ids | -3 | 38,800 | 18,018 | 6.92% | 6.56% | 7.08% |
| equal_numeric_ids | -2 | 38,800 | 18,047 | 7.02% | 7.10% | 6.99% |
| equal_numeric_ids | -1 | 38,800 | 18,235 | 6.64% | 6.85% | 6.55% |
| equal_numeric_ids | 0 | 38,800 | 18,651 | 8.97% | 9.24% | 8.86% |
| equal_numeric_ids | 1 | 38,800 | 18,297 | 6.76% | 7.08% | 6.62% |
| equal_numeric_ids | 2 | 38,800 | 18,255 | 6.39% | 6.37% | 6.40% |
| equal_numeric_ids | 3 | 38,800 | 18,269 | 6.48% | 6.61% | 6.42% |
| cyclic_permutation_control | 0 | 38,800 | 14,739 | 6.95% | 7.28% | 6.81% |

Zero shift has the strongest boundary alignment both before and after DST. Continuous
schedules also match many deliberately shifted records, so join rate alone cannot prove
timezone correctness. The control is a diagnostic, not a calibrated confidence interval.
All reported interaction counts remain conditional on the supported equal-ID mapping.

## 5. Matching algorithm, measured coverage, and usable interactions

1. Parse IDs without precision loss and timestamps using their explicit source formats.
2. Derive schedule end by converting local start to UTC, adding `duration`, and converting back.
3. Match equal numeric channel IDs and half-open intervals `[start, end)`:
   `overlap_start = max(view_start, airing_start)`,
   `overlap_end = min(view_stop, airing_end)`; require strictly positive UTC elapsed overlap.
4. Expand intervals into calendar-day buckets and process viewing one week at a time.
   Keep each session-airing pair once, including sessions spanning midnight.
5. Retain all sequential overlapping airings rather than just the program at session start.
   Quarantine simultaneous schedule overlaps. Exclude nonpositive schedule durations,
   duplicate session copies, duration-inconsistent sessions, and sessions longer than 24 hours.
6. Only positive `tv_show_id` values enter interactions. The default prepared interaction
   table additionally requires at least 60 seconds on an airing. This is an explicit
   provisional relevance policy, not a recovered Kaggle labeling rule.

| Viewing outcome | Records | Share of all viewing |
| --- | --- | --- |
| At least one identified program | 2,964,378 | 50.9169% |
| Schedule overlap, only unknown program IDs | 2,444,312 | 41.9842% |
| No schedule overlap | 413,297 | 7.0989% |

Any schedule overlap: 5,408,690 sessions.
Unmatched: 113,579 missing-channel records and
299,718 shared-channel records outside available schedule intervals.
Of the latter, 297,331 end before their
channel's schedule begins, 2,009 start after it ends,
and 378 fall in internal gaps.
Some channels first appear in schedules in April even though viewing begins in March;
see `unmatched_by_channel.csv` for exact attribution. Do not fill these gaps from another channel.

408,683 matched sessions are only partially attributable to identified
programs. 522,994 sessions span multiple distinct identified items.
78 source records are excluded by quality gates. With any positive
overlap, 2,964,330 quality-approved sessions generate
3,846,979 session-airing edges.

### Overlap-threshold sensitivity

| Minimum seconds | Session-airing edges | Sessions | Users | Items | Unique user-item pairs |
| --- | --- | --- | --- | --- | --- |
| 1 | 3,846,979 | 2,964,330 | 4,895 | 4,363 | 1,371,528 |
| 30 | 3,793,936 | 2,951,930 | 4,895 | 4,363 | 1,361,124 |
| 60 | 3,741,530 | 2,936,467 | 4,895 | 4,362 | 1,350,278 |
| 300 | 2,377,966 | 1,747,939 | 4,890 | 4,272 | 934,011 |
| 600 | 1,815,994 | 1,317,891 | 4,888 | 4,235 | 767,281 |

**Prepared default:** 3,741,530 interactions from 2,936,467 sessions,
4,895 users, 4,362 identified items, and 1,350,278
distinct user-item pairs. An interaction is a session-airing edge, not a unique user-item label.
4,914 users have positive raw histories;
19 have no usable identified history under the default policy.
Of the 1,260 submission users, 1258 have usable histories;
2 need a cold-user fallback.

The ready-to-use local DuckDB table is `interactions`; split-safe views are
`train_interactions` and `validation_interactions`. The cache also retains `viewing`,
`schedule`, `historical_schedule`, `match_edges`, and `session_quality` for inspection.
All detailed user-level tables stay in the local cache, not in tracked results.

### Independent verification

14 of 14 real-session checks passed.
A separate Python `csv.DictReader` scan reads both historical schedule files and computes
UTC overlaps with `datetime`/`zoneinfo`, independently of DuckDB joins and date parsing.
The complete expected set of airings, IDs, source row numbers, and seconds must equal
the SQL result for every example. Cases include missing channels, schedule gaps, zero
IDs, sequential programs, simultaneous airings, midnight, DST, exact boundaries,
long sessions, and the train/validation boundary.

| Viewing data row | Case | Channel | Start (local offset) | Stop (local offset) | Identified overlaps |
| --- | --- | --- | --- | --- | --- |
| 26 | unmatched_channel | 1174 | 2020-03-09 00:00:41+02:00 | 2020-03-09 00:01:42+02:00 | No identified item |
| 2418 | single_identified | 1495 | 2020-03-09 01:35:55+02:00 | 2020-03-09 01:37:07+02:00 | 12001732 (72s) |
| 411360 | ambiguous | 1020 | 2020-03-16 06:20:07+02:00 | 2020-03-16 06:47:58+02:00 | No identified item |
| 1250740 | dst_crossing_identified | 7 | 2020-03-28 22:56:59+02:00 | 2020-03-29 04:11:55+03:00 | 90074731 (2581s), 90082541 (5400s), 200469 (2700s) |
| 3945174 | validation | 23 | 2020-05-18 00:00:05+03:00 | 2020-05-18 00:01:36+03:00 | 90082248 (91s) |

Row numbers above exclude the CSV header. For example, viewing row 2,418 lies wholly
inside channel 1495's 01:16-01:39 airing on March 9: item 12001732, exactly 72 seconds.
Viewing row 3,945,174 overlaps channel 23's May 17 23:25-May 18 01:00 airing for 91
seconds; its schedule record is in the earlier file despite being a validation session.
Both examples were also checked directly against the source CSV rows.

Full-set invariants all passed: no repeated session-airing pairs, nonpositive overlaps,
zero/negative interaction items, quality-gate violations, overlap arithmetic errors,
overcounted unambiguous intervals, temporal split violations, or final-schedule leakage.
All validation label items occur in the validation candidate catalog.

## 6. Leakage-free chronological validation design

Use local-calendar ISO-week boundaries, converted consistently when working in UTC:

| Period | Inclusive start | Exclusive cutoff | Raw sessions by start | Usable interactions | Usable users | Items |
| --- | --- | --- | --- | --- | --- | --- |
| train | 2020-03-09 00:00:00 | 2020-05-18 00:00:00 | 3,945,173 | 2,310,865 | 4,841 | 2,972 |
| validation | 2020-05-18 00:00:00 | 2020-07-27 00:00:00 | 1,876,814 | 1,428,512 | 4,806 | 3,229 |
| refit | 2020-03-09 00:00:00 | 2020-07-27 00:00:00 | 5,821,987 | 3,741,530 | 4,895 | 4,362 |

- **Training weeks 11-20:** March 9-May 17. Features only from sessions completed
  by May 18 00:00. 1007 raw sessions cross this cutoff;
  exclude them from both holdout partitions, rather than exposing post-cutoff duration.
  They can return for final refitting when their complete history is available.
- **Validation weeks 21-30:** May 18-July 26. Freeze all fitted statistics, item/user
  vocabularies derived from behavior, popularity, recency, and neighbor weights at May 18.
  Validation viewing supplies labels only. Build relevance as distinct positive IDs
  from quality-approved validation overlaps under a declared minimum-duration policy.
- **Final prediction weeks 31-42:** July 27-October 18, exclusive cutoff October 19
  00:00. After choices are fixed, refit on weeks 11-30 and rank for all submission users.
  No observed final viewing is available; no final performance can be measured locally.
- Reject schedule spillover at/after the target end. Retain prior-file airings that
  overlap the target start. The code explicitly keeps the final schedule out of
  historical matching and verifies that training interactions only use the first schedule.

### What schedule information is available at prediction time?

The supplied files expose channel, start, elapsed duration, positive/unknown program ID,
title, category, genres, production year, director, and actors for future broadcasts.
For a **competition protocol that explicitly grants the full target-period schedule**, these
are exogenous candidate/content features: the validation union has 6,384 positive items
and the final union 6,649, including prior-file boundary spillover. No future viewing
counts, audience popularity, or labels may be used as features.

However, these are archive exports with no publication/revision timestamps. The files
cannot prove that a ten- or twelve-week schedule was actually known at the historical
prediction origin. Treat full-horizon schedule access as a competition assumption that
must be checked against its rules. For an operational backtest, use timestamped schedule
snapshots available at each origin (or rolling forecasts) and only the documented lead time.
Do not claim an operationally leakage-free schedule snapshot from these exports alone.

### Evaluation cohorts and metric

Validation has 4806 labeled users, including
54 without usable training history. There are
1390 validation items unseen in training interactions.
89 training-history users have no
identified validation positives. Among submission users, 10
lack training history and 14 lack identified validation labels.

Report at least: (1) all 1,260 submission users with AP=0 for empty observable truth;
(2) their 1246 nonempty-truth users;
(3) warm and cold cohorts separately. The primary convention must match the official
competition rule once obtained; empty observed truth can reflect missing IDs, not no interest.
Use the rest of the users as an additional diagnostic cohort, not an undisclosed replacement.

For nonempty distinct ground-truth set G, use
`AP@5 = sum(P@k * relevant_at_k, k=1..5) / min(5, |G|)`;
MAP@5 is the mean over the declared user cohort. Rank five distinct positive IDs and
count each hit once. There are 141,366 repeated
train/validation user-item pairs: do not automatically exclude previously watched shows.
The official relevance threshold, repeat policy, and empty-truth convention are not
recoverable from the CSV template and must not be fabricated.

Choose tuning settings on chronological inner folds within weeks 11-20, then use
weeks 21-30 as the primary holdout. If repeatedly tuning on that holdout, label its score
as development performance rather than an unbiased final estimate. Report sensitivity
to short overlaps, long-session filtering, unknown-ID coverage, and the 10- vs 12-week horizon.

## 7. Exact observed submission format

- File is comma-delimited UTF-8-compatible ASCII, LF line endings, no BOM or index.
- Header exactly: **`user_id,tv_show_id`**.
- Exactly **1,260 data rows**, one per unique template user; preserve the template's
  exact decimal ID strings and row order when writing predictions.
- Second column contains **five integer program IDs separated by single spaces**,
  ranked best first; no Python-list brackets, commas inside the list, or decimal suffixes.
- All source rows currently contain **`0 0 0 0 0`**, which are placeholders, not valid items.
  Replace them with five distinct positive eligible IDs. Output user IDs span 17-20 digits.
- The template establishes structure, not the official candidate eligibility rules or
  relevance/metric implementation. No Kaggle submission has been produced or uploaded.

## 8. Recommended next step toward a strong MAP@5 baseline

Confirm the organizer's channel namespace, clock convention, target-schedule access,
label threshold, and item eligibility. Then implement a transparent hybrid baseline:
recency-weighted personal repeat preferences plus per-user channel/category preferences,
backed off to training-only recent global popularity and restricted to known eligible
target-period items. Use channel preferences from unknown-ID sessions as training-only
side information without turning zero into an item. Union overlapping histories before
computing watch-time weights. New users receive a popularity/content fallback; unseen
target items require channel/content signals. Compare against pure recent popularity,
then evaluate item-item similarity or implicit-feedback factorization under the same split.
No model or MAP@5 result is claimed in this audit.

## 9. Artifacts, runtime, and limitations

Scripts: `data/src/audit.py`, `match.py`, `verify.py`, `report.py`, `run_pipeline.py`,
and shared `common.py`; dependency versions are pinned in `requirements.txt`.
Numerical outputs: `results/audit.json`, `matching.json`, `verification.json`,
`verified_examples.json`, and CSV breakdowns of weeks, channels, thresholds, splits,
missing schedule coverage, overlaps, and user-history distributions.
All raw files and pre-existing directories are preserved. Root-level `reports/` and
`results/` hold deliverables; existing empty `data/reports/` and `data/results/` remain.

Completed stage runtimes: audit **71.409s**, matching including probes
and independent checks **162.642s**, verification including rehashing
**24.028s**. These are measurements on this machine, not benchmarks.

End-to-end analysis wall time: **258.464s** (report rendering excluded).
Python 3.12.0; DuckDB 1.4.3; Windows.
Peak process working set: **1015.6 MiB**.
Final cache database size: **0.786 GiB**;
cache: `D:\sweet-tv-midterm-cache`. The working set measures resident process memory,
not total committed virtual memory. See `results/run_manifest.json` for timestamps.

Remaining limits: channel identity lacks an authoritative crosswalk; the observed clocks
support a Kyiv-compatible timezone but do not encode it explicitly; source logging and
upstream duration filtering are undocumented; unknown program IDs are highly prevalent;
publication-time schedule availability and the Kaggle truth policy are not supplied.
Algorithmic verification establishes the stated matching procedure, not hidden semantic
ground truth. All these limitations apply to the usable-interaction counts above.
