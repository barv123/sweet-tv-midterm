"""Render measured JSON/CSV outputs into the English audit report."""

import csv
import json

from common import RESULTS, ROOT, SCHEDULES


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def table(headers, records):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(str(value) for value in record) + " |" for record in records])


def n(value):
    return f"{value:,}"


def run():
    a, m, v = load("audit.json"), load("matching.json"), load("verification.json")
    datasets = a["datasets"]
    viewing = datasets["dataset11-30.csv"]
    cov = m["coverage"]
    chosen = next(row for row in m["interaction_thresholds"] if row["minimum_overlap_seconds"] == 60)
    evidence = a["channel_id_evidence"]
    train, validation, refit = m["split"]
    manifest_path = RESULTS / "run_manifest.json"
    manifest = load("run_manifest.json") if manifest_path.exists() else None
    examples = load("verified_examples.json")
    with (RESULTS / "viewing_by_week.csv").open(encoding="utf-8", newline="") as handle:
        weekly = list(csv.DictReader(handle))
    smallest = min(weekly, key=lambda row: int(row["viewing_rows"]))
    largest = max(weekly, key=lambda row: int(row["viewing_rows"]))
    overview = []
    for filename, info in datasets.items():
        detail = info.get("coverage", {})
        overview.append([filename, n(info["rows"]), len(info["columns"]),
            n(detail.get("unique_users", info.get("format", {}).get("unique_users", 0))) if not filename.startswith("export") else "N/A",
            detail.get("unique_channels", "N/A"), detail.get("identified_programs", "N/A")])
    dates = [[filename, info["coverage"]["first_start"], info["coverage"]["last_start"],
              info["coverage"].get("last_stop", info["coverage"].get("last_end"))]
             for filename, info in datasets.items() if "coverage" in info]
    missing = []
    for column in datasets[SCHEDULES[0]]["columns"]:
        missing.append([column] + [f"{datasets[file]['missing'][column]:,} ({100*datasets[file]['missing'][column]/datasets[file]['rows']:.2f}%)" for file in SCHEDULES])
    dtype_rows = [[column, dtype] for column, dtype in datasets[SCHEDULES[0]]["logical_types"].items()]
    probe_rows = [[row["channel_mapping"], row["viewing_shift_hours"], n(row["sample_rows"]), n(row["identified_matches"]),
        f"{100*row['stops_within_120s_of_boundary']/row['sample_rows']:.2f}%",
        f"{100*row['before_dst_stop_aligned']/row['before_dst_rows']:.2f}%",
        f"{100*row['after_dst_stop_aligned']/row['after_dst_rows']:.2f}%"] for row in m["timezone_probes"]]
    thresholds = [[row["minimum_overlap_seconds"], n(row["airing_interactions"]), n(row["sessions"]), n(row["users"]), n(row["items"]), n(row["user_item_pairs"])]
                  for row in m["interaction_thresholds"]]
    selected_examples = []
    for example in examples:
        if any(case in example["cases"] for case in ("single_identified", "validation", "unmatched_channel", "ambiguous", "dst_crossing_identified")):
            positive = [f"{edge['tv_show_id']} ({edge['overlap_seconds']}s)" for edge in example["schedule_overlaps"] if edge["tv_show_id"] > 0]
            selected_examples.append([example["viewing_source_row"], ", ".join(example["cases"]), example["channel_id"],
                example["start_time"], example["stop_time"], ", ".join(positive) or "No identified item"])
    report = f"""# SWEET.TV data audit and recommendation data pipeline

## 1. Scope and reproducibility

All five local CSVs were parsed in full, with strict CSV validation and no skipped bad rows.
Counts below are data records excluding headers, not physical newline estimates.
Inputs were read-only; all five SHA-256 checksums matched before and after processing.
No recommendation model was trained and no predictions were generated.

Reproduce from the repository root (PowerShell):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
$env:SWEET_TV_CACHE = "D:\\sweet-tv-midterm-cache"
.\\.venv\\Scripts\\python.exe data/src/run_pipeline.py
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

{table(['File', 'Rows', 'Columns', 'Users', 'Channels', 'Positive program IDs'], overview)}

Across all schedules: **{n(a['schedule_integrity']['channels'])} channel IDs** and
**{n(a['schedule_integrity']['identified_programs'])} positive program IDs**. Program titles are
not item identifiers. Schedule-only tables have no user dimension. Viewing rows have
no direct program ID; submission item lists are all placeholders.

### Columns and logical types

Viewing: `user_id` is an exact decimal identifier stored as VARCHAR; `vsetv_id` is BIGINT;
`start_time` and `stop_time` parse as timestamps; **`duraton`** is the actual misspelled
source column, interpreted as integer elapsed seconds. All five columns have zero
missing values. There are no invalid numeric IDs or unparsable timestamps.
{n(viewing['identifier_checks']['users_above_signed_int64_rows'])} rows have user IDs above the
signed-int64 maximum. Never load IDs through floating point or signed int64.

All three schedule files share exactly these columns/types:

{table(['Column', 'Validated logical type'], dtype_rows)}

`year_of_production` remains text because values include ranges/open-ended years.
`duration` is integer elapsed seconds. No schedule identifier or timestamp failed
conversion. Native CSV fields are text; logical types above describe explicit parsing.

Submission columns are exactly `user_id,tv_show_id`; both are strings during parsing.
The second is a five-token list, not a scalar numeric column. Neither is missing.

### Schedule missingness

Missing means an empty, null, or whitespace-only field. Zero is not counted as missing.

{table(['Column', 'Weeks 11-20', 'Weeks 21-30', 'Weeks 31-42'], missing)}

### Actual timestamps and coverage

{table(['File', 'Earliest start', 'Latest start', 'Latest stop/end'], dates)}

Viewing format: `%Y-%m-%d %H:%M:%S`; schedule format: `%d.%m.%Y %H:%M:%S`.
Both omit timezone offsets. Schedule end is derived, not a source column.
The minimum/maximum schedule ends above are unchanged by the DST correction below.
Viewing covers ISO weeks 11-30 of **2020**; there are no final-period viewing labels.
Schedules contain broadcast-day spillover beyond their named periods; never split by
filename alone. Schedule rows outside named periods are counted in `matching.json`.

`viewing_by_week.csv` and `schedule_by_week.csv` give all weekly counts and bounds.
Viewing activity ranges from {n(int(smallest['viewing_rows']))} sessions in week {smallest['iso_week']}
to {n(int(largest['viewing_rows']))} in week {largest['iso_week']}. User/session activity drops
substantially during June and rebounds in July. These files alone cannot establish
whether this is user behavior, sampling, or logging coverage; it creates distribution shift.

## 3. Data quality and identifier integrity

### Viewing durations

All {n(viewing['rows'])} rows have positive reported durations; minimum
{viewing['duration']['minimum']} seconds, median {viewing['duration']['quantiles'][3]:.0f} seconds,
mean {viewing['duration']['mean']:.2f} seconds, p95 {viewing['duration']['quantiles'][6]:.0f} seconds,
p99 {viewing['duration']['quantiles'][7]:.2f} seconds, maximum
{n(viewing['duration']['maximum'])} seconds ({viewing['duration']['maximum']/3600:.2f} hours).
There are {viewing['duration']['over_24_hours']} sessions longer than 24 hours and no sessions
at or below 60 seconds. This is consistent with upstream filtering, but its policy is unknown.
Exact p1/p5/p25/p50/p75/p90/p95/p99/p99.9, standard deviations, and schedule-duration
distributions are in `audit.json`. The schedules contain one zero-duration row, excluded
from matching, and no negative durations.

### Duplicates, overlaps, and inconsistent keys

- Exact duplicate viewing rows (excess copies): **{viewing['duplicate_rows_excess']}**; retain the first source record.
- Exact duplicate rows in each schedule and submission: **0**. Semantic duplicate schedule rows: **0**.
- Same `(user, channel, start)` with different end/duration: **{viewing['conflicting_session_keys']} keys**,
  affecting {v['conflicting_session_keys']['affected_rows']} records and {v['conflicting_session_keys']['affected_users']} users.
- {n(viewing['user_overlapping_records']['rows_overlapping_previous_running_end'])} viewing rows overlap a previous
  running interval for the same user; {n(viewing['user_overlapping_records']['users_with_overlap'])} users are affected.
  After exact deduplication, {n(v['same_user_overlaps']['same_user_channel_overlapping_rows'])} rows still overlap on
  the same user/channel. These may be multiple devices, restarts, or logging duplication;
  the data do not identify which. They remain separate source events, not independent watch-time evidence.
  Downstream weights must union overlapping intervals per user/item before summing time;
  binary user-item labels are already deduplicated for evaluation.
- {a['schedule_integrity']['conflicting_channel_start_keys']} conflicting schedule `(channel,start)` key;
  {a['schedule_integrity']['overlap']['overlapping_rows']} overlapping schedule rows across all periods;
  {cov['ambiguous_rows']} historical viewing sessions overlap simultaneous airings and are quarantined.
  Consecutive programs in a long session are valid multiple interactions, not ambiguity.
- {a['schedule_integrity']['channel_ids_with_multiple_titles']} channel IDs have multiple titles;
  {a['schedule_integrity']['channel_titles_with_multiple_ids']} channel title maps to multiple IDs.
  {n(a['schedule_integrity']['program_ids_with_multiple_titles'])} positive program IDs have multiple titles and
  {a['schedule_integrity']['program_titles_with_multiple_positive_ids']} exact titles map to multiple positive IDs.
  Episode/season/prefix changes can explain some variation; no title-based ID merging was performed.

### Unknown program IDs

{table(['Schedule', 'Rows with ID 0', 'Percentage'], [[file, n(datasets[file]['coverage']['zero_program_rows']), f"{datasets[file]['coverage']['zero_program_percent']:.4f}%"] for file in SCHEDULES])}

Overall: {n(v['zero_id_summary']['zero_id_rows'])} / {n(a['schedule_integrity']['rows'])}
({v['zero_id_summary']['zero_id_percent']:.4f}%). **ID 0 is never a recommendation item.**
Unknown airings contribute only to coverage diagnostics; their titles are not used to
invent IDs. The missing metadata and item IDs make identifiable viewing a biased subset.

## 4. Channel mapping and timezone investigation

### Evidence for channel identity

{evidence['shared_numeric_channel_ids']} of {evidence['viewing_channels']} viewing IDs occur as schedule
channel IDs, covering {n(evidence['rows_with_shared_numeric_channel_id'])} viewing records
({100*evidence['rows_with_shared_numeric_channel_id']/viewing['rows']:.4f}%). Missing IDs are
`{', '.join(map(str, evidence['viewing_only_channel_ids']))}`.
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
{viewing['coverage']['duration_mismatch']} rows, always by +3,600 seconds, starting March 28-29.
All **{v['kyiv_dst_check']['resolved_as_kyiv_elapsed_seconds']}** resolve exactly when interpreted as
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
After correction, **{cov['timezone_adjusted_duration_mismatches']} viewing-duration mismatches remain**.

### Offset sensitivity and channel negative control

Deterministic sample: every 100th source row in training, restricted to March 10-May 17;
{m['timezone_probes'][0]['sample_rows']:,} sessions. Only training sessions enter the overlap
and boundary-alignment measurements. The negative-control permutation uses the shared-ID
inventory from the full audit; it is a diagnostic, not a fitted prediction feature.
Shifts are applied to viewing timestamps relative to schedules.
Stop-boundary alignment means within 120 seconds of a scheduled program start.

{table(['Mapping', 'Shift (hours)', 'Sample', 'Identified matches', 'Stop alignment', 'Before DST', 'After DST'], probe_rows)}

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

{table(['Viewing outcome', 'Records', 'Share of all viewing'], [[label, n(count), f'{100*count/viewing["rows"]:.4f}%'] for label, count in [
('At least one identified program', cov['identified_matched_rows']),
('Schedule overlap, only unknown program IDs', cov['only_unknown_program_rows']),
('No schedule overlap', cov['no_schedule_overlap_rows'])]])}

Any schedule overlap: {n(cov['any_schedule_matched_rows'])} sessions.
Unmatched: {n(cov['missing_channel_rows'])} missing-channel records and
{n(cov['uncovered_time_rows'])} shared-channel records outside available schedule intervals.
Of the latter, {n(v['unmatched_time_breakdown']['before_channel_schedule_begins'])} end before their
channel's schedule begins, {n(v['unmatched_time_breakdown']['after_channel_schedule_ends'])} start after it ends,
and {n(v['unmatched_time_breakdown']['inside_channel_coverage_gaps'])} fall in internal gaps.
Some channels first appear in schedules in April even though viewing begins in March;
see `unmatched_by_channel.csv` for exact attribution. Do not fill these gaps from another channel.

{n(cov['partially_identified_rows'])} matched sessions are only partially attributable to identified
programs. {n(cov['multi_identified_item_rows'])} sessions span multiple distinct identified items.
{cov['excluded_quality_rows']} source records are excluded by quality gates. With any positive
overlap, {n(cov['usable_matched_sessions'])} quality-approved sessions generate
{n(cov['usable_identified_edges'])} session-airing edges.

### Overlap-threshold sensitivity

{table(['Minimum seconds', 'Session-airing edges', 'Sessions', 'Users', 'Items', 'Unique user-item pairs'], thresholds)}

**Prepared default:** {n(chosen['airing_interactions'])} interactions from {n(chosen['sessions'])} sessions,
{n(chosen['users'])} users, {n(chosen['items'])} identified items, and {n(chosen['user_item_pairs'])}
distinct user-item pairs. An interaction is a session-airing edge, not a unique user-item label.
{n(a['users']['valid_interval_users'])} users have positive raw histories;
{a['users']['all_viewing_users']-chosen['users']} have no usable identified history under the default policy.
Of the 1,260 submission users, {refit['submission_users_with_history']} have usable histories;
{m['submission_history']['without_usable_history']} need a cold-user fallback.

The ready-to-use local DuckDB table is `interactions`; split-safe views are
`train_interactions` and `validation_interactions`. The cache also retains `viewing`,
`schedule`, `historical_schedule`, `match_edges`, and `session_quality` for inspection.
All detailed user-level tables stay in the local cache, not in tracked results.

### Independent verification

{m['independent_verification']['passed']} of {m['independent_verification']['examples']} real-session checks passed.
A separate Python `csv.DictReader` scan reads both historical schedule files and computes
UTC overlaps with `datetime`/`zoneinfo`, independently of DuckDB joins and date parsing.
The complete expected set of airings, IDs, source row numbers, and seconds must equal
the SQL result for every example. Cases include missing channels, schedule gaps, zero
IDs, sequential programs, simultaneous airings, midnight, DST, exact boundaries,
long sessions, and the train/validation boundary.

{table(['Viewing data row', 'Case', 'Channel', 'Start (local offset)', 'Stop (local offset)', 'Identified overlaps'], selected_examples)}

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

{table(['Period', 'Inclusive start', 'Exclusive cutoff', 'Raw sessions by start', 'Usable interactions', 'Usable users', 'Items'], [[row['period'], row['start_inclusive'], row['end_exclusive'], n(row['start_assigned_raw_sessions']), n(row['usable_interactions']), n(row['usable_users']), n(row['usable_items'])] for row in m['split']])}

- **Training weeks 11-20:** March 9-May 17. Features only from sessions completed
  by May 18 00:00. {train['cutoff_crossing_sessions']} raw sessions cross this cutoff;
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

Validation has {m['validation_cohorts']['validation_users']} labeled users, including
{m['validation_cohorts']['cold_users']} without usable training history. There are
{m['validation_cohorts']['cold_items']} validation items unseen in training interactions.
{m['validation_cohorts']['users_with_train_but_no_validation_labels']} training-history users have no
identified validation positives. Among submission users, {m['submission_history']['without_train_history']}
lack training history and {m['submission_history']['without_validation_labels']} lack identified validation labels.

Report at least: (1) all 1,260 submission users with AP=0 for empty observable truth;
(2) their {m['validation_cohorts']['submission_users_with_labels']} nonempty-truth users;
(3) warm and cold cohorts separately. The primary convention must match the official
competition rule once obtained; empty observed truth can reflect missing IDs, not no interest.
Use the rest of the users as an additional diagnostic cohort, not an undisclosed replacement.

For nonempty distinct ground-truth set G, use
`AP@5 = sum(P@k * relevant_at_k, k=1..5) / min(5, |G|)`;
MAP@5 is the mean over the declared user cohort. Rank five distinct positive IDs and
count each hit once. There are {n(m['validation_cohorts']['repeated_train_user_item_pairs'])} repeated
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

Completed stage runtimes: audit **{a['runtime_seconds']:.3f}s**, matching including probes
and independent checks **{m['runtime_seconds']:.3f}s**, verification including rehashing
**{v['runtime_seconds']:.3f}s**. These are measurements on this machine, not benchmarks.
"""
    if manifest:
        peak = manifest.get("peak_process_working_set_bytes")
        report += f"""
End-to-end analysis wall time: **{manifest['analysis_wall_seconds']:.3f}s** (report rendering excluded).
Python {manifest['python_version']}; DuckDB {a['duckdb_version']}; {manifest['platform']}.
Peak process working set: **{peak / 1024**2:.1f} MiB**.
Final cache database size: **{manifest['cache_database_bytes'] / 1024**3:.3f} GiB**;
cache: `{manifest['cache_directory']}`. The working set measures resident process memory,
not total committed virtual memory. See `results/run_manifest.json` for timestamps.
""" if peak is not None else "\nPeak process memory was not measured on this platform.\n"
    report += """
Remaining limits: channel identity lacks an authoritative crosswalk; the observed clocks
support a Kyiv-compatible timezone but do not encode it explicitly; source logging and
upstream duration filtering are undocumented; unknown program IDs are highly prevalent;
publication-time schedule availability and the Kaggle truth policy are not supplied.
Algorithmic verification establishes the stated matching procedure, not hidden semantic
ground truth. All these limitations apply to the usable-interaction counts above.
"""
    directory = ROOT / "reports"
    directory.mkdir(exist_ok=True)
    (directory / "data_audit.md").write_text(report, encoding="utf-8")
    print("Wrote reports/data_audit.md", flush=True)


if __name__ == "__main__":
    run()
