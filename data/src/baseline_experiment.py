"""First fixed-configuration experiment: popularity and a separate repeat heuristic.

Run after run_pipeline.py, using the same SWEET_TV_CACHE setting.
All user behavior used for scoring is materialized before the prediction cutoff.
Validation labels are loaded only after validation rankings have been frozen.
The first submission always uses popularity, not a post-hoc model selection rule.
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import time

from common import CACHE, RAW, RESULTS, ROOT, SCHEDULES, connect, literal, one, rows, save_csv, save_json
from metrics import average_precision_at_k, mean_average_precision_at_k
from validate_submission import candidate_ids, read_template, validate

BEGIN = "2020-03-09 00:00:00"
CUT = "2020-05-18 00:00:00"
FINAL = "2020-07-27 00:00:00"
TARGET_END = "2020-10-19 00:00:00"
HALF_LIFE_DAYS = 14


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def reconcile(con):
    totals = one(con, f"""SELECT count(*) AS all_usable_interactions,
        count(*) FILTER (WHERE session_start>=TIMESTAMP '{BEGIN}' AND session_end<=TIMESTAMP '{CUT}') AS training,
        count(*) FILTER (WHERE session_start>=TIMESTAMP '{CUT}' AND session_end<=TIMESTAMP '{FINAL}') AS validation,
        count(*) FILTER (WHERE session_start<TIMESTAMP '{CUT}' AND session_end>TIMESTAMP '{CUT}') AS crossing_interactions,
        count(DISTINCT viewing_id) FILTER (WHERE session_start<TIMESTAMP '{CUT}' AND session_end>TIMESTAMP '{CUT}') AS crossing_sessions_with_interactions
        FROM interactions""")
    totals["raw_cutoff_crossing_sessions"] = con.execute(f"""SELECT count(*) FROM viewing
        WHERE start_time<TIMESTAMP '{CUT}' AND end_time>TIMESTAMP '{CUT}'""").fetchone()[0]
    totals["crossing_sessions_without_default_interactions"] = (
        totals["raw_cutoff_crossing_sessions"] - totals["crossing_sessions_with_interactions"])
    totals["difference"] = totals["all_usable_interactions"] - totals["training"] - totals["validation"]
    require(totals["difference"] == totals["crossing_interactions"], "Unexplained interaction partition difference")
    previous = json.loads((RESULTS / "matching.json").read_text(encoding="utf-8"))
    require([totals["training"], totals["validation"], totals["all_usable_interactions"]] ==
            [record["usable_interactions"] for record in previous["split"]], "Cached counts differ from the completed audit")
    totals["crossing_edge_positions"] = one(con, f"""SELECT
        count(*) FILTER (WHERE overlap_end<=TIMESTAMP '{CUT}') AS entirely_before_cutoff,
        count(*) FILTER (WHERE overlap_start>=TIMESTAMP '{CUT}') AS entirely_after_cutoff,
        count(*) FILTER (WHERE overlap_start<TIMESTAMP '{CUT}' AND overlap_end>TIMESTAMP '{CUT}') AS straddling_cutoff
        FROM interactions WHERE session_start<TIMESTAMP '{CUT}' AND session_end>TIMESTAMP '{CUT}'""")
    require(sum(totals["crossing_edge_positions"].values()) == totals["crossing_interactions"], "Crossing edge positions do not partition")
    totals["crossing_session_reasons"] = rows(con, f"""WITH counts AS (
        SELECT viewing_id,count(*) AS n FROM interactions GROUP BY viewing_id)
        SELECT CASE WHEN n>0 THEN 'has_usable_default_interactions'
          WHEN NOT q.usable THEN 'excluded_by_quality_gate'
          WHEN q.identified_edges=0 THEN 'no_identified_program_overlap'
          ELSE 'all_identified_overlaps_under_60_seconds' END AS reason,count(*) AS sessions
        FROM session_quality q LEFT JOIN counts USING(viewing_id)
        WHERE q.start_time<TIMESTAMP '{CUT}' AND q.end_time>TIMESTAMP '{CUT}' GROUP BY reason ORDER BY reason""")
    checks = {
        "train_validation_shared_viewing_ids": "SELECT count(*) FROM (SELECT viewing_id FROM train_interactions INTERSECT SELECT viewing_id FROM validation_interactions)",
        "training_after_cutoff": f"SELECT count(*) FROM train_interactions WHERE session_start<TIMESTAMP '{BEGIN}' OR session_end>TIMESTAMP '{CUT}' OR overlap_end>TIMESTAMP '{CUT}'",
        "validation_outside_period": f"SELECT count(*) FROM validation_interactions WHERE session_start<TIMESTAMP '{CUT}' OR session_end>TIMESTAMP '{FINAL}'",
        "full_history_after_final_cutoff": f"SELECT count(*) FROM interactions WHERE session_end>TIMESTAMP '{FINAL}' OR session_start<TIMESTAMP '{BEGIN}'",
        "future_schedule_in_historical_interactions": f"SELECT count(*) FROM interactions i JOIN schedule s USING(schedule_id) WHERE s.source={literal(SCHEDULES[2])}",
    }
    totals["leakage_checks"] = {name: con.execute(sql).fetchone()[0] for name, sql in checks.items()}
    require(all(value == 0 for value in totals["leakage_checks"].values()), "Temporal leakage detected")
    return totals


def independently_count_viewers(con, eligible):
    """Streaming Python sets verify SQL COUNT(DISTINCT user_id), one item at a time."""
    cursor = con.execute("SELECT tv_show_id,user_id FROM experiment_history ORDER BY tv_show_id")
    counts = {}
    item = None
    viewers = set()
    while batch := cursor.fetchmany(10000):
        for current, user in batch:
            if current != item:
                if item in eligible:
                    counts[item] = len(viewers)
                item, viewers = current, set()
            viewers.add(user)
    if item in eligible:
        counts[item] = len(viewers)
    return {candidate: counts.get(candidate, 0) for candidate in eligible}


def fit_and_predict(con, users, cutoff, target_end, files, period, personalized=True):
    tick = time.perf_counter()
    # This is the only source of behavioral features, shared by both methods.
    con.execute(f"""CREATE OR REPLACE TEMP TABLE experiment_history AS SELECT * FROM interactions
        WHERE session_start>=TIMESTAMP '{BEGIN}' AND session_end<=TIMESTAMP {literal(cutoff)}""")
    bounds = one(con, """SELECT count(*) AS interaction_rows,count(DISTINCT user_id) AS users,
        min(session_start) AS first_session_start,max(session_end) AS last_session_end,
        max(overlap_end) AS last_overlap_end FROM experiment_history""")
    require(bounds["last_session_end"] <= datetime.fromisoformat(cutoff), "Future history entered fitting")
    eligible = candidate_ids(con, cutoff, target_end, files)
    require(len(eligible) >= 5, "Fewer than five eligible target-period items")
    con.execute("CREATE OR REPLACE TEMP TABLE experiment_candidates(tv_show_id BIGINT PRIMARY KEY)")
    con.executemany("INSERT INTO experiment_candidates VALUES (?)", [(item,) for item in sorted(eligible)])
    popularity = rows(con, """WITH p AS (SELECT tv_show_id,count(DISTINCT user_id) AS viewers
        FROM experiment_history GROUP BY tv_show_id)
        SELECT c.tv_show_id,coalesce(p.viewers,0) AS distinct_viewers FROM experiment_candidates c
        LEFT JOIN p USING(tv_show_id) ORDER BY distinct_viewers DESC,c.tv_show_id ASC""")
    scores = {row["tv_show_id"]: row["distinct_viewers"] for row in popularity}
    verification_start = time.perf_counter()
    require(scores == independently_count_viewers(con, eligible), "Python and SQL viewer counts differ")
    independent_seconds = time.perf_counter() - verification_start
    ranking = [row["tv_show_id"] for row in popularity]
    require(ranking == sorted(eligible, key=lambda item: (-scores[item], item)), "Incorrect popularity tie-breaking")
    save_csv(f"popularity_{period}.csv", [{"rank": rank, **row} for rank, row in enumerate(popularity, 1)])
    history_users = {row[0] for row in con.execute("SELECT DISTINCT user_id FROM experiment_history").fetchall()}
    predictions = {"popularity": {user: ranking[:5] for user in users}}
    info = {"period": period, "prediction_cutoff_local": cutoff, "target_end_exclusive_local": target_end,
        "history": bounds, "candidate_count": len(eligible), "candidates_with_positive_popularity": sum(score > 0 for score in scores.values()),
        "forecast_users": len(users), "users_without_usable_history": sum(user not in history_users for user in users),
        "global_top5": ranking[:5], "popularity_ties": "distinct viewers descending, numeric tv_show_id ascending",
        "independent_distinct_viewer_count_check": True, "independent_count_seconds": round(independent_seconds, 6),
        "popularity_fit_predict_seconds_including_verification": round(time.perf_counter() - tick, 6)}
    if personalized:
        personal_start = time.perf_counter()
        con.execute("CREATE OR REPLACE TEMP TABLE experiment_popularity(tv_show_id BIGINT,viewers BIGINT)")
        con.executemany("INSERT INTO experiment_popularity VALUES (?,?)", list(scores.items()))
        # One vote per user/item/local watch date avoids rewarding fragmented sessions.
        con.execute(f"""CREATE OR REPLACE TEMP TABLE personal_scores AS WITH daily AS (
            SELECT DISTINCT h.user_id,h.tv_show_id,cast(h.overlap_start AS DATE) AS watch_day
            FROM experiment_history h JOIN experiment_users u USING(user_id)
            JOIN experiment_candidates c USING(tv_show_id)), scored AS (
            SELECT user_id,tv_show_id,round(sum(pow(2.0,
                -date_diff('day',watch_day,DATE {literal(cutoff[:10])})/{float(HALF_LIFE_DAYS)})),12) AS score
            FROM daily GROUP BY user_id,tv_show_id)
            SELECT s.*,p.viewers,row_number() OVER (PARTITION BY user_id ORDER BY score DESC,viewers DESC,tv_show_id ASC) AS rank
            FROM scored s JOIN experiment_popularity p USING(tv_show_id)""")
        selected = {user: [] for user in users}
        for user, item in con.execute("SELECT user_id,tv_show_id FROM personal_scores WHERE rank<=5 ORDER BY user_id,rank").fetchall():
            selected[user].append(item)
        personal_counts = {user: len(items) for user, items in selected.items()}
        for user, items in selected.items():
            for item in ranking:
                if len(items) == 5:
                    break
                if item not in items:
                    items.append(item)
        predictions["repeat_recency_heuristic"] = selected
        info["personalized"] = {"half_life_days": HALF_LIFE_DAYS,
            "formula": "sum over distinct local overlap-start dates d: 2^(-(cutoff_date-d).days/14), rounded to 12 decimals",
            "tie_breaking": "personal score descending, global distinct viewers descending, numeric item ID ascending",
            "fully_popularity_fallback_users": sum(count == 0 for count in personal_counts.values()),
            "users_needing_any_popularity_padding": sum(count < 5 for count in personal_counts.values()),
            "additional_fit_predict_seconds": round(time.perf_counter() - personal_start, 6)}
        require(all(selected[user] == ranking[:5] for user in users if user not in history_users), "Cold-user fallback inconsistent")
    for method, method_predictions in predictions.items():
        require(list(method_predictions) == users, f"Wrong user cohort for {method}")
        require(all(len(items) == 5 and len(set(items)) == 5 and set(items) <= eligible for items in method_predictions.values()),
                f"Invalid ranked list for {method}")
    info["total_fit_predict_seconds"] = round(time.perf_counter() - tick, 6)
    return predictions, info, eligible, history_users


def evaluate(con, users, predictions, eligible, history_users):
    tick = time.perf_counter()
    # Labels become visible only after both prediction dictionaries are finalized.
    con.execute("""CREATE OR REPLACE TEMP TABLE experiment_truth AS SELECT DISTINCT user_id,tv_show_id
        FROM validation_interactions JOIN experiment_users USING(user_id)""")
    truth = {user: set() for user in users}
    for user, item in con.execute("SELECT user_id,tv_show_id FROM experiment_truth").fetchall():
        truth[user].add(item)
    require(all(items <= eligible for items in truth.values()), "Validation positives outside target candidate pool")
    cohorts = {
        "all_template_users": users,
        "nonempty_truth": [user for user in users if truth[user]],
        "warm_history": [user for user in users if user in history_users],
        "cold_history": [user for user in users if user not in history_users],
    }
    records, per_user = [], []
    for method, recommendations in predictions.items():
        for user_index, user in enumerate(users, 1):
            per_user.append({"template_row": user_index, "method": method, "has_training_history": user in history_users,
                "relevant_items": len(truth[user]), "hits_at_5": len(truth[user] & set(recommendations[user])),
                "ap_at_5": average_precision_at_k(truth[user], recommendations[user]),
                "predicted_ids": " ".join(map(str, recommendations[user]))})
        for name, cohort in cohorts.items():
            records.append({"method": method, "cohort": name, "evaluated_users": len(cohort),
                "excluded_template_users": len(users) - len(cohort),
                "empty_truth_users": sum(not truth[user] for user in cohort), "eligible_candidates": len(eligible),
                "relevant_user_item_pairs": sum(len(truth[user]) for user in cohort),
                "distinct_relevant_items": len(set().union(*(truth[user] for user in cohort))),
                "map_at_5": mean_average_precision_at_k(truth, recommendations, cohort) if cohort else None})
    # Independent SQL AP@5 calculation validates every evaluated user, not just the mean.
    con.execute("CREATE OR REPLACE TEMP TABLE experiment_predictions(method VARCHAR,user_id VARCHAR,rank INTEGER,tv_show_id BIGINT)")
    con.executemany("INSERT INTO experiment_predictions VALUES (?,?,?,?)", [
        (method, user, rank, item) for method, recommendation in predictions.items()
        for user, items in recommendation.items() for rank, item in enumerate(items, 1)])
    independent = rows(con, """WITH hit_rows AS (
        SELECT p.*,CASE WHEN t.tv_show_id IS NULL THEN 0 ELSE 1 END AS hit
        FROM experiment_predictions p LEFT JOIN experiment_truth t USING(user_id,tv_show_id)),
        cumulative AS (SELECT *,sum(hit) OVER (PARTITION BY method,user_id ORDER BY rank) AS cumulative_hits FROM hit_rows),
        denominators AS (SELECT user_id,count(*) AS n FROM experiment_truth GROUP BY user_id)
        SELECT c.method,c.user_id,CASE WHEN coalesce(max(d.n),0)=0 THEN 0.0
          ELSE sum(c.hit*c.cumulative_hits::DOUBLE/c.rank)/least(5,max(d.n)) END AS ap
        FROM cumulative c LEFT JOIN denominators d USING(user_id) GROUP BY c.method,c.user_id""")
    max_difference = 0.0
    for row in independent:
        actual = average_precision_at_k(truth[row["user_id"]], predictions[row["method"]][row["user_id"]])
        max_difference = max(max_difference, abs(actual - row["ap"]))
    require(len(independent) == len(users) * len(predictions), "Independent evaluator skipped users")
    require(max_difference < 1e-12, "Python and SQL AP@5 differ")
    manual_actual = average_precision_at_k({11, 22, 33, 44}, [11, 99, 22, 88, 33])
    require(math.isclose(manual_actual, 17 / 30, abs_tol=1e-12), "Manual AP@5 example failed")
    save_csv("baseline_evaluation.csv", records)
    save_csv("baseline_per_user_evaluation.csv", per_user)
    return {"scores": records, "independent_python_sql_ap_rows_checked": len(independent),
        "maximum_ap_difference": max_difference, "manual_example": {
            "relevant": [11, 22, 33, 44], "predicted": [11, 99, 22, 88, 33],
            "hit_ranks": [1, 3, 5], "precision_at_hit_ranks": [1.0, 2 / 3, 3 / 5],
            "denominator": 4, "calculation": "(1 + 2/3 + 3/5) / 4 = 17/30", "ap_at_5": manual_actual},
        "relevance_distribution": one(con, """SELECT min(n) AS min_items,max(n) AS max_items,avg(n) AS mean_items,
            quantile_cont(n,[0.25,0.5,0.75,0.9]) AS quantiles FROM (
                SELECT u.user_id,count(t.tv_show_id) n FROM experiment_users u LEFT JOIN experiment_truth t USING(user_id) GROUP BY u.user_id)"""),
        "runtime_seconds": round(time.perf_counter() - tick, 6)}


def raw_integrity():
    audit = json.loads((RESULTS / "audit.json").read_text(encoding="utf-8"))
    checks = {}
    for filename, metadata in audit["datasets"].items():
        with (RAW / filename).open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        require(actual == metadata["sha256"], f"Raw input differs from completed audit: {filename}")
        checks[filename] = actual
    return checks


def write_report(result):
    r = result["reconciliation"]
    scores = result["evaluation"]["scores"]
    primary = [row for row in scores if row["cohort"] == "all_template_users"]
    fit = result["validation_fit"]
    final = result["final_fit"]
    checked = result["submission_validation"]
    table = "\n".join(f"| {row['method']} | {row['cohort']} | {row['evaluated_users']} | {row['excluded_template_users']} | {row['map_at_5']:.8f} |" for row in scores if row["map_at_5"] is not None)
    report = f"""# First recommendation experiment

## Resolved audit discrepancy

**{r['all_usable_interactions']:,} = {r['training']:,} training + {r['validation']:,} validation + {r['crossing_interactions']:,} cutoff-crossing interactions.**

The {r['raw_cutoff_crossing_sessions']:,} raw sessions starting before May 18 00:00 and ending after it contain
{r['crossing_sessions_with_interactions']:,} sessions with default usable interactions and
{r['crossing_sessions_without_default_interactions']:,} with none. One session can overlap several airings;
session counts and session-airing interaction counts are different units.
Of the {r['crossing_interactions']:,} edges, {r['crossing_edge_positions']['entirely_before_cutoff']:,} lie entirely before the cutoff,
{r['crossing_edge_positions']['entirely_after_cutoff']:,} entirely after it, and
{r['crossing_edge_positions']['straddling_cutoff']:,} straddle it.
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
  **{fit['candidate_count']:,}** eligible validation items; **0 users are excluded from the primary metric**.
- Candidates are positive IDs from positive-duration scheduled airings intersecting
  `[May 18, July 27)`, including the previous schedule file's boundary spillover.
  No candidate is selected using future viewing counts; never use item 0.
- Labels are distinct user/item pairs from the audit's >=60-second, quality-approved
  validation interactions. Previously watched items are allowed.
- The primary cohort has **{primary[0]['relevant_user_item_pairs']:,} relevant user-item pairs**,
  **{primary[0]['distinct_relevant_items']:,} distinct relevant programs**, and
  **{primary[0]['empty_truth_users']} empty-truth users**. All positives are in the candidate pool.
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

There are {fit['users_without_usable_history']} template users without usable training history.
The heuristic uses a complete popularity fallback for {fit['personalized']['fully_popularity_fallback_users']} users
with no eligible personal candidates and popularity padding for
{fit['personalized']['users_needing_any_popularity_padding']} users with fewer than five personal candidates.
Cold-user predictions are checked to equal the popularity baseline exactly.

## Observed MAP@5

| Method | Cohort | Evaluated | Excluded template users | MAP@5 |
| --- | --- | --- | --- | --- |
{table}

`AP@5 = sum(precision@rank for first-time relevant hits) / min(5, number of distinct relevant items)`.
Empty ground truth receives AP=0. User means are unweighted. Duplicate predictions never
earn additional hits and are rejected in actual ranked outputs.

Independent manual example: relevance `{{11,22,33,44}}`, ranking `[11,99,22,88,33]`.
Hits at ranks 1,3,5 give `(1 + 2/3 + 3/5) / 4 = 17/30 = 0.5666666667`.
Python and an independent SQL cumulative-hit implementation agreed on all
**{result['evaluation']['independent_python_sql_ap_rows_checked']:,}** method/user AP values
(maximum difference {result['evaluation']['maximum_ap_difference']:.3g}).
`test_metrics.py` also checks capped denominators, empty labels, duplicate handling,
rank truncation, cohort inclusion, and missing predictions.

## First Kaggle submission

**`results/submission_baseline.csv` uses popularity**, preselected before comparing scores.
Refit on all {final['history']['interaction_rows']:,} usable interactions completed by July 27;
no final-period viewing is available or accessed. The target is `[July 27, October 19)`.
The final schedule candidate pool contains **{final['candidate_count']:,}** positive items,
including previous-file boundary spillover. All 1,260 users receive the final popularity
top five: `{' '.join(map(str, final['global_top5']))}`.

Programmatic validation passed:
- Exact original template order, **{checked['rows']} rows / {checked['unique_users']} unique users**.
- Exact header `user_id,tv_show_id`, LF line endings, no BOM/index/missing fields.
- **{checked['prediction_slots']:,} prediction slots**, five distinct positive eligible IDs per user.
- Invalid formatting rows: {checked['invalid_format_rows']}; ineligible-item rows: {checked['ineligible_item_rows']};
  repeated-item rows: {checked['repeated_item_rows']}.
- CSV SHA-256: `{checked['sha256']}`.
- All five raw CSV checksums match the completed audit.

This establishes local structural and candidate validity. The file has **not** been uploaded
to Kaggle; no leaderboard or private-score result is claimed.

## Runtime and reproduction

- Validation popularity fit/predict including independent count verification: {fit['popularity_fit_predict_seconds_including_verification']:.3f}s.
- Personalized heuristic additional fit/predict: {fit['personalized']['additional_fit_predict_seconds']:.3f}s.
- Evaluation plus independent SQL checks: {result['evaluation']['runtime_seconds']:.3f}s.
- Final popularity refit/predict including verification: {final['total_fit_predict_seconds']:.3f}s.
- Total measured experiment including reconciliation, input hashing, and submission checks:
  **{result['runtime_seconds']:.3f}s**, excluding report rendering and the prerequisite audit.

```powershell
$env:SWEET_TV_CACHE = "D:\\sweet-tv-midterm-cache"
.\\.venv\\Scripts\\python.exe data/src/test_metrics.py
.\\.venv\\Scripts\\python.exe data/src/baseline_experiment.py
.\\.venv\\Scripts\\python.exe data/src/validate_submission.py
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
"""
    (ROOT / "reports" / "baseline_experiment.md").write_text(report, encoding="utf-8")


def run():
    started = time.perf_counter()
    result = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"half_life_days": HALF_LIFE_DAYS, "minimum_overlap_seconds": 60,
                   "submission_method": "popularity", "primary_cohort": "all 1260 template users", "empty_truth_ap": 0.0}}
    result["source_sha256"] = raw_integrity()
    con = connect()
    users = read_template()
    con.execute("CREATE TEMP TABLE experiment_users(user_id VARCHAR PRIMARY KEY,template_row INTEGER)")
    con.executemany("INSERT INTO experiment_users VALUES (?,?)", [(user, index) for index, user in enumerate(users, 1)])
    result["reconciliation"] = reconcile(con)
    save_json("baseline_reconciliation.json", result["reconciliation"])
    print(json.dumps(result["reconciliation"], indent=2), flush=True)
    validation_predictions, result["validation_fit"], eligible, history_users = fit_and_predict(
        con, users, CUT, FINAL, SCHEDULES[:2], "validation")
    result["evaluation"] = evaluate(con, users, validation_predictions, eligible, history_users)
    print(json.dumps(result["evaluation"]["scores"], indent=2), flush=True)
    final_predictions, result["final_fit"], final_eligible, _ = fit_and_predict(
        con, users, FINAL, TARGET_END, SCHEDULES[1:], "final", personalized=False)
    destination = RESULTS / "submission_baseline.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["user_id", "tv_show_id"])
        writer.writerows((user, " ".join(map(str, final_predictions["popularity"][user]))) for user in users)
    result["submission_validation"] = validate(destination, users, final_eligible)
    result["all_five_raw_hashes_unchanged_after_experiment"] = raw_integrity() == result["source_sha256"]
    require(result["all_five_raw_hashes_unchanged_after_experiment"], "Raw inputs changed during experiment")
    con.close()
    result["runtime_seconds"] = round(time.perf_counter() - started, 6)
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json("baseline_experiment.json", result)
    save_json("submission_baseline_validation.json", result["submission_validation"])
    write_report(result)
    print(f"Completed experiment in {result['runtime_seconds']:.3f}s; submission validation passed", flush=True)


if __name__ == "__main__":
    run()
