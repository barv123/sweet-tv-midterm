"""One fixed weeks-15-16 inner holdout plus compact validation-user case study.

Reuses the audited DuckDB cache and saved full-validation predictions. Does not rerun
the advanced model experiment, alter the established weeks-11-20 / 21-30 protocol,
or train/tune a new hybrid setting.
"""

import hashlib
import json
import time

import numpy as np
from scipy import sparse

from advanced_common import load_json
from advanced_experiment import collect_rankings, input_matrices
from baseline_experiment import BEGIN, raw_integrity, require
from common import RAW, RESULTS, SCHEDULES, TIMEZONE, connect, literal, one, rows, save_csv, save_json
from metrics import average_precision_at_k
from recommendation_methods import mine_association_rules, reciprocal_rank_fusion, rule_score_blocks
from validate_submission import candidate_ids, read_template, validate

INNER_START = "2020-04-06 00:00:00"  # ISO week 15
INNER_END = "2020-04-20 00:00:00"    # ISO week 17; exclusive
RULE_CONFIG = {"min_pair_count": 25, "min_lift": 1.0, "min_confidence": 0.1,
               "min_item_viewers": 20, "max_rules": 100}
METHODS = ("popularity", "repeat_recency", "hybrid_rrf60")


def build_case_study(con):
    users = read_template()
    selected_rows = (2, 44)
    selected_users = {row: users[row-1] for row in selected_rows}
    needed = {"popularity": "popularity", "content_metadata_title": "content_metadata_title",
              "association_count25_lift1": "association_count25_lift1", "hybrid_rrf60": "hybrid_rrf60"}
    stored = {}
    with (RESULTS / "advanced_per_user_evaluation.csv").open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            row = int(record["template_row"])
            if row in selected_rows and record["method"] in needed:
                stored[(row, record["method"])] = record
    require(len(stored) == len(selected_rows)*len(needed), "Saved full-validation ranking evidence is incomplete")
    # Resolve template row numbers from the exact original CSV order.
    con.execute("CREATE OR REPLACE TEMP TABLE case_template(user_id VARCHAR,template_row INTEGER)")
    con.executemany("INSERT INTO case_template VALUES (?,?)", [(u, i) for i, u in enumerate(users, 1)])
    truth = {row: set() for row in selected_rows}
    for row, item in con.execute("""SELECT c.template_row,v.tv_show_id FROM validation_interactions v
        JOIN case_template c USING(user_id) WHERE c.template_row IN (2,44)""").fetchall():
        truth[row].add(item)
    validation_candidates = candidate_ids(con, "2020-05-18", "2020-07-27", SCHEDULES[:2])
    require(all(items <= validation_candidates for items in truth.values()), "Case-study truth outside validation candidates")
    metadata = rows(con, """WITH counts AS (SELECT tv_show_id,title,category,count(*) AS n
        FROM advanced_metadata WHERE tv_show_id>0 GROUP BY tv_show_id,title,category), ranked AS (
        SELECT *,row_number() OVER (PARTITION BY tv_show_id ORDER BY n DESC,title ASC,category ASC) AS choice
        FROM counts)
        SELECT tv_show_id,title,category FROM ranked WHERE choice=1""")
    metadata = {record["tv_show_id"]: record for record in metadata}
    evidence = []
    for template_row in selected_rows:
        relevant = truth[template_row]
        for method in needed:
            record = stored[(template_row, method)]
            predicted = [int(item) for item in record["predicted_ids"].split()]
            require(len(predicted) == len(set(predicted)) == 5, "Saved example ranking is not exactly five unique items")
            require(set(predicted) <= validation_candidates, "Saved example includes an ineligible item")
            ap = average_precision_at_k(relevant, predicted)
            require(abs(ap-float(record["ap_at_5"])) < 1e-12, "Case-study AP does not match saved validation result")
            for rank, item in enumerate(predicted, 1):
                meta = metadata.get(item, {})
                evidence.append({"template_row": template_row, "relevant_items_for_user": len(relevant),
                    "method": method, "user_ap_at_5": ap, "rank": rank, "tv_show_id": item,
                    "representative_schedule_title": meta.get("title", ""),
                    "category": meta.get("category", ""), "heldout_relevant": item in relevant,
                    "title_rule": "most frequent schedule title; lexical tie-break"})
    save_csv("validation_user_top5_evidence.csv", evidence)
    return {"template_rows": list(selected_rows), "relevant_items": {str(row): len(truth[row]) for row in selected_rows},
        "methods": list(needed), "evidence_rows": len(evidence), "evidence_file": "results/validation_user_top5_evidence.csv",
        "all_predictions_eligible_and_unique": True, "AP_recomputed_against_saved_labels": True}


def run_inner_holdout(con):
    started = time.perf_counter()
    users = read_template()
    con.execute("CREATE OR REPLACE TEMP TABLE adv_users(user_id VARCHAR,template_row INTEGER)")
    con.executemany("INSERT INTO adv_users VALUES (?,?)", [(u, i) for i, u in enumerate(users, 1)])
    con.execute(f"""CREATE OR REPLACE TEMP TABLE inner_train AS SELECT * FROM interactions
        WHERE session_start>=TIMESTAMP '{BEGIN}' AND session_end<=TIMESTAMP '{INNER_START}'""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE inner_validation AS SELECT * FROM interactions
        WHERE session_start>=TIMESTAMP '{INNER_START}' AND session_end<=TIMESTAMP '{INNER_END}'""")
    candidate_set = candidate_ids(con, INNER_START, INNER_END, SCHEDULES[:1])
    require(len(candidate_set) >= 5 and all(item > 0 for item in candidate_set), "Inner holdout candidate universe invalid")
    candidate_ids_ordered = np.array(sorted(candidate_set), dtype=np.int64)
    con.execute("CREATE OR REPLACE TEMP TABLE inner_candidates(tv_show_id BIGINT)")
    con.executemany("INSERT INTO inner_candidates VALUES (?)", [(int(item),) for item in candidate_ids_ordered])

    popularity_start = time.perf_counter()
    popularity_rows = rows(con, """SELECT c.tv_show_id,coalesce(p.viewers,0) AS viewers FROM inner_candidates c
        LEFT JOIN (SELECT tv_show_id,count(DISTINCT user_id) AS viewers FROM inner_train GROUP BY tv_show_id) p USING(tv_show_id)
        ORDER BY viewers DESC,c.tv_show_id""")
    popularity_lookup = {row["tv_show_id"]: row["viewers"] for row in popularity_rows}
    global_ranking = [row["tv_show_id"] for row in popularity_rows]
    popularity_scores = np.array([popularity_lookup[int(item)] for item in candidate_ids_ordered], dtype=np.int64)
    popularity_seconds = time.perf_counter()-popularity_start

    matrix_start = time.perf_counter()
    binary, personal, item_ids, candidate_columns, train_users = input_matrices(
        con, users, candidate_ids_ordered, history_table="inner_train", recency_cutoff=INNER_START)
    matrix_preparation_seconds = time.perf_counter()-matrix_start
    cold_users = set(users) - train_users
    ranked = {"popularity": {user: global_ranking[:100] for user in users}}

    repeat_start = time.perf_counter()
    repeat_rankings, repeat_info = collect_rankings(
        ((0, personal[:, candidate_columns].toarray()),), users, candidate_ids_ordered,
        popularity_scores, cold_users, global_ranking)
    ranked["repeat_recency"] = repeat_rankings
    repeat_seconds = time.perf_counter()-repeat_start

    association_start = time.perf_counter()
    rules, rule_info = mine_association_rules(binary, item_ids, candidate_columns, **RULE_CONFIG)
    rule_rankings, rule_coverage = collect_rankings(
        rule_score_blocks(personal, rules, candidate_columns), users, candidate_ids_ordered,
        popularity_scores, cold_users, global_ranking)
    ranked["association_rules"] = rule_rankings
    association_seconds = time.perf_counter()-association_start
    rule_evidence_mask = np.asarray(rules[:, candidate_columns].getnnz(axis=0)).ravel() > 0
    repeat_evidence_mask = np.asarray(personal[:, candidate_columns].getnnz(axis=0)).ravel() > 0
    popularity_evidence_mask = popularity_scores > 0
    hybrid_evidence_count = int(np.sum(rule_evidence_mask | repeat_evidence_mask | popularity_evidence_mask))
    hybrid_start = time.perf_counter()
    ranked["hybrid_rrf60"] = {user: reciprocal_rank_fusion(
        [rule_rankings[user][:100], repeat_rankings[user][:100], global_ranking[:100]],
        candidate_ids_ordered, popularity_scores, constant=60)[:100] for user in users}
    hybrid_seconds = time.perf_counter()-hybrid_start

    # The AP targets are consumed only after train-only model forecasts are frozen.
    con.execute("""CREATE OR REPLACE TEMP TABLE inner_truth AS SELECT DISTINCT v.user_id,v.tv_show_id
        FROM inner_validation v JOIN adv_users USING(user_id)""")
    truth = {user: set() for user in users}
    for user, item in con.execute("SELECT user_id,tv_show_id FROM inner_truth").fetchall():
        truth[user].add(item)
    require(all(items <= candidate_set for items in truth.values()), "Inner validation truth outside schedule candidates")
    prefit_scores = {}
    per_user = []
    for method in METHODS:
        rows_for_users = []
        aps = []
        for index, user in enumerate(users, 1):
            items = ranked[method][user][:5]
            require(len(items) == len(set(items)) == 5 and set(items) <= candidate_set,
                    f"Inner holdout invalid top five: {method}")
            if user in cold_users:
                require(items == global_ranking[:5], f"Inner holdout cold fallback mismatch: {method}")
            ap = average_precision_at_k(truth[user], items)
            aps.append(ap)
            rows_for_users.append({"method": method, "template_row": index,
                "has_training_history": user not in cold_users, "relevant_items": len(truth[user]),
                "hits_at_5": len(set(items)&truth[user]), "ap_at_5": ap,
                "predicted_ids": " ".join(map(str, items))})
        prefit_scores[method] = np.array(aps)
        if method == "popularity":
            per_user = rows_for_users
        else:
            per_user.extend(rows_for_users)
    save_csv("inner_holdout_per_user.csv", per_user)
    # Independently compute AP from the serialized predictions using SQL window functions.
    con.execute("CREATE OR REPLACE TEMP TABLE inner_predictions(method VARCHAR,user_id VARCHAR,rank INTEGER,tv_show_id BIGINT)")
    con.executemany("INSERT INTO inner_predictions VALUES (?,?,?,?)", [
        (method, user, rank, item) for method in METHODS for user in users
        for rank, item in enumerate(ranked[method][user][:5], 1)])
    sql_ap = rows(con, """WITH h AS (SELECT p.*,CASE WHEN t.tv_show_id IS NULL THEN 0 ELSE 1 END hit
        FROM inner_predictions p LEFT JOIN inner_truth t USING(user_id,tv_show_id)), c AS (
        SELECT *,sum(hit) OVER (PARTITION BY method,user_id ORDER BY rank) cumulative_hits FROM h),
        d AS (SELECT user_id,count(*) n FROM inner_truth GROUP BY user_id)
        SELECT c.method,c.user_id,CASE WHEN coalesce(max(d.n),0)=0 THEN 0.0
          ELSE sum(c.hit*c.cumulative_hits::DOUBLE/c.rank)/least(5,max(d.n)) END ap
        FROM c LEFT JOIN d USING(user_id) GROUP BY c.method,c.user_id""")
    max_ap_error = max(abs(row["ap"]-average_precision_at_k(truth[row["user_id"]],
        ranked[row["method"]][row["user_id"]][:5])) for row in sql_ap)
    require(len(sql_ap) == len(METHODS)*len(users) and max_ap_error < 1e-12,
            "Independent SQL MAP calculation failed")

    comparison = []
    eval_rows = []
    for method in METHODS:
        aps = prefit_scores[method]
        recommendations = {user: ranked[method][user][:5] for user in users}
        unique_recommended = {item for items in recommendations.values() for item in items}
        comparison.append({"method": method, "map_at_5": float(aps.mean()), "evaluated_users": len(users),
            "excluded_users": 0, "empty_truth_users": sum(not truth[u] for u in users),
            "candidate_count": len(candidate_set), "distinct_relevant_user_item_pairs": sum(map(len, truth.values())),
            "distinct_relevant_items": len(set().union(*truth.values())),
            "users_without_training_history": len(cold_users),
            "cold_user_map_at_5": float(np.mean([aps[i] for i,u in enumerate(users) if u in cold_users])) if cold_users else None,
            "users_with_zero_hits": int(sum(not (set(recommendations[u]) & truth[u]) for u in users)),
            "unique_recommended_items": len(unique_recommended),
            "candidates_with_training_evidence": (sum(v > 0 for v in popularity_lookup.values()) if method == "popularity"
                else hybrid_evidence_count if method == "hybrid_rrf60"
                else repeat_info["candidates_with_positive_evidence"]),
            "runtime_seconds": {"popularity": popularity_seconds,
                "repeat_recency": repeat_seconds, "hybrid_rrf60": popularity_seconds+repeat_seconds+association_seconds+hybrid_seconds}[method]})
        eval_rows.append({"method": method, "map_at_5": float(aps.mean()), "users": len(users)})
    # Method runtimes are separately timed below; each ranking was computed once above.
    result = {"fixed_protocol": {"train_start_inclusive": BEGIN, "train_end_exclusive": INNER_START,
        "validation_start_inclusive": INNER_START, "validation_end_exclusive": INNER_END,
        "training_iso_weeks": [11,12,13,14], "validation_iso_weeks": [15,16],
        "schedule_file_for_candidates": SCHEDULES[0], "minimum_overlap_seconds": 60,
        "allow_previously_seen_items": True, "population_rule_parameters": RULE_CONFIG,
        "hybrid_rrf_constant": 60, "hybrid_components": ["association_rules", "repeat_recency", "popularity"],
        "no_new_holdout_tuning": True},
        "temporal_checks": {"training_rows_after_cutoff": con.execute(f"SELECT count(*) FROM inner_train WHERE session_end>TIMESTAMP '{INNER_START}' OR overlap_end>TIMESTAMP '{INNER_START}'").fetchone()[0],
            "validation_rows_outside_window": con.execute(f"SELECT count(*) FROM inner_validation WHERE session_start<TIMESTAMP '{INNER_START}' OR session_end>TIMESTAMP '{INNER_END}'").fetchone()[0],
            "shared_viewing_ids": con.execute("SELECT count(*) FROM (SELECT viewing_id FROM inner_train INTERSECT SELECT viewing_id FROM inner_validation)").fetchone()[0],
            "training_interactions_using_later_schedule_file": con.execute(f"SELECT count(*) FROM inner_train i JOIN schedule s USING(schedule_id) WHERE s.source<>{literal(SCHEDULES[0])}").fetchone()[0],
            "labels_outside_candidate_set": con.execute("SELECT count(*) FROM inner_truth WHERE tv_show_id NOT IN (SELECT tv_show_id FROM inner_candidates)").fetchone()[0]},
        "training_data": one(con, "SELECT count(*) interactions,count(DISTINCT viewing_id) sessions,count(DISTINCT user_id) users,max(session_end) last_session_end FROM inner_train"),
        "validation_data": one(con, "SELECT count(*) interactions,count(DISTINCT viewing_id) sessions,count(DISTINCT user_id) users FROM inner_validation"),
        "candidate_schedule": one(con, f"""SELECT count(*) FILTER (WHERE tv_show_id>0 AND duration>0 AND start_time<TIMESTAMP '{INNER_END}'
          AND timezone('{TIMEZONE}',timezone('{TIMEZONE}',start_time)+duration*INTERVAL '1 second')>TIMESTAMP '{INNER_START}') AS intersecting_positive_airings,
          count(DISTINCT tv_show_id) FILTER (WHERE tv_show_id>0 AND duration>0 AND start_time<TIMESTAMP '{INNER_END}'
          AND timezone('{TIMEZONE}',timezone('{TIMEZONE}',start_time)+duration*INTERVAL '1 second')>TIMESTAMP '{INNER_START}') AS eligible_programs
          FROM schedule WHERE source={literal(SCHEDULES[0])}"""),
        "users": len(users), "cold_users": len(cold_users), "relevant_distribution": one(con, """SELECT min(n) min_relevant,max(n) max_relevant,avg(n) mean_relevant,
          quantile_cont(n,[0.25,0.5,0.75,0.9]) quantiles,count(*) FILTER (WHERE n=0) empty_users,
          sum(n) relevant_pairs FROM (SELECT u.user_id,count(t.tv_show_id) n FROM adv_users u LEFT JOIN inner_truth t USING(user_id) GROUP BY u.user_id)"""),
         "already_seen_validation_pairs": con.execute("""SELECT count(*) FROM inner_truth t JOIN
          (SELECT DISTINCT user_id,tv_show_id FROM inner_train) h USING(user_id,tv_show_id)""").fetchone()[0],
         "association_rule_fit": {key: value for key,value in rule_info.items() if key != "sample_rules"},
         "comparison": comparison, "independent_sql_ap": {"values_checked": len(sql_ap), "max_absolute_error": max_ap_error},
        "candidate_ids_are_positive": min(candidate_set) > 0, "cold_users_use_global_popularity": True}
    require(all(value == 0 for value in result["temporal_checks"].values()), "Inner holdout leakage or candidate mismatch")
    require(result["independent_sql_ap"]["max_absolute_error"] < 1e-12, "MAP verification mismatch")
    result["shared_matrix_preparation_seconds"] = matrix_preparation_seconds
    popular_aps = prefit_scores["popularity"]
    hybrid_aps = prefit_scores["hybrid_rrf60"]
    paired_delta = hybrid_aps-popular_aps
    result["hybrid_minus_popularity"] = {"absolute_map_delta": float(hybrid_aps.mean()-popular_aps.mean()),
        "users_hybrid_better": int(np.sum(paired_delta > 1e-12)), "users_tied": int(np.sum(np.abs(paired_delta)<=1e-12)),
        "users_hybrid_worse": int(np.sum(paired_delta < -1e-12))}
    result["runtime_seconds"] = round(time.perf_counter()-started, 3)
    save_csv("inner_holdout_evaluation.csv", comparison)
    return result


def run():
    started = time.perf_counter()
    hashes_before = raw_integrity()
    con = connect()
    result = run_inner_holdout(con)
    result["original_validation_case_study"] = build_case_study(con)
    con.close()
    require(raw_integrity() == hashes_before, "Raw CSVs changed")
    # Candidate output from the prior experiment must remain valid and unchanged.
    validation = load_json("advanced_submission_validation.json")
    with (RESULTS / "advanced_candidate_submission.csv").open("rb") as handle:
        submission_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    require(submission_hash == validation["submission_validation"]["sha256"], "Advanced candidate submission changed")
    result["final_submission_sha256_rechecked"] = submission_hash
    result["all_raw_hashes_unchanged"] = True
    con = connect()
    final_candidates = candidate_ids(con, "2020-07-27", "2020-10-19", SCHEDULES[1:])
    result["final_candidate_submission_validation"] = validate(
        RESULTS / "advanced_candidate_submission.csv", read_template(), final_candidates)
    con.close()
    result["total_runtime_seconds"] = round(time.perf_counter()-started, 3)
    save_json("pre_kaggle_reliability.json", result)
    print(json.dumps({"comparison": result["comparison"], "hybrid_minus_popularity": result["hybrid_minus_popularity"],
        "temporal_checks": result["temporal_checks"], "candidates": result["candidate_schedule"],
        "case_study": result["original_validation_case_study"], "runtime_seconds": result["total_runtime_seconds"]}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    run()
