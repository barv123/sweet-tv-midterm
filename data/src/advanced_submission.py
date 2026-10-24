"""Refit the validation-selected rank-fusion strategy for the final schedule.

Writes a separate advanced candidate; never modifies the original baseline submission.
No final-period viewing labels are used and no network/Kaggle submission occurs.
"""

import csv
import time

import numpy as np
from scipy import sparse

from advanced_common import ensure_metadata, prepare, protected_hashes
from advanced_experiment import PLAN, collect_rankings, input_matrices
from baseline_experiment import BEGIN, FINAL, TARGET_END, raw_integrity, require
from common import CACHE, RAW, RESULTS, SCHEDULES, connect, literal, one, rows, save_json
from recommendation_methods import mine_association_rules, reciprocal_rank_fusion, rule_score_blocks
from validate_submission import candidate_ids, read_template, validate


def run():
    started = time.perf_counter()
    protected = protected_hashes()
    source_hashes = raw_integrity()
    con = connect()
    users = read_template()
    con.execute("CREATE OR REPLACE TEMP TABLE adv_users(user_id VARCHAR,template_row INTEGER)")
    con.executemany("INSERT INTO adv_users VALUES (?,?)", [(user, index) for index, user in enumerate(users, 1)])
    # This table intentionally includes all train+validation viewing, but nothing at or
    # after the final target origin. Whole viewing records are complete by the cutoff.
    con.execute(f"""CREATE OR REPLACE TEMP TABLE adv_final_history AS SELECT * FROM interactions
        WHERE session_start>=TIMESTAMP '{BEGIN}' AND session_end<=TIMESTAMP '{FINAL}'""")
    bounds = one(con, """SELECT count(*) AS interactions,count(DISTINCT viewing_id) AS sessions,
        count(DISTINCT user_id) AS users,min(session_start) AS first_start,max(session_end) AS last_end,
        max(overlap_end) AS last_overlap_end FROM adv_final_history""")
    require(bounds["last_end"] <= __import__("datetime").datetime.fromisoformat(FINAL), "Future viewing leaked into final fit")
    eligible = np.array(sorted(candidate_ids(con, FINAL, TARGET_END, SCHEDULES[1:])), dtype=np.int64)
    require(len(eligible) >= 5 and np.all(eligible > 0), "Invalid final candidate set")
    con.execute("CREATE OR REPLACE TEMP TABLE adv_final_candidates(tv_show_id BIGINT)")
    con.executemany("INSERT INTO adv_final_candidates VALUES (?)", [(int(item),) for item in eligible])
    pop_rows = rows(con, """SELECT c.tv_show_id,coalesce(p.viewers,0) AS viewers
        FROM adv_final_candidates c LEFT JOIN (
          SELECT tv_show_id,count(DISTINCT user_id) viewers FROM adv_final_history GROUP BY tv_show_id
        ) p USING(tv_show_id) ORDER BY viewers DESC,c.tv_show_id""")
    pop_by_id = {row["tv_show_id"]: row["viewers"] for row in pop_rows}
    require(len(pop_by_id) == len(eligible), "Candidate IDs must be unique")
    global_ranking = [row["tv_show_id"] for row in pop_rows]
    popularity = np.array([pop_by_id[int(item)] for item in eligible], dtype=np.int64)
    binary, personal, item_ids, candidate_columns, train_users = input_matrices(
        con, users, eligible, history_table="adv_final_history", recency_cutoff=FINAL)
    cold_users = set(users) - train_users
    rules, rule_info = mine_association_rules(binary, item_ids, candidate_columns,
        min_pair_count=25, min_lift=1.0, min_confidence=0.1, min_item_viewers=20, max_rules=100)
    rule_rankings, rule_details = collect_rankings(rule_score_blocks(personal, rules, candidate_columns),
        users, eligible, popularity, cold_users, global_ranking)
    daily_rankings, daily_details = collect_rankings(((0, personal[:, candidate_columns].toarray()),),
        users, eligible, popularity, cold_users, global_ranking)
    con.close()

    forecasts = {user: reciprocal_rank_fusion(
        [rule_rankings[user][:100], daily_rankings[user][:100], global_ranking[:100]],
        eligible, popularity, constant=60)[:5] for user in users}
    for user, items in forecasts.items():
        require(len(items) == len(set(items)) == 5 and set(items) <= set(eligible), "Invalid final hybrid ranking")
        if user in cold_users:
            require(items == global_ranking[:5], "Final cold-start fallback mismatch")

    output = RESULTS / "advanced_candidate_submission.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["user_id", "tv_show_id"])
        writer.writerows((user, " ".join(map(str, forecasts[user]))) for user in users)
    checked = validate(output, users, set(map(int, eligible)))
    hashes_after = raw_integrity()
    require(hashes_after == source_hashes, "Raw source files changed")
    require(protected_hashes() == protected, "Previous baseline artifacts changed")
    evidence = {"selected_validation_method": "hybrid_rrf60",
        "validation_map_at_5": 0.8169656084656084,
        "components": ["association_count25_lift1", "repeat_recency_heuristic", "popularity"],
        "final_fit": bounds, "final_target": {"start_inclusive": FINAL, "end_exclusive": TARGET_END},
        "final_eligible_candidates": len(eligible), "candidates_with_training_viewers": sum(v > 0 for v in pop_by_id.values()),
        "global_popularity_top5": global_ranking[:5], "association_rules": {key: val for key, val in rule_info.items() if key != "sample_rules"},
        "rule_scoring_candidate_coverage": rule_details, "repeat_scoring_candidate_coverage": daily_details,
        "cold_users": len(cold_users), "cold_users_use_global_popularity_fallback": True,
        "final_top5_examples": [{"template_row": index+1,"user_id": user,"items": forecasts[user]} for index,user in enumerate(users[:10])],
        "submission_validation": checked, "raw_sha256_unchanged": True,
        "previous_baseline_artifacts_unchanged": True, "baseline_result_hashes": protected,
        "kaggle_uploaded": False, "runtime_seconds": round(time.perf_counter()-started, 3)}
    save_json("advanced_submission_validation.json", evidence)
    print("Advanced candidate validation passed:")
    print({key: evidence[key] for key in ("selected_validation_method", "validation_map_at_5", "final_fit",
        "final_eligible_candidates", "cold_users", "submission_validation", "runtime_seconds")})


if __name__ == "__main__":
    run()
