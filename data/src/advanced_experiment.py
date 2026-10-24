"""Small fixed comparison of content filtering, association rules, and rank fusion.

Requires the existing cache and a successful critical_validation_audit.py run.
Does not regenerate the audit, baseline files, or Kaggle submission.
"""

from datetime import datetime, timezone
import json
import time

import numpy as np
from scipy import sparse

from advanced_common import ensure_metadata, load_json, prepare, protected_hashes
from baseline_experiment import CUT, FINAL, raw_integrity, require
from common import CACHE, RESULTS, connect, literal, one, rows, save_csv, save_json
from metrics import average_precision_at_k
from recommendation_methods import (content_features, content_score_blocks, mine_association_rules,
                                    ranked_indices, reciprocal_rank_fusion, rule_score_blocks)

PLAN = {
    "content": [{"name": "content_metadata", "include_title": False},
                {"name": "content_metadata_title", "include_title": True}],
    "association": [{"name": "association_count25_lift1", "min_pair_count": 25, "min_lift": 1.0},
                    {"name": "association_count50_lift1_2", "min_pair_count": 50, "min_lift": 1.2}],
    "fixed": {"history_half_life_days": 14, "max_rules_per_antecedent": 100,
              "min_confidence": 0.1, "min_item_viewers": 20, "ranking_depth": 100},
    "hybrid": "One equal-weight reciprocal-rank fusion: strongest advanced configuration + repeat/recency + popularity; k=60, top100 inputs",
    "selection": "Maximum validation MAP@5; exact ties resolved by method name. These are development scores after validation selection.",
}


def input_matrices(con, users, candidate_ids, history_table="adv_train", recency_cutoff=CUT):
    train_users = [row[0] for row in con.execute(f"SELECT DISTINCT user_id FROM {history_table} ORDER BY user_id").fetchall()]
    train_items = {row[0] for row in con.execute(f"SELECT DISTINCT tv_show_id FROM {history_table}").fetchall()}
    item_ids = np.array(sorted(train_items | set(map(int, candidate_ids))), dtype=np.int64)
    item_index = {int(item): index for index, item in enumerate(item_ids)}
    user_index = {user: index for index, user in enumerate(train_users)}
    row_parts, col_parts = [], []
    cursor = con.execute(f"SELECT DISTINCT user_id,tv_show_id FROM {history_table}")
    while batch := cursor.fetchmany(20000):
        row_parts.append(np.fromiter((user_index[user] for user, _ in batch), dtype=np.int32, count=len(batch)))
        col_parts.append(np.fromiter((item_index[item] for _, item in batch), dtype=np.int32, count=len(batch)))
    rr, cc = np.concatenate(row_parts), np.concatenate(col_parts)
    binary = sparse.csr_matrix((np.ones(len(rr), dtype=np.int32), (rr, cc)), shape=(len(train_users), len(item_ids)))
    row_parts, col_parts, value_parts = [], [], []
    evaluation_index = {user: index for index, user in enumerate(users)}
    cursor = con.execute(f"""WITH daily AS (SELECT DISTINCT user_id,tv_show_id,cast(overlap_start AS DATE) AS d
        FROM {history_table} JOIN adv_users USING(user_id))
        SELECT user_id,tv_show_id,round(sum(pow(2.0,-date_diff('day',d,DATE '{recency_cutoff[:10]}')/14.0)),12) AS weight
        FROM daily GROUP BY user_id,tv_show_id""")
    while batch := cursor.fetchmany(20000):
        row_parts.append(np.fromiter((evaluation_index[user] for user, _, _ in batch), dtype=np.int32, count=len(batch)))
        col_parts.append(np.fromiter((item_index[item] for _, item, _ in batch), dtype=np.int32, count=len(batch)))
        value_parts.append(np.fromiter((weight for _, _, weight in batch), dtype=np.float64, count=len(batch)))
    rr, cc, vv = np.concatenate(row_parts), np.concatenate(col_parts), np.concatenate(value_parts)
    personal = sparse.csr_matrix((vv, (rr, cc)), shape=(len(users), len(item_ids)), dtype=np.float64)
    candidates = np.array([item_index[int(item)] for item in candidate_ids], dtype=np.int32)
    return binary, personal, item_ids, candidates, set(train_users)


def canonical_metadata(con):
    # Prefer pre-cutoff records even when target metadata contains later edits.
    # Target-period metadata is used only for items with no pre-cutoff record.
    records = rows(con, f"""WITH available AS (
        SELECT *,CASE WHEN start_time<TIMESTAMP '{CUT}' THEN 0 ELSE 1 END AS phase
        FROM advanced_metadata WHERE duration>0 AND start_time<TIMESTAMP '{FINAL}'),
        prioritized AS (SELECT *,min(phase) OVER (PARTITION BY tv_show_id) AS best_phase FROM available)
        SELECT tv_show_id,min(phase) AS metadata_period,min(start_time) AS first_metadata_time,
          first(title ORDER BY start_time,title,channel_id) AS title,
          CASE WHEN count(DISTINCT category)=1 THEN min(category) END AS category,
          CASE WHEN count(DISTINCT (genre1,genre2,genre3))=1 THEN min(genre1) END AS genre1,
          CASE WHEN count(DISTINCT (genre1,genre2,genre3))=1 THEN min(genre2) END AS genre2,
          CASE WHEN count(DISTINCT (genre1,genre2,genre3))=1 THEN min(genre3) END AS genre3,
          CASE WHEN count(DISTINCT production_year)<=1 THEN min(production_year) END AS production_year,
          CASE WHEN count(DISTINCT director)<=1 THEN min(director) END AS director
        FROM prioritized WHERE phase=best_phase GROUP BY tv_show_id ORDER BY tv_show_id""")
    return {row["tv_show_id"]: row for row in records}


def collect_rankings(blocks, users, candidate_ids, popularity, cold_users, global_ranking):
    output, positive_counts, scored_union = {}, [], np.zeros(len(candidate_ids), dtype=bool)
    independent_rank_checks = 0
    for first, scores in blocks:
        for offset, score in enumerate(scores):
            index = first + offset
            positive = np.round(score, 12) > 0
            positive_counts.append(int(positive.sum()))
            scored_union |= positive
            order = ranked_indices(score, candidate_ids, popularity, 100)
            ranking = candidate_ids[order].tolist()
            require(len(set(ranking[:5])) == 5 and min(ranking[:5]) > 0, "Invalid top-five ranking")
            if users[index] in cold_users:
                require(ranking[:5] == global_ranking[:5], "Cold-start fallback changed")
            if index in (0, 1, 11, 43, 86, 113):
                independent = sorted(range(len(score)), key=lambda j: (-round(float(score[j]), 12), -int(popularity[j]), int(candidate_ids[j])))[:100]
                require(independent == order.tolist(), "Independent ranking/tie-breaking mismatch")
                independent_rank_checks += 1
            output[users[index]] = ranking
    require(len(output) == len(users), "Incomplete forecast cohort")
    return output, {"candidate_count": len(candidate_ids), "candidates_with_positive_evidence": int(scored_union.sum()),
        "full_popularity_fallback_users": sum(n == 0 for n in positive_counts),
        "users_with_fewer_than_5_scored_candidates": sum(n < 5 for n in positive_counts),
        "median_positive_candidates_per_user": float(np.median(positive_counts)),
        "independent_rank_checks": independent_rank_checks}


def summarize(name, ranking, users, truth, cold_users, training_items, runtime, details, reference_scores):
    aps = np.array([average_precision_at_k(truth[user], ranking[user][:5]) for user in users])
    cold = np.array([user in cold_users for user in users])
    nonempty = np.array([bool(truth[user]) for user in users])
    predicted = {item for user in users for item in ranking[user][:5]}
    report = {"method": name, "map_at_5": float(aps.mean()), "evaluated_users": len(users), "excluded_users": 0,
        "nonempty_truth_map": float(aps[nonempty].mean()), "cold_user_map": float(aps[cold].mean()),
        "warm_user_map": float(aps[~cold].mean()), "empty_truth_users": int((~nonempty).sum()),
        "unique_recommended_items": len(predicted), "recommended_items_unseen_in_training_behavior": len(predicted-training_items),
        "users_with_zero_hits": int(np.sum(aps == 0)), "runtime_seconds": runtime, **details}
    failures = np.argsort(aps - reference_scores, kind="stable")[:5]
    report["largest_losses_vs_popularity"] = [{"template_row": int(index+1), "relevant_items": len(truth[users[index]]),
        "map_difference": float(aps[index] - reference_scores[index]), "ap_at_5": float(aps[index]),
        "predictions": ranking[users[index]][:5], "hits": [int(item in truth[users[index]]) for item in ranking[users[index]][:5]]}
        for index in failures]
    return report, aps


def independent_rule_checks(binary, item_ids, rules):
    locations = {int(item): index for index, item in enumerate(item_ids)}
    checked = []
    for record in rules[:6]:
        a = set(binary[:, locations[record["antecedent"]]].nonzero()[0].tolist())
        b = set(binary[:, locations[record["consequent"]]].nonzero()[0].tolist())
        count = len(a & b)
        require(count == record["users_both"], "Rule pair count mismatch")
        require(abs(count/len(a) - record["confidence"]) < 1e-12, "Rule confidence mismatch")
        require(abs((count/len(a))/(len(b)/binary.shape[0]) - record["lift"]) < 1e-12, "Rule lift mismatch")
        checked.append({"antecedent": record["antecedent"], "consequent": record["consequent"], "independent_users_both": count, "passed": True})
    return checked


def verify_all_ap(con, per_user_file, predictions, users, truth):
    result = rows(con, f"""WITH p AS (
        SELECT t.method,u.user_id,p.rank,cast(p.item AS BIGINT) AS tv_show_id
        FROM read_csv({literal(per_user_file)},header=true) t JOIN adv_users u USING(template_row),
        unnest(string_split(t.predicted_ids,' ')) WITH ORDINALITY p(item,rank)),
        hit AS (SELECT p.*,CASE WHEN t.tv_show_id IS NULL THEN 0 ELSE 1 END AS hit FROM p LEFT JOIN adv_truth t USING(user_id,tv_show_id)),
        accumulated AS (SELECT *,sum(hit) OVER (PARTITION BY method,user_id ORDER BY rank) AS n FROM hit),
        denominator AS (SELECT user_id,count(*) AS n FROM adv_truth GROUP BY user_id)
        SELECT a.method,a.user_id,CASE WHEN coalesce(max(d.n),0)=0 THEN 0.0 ELSE
          sum(a.hit*a.n::DOUBLE/a.rank)/least(5,max(d.n)) END AS ap
        FROM accumulated a LEFT JOIN denominator d USING(user_id) GROUP BY a.method,a.user_id""")
    max_error = max(abs(row["ap"]-average_precision_at_k(truth[row["user_id"]], predictions[row["method"]][row["user_id"]][:5])) for row in result)
    require(len(result) == len(users)*len(predictions) and max_error < 1e-12, "Independent AP validation failed")
    return {"method_user_values_checked": len(result), "maximum_absolute_error": max_error, "passed": True}


def run():
    started = time.perf_counter()
    audit = load_json("advanced_validation_audit.json")
    require(all(value == 0 for value in audit["provenance_checks"].values()), "Resolve validation errors before fitting")
    protected = protected_hashes()
    save_json("advanced_plan.json", PLAN)
    con = connect()
    users, truth, pop, baseline = prepare(con)
    ensure_metadata(con)
    # Candidate order is numeric; popularity is only an explicit tie-break/fallback.
    scores_by_id = {row["tv_show_id"]: row["viewers"] for row in pop}
    candidate_ids = np.array(sorted(scores_by_id), dtype=np.int64)
    pop_scores = np.array([scores_by_id[int(item)] for item in candidate_ids], dtype=np.int64)
    global_ranking = [row["tv_show_id"] for row in pop]
    prep_start = time.perf_counter()
    binary, personal, item_ids, candidate_columns, train_users = input_matrices(con, users, candidate_ids)
    cold_users = set(users) - train_users
    metadata = canonical_metadata(con)
    training_items = set(item_ids[np.asarray(binary.sum(axis=0)).ravel() > 0].tolist())
    metadata_stats = {"items": len(metadata), "pre_cutoff_records": sum(r["metadata_period"] == 0 for r in metadata.values()),
        "target_only_records": sum(r["metadata_period"] == 1 for r in metadata.values()),
        "missing_category": sum(not r["category"] for r in metadata.values()),
        "missing_genres": sum(not any(r[k] for k in ("genre1", "genre2", "genre3")) for r in metadata.values()),
        "missing_director": sum(not r["director"] for r in metadata.values()),
        "missing_year": sum(not r["production_year"] for r in metadata.values())}
    model_directory = CACHE / "advanced_models"
    model_directory.mkdir(exist_ok=True)
    np.save(model_directory / "item_ids.npy", item_ids)
    sparse.save_npz(model_directory / "training_user_item_binary.npz", binary)
    sparse.save_npz(model_directory / "template_user_item_recency.npz", personal)
    preparation_seconds = time.perf_counter()-prep_start
    predictions = {key: value for key, value in baseline.items()}
    reference_aps = np.array([average_precision_at_k(truth[user], baseline["popularity"][user]) for user in users])
    reports, method_aps = [], {}
    old_run = load_json("baseline_experiment.json")
    for name in baseline:
        runtime = old_run["validation_fit"]["popularity_fit_predict_seconds_including_verification"]
        if name != "popularity":
            runtime += old_run["validation_fit"]["personalized"]["additional_fit_predict_seconds"]
        details = {"candidate_count": len(candidate_ids), "runtime_source": "preserved baseline run, not a fresh timing",
            "candidates_with_positive_evidence": sum(row["viewers"] > 0 for row in pop) if name == "popularity" else int(np.sum(personal[:, candidate_columns].getnnz(axis=0) > 0)),
            "full_popularity_fallback_users": len(users) if name == "popularity" else old_run["validation_fit"]["personalized"]["fully_popularity_fallback_users"]}
        record, aps = summarize(name, baseline[name], users, truth, cold_users, training_items, runtime, details, reference_aps)
        reports.append(record)
        method_aps[name] = aps
    # Regenerate the full repeat ranking only to supply the documented top100 fusion input.
    repeat_rankings, _ = collect_rankings(((0, personal[:, candidate_columns].toarray()),), users, candidate_ids, pop_scores, cold_users, global_ranking)
    require(all(repeat_rankings[user][:5] == baseline["repeat_recency_heuristic"][user] for user in users), "Repeat ranking changed")
    for config in PLAN["content"]:
        tick = time.perf_counter()
        features, feature_info = content_features(metadata, item_ids, config["include_title"])
        ranking, details = collect_rankings(content_score_blocks(personal, features, candidate_columns), users,
            candidate_ids, pop_scores, cold_users, global_ranking)
        details.update(feature_info)
        details["profile_weighting"] = "14-day-decayed distinct watching dates; L2-normalized metadata centroid"
        predictions[config["name"]] = ranking
        sparse.save_npz(model_directory / (config["name"] + ".npz"), features)
        record, aps = summarize(config["name"], ranking, users, truth, cold_users, training_items, time.perf_counter()-tick, details, reference_aps)
        reports.append(record)
        method_aps[config["name"]] = aps
        print(f"{config['name']}: MAP@5={record['map_at_5']:.8f}, {record['runtime_seconds']:.2f}s", flush=True)
    rule_checks = {}
    for config in PLAN["association"]:
        tick = time.perf_counter()
        rules, rule_info = mine_association_rules(binary, item_ids, candidate_columns,
            min_pair_count=config["min_pair_count"], min_lift=config["min_lift"])
        ranking, details = collect_rankings(rule_score_blocks(personal, rules, candidate_columns), users,
            candidate_ids, pop_scores, cold_users, global_ranking)
        rule_checks[config["name"]] = independent_rule_checks(binary, item_ids, rule_info["sample_rules"])
        save_csv(config["name"] + "_rule_examples.csv", rule_info.pop("sample_rules"))
        details.update(rule_info)
        predictions[config["name"]] = ranking
        sparse.save_npz(model_directory / (config["name"] + ".npz"), rules)
        record, aps = summarize(config["name"], ranking, users, truth, cold_users, training_items, time.perf_counter()-tick, details, reference_aps)
        reports.append(record)
        method_aps[config["name"]] = aps
        print(f"{config['name']}: MAP@5={record['map_at_5']:.8f}, {record['runtime_seconds']:.2f}s", flush=True)
    best_advanced = sorted(reports[2:], key=lambda row: (-row["map_at_5"], row["method"]))[0]["method"]
    hybrid_start = time.perf_counter()
    hybrid = {user: reciprocal_rank_fusion([predictions[best_advanced][user][:100], repeat_rankings[user][:100], global_ranking[:100]],
        candidate_ids, pop_scores) for user in users}
    hybrid_name = "hybrid_rrf60"
    predictions[hybrid_name] = hybrid
    record, aps = summarize(hybrid_name, hybrid, users, truth, cold_users, training_items, time.perf_counter()-hybrid_start,
        {"candidate_count": len(candidate_ids), "components": [best_advanced, "repeat_recency_heuristic", "popularity"],
         "fusion_constant": 60, "equal_weights": True, "full_popularity_fallback_users": len(cold_users),
         "candidates_with_positive_evidence": len({item for user in users for ranking in (predictions[best_advanced][user][:100], repeat_rankings[user][:100], global_ranking[:100]) for item in ranking})}, reference_aps)
    reports.append(record)
    method_aps[hybrid_name] = aps
    print(f"{hybrid_name}: MAP@5={record['map_at_5']:.8f}, {record['runtime_seconds']:.2f}s", flush=True)
    # Actual-output tests apply to every method, including preserved baselines and hybrid.
    eligible = set(candidate_ids.tolist())
    for name, forecast in predictions.items():
        require(set(forecast) == set(users), f"Cohort mismatch: {name}")
        for user in users:
            top5 = forecast[user][:5]
            require(len(top5) == len(set(top5)) == 5 and set(top5) <= eligible, f"Invalid recommendations: {name}")
            if user in cold_users:
                require(top5 == global_ranking[:5], f"Cold-user fallback mismatch: {name}")
    per_user = [{"method": name, "template_row": index+1, "relevant_items": len(truth[user]),
        "cold_history": user in cold_users, "ap_at_5": float(method_aps[name][index]),
        "hits_at_5": len(set(forecast[user][:5]) & truth[user]), "predicted_ids": " ".join(map(str, forecast[user][:5]))}
        for name, forecast in predictions.items() for index, user in enumerate(users)]
    save_csv("advanced_per_user_evaluation.csv", per_user)
    independent_ap = verify_all_ap(con, RESULTS / "advanced_per_user_evaluation.csv", predictions, users, truth)
    raw_integrity()
    require(protected_hashes() == protected, "Previous baseline artifacts were modified")
    table_columns = ("method", "map_at_5", "runtime_seconds", "candidate_count", "candidates_with_positive_evidence",
        "unique_recommended_items", "recommended_items_unseen_in_training_behavior", "full_popularity_fallback_users",
        "users_with_zero_hits", "evaluated_users", "excluded_users")
    save_csv("advanced_comparison.csv", [{key: record[key] for key in table_columns} for record in reports])
    best = sorted(reports, key=lambda row: (-row["map_at_5"], row["method"]))[0]
    # Paired uncertainty on the selected configuration is descriptive, not selection-adjusted.
    delta = method_aps[best["method"]] - reference_aps
    mean_delta = float(delta.mean())
    se = float(delta.std(ddof=1)/np.sqrt(len(delta)))
    result = {"plan": PLAN, "comparison": reports, "best_method": best["method"], "best_advanced_method": best_advanced,
        "best_minus_popularity": {"mean": mean_delta, "normal_95_interval": [mean_delta-1.96*se, mean_delta+1.96*se],
            "selection_adjusted": False},
        "protocol": {"users": len(users), "excluded": 0, "candidate_count": len(candidate_ids),
            "relevant_user_item_pairs": sum(map(len, truth.values())), "cold_users": len(cold_users), "unchanged": True},
        "metadata": metadata_stats, "training_matrix": {"users": binary.shape[0], "item_columns": binary.shape[1],
            "nonzero_user_item_pairs": binary.nnz, "last_training_session_end": str(con.execute("SELECT max(session_end) FROM adv_train").fetchone()[0])},
        "verification": {"eligible_unique_five_forecasts_checked": len(users)*len(predictions),
            "cold_start_lists_checked": len(cold_users)*len(predictions), "independent_ap": independent_ap,
            "independent_rule_checks": rule_checks, "all_raw_hashes_match_audit": True, "baseline_results_unchanged": True,
            "baseline_result_hashes": protected},
        "shared_matrix_preparation_seconds": preparation_seconds, "runtime_seconds": round(time.perf_counter()-started, 3),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__}
    save_json("advanced_experiment.json", result)
    con.close()
    print(f"Best development strategy: {best['method']} ({best['map_at_5']:.8f}); total {result['runtime_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    run()
