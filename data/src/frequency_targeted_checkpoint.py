"""Two fixed frequency variations using cached joins and the existing strict evaluator."""

import hashlib
import json
import time

import numpy as np

from advanced_common import load_json
from baseline_experiment import FINAL, require
from common import RESULTS, SCHEDULES, TIMEZONE, connect, literal, save_json
from metrics import average_precision_at_k
from personal_frequency_no_decay import (
    FOLDS, cached_events, eval_fold, predict_no_decay, train_table, write_new_submission,
)
from recommendation_methods import ranked_indices
from validate_submission import candidate_ids, read_template, validate


def variants(con, users, fold, training, eligible, popularity, methods):
    ids = np.array(sorted(eligible), dtype=np.int64)
    positions = {int(item): i for i, item in enumerate(ids)}
    pop = np.array([popularity[int(item)] for item in ids])
    counts = {user: {} for user in users}
    if "distinct_days" in methods:
        query = f"""SELECT user_id,tv_show_id,count(DISTINCT CAST(overlap_start AS DATE))
            FROM freq_edges JOIN adv_users USING(user_id)
            WHERE session_start>='2020-03-09 00:00:00'
              AND session_end<=TIMESTAMP {literal(fold['start'])} AND coverage_fraction>0.50
            GROUP BY user_id,tv_show_id"""
        day_counts = {user: {} for user in users}
        for user, item, count in con.execute(query).fetchall():
            if item in positions:
                day_counts[user][positions[item]] = count
    exposure = np.ones(len(ids))
    evidence = {}
    if "schedule_exposure" in methods:
        for user, item, count in con.execute(f"""SELECT user_id,tv_show_id,count(DISTINCT schedule_id)
                FROM {training} JOIN adv_users USING(user_id) GROUP BY user_id,tv_show_id""").fetchall():
            if item in positions:
                counts[user][positions[item]] = count
        schedule_counts = dict(con.execute(f"""SELECT tv_show_id,count(DISTINCT schedule_id)
            FROM schedule WHERE tv_show_id>0 AND duration>0
              AND source IN ({','.join(literal(s) for s in fold['sources'])})
              AND start_time<TIMESTAMP {literal(fold['end'])}
              AND timezone('{TIMEZONE}',timezone('{TIMEZONE}',start_time)+duration*INTERVAL '1 second')
                  >TIMESTAMP {literal(fold['start'])}
            GROUP BY tv_show_id""").fetchall())
        require(set(schedule_counts) == eligible, "Exposure candidates differ from strict candidates")
        scheduled = np.array([schedule_counts[int(item)] for item in ids], dtype=float)
        scale = float(np.quantile(scheduled, 0.95))
        exposure += 0.15 * np.minimum(np.log1p(scheduled) / np.log1p(scale), 1.0)
        require(np.all((exposure >= 1.0) & (exposure <= 1.15)), "Exposure boost outside bounds")
        evidence = {"schedule_airings": int(scheduled.sum()), "schedule_count_p95": scale,
                    "multiplier_min": float(exposure.min()), "multiplier_max": float(exposure.max())}
    predictions = {method: {} for method in methods}
    for method in methods:
        values = day_counts if method == "distinct_days" else counts
        for user in users:
            scores = np.zeros(len(ids))
            for index, count in values[user].items():
                scores[index] = count
            if method == "schedule_exposure":
                scores *= exposure
            top = ids[ranked_indices(scores, ids, pop, 5)].tolist()
            require(len(set(top)) == 5 and min(top) > 0 and set(top) <= eligible, "Invalid predictions")
            predictions[method][user] = top
    return predictions, evidence


def main():
    started = time.perf_counter()
    baseline_path = RESULTS / "submission_personal_frequency_no_decay_strict80.csv"
    original_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    users = read_template()
    con = connect()
    print("Loading existing cached joins", flush=True)
    cached_events(con)
    con.execute("CREATE OR REPLACE TEMP TABLE adv_users(user_id VARCHAR,template_row INTEGER)")
    con.executemany("INSERT INTO adv_users VALUES (?,?)", [(u, i) for i, u in enumerate(users, 1)])
    reference = load_json("strict80_summary.json")
    previous = load_json("frequency_no_decay_results.json")["no_decay_MAP_at_5"]["by_fold"]
    methods = ("distinct_days", "schedule_exposure")
    result = {"configuration": {
        "distinct_days": "distinct local overlap_start dates of individual >50% qualifying rows",
        "schedule_exposure": "airing frequency * (1 + 0.15 * min(log1p(schedule_count)/log1p(schedule_count_p95),1))",
        "tie_break": "unchanged training distinct-user popularity, then program ID",
        "selection": "outer delta >=0.002; inner delta >=-0.002; paired outer normal 95% lower bound >0",
        "schedule_sources": "same period/source restrictions as existing candidate_ids; schedule only, no future viewing",
        "train_gt50_truth_gt80": True, "empty_truth_users_included": True,
        "kaggle_scores_used": False}, "baseline_sha256_before": original_hash, "folds": []}
    for fold in FOLDS:
        baseline = eval_fold(con, users, fold, reference)
        require(abs(baseline["map_at_5"] - previous[fold["name"]]) < 1e-12, "Baseline failed reproduction")
        predictions, exposure_evidence = variants(con, users, fold,
            f"{fold['name']}_no_decay_train", baseline["eligible"], baseline["popularity"], methods)
        base_ap = np.array([average_precision_at_k(baseline["truth"][u], baseline["rankings"][u][:5]) for u in users])
        metrics = {"baseline": {"map_at_5": baseline["map_at_5"]}}
        for method in methods:
            ap = np.array([average_precision_at_k(baseline["truth"][u], predictions[method][u]) for u in users])
            delta = ap - base_ap
            margin = 1.96 * float(delta.std(ddof=1)) / np.sqrt(len(users))
            metrics[method] = {"map_at_5": float(ap.mean()), "delta": float(delta.mean()),
                "paired_normal_95_interval": [float(delta.mean()-margin), float(delta.mean()+margin)],
                "changed_users": sum(predictions[method][u] != baseline["rankings"][u][:5] for u in users)}
        result["folds"].append({"name": fold["name"], "users": len(users),
            "candidate_count": baseline["candidate_count"], "strict_relevant_pairs": baseline["strict_relevant_pairs"],
            "temporal_checks": baseline["temporal_checks"], "baseline_sql_max_error": baseline["max_sql_ap_error"],
            "baseline_reproduced": True, "metrics": metrics, "exposure": exposure_evidence})
        save_json("frequency_targeted_checkpoint.json", result)
        print(json.dumps(result["folds"][-1]), flush=True)
    inner, outer = [f["metrics"] for f in result["folds"]]
    qualified = [m for m in methods if outer[m]["delta"] >= 0.002 and inner[m]["delta"] >= -0.002
                 and outer[m]["paired_normal_95_interval"][0] > 0]
    winner = max(qualified, key=lambda m: outer[m]["map_at_5"]) if qualified else None
    result["winner"] = winner
    result["new_submission"] = None
    final_fold = {"name": "targeted_final", "start": FINAL, "end": "2020-10-19 00:00:00", "sources": SCHEDULES[1:]}
    eligible = candidate_ids(con, FINAL, final_fold["end"], final_fold["sources"])
    result["preserved_baseline_validation"] = validate(baseline_path, users, eligible)
    if winner:
        print("Refitting selected variation: " + winner, flush=True)
        training = train_table(con, final_fold)
        _, _, popularity, stats = predict_no_decay(con, users, training, eligible)
        predictions, evidence = variants(con, users, final_fold, training, eligible, popularity, (winner,))
        output = RESULTS / f"submission_frequency_targeted_{winner}_strict80.csv"
        require(not output.exists(), "Refusing to overwrite existing candidate")
        write_new_submission(output, users, predictions[winner], eligible)
        result["new_submission"] = validate(output, users, eligible)
        result["final_fit"] = stats
        result["final_exposure"] = evidence
    con.close()
    result["baseline_sha256_after"] = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    require(result["baseline_sha256_after"] == original_hash, "Best baseline changed")
    result["runtime_seconds"] = round(time.perf_counter()-started, 3)
    result["go"] = winner is not None
    save_json("frequency_targeted_checkpoint.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "folds"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
