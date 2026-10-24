"""Four frozen, independent augmentations of the strict no-decay champion.

Reuse saved strict association Top-5 evidence, not proxy-trained rule caches.
Run with the existing model environment and SWEET_TV_CACHE configured.
"""

import csv
import hashlib
import json
import time

import numpy as np

from advanced_common import load_json
from baseline_experiment import FINAL, require
from common import RESULTS, SCHEDULES, connect, literal, save_csv, save_json
from metrics import average_precision_at_k
from personal_frequency_no_decay import (
    FOLDS, cached_events, eval_fold, predict_no_decay, train_table, write_new_submission,
)
from recommendation_methods import ranked_indices
from validate_submission import candidate_ids, read_template, validate

SETTINGS = {"A_days_005": ("days", 0.05), "A_days_010": ("days", 0.10),
            "B_rules_003": ("rules", 0.03), "B_rules_006": ("rules", 0.06)}
EXPECTED = (0.4227206790123458, 0.5341366843033509)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_arrays(con, users, fold, eligible):
    ids = np.array(sorted(eligible), dtype=np.int64)
    ui = {u: i for i, u in enumerate(users)}
    ii = {int(item): i for i, item in enumerate(ids)}
    frequency = np.zeros((len(users), len(ids)))
    days = np.zeros_like(frequency)
    for user, item, count, ndays in con.execute(f"""SELECT user_id,tv_show_id,
        count(DISTINCT schedule_id),count(DISTINCT CAST(overlap_start AS DATE))
        FROM freq_edges JOIN adv_users USING(user_id)
        WHERE session_start>='2020-03-09 00:00:00'
          AND session_end<=TIMESTAMP {literal(fold['start'])} AND coverage_fraction>0.50
        GROUP BY user_id,tv_show_id""").fetchall():
        if item in ii:
            frequency[ui[user], ii[item]] = count
            days[ui[user], ii[item]] = ndays
    return ids, frequency, days


def augment(users, ids, frequency, days, popularity, base, association, settings):
    pop = np.array([popularity[int(item)] for item in ids])
    index = {int(item): i for i, item in enumerate(ids)}
    output = {name: {} for name in settings}
    for row, user in enumerate(users):
        # Verify the unmodified feature extraction reproduces champion ranking.
        raw_top = ids[ranked_indices(frequency[row], ids, pop, 100)].tolist()
        require(raw_top == base[user][:100], "Frequency feature reproduction mismatch")
        normalized_frequency = frequency[row] / max(float(frequency[row].max()), 1.0)
        normalized_days = days[row] / max(float(days[row].max()), 1.0)
        for name, (kind, weight) in settings.items():
            if kind == "days":
                score = normalized_frequency + weight * normalized_days
            else:
                score = np.zeros(len(ids))
                for rank, item in enumerate(base[user][:100], 1):
                    score[index[item]] += 1.0 / (60 + rank)
                for rank, item in enumerate(association[user][:5], 1):
                    score[index[item]] += weight / (60 + rank)
            top = ids[ranked_indices(score, ids, pop, 5)].tolist()
            require(len(set(top)) == 5 and min(top) > 0, "Invalid augmentation ranking")
            output[name][user] = top
    return output


def summarize(users, truth, base, prediction):
    before = np.array([average_precision_at_k(truth[u], base[u][:5]) for u in users])
    after = np.array([average_precision_at_k(truth[u], prediction[u][:5]) for u in users])
    delta = after - before
    changed = sum(prediction[u][:5] != base[u][:5] for u in users)
    membership = sum(set(prediction[u][:5]) != set(base[u][:5]) for u in users)
    positive = np.sort(delta[delta > 1e-12])[::-1]
    top10 = float(positive[:10].sum())
    result = {"map_at_5": float(after.mean()), "paired_mean_delta": float(delta.mean()),
        "paired_median_delta": float(np.median(delta)),
        "paired_delta_quantiles": dict(zip(("min", "p10", "p25", "p75", "p90", "max"),
            map(float, np.quantile(delta, [0, .1, .25, .75, .9, 1])))),
        "improved_users": int(np.sum(delta > 1e-12)), "worsened_users": int(np.sum(delta < -1e-12)),
        "unchanged_ap_users": int(np.sum(np.abs(delta) <= 1e-12)),
        "top5_changed_users": changed, "top5_changed_percent": 100 * changed / len(users),
        "top5_membership_changed_users": membership,
        "gross_positive_delta": float(positive.sum()), "gross_negative_delta": float(delta[delta < 0].sum()),
        "top10_share_of_positive_gain": top10 / float(positive.sum()) if len(positive) else 0.0,
        "mean_delta_excluding_top10_gainers": float((delta.sum()-top10)/(len(users)-min(10, len(positive))))}
    return result, before, after


def final_association(con, users, fold, ids, popularity):
    # Only needed if a rule variation wins. No compatible final strict rule cache exists.
    from advanced_experiment import collect_rankings, input_matrices
    from recommendation_methods import mine_association_rules, rule_score_blocks
    from strict_relevance_experiment import RULES
    con.execute(f"""CREATE OR REPLACE TEMP TABLE conservative_rule_train AS
        SELECT * FROM freq_edges WHERE session_start>='2020-03-09 00:00:00'
        AND session_end<=TIMESTAMP {literal(fold['start'])} AND coverage_fraction>0.50""")
    binary, personal, items, columns, train_users = input_matrices(con, users, ids,
        history_table="conservative_rule_train", recency_cutoff=fold["start"])
    rules, info = mine_association_rules(binary, items, columns, **RULES)
    pop = np.array([popularity[int(i)] for i in ids])
    global_order = ids[ranked_indices(pop, ids, pop, 100)].tolist()
    ranking, _ = collect_rankings(rule_score_blocks(personal, rules, columns), users, ids,
        pop, set(users)-train_users, global_order)
    return ranking, info["retained_rules"]


def main():
    started = time.perf_counter()
    protected = {p.name: digest(p) for p in RESULTS.glob("submission*.csv")}
    users = read_template()
    reference = load_json("strict80_summary.json")
    con = connect()
    cached_events(con)
    con.execute("CREATE OR REPLACE TEMP TABLE adv_users(user_id VARCHAR,template_row INTEGER)")
    con.executemany("INSERT INTO adv_users VALUES (?,?)", [(u, i) for i, u in enumerate(users, 1)])
    bases = []
    for fold, expected in zip(FOLDS, EXPECTED):
        baseline = eval_fold(con, users, fold, reference)
        require(abs(baseline["map_at_5"]-expected) < 1e-12,
                f"STOP: {fold['name']} expected {expected}, measured {baseline['map_at_5']}")
        bases.append(baseline)
        print(f"Champion reproduced {fold['name']}: {baseline['map_at_5']:.10f}", flush=True)
    cached_rules = {f["name"]: {} for f in FOLDS}
    evidence_path = RESULTS / "strict80_per_user.csv"
    with evidence_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "association_rules" and row["fold"] in cached_rules:
                cached_rules[row["fold"]][users[int(row["template_row"])-1]] = list(map(int, row["predicted_ids"].split()))
    result = {"settings": SETTINGS, "day_formula": "F/max_user_eligible_F + weight * D/max_user_eligible_D; zero maxima use 1",
        "rule_formula": "RRF60 frequency top100 weight 1 + saved strict association top5 weight w",
        "day_definition": "distinct local overlap_start dates from individual >50% pre-cutoff rows",
        "rule_evidence_path": str(evidence_path), "rule_evidence_sha256": digest(evidence_path),
        "rule_evidence_depth": 5, "proxy_rule_matrices_used": False,
        "selection_rule": "inner delta >=.002, outer >=.005; both folds >=50 gainers, gainers>losers, and delta excluding top10 gainers >0",
        "kaggle_scores_used_as_targets": False, "folds": [], "winner": None, "new_submission": None}
    paired_rows = []
    for fold, baseline in zip(FOLDS, bases):
        association = cached_rules[fold["name"]]
        require(set(association) == set(users), "Incomplete saved strict rule predictions")
        require(all(len(v) == len(set(v)) == 5 and set(v) <= baseline["eligible"] for v in association.values()),
                "Saved rules violate strict candidate constraints")
        rule_map = float(np.mean([average_precision_at_k(baseline["truth"][u], association[u]) for u in users]))
        saved_fold = next(f for f in reference["folds"] if f["name"] == fold["name"])
        saved_map = next(r["map_at_5"] for r in saved_fold["comparison"] if r["method"] == "association_rules")
        require(abs(rule_map-saved_map) < 1e-12, "Strict association evidence failed reproduction")
        ids, frequency, days = feature_arrays(con, users, fold, baseline["eligible"])
        predictions = augment(users, ids, frequency, days, baseline["popularity"], baseline["rankings"], association, SETTINGS)
        metrics = {}
        for name, prediction in predictions.items():
            metrics[name], before, after = summarize(users, baseline["truth"], baseline["rankings"], prediction)
            paired_rows.extend({"fold": fold["name"], "method": name, "template_row": i+1,
                "champion_ap_at_5": float(before[i]), "challenger_ap_at_5": float(after[i]),
                "delta": float(after[i]-before[i]), "predicted_ids": " ".join(map(str, prediction[u]))}
                for i, u in enumerate(users))
        result["folds"].append({"name": fold["name"], "champion_map_at_5": baseline["map_at_5"],
            "users": len(users), "candidate_count": len(ids), "temporal_checks": baseline["temporal_checks"],
            "baseline_sql_ap_error": baseline["max_sql_ap_error"], "cached_association_map": rule_map, "metrics": metrics})
        save_json("frequency_conservative_checkpoint.json", result)
        print(json.dumps(result["folds"][-1]), flush=True)
    inner, outer = [f["metrics"] for f in result["folds"]]
    qualified = [name for name in SETTINGS if inner[name]["paired_mean_delta"] >= .002
        and outer[name]["paired_mean_delta"] >= .005
        and all(m[name]["improved_users"] >= 50 and m[name]["improved_users"] > m[name]["worsened_users"]
                and m[name]["mean_delta_excluding_top10_gainers"] > 0 for m in (inner, outer))]
    winner = max(qualified, key=lambda n: outer[n]["map_at_5"]) if qualified else None
    result["winner"] = winner
    save_csv("frequency_conservative_per_user.csv", paired_rows)
    if winner:
        fold = {"name": "conservative_final", "start": FINAL, "end": "2020-10-19 00:00:00", "sources": SCHEDULES[1:]}
        eligible = candidate_ids(con, FINAL, fold["end"], fold["sources"])
        training = train_table(con, fold)
        base, _, popularity, stats = predict_no_decay(con, users, training, eligible)
        ids, frequency, days = feature_arrays(con, users, fold, eligible)
        association = {}
        if SETTINGS[winner][0] == "rules":
            association, rule_count = final_association(con, users, fold, ids, popularity)
            result["final_rules_retained"] = rule_count
        predictions = augment(users, ids, frequency, days, popularity, base, association, {winner: SETTINGS[winner]})[winner]
        path = RESULTS / f"submission_frequency_conservative_{winner}_strict80.csv"
        require(not path.exists(), "Refusing to overwrite a previous submission")
        latest = con.execute(f"SELECT max(last_session_end) FROM {training}").fetchone()[0]
        require(str(latest) <= FINAL, "Final training passes cutoff")
        write_new_submission(path, users, predictions, eligible)
        result["new_submission"] = validate(path, users, eligible)
        result["final_training"] = {"stats": stats, "last_session_end": str(latest), "cutoff": FINAL}
    con.close()
    require(all(digest(RESULTS / name) == value for name, value in protected.items()), "Prior submission changed")
    result["prior_submission_hashes_preserved"] = protected
    result["runtime_seconds"] = round(time.perf_counter()-started, 3)
    result["go"] = winner is not None
    save_json("frequency_conservative_checkpoint.json", result)
    print(json.dumps({"winner": winner, "go": result["go"], "new_submission": result["new_submission"],
                      "runtime_seconds": result["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
