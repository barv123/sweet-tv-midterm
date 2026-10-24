"""Bounded final sprint. Stages: prepare, A, B, C, select. Private caches only."""
import argparse
import csv
import json
import pickle
import time

import numpy as np
from scipy import sparse

from advanced_common import load_json
from baseline_experiment import FINAL, require
from common import CACHE, RESULTS, SCHEDULES, connect, literal, save_csv, save_json
from frequency_conservative_checkpoint import digest, final_association, summarize
from personal_frequency_no_decay import FOLDS, cached_events, eval_fold, predict_no_decay, train_table, write_new_submission
from recommendation_methods import ranked_indices
from validate_submission import candidate_ids, read_template, validate

STATE = CACHE / "final_competitive_sprint.pkl"
METRICS = "final_competitive_sprint.json"
SETTINGS = {"A_low025": ["A", .25, 0.0], "A_low050_high025": ["A", .50, .25],
            "B_cf003": ["B", .03], "B_cf006": ["B", .06],
            "C_weak0035": ["C", 3, .35, .06], "C_weak0050": ["C", 5, .50, .10]}


def save_state(state):
    with STATE.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_state():
    with STATE.open("rb") as handle:
        return pickle.load(handle)


def fuse(base, assoc, ids, pop, weight=.03, cf=None, cf_weight=0):
    positions = {int(item): i for i, item in enumerate(ids)}
    scores = np.zeros(len(ids))
    for rank, item in enumerate(base[:100], 1):
        scores[positions[item]] += 1 / (60 + rank)
    for rank, item in enumerate(assoc[:5], 1):
        scores[positions[item]] += weight / (60 + rank)
    if cf is not None:
        for rank, item in enumerate(cf[:20], 1):
            scores[positions[item]] += cf_weight / (60 + rank)
    return ids[ranked_indices(scores, ids, pop, 5)].tolist()


def features(con, users, fold, base, association):
    ids = np.array(sorted(base["eligible"]), dtype=np.int64)
    positions = {int(v): i for i, v in enumerate(ids)}
    userpos = {u: i for i, u in enumerate(users)}
    arrays = [np.zeros((len(users), len(ids)), dtype=np.float32) for _ in range(3)]
    # max coverage selects one qualifying row; it never sums sessions or airings.
    con.execute(f"""CREATE OR REPLACE TEMP TABLE sprint_airings AS
        SELECT user_id,tv_show_id,schedule_id,max(coverage_fraction) coverage
        FROM freq_edges WHERE session_start>='2020-03-09 00:00:00'
        AND session_end<=TIMESTAMP {literal(fold['start'])} AND coverage_fraction>0.30
        GROUP BY user_id,tv_show_id,schedule_id""")
    for user, item, high50, low, high80 in con.execute("""SELECT user_id,tv_show_id,
        count(*) FILTER(WHERE coverage>.50), count(*) FILTER(WHERE coverage<=.50),
        count(*) FILTER(WHERE coverage>.80) FROM sprint_airings JOIN adv_users USING(user_id)
        GROUP BY user_id,tv_show_id""").fetchall():
        if item in positions:
            for arr, value in zip(arrays, (high50, low, high80)):
                arr[userpos[user], positions[item]] = value
    f, low, high = arrays
    training = f"{fold['name']}_no_decay_train"
    pairs = con.execute(f"SELECT user_id,tv_show_id,count(*) FROM {training} GROUP BY user_id,tv_show_id").fetchall()
    allusers = sorted({r[0] for r in pairs})
    items = np.array(sorted({r[1] for r in pairs} | set(positions)), dtype=np.int64)
    up = {u: i for i, u in enumerate(allusers)}
    ip = {int(v): i for i, v in enumerate(items)}
    rr = np.array([up[u] for u, _, _ in pairs])
    cc = np.array([ip[item] for _, item, _ in pairs])
    binary = sparse.csr_matrix((np.ones(len(pairs), dtype=np.float32), (rr, cc)), shape=(len(allusers), len(items)))
    personal_pairs = [(userpos[u], ip[item], count) for u, item, count in pairs if u in userpos]
    history = sparse.csr_matrix(([np.log1p(n) for _, _, n in personal_pairs],
        ([u for u, _, _ in personal_pairs], [i for _, i, _ in personal_pairs])), shape=(len(users), len(items)))
    total = np.zeros(len(users)); peak = np.zeros(len(users)); top_share = np.zeros(len(users))
    by_user = {u: [] for u in users}
    for u, _, n in pairs:
        if u in by_user:
            by_user[u].append(n)
    for i, u in enumerate(users):
        counts = sorted(by_user[u], reverse=True)
        total[i] = sum(counts)
        peak[i] = counts[0] if counts else 0
        top_share[i] = sum(counts[:5]) / max(total[i], 1)
    pop = np.array([base["popularity"][int(v)] for v in ids])
    champion = {}
    for i, u in enumerate(users):
        require(ids[ranked_indices(f[i], ids, pop, 100)].tolist() == base["rankings"][u][:100], "Frequency aggregate mismatch")
        champion[u] = fuse(base["rankings"][u], association[u], ids, pop)
    return {"fold": fold, "users": users, "ids": ids, "pop": pop, "frequency": f, "low": low,
        "high": high, "base": base["rankings"], "champion": champion, "association": association,
        "truth": base.get("truth"), "binary": binary, "history": history, "items": items,
        "candidate_columns": np.array([ip[int(v)] for v in ids]), "total": total, "peak": peak, "share": top_share}


def diagnosis(ctx):
    users, truth = ctx["users"], ctx["truth"]
    total_labels = sum(len(truth[u]) for u in users)
    missed = sum(len(truth[u] - set(ctx["base"][u][:5])) for u in users)
    added = [(u, item) for u in users for item in set(ctx["champion"][u])-set(ctx["base"][u][:5])]
    itempos = {int(v): i for i, v in enumerate(ctx["ids"])}
    userpos = {u: i for i, u in enumerate(users)}
    return {"cold_users": int(np.sum(ctx["total"] == 0)),
        "weak_or_diffuse_users": int(np.sum((ctx["total"] > 0) & ((ctx["peak"] < 3) | (ctx["share"] < .35)))),
        "users_with_relevant_program_outside_historical_top5": sum(bool(truth[u]-set(ctx["base"][u][:5])) for u in users),
        "relevant_pairs_outside_historical_top5": missed, "relevant_pairs": total_labels,
        "outside_top5_percent": 100*missed/max(total_labels, 1),
        "association_added_top5_slots": len(added),
        "association_added_never_gt50_watched_slots": int(sum(ctx["frequency"][userpos[u], itempos[item]] == 0 for u, item in added)),
        "association_reorder_only_users": sum(ctx["champion"][u] != ctx["base"][u][:5] and set(ctx["champion"][u]) == set(ctx["base"][u][:5]) for u in users),
        "eligible_30_to_50_airing_signals": int(ctx["low"].sum()),
        "users_with_30_to_50_signal": int(np.sum(ctx["low"].sum(axis=1)>0)),
        "sparse_binary_shape": ctx["binary"].shape, "sparse_binary_nonzeros": int(ctx["binary"].nnz)}


def prepare():
    tick = time.perf_counter()
    users = read_template(); ref = load_json("strict80_summary.json")
    previous = load_json("frequency_conservative_checkpoint.json")
    assoc = {fold["name"]: {} for fold in FOLDS}
    old_champion = {fold["name"]: {} for fold in FOLDS}
    for filename, method, output in (("strict80_per_user.csv", "association_rules", assoc),
            ("frequency_conservative_per_user.csv", "B_rules_003", old_champion)):
        with (RESULTS / filename).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["method"] == method:
                    output[row["fold"]][users[int(row["template_row"])-1]] = list(map(int, row["predicted_ids"].split()))
    con = connect(); cached_events(con)
    con.execute("CREATE OR REPLACE TEMP TABLE adv_users(user_id VARCHAR,template_row INTEGER)")
    con.executemany("INSERT INTO adv_users VALUES (?,?)", [(u, i) for i, u in enumerate(users, 1)])
    result = {"settings": SETTINGS, "selection": "inner gain >=.002; outer >=.005; both gainers>losers, >=50 gainers, positive mean after removing top10 gains",
        "configuration": {"A": "F_gt50 + low_weight*F_30_to_50 + high_bonus*F_gt80, then existing .03 association RRF",
        "B": "champion RRF + weighted top20 item-CF RRF; cosine co-view shrink n/(n+20), min co-view 5, min viewers 10, top40 neighbors",
        "C": "weak if pre-cutoff max airing count below threshold OR top5 count share below threshold; weak association weight .06/.10; others .03; cold popularity"},
        "protected_hashes": {p.name: digest(p) for p in RESULTS.glob("submission*.csv")}, "folds": [], "experiments": {}}
    contexts = []
    for fold, prev in zip(FOLDS, previous["folds"]):
        base = eval_fold(con, users, fold, ref)
        ctx = features(con, users, fold, base, assoc[fold["name"]])
        require(ctx["champion"] == old_champion[fold["name"]], "Champion prediction reproduction failed")
        metric, _, _ = summarize(users, ctx["truth"], ctx["champion"], ctx["champion"])
        expected = prev["metrics"]["B_rules_003"]["map_at_5"]
        require(abs(metric["map_at_5"]-expected)<1e-12, f"Champion mismatch: {metric['map_at_5']} vs {expected}")
        record = {"name": fold["name"], "champion_map_at_5": metric["map_at_5"],
            "temporal_checks": base["temporal_checks"], "diagnosis": diagnosis(ctx)}
        result["folds"].append(record); contexts.append(ctx)
        save_json(METRICS, result)
        print(json.dumps(record), flush=True)
    con.close()
    save_state(contexts)
    result["prepare_seconds"] = round(time.perf_counter()-tick, 3)
    save_json(METRICS, result)


def item_cf(ctx):
    tick = time.perf_counter()
    binary = ctx["binary"].tocsc(); items = ctx["items"]; columns = ctx["candidate_columns"]
    counts = np.asarray(binary.sum(axis=0)).ravel()
    target = binary[:, columns]
    rr, cc, vv = [], [], []
    antecedents = np.flatnonzero(counts >= 10)
    for first in range(0, len(antecedents), 128):
        subset = antecedents[first:first+128]
        block = (binary[:, subset].T @ target).tocsr()
        for local, source in enumerate(subset):
            start, end = block.indptr[local:local+2]
            dest, together = block.indices[start:end], block.data[start:end]
            keep = (together >= 5) & (counts[columns[dest]] >= 10) & (columns[dest] != source)
            dest, together = dest[keep], together[keep]
            weights = together / np.sqrt(counts[source]*counts[columns[dest]]) * together/(together+20)
            order = np.lexsort((ctx["ids"][dest], -weights))[:40]
            rr.extend([int(source)]*len(order)); cc.extend(dest[order].tolist()); vv.extend(weights[order].tolist())
    similarity = sparse.csr_matrix((vv, (rr, cc)), shape=(len(items), len(columns)))
    rankings = {}
    for first in range(0, len(ctx["users"]), 64):
        scores = (ctx["history"][first:first+64] @ similarity).toarray()
        for offset, score in enumerate(scores):
            order = ranked_indices(score, ctx["ids"], ctx["pop"], 20)
            rankings[ctx["users"][first+offset]] = [int(ctx["ids"][j]) for j in order if score[j]>0]
    return rankings, {"seconds": round(time.perf_counter()-tick, 3), "neighbors": int(similarity.nnz),
                      "block_size": 128, "max_neighbors": 40}


def predict(ctx, setting, cf=None):
    kind = setting[0]; output = {}
    for i, u in enumerate(ctx["users"]):
        if ctx["total"][i] == 0:
            output[u] = ctx["base"][u][:5]
            continue
        base = ctx["base"][u]; weight = .03
        if kind == "A":
            score = ctx["frequency"][i] + setting[1]*ctx["low"][i] + setting[2]*ctx["high"][i]
            base = ctx["ids"][ranked_indices(score, ctx["ids"], ctx["pop"], 100)].tolist()
        if kind == "C" and (ctx["peak"][i] < setting[1] or ctx["share"][i] < setting[2]):
            weight = setting[3]
        output[u] = fuse(base, ctx["association"][u], ctx["ids"], ctx["pop"], weight,
            cf[u] if kind == "B" else None, setting[1] if kind == "B" else 0)
    return output


def experiment(kind):
    tick = time.perf_counter(); contexts = load_state(); result = load_json(METRICS)
    cf = {}; cf_info = {}
    if kind == "B":
        for ctx in contexts:
            cf[ctx["fold"]["name"]], cf_info[ctx["fold"]["name"]] = item_cf(ctx)
    for name, setting in SETTINGS.items():
        if setting[0] != kind:
            continue
        start = time.perf_counter(); records = []; paired = []
        for ctx in contexts:
            predictions = predict(ctx, setting, cf.get(ctx["fold"]["name"]))
            require(all(len(set(v)) == 5 and min(v)>0 and set(v)<=set(ctx["ids"]) for v in predictions.values()), "Invalid top5")
            metric, before, after = summarize(ctx["users"], ctx["truth"], ctx["champion"], predictions)
            metric["fold"] = ctx["fold"]["name"]
            metric["membership_change_percent"] = 100*metric["top5_membership_changed_users"]/len(ctx["users"])
            metric["reorder_only_users"] = metric["top5_changed_users"]-metric["top5_membership_changed_users"]
            records.append(metric)
            paired.extend({"fold": metric["fold"], "template_row": i+1, "champion_ap": float(before[i]),
                "ap": float(after[i]), "delta": float(after[i]-before[i]), "predicted_ids": " ".join(map(str, predictions[u]))}
                for i, u in enumerate(ctx["users"]))
        result["experiments"][name] = {"folds": records, "scoring_seconds": round(time.perf_counter()-start, 3), "shared_cf_fit": cf_info}
        save_csv(f"final_sprint_{name}_per_user.csv", paired)
        save_json(METRICS, result)
        print(json.dumps({"method": name, **result["experiments"][name]}), flush=True)
    result[f"stage_{kind}_seconds"] = round(time.perf_counter()-tick, 3)
    save_json(METRICS, result)


def select():
    tick = time.perf_counter(); result = load_json(METRICS)
    qualified = []
    comparison = []
    for name, experiment in result["experiments"].items():
        inner, outer = experiment["folds"]
        for m in (inner, outer):
            comparison.append({"method": name, **{k:v for k,v in m.items() if not isinstance(v, dict)}})
        if (inner["paired_mean_delta"] >= .002 and outer["paired_mean_delta"] >= .005
            and all(m["improved_users"] >=50 and m["improved_users"]>m["worsened_users"]
                and m["mean_delta_excluding_top10_gainers"]>0 for m in (inner, outer))):
            qualified.append(name)
    winner = max(qualified, key=lambda n: result["experiments"][n]["folds"][1]["map_at_5"]) if qualified else None
    result["winner"] = winner; result["new_submission"] = None
    save_csv("final_competitive_comparison.csv", comparison)
    save_json(METRICS, result)
    if winner:
        con = connect(); cached_events(con); users = read_template()
        con.execute("CREATE OR REPLACE TEMP TABLE adv_users(user_id VARCHAR,template_row INTEGER)")
        con.executemany("INSERT INTO adv_users VALUES (?,?)", [(u,i) for i,u in enumerate(users,1)])
        fold = {"name": "final_sprint", "start": FINAL, "end": "2020-10-19 00:00:00", "sources": SCHEDULES[1:]}
        training = train_table(con, fold); eligible = candidate_ids(con, FINAL, fold["end"], fold["sources"])
        rankings, _, pop, stats = predict_no_decay(con, users, training, eligible)
        ids = np.array(sorted(eligible))
        association, rule_count = final_association(con, users, fold, ids, pop)
        ctx = features(con, users, fold, {"eligible":eligible,"rankings":rankings,"popularity":pop}, association)
        champion_path = RESULTS / "submission_frequency_conservative_B_rules_003_strict80.csv"
        with champion_path.open(encoding="utf-8", newline="") as handle:
            saved = {r["user_id"]: list(map(int,r["tv_show_id"].split())) for r in csv.DictReader(handle)}
        require(ctx["champion"]==saved,"Final champion reproduction failed")
        cf, info = item_cf(ctx) if SETTINGS[winner][0]=="B" else (None, None)
        predictions = predict(ctx, SETTINGS[winner], cf)
        last = con.execute("SELECT max(session_end) FROM freq_edges WHERE session_start>='2020-03-09 00:00:00' AND session_end<=CAST(? AS TIMESTAMP) AND coverage_fraction>.30", [FINAL]).fetchone()[0]
        require(str(last)<=FINAL,"Future training leakage")
        path = RESULTS / f"submission_final_sprint_{winner}_strict80.csv"
        require(not path.exists(),"Refusing to overwrite prior CSV")
        write_new_submission(path, users, predictions, eligible)
        result["new_submission"] = validate(path,users,eligible)
        result["final_fit"] = {"latest_training_session_end":str(last),"cutoff":FINAL,"rules":rule_count,"cf":info,"stats":stats}
        con.close()
    require(all(digest(RESULTS/name)==value for name,value in result["protected_hashes"].items()),"Prior CSV changed")
    result["prior_submissions_preserved"] = True
    result["selection_refit_seconds"] = round(time.perf_counter()-tick,3)
    result["total_compute_seconds"] = result["prepare_seconds"]+sum(result.get(f"stage_{k}_seconds",0) for k in "ABC")+result["selection_refit_seconds"]
    save_json(METRICS,result)
    print(json.dumps({k:result[k] for k in ("winner","new_submission","total_compute_seconds","prior_submissions_preserved")}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=("prepare","A","B","C","select"))
    stage=parser.parse_args().stage
    if stage=="prepare": prepare()
    elif stage=="select": select()
    else: experiment(stage)
