"""Fast A-only strict-label evaluation and final candidate generation.

Uses cached interval edges, strict >80% labels, and >50% per-airing training signals.
One user-program airing counts once (quality-filtered duplicate rows are already removed).
This intentionally evaluates only the no-decay frequency baseline; it does not rerun
content, association, or hybrid models.
"""

import csv
import json
import time
from fractions import Fraction

import numpy as np

from advanced_common import ensure_metadata, load_json
from baseline_experiment import BEGIN, FINAL, raw_integrity, require
from common import RESULTS, SCHEDULES, connect, literal, rows, save_csv, save_json
from metrics import average_precision_at_k
from recommendation_methods import ranked_indices
from validate_submission import candidate_ids, read_template, validate

TRAIN_FRACTION=0.50
LABEL_FRACTION=0.80
FOLDS=(
    {"name":"inner_weeks15_16","start":"2020-04-06 00:00:00","end":"2020-04-20 00:00:00","sources":SCHEDULES[:1]},
    {"name":"outer_weeks21_30","start":"2020-05-18 00:00:00","end":"2020-07-27 00:00:00","sources":SCHEDULES[:2]},
)


def cached_events(con):
    con.execute("""CREATE OR REPLACE TEMP TABLE freq_edges AS SELECT e.viewing_id,v.user_id,v.channel_id,
      e.schedule_id,s.start_time airing_start,s.duration airing_seconds,e.tv_show_id,
      v.start_time session_start,v.end_time session_end,e.overlap_start,e.overlap_seconds,
      e.overlap_seconds::DOUBLE/s.duration coverage_fraction
      FROM match_edges e JOIN viewing v USING(viewing_id)
      JOIN session_quality q USING(viewing_id) JOIN schedule s USING(schedule_id)
      WHERE q.usable AND e.tv_show_id>0 AND s.duration>0 AND e.overlap_seconds>0""")


def train_table(con, fold):
    name=f"{fold['name']}_no_decay_train"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE {name} AS SELECT user_id,tv_show_id,schedule_id,
      min(airing_start) airing_start,max(session_end) last_session_end
      FROM freq_edges WHERE session_start>='2020-03-09 00:00:00'
       AND session_end<=TIMESTAMP '{fold['start']}' AND coverage_fraction>{TRAIN_FRACTION}
      GROUP BY user_id,tv_show_id,schedule_id""")
    return name


def predict_no_decay(con, users, training, eligible):
    candidates=np.array(sorted(eligible),dtype=np.int64)
    con.execute("CREATE OR REPLACE TEMP TABLE freq_candidates(tv_show_id BIGINT)")
    con.executemany("INSERT INTO freq_candidates VALUES (?)",[(int(x),) for x in candidates])
    popularity_rows=rows(con,f"""SELECT c.tv_show_id,coalesce(p.viewers,0) viewers FROM freq_candidates c
      LEFT JOIN (SELECT tv_show_id,count(DISTINCT user_id) viewers FROM {training} GROUP BY tv_show_id) p USING(tv_show_id)
      ORDER BY viewers DESC,c.tv_show_id""")
    popularity={r["tv_show_id"]:r["viewers"] for r in popularity_rows}
    popularity_vector=np.array([popularity[int(x)] for x in candidates],dtype=np.int64)
    global_order=[r["tv_show_id"] for r in popularity_rows]
    personal_rows=rows(con,f"""SELECT user_id,tv_show_id,count(DISTINCT schedule_id) airing_frequency
      FROM {training} JOIN adv_users USING(user_id) JOIN freq_candidates USING(tv_show_id)
      GROUP BY user_id,tv_show_id""")
    by_user={u:{} for u in users}
    for row in personal_rows: by_user[row["user_id"]][row["tv_show_id"]]=row["airing_frequency"]
    positions={int(item):i for i,item in enumerate(candidates)}
    rankings={}
    for user in users:
        scores=np.zeros(len(candidates),dtype=np.float64)
        for item,count in by_user[user].items(): scores[positions[int(item)]]=float(count)
        rankings[user]=candidates[ranked_indices(scores,candidates,popularity_vector,100)].tolist()
    return rankings,global_order,popularity,{"eligible_candidates":len(candidates),
        "candidates_with_popularity":sum(value>0 for value in popularity.values()),
        "training_users":len({r["user_id"] for r in personal_rows}),
        "personal_user_program_pairs":len(personal_rows),
        "training_airing_occurrences":con.execute(f"SELECT count(*) FROM {training}").fetchone()[0]}


def eval_fold(con,users,fold,strict_reference):
    started=time.perf_counter()
    training=train_table(con,fold)
    eligible=candidate_ids(con,fold["start"],fold["end"],fold["sources"])
    rankings,global_order,popularity,training_stats=predict_no_decay(con,users,training,eligible)
    truth_table=f"{fold['name']}_no_decay_truth"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE {truth_table} AS SELECT DISTINCT e.user_id,e.tv_show_id
      FROM freq_edges e JOIN adv_users USING(user_id)
      WHERE e.session_start>=TIMESTAMP '{fold['start']}' AND e.session_end<=TIMESTAMP '{fold['end']}'
        AND e.coverage_fraction>{LABEL_FRACTION}""")
    truth={user:set() for user in users}
    for user,item in con.execute(f"SELECT user_id,tv_show_id FROM {truth_table}").fetchall():truth[user].add(item)
    proxy={user:set() for user in users}
    for user,item in con.execute(f"""SELECT DISTINCT i.user_id,i.tv_show_id FROM interactions i JOIN adv_users USING(user_id)
      WHERE i.session_start>=TIMESTAMP '{fold['start']}' AND i.session_end<=TIMESTAMP '{fold['end']}'""").fetchall():proxy[user].add(item)
    require(all(items<=eligible for items in truth.values()),"Strict label outside candidate pool")
    ap=[];user_scores=[]
    for rownum,user in enumerate(users,1):
        top=rankings[user][:5]
        require(len(top)==len(set(top))==5 and set(top)<=eligible and all(x>0 for x in top),"Invalid no-decay top five")
        value=average_precision_at_k(truth[user],top);ap.append(value)
        user_scores.append({"template_row":rownum,"strict_relevant_items":len(truth[user]),
            "ap_at_5":value,"hits_at_5":len(set(top)&truth[user]),"predicted_ids":" ".join(map(str,top))})
    score=float(np.mean(ap))
    prior_fold=next(f for f in strict_reference["folds"] if f["name"]==fold["name"])
    prior_d=next(r["map_at_5"] for r in prior_fold["comparison"] if r["method"]=="repeat_recency")
    sqlname=f"{fold['name']}_no_decay_prediction_check"
    con.execute(f"CREATE OR REPLACE TEMP TABLE {sqlname}(user_id VARCHAR,rank INTEGER,tv_show_id BIGINT)")
    con.executemany(f"INSERT INTO {sqlname} VALUES (?,?,?)",[(user,k,item) for user in users
        for k,item in enumerate(rankings[user][:5],1)])
    sql=con.execute(f"""WITH h AS (SELECT p.*,CASE WHEN t.tv_show_id IS NULL THEN 0 ELSE 1 END hit
      FROM {sqlname} p LEFT JOIN {truth_table} t USING(user_id,tv_show_id)), c AS
      (SELECT *,sum(hit) OVER(PARTITION BY user_id ORDER BY rank) cum FROM h), d AS
      (SELECT user_id,count(*) n FROM {truth_table} GROUP BY user_id)
      SELECT c.user_id,CASE WHEN coalesce(max(d.n),0)=0 THEN 0.0 ELSE
        sum(c.hit*c.cum::DOUBLE/c.rank)/least(5,max(d.n)) END ap
      FROM c LEFT JOIN d USING(user_id) GROUP BY c.user_id""").fetchall()
    max_error=max(abs(row[1]-average_precision_at_k(truth[row[0]],rankings[row[0]][:5])) for row in sql)
    require(len(sql)==len(users) and max_error<1e-12,"Independent MAP@5 check failed")
    temporal={"training_after_cutoff":con.execute(f"SELECT count(*) FROM {training} WHERE last_session_end>TIMESTAMP '{fold['start']}'").fetchone()[0],
        "truth_missing_in_window":con.execute(f"""SELECT count(*) FROM {truth_table} t WHERE NOT EXISTS (
          SELECT 1 FROM freq_edges e WHERE e.user_id=t.user_id AND e.tv_show_id=t.tv_show_id
           AND e.session_start>=TIMESTAMP '{fold['start']}' AND e.session_end<=TIMESTAMP '{fold['end']}'
           AND e.coverage_fraction>{LABEL_FRACTION})""").fetchone()[0],
        "truth_outside_candidates":con.execute(f"SELECT count(*) FROM {truth_table} WHERE tv_show_id NOT IN (SELECT tv_show_id FROM freq_candidates)").fetchone()[0]}
    require(all(v==0 for v in temporal.values()),"No-decay leakage check failed")
    # Check the raw >50 signal count against the completed strict-label run.
    prior_train_rows=next(f for f in prior_fold["comparison"] if f["method"]=="repeat_recency")["train50_event_airing_signals"]
    raw_gt50=con.execute(f"""SELECT count(*) FROM freq_edges WHERE session_start>='2020-03-09 00:00:00'
       AND session_end<=TIMESTAMP '{fold['start']}' AND coverage_fraction>{TRAIN_FRACTION}""").fetchone()[0]
    require(raw_gt50==prior_train_rows,"Train >50 airing signal count differs from saved strict pipeline")
    return {"fold":fold["name"],"map_at_5":score,"users":len(users),"excluded_users":0,
        "candidate_count":len(eligible),"strict_relevant_pairs":sum(map(len,truth.values())),
        "proxy60_relevant_pairs":sum(map(len,proxy.values())),"proxy_only_pairs":sum(len(proxy[u]-truth[u]) for u in users),
        "empty_truth_users":sum(not truth[u] for u in users),"training_gt50_raw_airing_signals":raw_gt50,
        "training_gt50_deduped_airing_occurrences":training_stats["training_airing_occurrences"],
        "training_user_program_pairs":training_stats["personal_user_program_pairs"],
        "cold_users":len(users)-training_stats["training_users"],"temporal_checks":temporal,
        "prior_D_14day_map_at_5":prior_d,"independent_sql_ap_values":len(sql),"max_sql_ap_error":max_error,
        "training_stats":training_stats,"user_scores":user_scores,"rankings":rankings,
        "truth":truth,"proxy":proxy,"global_order":global_order,"popularity":popularity,
        "eligible":eligible,"runtime_seconds":round(time.perf_counter()-started,3)}


def main():
    started=time.perf_counter()
    baseline_hashes=raw_integrity()
    strict_summary=load_json("strict80_summary.json")
    con=connect();ensure_metadata(con);cached_events(con)
    users=read_template()
    con.execute("CREATE OR REPLACE TEMP TABLE adv_users(user_id VARCHAR,template_row INTEGER)")
    con.executemany("INSERT INTO adv_users VALUES (?,?)",[(user,i) for i,user in enumerate(users,1)])
    folds=[]
    for fold in FOLDS:
        result=eval_fold(con,users,fold,strict_summary)
        folds.append(result)
        save_json("frequency_no_decay_checkpoint.json",{"completed_folds":[
            {k:v for k,v in f.items() if k not in ("user_scores","rankings","truth","proxy","global_order","popularity","eligible")}
            for f in folds]})
        print(f"Completed no-decay {result['fold']}: MAP@5={result['map_at_5']:.8f}",flush=True)
    # Only the no-decay personal frequency model is newly evaluated in this checkpoint.
    best_repeat="personal_frequency"
    alternative=strict_summary["selected_robust_alternative"]
    example_user=min((r["template_row"] for f in folds for r in f["user_scores"] if r["ap_at_5"]>0),default=None)
    # Use the outer fold's strict truth for a hand-checkable no-decay example.
    outer=next(f for f in folds if f["fold"]=="outer_weeks21_30")
    candidate_examples=[]
    for row in outer["user_scores"]:
        if row["ap_at_5"]>0 and row["ap_at_5"]<1 and row["hits_at_5"]>=2:
            candidate_examples.append(row["template_row"])
    selected_row=min(candidate_examples) if candidate_examples else example_user
    user=users[selected_row-1]
    labels=outer["truth"][user]
    pred=outer["rankings"][user][:5]
    proxy=outer["proxy"][user]
    count_query=con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='outer_weeks21_30_no_decay_train'").fetchone()[0]
    require(count_query==1,"Training-airing cache table missing")
    outer_train="outer_weeks21_30_no_decay_train"
    counts={r[0]:r[1] for r in con.execute(f"SELECT tv_show_id,count(*) FROM {outer_train} WHERE user_id=? GROUP BY tv_show_id",[user]).fetchall()}
    hits=0;numerator=Fraction(0);example_rows=[]
    for rank,item in enumerate(pred,1):
        hit=item in labels
        if hit:hits+=1;numerator+=Fraction(hits,rank)
        example_rows.append({"rank":rank,"tv_show_id":item,"qualifying_airing_count":counts.get(item,0),
            "strict80_relevant":hit})
    denominator=min(5,len(labels));exact=numerator/denominator if denominator else Fraction(0)
    require(float(exact)==average_precision_at_k(labels,pred),"Hand AP@5 mismatch")
    hand_example={"fold":outer["fold"],"template_row":selected_row,"method":"personal_frequency",
        "strict_relevant_programs":len(labels),"proxy60_relevant_programs":len(proxy),
        "recommendations":example_rows,"AP_at_5_calculation":f"{numerator}/{denominator}",
        "AP_at_5_fraction":str(exact),"AP_at_5":float(exact)}
    final_fold={"name":"final_frequency_refit","start":FINAL,"end":"2020-10-19 00:00:00","sources":SCHEDULES[1:]}
    final_train=train_table(con,final_fold)
    final_eligible=candidate_ids(con,FINAL,final_fold["end"],final_fold["sources"])
    final_predictions,final_global,final_pop,final_stats=predict_no_decay(con,users,final_train,final_eligible)
    new_path=RESULTS/"submission_personal_frequency_no_decay_strict80.csv"
    write_new_submission(new_path,users,final_predictions,final_eligible)
    new_check=validate(new_path,users,final_eligible)
    new_check["path"]=str(new_path.relative_to(RESULTS.parent))
    con.close()
    require(raw_integrity()==baseline_hashes,"Original datasets changed")
    proxy_scores={"inner":{r["method"]:float(r["map_at_5"]) for r in csv.DictReader(
        (RESULTS/"inner_holdout_evaluation.csv").open(encoding="utf-8",newline=""))},
        "outer":{r["method"]:r["map_at_5"] for r in load_json("advanced_experiment.json")["comparison"]}}
    strict_maps={f["name"]:{r["method"]:r["map_at_5"] for r in f["comparison"]} for f in strict_summary["folds"]}
    combined=[]
    for fold in folds:
        ref=strict_maps[fold["fold"]]
        combined.append({"fold":fold["fold"],"popularity":ref["popularity"],
            "personal_frequency_no_decay":fold["map_at_5"],"repeat_recency_14d":ref["repeat_recency"],
            "content_based":ref["content_based"],"association_rules":ref["association_rules"],
            "hybrid_rrf60":ref["hybrid_rrf60"],"evaluated_users":1260,"excluded_users":0,
            "candidate_count":fold["candidate_count"],"strict_relevant_pairs":fold["strict_relevant_pairs"],
            "empty_truth_users":fold["empty_truth_users"]})
    result={"new_experiment":"A only: distinct airing-frequency count, no recency decay",
        "B_90day_and_C_30day":"Not run in this checkpoint",
        "D_existing_14day":"Reused from the completed strict80 comparison; not refit",
        "strict_gt50_train_strict_gt80_labels":True,
        "no_decay_MAP_at_5":{"mean":float(np.mean([f["map_at_5"] for f in folds])),
            "by_fold":{f["fold"]:f["map_at_5"] for f in folds}},
        "strict_comparison":combined,"prior_60second_proxy_scores":proxy_scores,
        "frequency_user_example":hand_example,"final_submission":{"method":"personal_frequency_no_decay",
        "path":str(new_path.relative_to(RESULTS.parent)),"validation":new_check,"final_fit":final_stats,
           "candidate_count":len(final_eligible),"kaggle_uploaded":False},
        "inputs_unchanged":True,"runtime_seconds":round(time.perf_counter()-started,3)}
    save_json("frequency_no_decay_results.json",result)
    save_csv("frequency_no_decay_comparison.csv",combined)
    save_json("frequency_user_example.json",hand_example)
    print(json.dumps({"no_decay_MAP_at_5":result["no_decay_MAP_at_5"],
        "strict_comparison":combined,"user_example":hand_example,
      "submission":result["final_submission"],"runtime_seconds":result["runtime_seconds"]},indent=2,default=str),flush=True)


def predict_no_decay_final(con,users,train_table,cutoff,eligible):
    con.execute("CREATE OR REPLACE TEMP TABLE frequency_candidates(tv_show_id BIGINT)")
    con.executemany("INSERT INTO frequency_candidates VALUES (?)",[(int(i),) for i in eligible])
    poprows=rows(con,f"""SELECT c.tv_show_id,coalesce(p.n,0) viewers FROM frequency_candidates c
      LEFT JOIN (SELECT tv_show_id,count(DISTINCT user_id) n FROM {train_table} GROUP BY tv_show_id) p USING(tv_show_id)
      ORDER BY viewers DESC,c.tv_show_id""")
    pop={r["tv_show_id"]:r["viewers"] for r in poprows};global_rank=[r["tv_show_id"] for r in poprows]
    ids=np.array(sorted(eligible),dtype=np.int64);popvec=np.array([pop[int(i)] for i in ids],dtype=np.int64)
    personal={u:{} for u in users}
    for row in rows(con,f"""SELECT a.user_id,a.tv_show_id,count(*) n FROM {train_table} a
      JOIN adv_users u USING(user_id) JOIN frequency_candidates c USING(tv_show_id)
      GROUP BY a.user_id,a.tv_show_id"""):
        personal[row["user_id"]][row["tv_show_id"]]=row["n"]
    prediction=rank_scores(personal,ids,popvec,users)
    return prediction,global_rank,pop,{"train_gt50_airing_occurrences":con.execute(f"SELECT count(*) FROM {train_table}").fetchone()[0],
        "distinct_user_program_airings":con.execute(f"SELECT count(*) FROM (SELECT DISTINCT user_id,tv_show_id,schedule_id FROM {train_table})").fetchone()[0],
        "train_users":con.execute(f"SELECT count(DISTINCT user_id) FROM {train_table}").fetchone()[0],
        "last_session_end":str(con.execute(f"SELECT max(last_session_end) FROM {train_table}").fetchone()[0])}


def write_new_submission(path,users,forecasts,eligible):
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n");writer.writerow(["user_id","tv_show_id"])
        for user in users:
            top=forecasts[user][:5]
            require(len(top)==len(set(top))==5 and set(top)<=eligible and min(top)>0,"No-decay submission invalid")
            writer.writerow([user," ".join(map(str,top))])


if __name__=="__main__":
    main()
