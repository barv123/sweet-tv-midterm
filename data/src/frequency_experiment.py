"""Fast, checkpointed A/B/C personal airing-frequency comparison.

Uses the existing interval cache and strict80 summary. Popularity/D and the content,
association, and hybrid MAP scores are reused from the completed strict-label run; this
script computes only the new A/B/C personal-frequency variants plus a lightweight D
consistency check. No audit, content model, or association model is rebuilt.
"""

import csv
from datetime import date
from fractions import Fraction
import hashlib
import json
import shutil
import time

import numpy as np

from advanced_common import ensure_metadata, load_json
from baseline_experiment import BEGIN, FINAL, raw_integrity, require
from common import RAW, RESULTS, SCHEDULES, connect, literal, one, rows, save_csv, save_json
from metrics import average_precision_at_k
from recommendation_methods import ranked_indices
from validate_submission import candidate_ids, read_template, validate

TRAIN_THRESHOLD=0.50   # a row is a training signal only when fraction > 0.50
LABEL_THRESHOLD=0.80   # a user/item is relevant only when an individual event > 0.80
FOLDS=(
    {"name":"inner_weeks15_16","start":"2020-04-06 00:00:00","end":"2020-04-20 00:00:00","sources":SCHEDULES[:1]},
    {"name":"outer_weeks21_30","start":"2020-05-18 00:00:00","end":"2020-07-27 00:00:00","sources":SCHEDULES[:2]},
)
PERSONAL_METHODS=("personal_frequency","frequency_decay90","frequency_decay30","existing_recency14")
METHODS=("popularity",)+PERSONAL_METHODS+("content_based","association_rules","hybrid_rrf60")


def unique_airing_frequency(schedule_ids):
    return len(set(schedule_ids))


def decayed_frequency(airing_days,cutoff_day,half_life_days):
    if half_life_days<=0: raise ValueError("Half-life must be positive")
    return round(sum(2.0**(-max((cutoff_day-day).days,0)/half_life_days) for day in airing_days),12)


def install_cached_edges(con):
    con.execute("""CREATE OR REPLACE TEMP TABLE frequency_edges AS SELECT
      e.viewing_id,v.user_id,v.channel_id,e.schedule_id,s.start_time AS airing_start,
      s.duration AS airing_seconds,e.tv_show_id,v.start_time AS session_start,v.end_time AS session_end,
      e.overlap_start,e.overlap_seconds,e.overlap_seconds::DOUBLE/s.duration AS coverage_fraction
      FROM match_edges e JOIN viewing v USING(viewing_id)
      JOIN session_quality q USING(viewing_id) JOIN schedule s USING(schedule_id)
      WHERE q.usable AND e.tv_show_id>0 AND s.duration>0 AND e.overlap_seconds>0""")


def build_train_airings(con,fold):
    table=f"{fold['name']}_train_gt50_airings"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE {table} AS SELECT user_id,tv_show_id,schedule_id,
      min(airing_start) airing_start,min(cast(airing_start AS DATE)) airing_day,
      max(session_end) last_session_end,max(overlap_start) overlap_start,
      max(coverage_fraction) max_fraction
      FROM frequency_edges WHERE session_start>=TIMESTAMP '{BEGIN}' AND session_end<=TIMESTAMP '{fold['start']}'
        AND coverage_fraction>{TRAIN_THRESHOLD}
      GROUP BY user_id,tv_show_id,schedule_id""")
    return table


def personal_scores(con,train_table,cutoff,users):
    cutoff_day=cutoff[:10]
    airing_counts=rows(con,f"""SELECT a.user_id,a.tv_show_id,count(*) AS airing_count,
      sum(pow(2.0,-date_diff('day',a.airing_day,DATE '{cutoff_day}')/90.0)) AS decay90,
      sum(pow(2.0,-date_diff('day',a.airing_day,DATE '{cutoff_day}')/30.0)) AS decay30,
      list(a.schedule_id ORDER BY a.schedule_id) AS schedule_ids,
      list(a.airing_day ORDER BY a.schedule_id) AS airing_days
      FROM {train_table} a JOIN adv_users u USING(user_id)
      JOIN freq_candidates c USING(tv_show_id) GROUP BY a.user_id,a.tv_show_id""")
    scores={m:{u:{} for u in users} for m in PERSONAL_METHODS}
    formula_examples=[]
    day=date.fromisoformat(cutoff_day)
    for row in airing_counts:
        user,item=row["user_id"],row["tv_show_id"]
        count=unique_airing_frequency(row["schedule_ids"])
        require(count==row["airing_count"],"Multiple log records inflated a per-airing frequency")
        scores["personal_frequency"][user][item]=float(count)
        scores["frequency_decay90"][user][item]=round(float(row["decay90"]),12)
        scores["frequency_decay30"][user][item]=round(float(row["decay30"]),12)
        if len(formula_examples)<12:
            formula_examples.append({"template_row":next(i for i,u in enumerate(users,1) if u==user),
                "tv_show_id":item,"airing_frequency":count,
                "manual_decay90":decayed_frequency(row["airing_days"],day,90),
                "manual_decay30":decayed_frequency(row["airing_days"],day,30)})
    daily=rows(con,f"""WITH dates AS (
      SELECT DISTINCT e.user_id,e.tv_show_id,cast(e.overlap_start AS DATE) watch_day
      FROM frequency_edges e JOIN adv_users u USING(user_id) JOIN freq_candidates c USING(tv_show_id)
      WHERE e.session_start>=TIMESTAMP '{BEGIN}' AND e.session_end<=TIMESTAMP '{cutoff}'
        AND e.coverage_fraction>{TRAIN_THRESHOLD})
      SELECT user_id,tv_show_id,round(sum(pow(2.0,-date_diff('day',watch_day,DATE '{cutoff_day}')/14.0)),12) score
      FROM dates GROUP BY user_id,tv_show_id""")
    for row in daily:
        scores["existing_recency14"][row["user_id"]][row["tv_show_id"]]=float(row["score"])
    return scores,{"score_formula_pairs_independently_checked":len(formula_examples),"examples":formula_examples,
        "deduplication":"one user/program/schedule_id for A/B/C; one user/program/local overlap-start day for D"}


def rank_scores(scores,candidate_ids,popularity,users,depth=100):
    positions={int(item):i for i,item in enumerate(candidate_ids)}
    rankings={}
    for user in users:
        row=np.zeros(len(candidate_ids),dtype=np.float64)
        for item,score in scores[user].items():
            if int(item) in positions: row[positions[int(item)]]=score
        rankings[user]=candidate_ids[ranked_indices(row,candidate_ids,popularity,depth)].tolist()
    return rankings


def evaluate_fold(con,users,fold,previous_strict):
    started=time.perf_counter()
    train=build_train_airings(con,fold)
    eligible=candidate_ids(con,fold["start"],fold["end"],fold["sources"])
    candidate_ids_array=np.array(sorted(eligible),dtype=np.int64)
    require(len(eligible)>=5 and min(eligible)>0,"Invalid candidate catalog")
    con.execute("CREATE OR REPLACE TEMP TABLE freq_candidates(tv_show_id BIGINT)")
    con.executemany("INSERT INTO freq_candidates VALUES (?)",[(int(i),) for i in candidate_ids_array])
    pop_rows=rows(con,f"""SELECT c.tv_show_id,coalesce(p.n,0) viewers FROM freq_candidates c
      LEFT JOIN (SELECT tv_show_id,count(DISTINCT user_id) n FROM {train} GROUP BY tv_show_id) p USING(tv_show_id)
      ORDER BY viewers DESC,c.tv_show_id""")
    pop={r["tv_show_id"]:r["viewers"] for r in pop_rows}
    global_popularity=[r["tv_show_id"] for r in pop_rows]
    pop_vector=np.array([pop[int(i)] for i in candidate_ids_array],dtype=np.int64)
    scores,score_checks=personal_scores(con,train,fold["start"],users)
    rankings={"popularity":{u:global_popularity[:100] for u in users}}
    for method in PERSONAL_METHODS:
        rankings[method]=rank_scores(scores[method],candidate_ids_array,pop_vector,users)

    truth_table=f"{fold['name']}_gt80_frequency"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE {truth_table} AS SELECT DISTINCT e.user_id,e.tv_show_id
      FROM frequency_edges e JOIN adv_users u USING(user_id)
      WHERE e.session_start>=TIMESTAMP '{fold['start']}' AND e.session_end<=TIMESTAMP '{fold['end']}'
        AND e.coverage_fraction>{LABEL_THRESHOLD}""")
    truth={u:set() for u in users}
    for u,item in con.execute(f"SELECT user_id,tv_show_id FROM {truth_table}").fetchall(): truth[u].add(item)
    proxy={u:set() for u in users}
    for u,item in con.execute(f"""SELECT DISTINCT i.user_id,i.tv_show_id FROM interactions i JOIN adv_users u USING(user_id)
      WHERE i.session_start>=TIMESTAMP '{fold['start']}' AND i.session_end<=TIMESTAMP '{fold['end']}'""").fetchall(): proxy[u].add(item)
    require(all(labels<=eligible for labels in truth.values()),"Strict truth outside schedule candidates")
    maps={}; per_user=[]; comparison=[]
    for method in METHODS:
        if method=="content_based":
            strict_row=next(r for r in previous_strict[fold["name"]] if r["method"]==method)
            maps[method]=strict_row["map_at_5"]
            comparison.append({"fold":fold["name"],"method":method,"strict80_map_at_5":maps[method],
                "users":1260,"excluded_users":0,"strict_pairs":sum(map(len,truth.values())),
                "candidates":len(eligible),"metrics_reused_from_saved_strict80_run":True})
            continue
        if method=="association_rules":
            strict_row=next(r for r in previous_strict[fold["name"]] if r["method"]==method)
            maps[method]=strict_row["map_at_5"]
            comparison.append({"fold":fold["name"],"method":method,"strict80_map_at_5":maps[method],
                "users":1260,"excluded_users":0,"strict_pairs":sum(map(len,truth.values())),
                "candidates":len(eligible),"metrics_reused_from_saved_strict80_run":True})
            continue
        aps=[]
        for rownum,user in enumerate(users,1):
            top=rankings[method][user][:5]
            require(len(top)==len(set(top))==5 and set(top)<=eligible and all(i>0 for i in top),
                f"Invalid strict frequency top five ({fold['name']}/{method})")
            aps.append(average_precision_at_k(truth[user],top))
            per_user.append({"fold":fold["name"],"method":method,"template_row":rownum,
                "strict_relevant_items":len(truth[user]),"ap_at_5":aps[-1],
                "hits_at_5":len(set(top)&truth[user]),"predicted_ids":" ".join(map(str,top))})
        maps[method]=float(np.mean(aps))
        comparison.append({"fold":fold["name"],"method":method,"strict80_map_at_5":maps[method],
            "users":len(users),"excluded_users":0,"strict_pairs":sum(map(len,truth.values())),
            "proxy60_pairs":sum(map(len,proxy.values())),"proxy_only_pairs":sum(len(proxy[u]-truth[u]) for u in users),
            "empty_truth_users":sum(not truth[u] for u in users),"candidates":len(eligible),
            "training_gt50_airing_signals":con.execute(f"SELECT count(*) FROM {train}").fetchone()[0],
            "training_user_program_airings":con.execute(f"SELECT count(*) FROM {train}").fetchone()[0],
            "users_with_no_top5_hit":sum(not(set(rankings[method][u][:5])&truth[u]) for u in users)})
    # Compare newly computed D with its already recorded strict80 score.
    old_d=next(r["map_at_5"] for r in previous_strict[fold["name"]] if r["method"]=="repeat_recency")
    require(abs(maps["existing_recency14"]-old_d)<1e-12,"D did not reproduce the saved strict repeat baseline")
    sql_name=f"{fold['name']}_frequency_check_predictions"
    checked_methods=tuple(PERSONAL_METHODS)+("popularity",)
    con.execute(f"CREATE OR REPLACE TEMP TABLE {sql_name}(method VARCHAR,user_id VARCHAR,rank INTEGER,tv_show_id BIGINT)")
    con.executemany(f"INSERT INTO {sql_name} VALUES (?,?,?,?)",[(m,u,k,i) for m in checked_methods for u in users
        for k,i in enumerate(rankings[m][u][:5],1)])
    sql=rows(con,f"""WITH h AS (SELECT p.*,CASE WHEN t.tv_show_id IS NULL THEN 0 ELSE 1 END hit
      FROM {sql_name} p LEFT JOIN {truth_table} t USING(user_id,tv_show_id)), c AS
      (SELECT *,sum(hit) OVER(PARTITION BY method,user_id ORDER BY rank) cum FROM h),d AS
      (SELECT user_id,count(*) n FROM {truth_table} GROUP BY user_id)
      SELECT c.method,c.user_id,CASE WHEN coalesce(max(d.n),0)=0 THEN 0.0 ELSE
      sum(c.hit*c.cum::DOUBLE/c.rank)/least(5,max(d.n)) END ap FROM c LEFT JOIN d USING(user_id)
      GROUP BY c.method,c.user_id""")
    max_error=max(abs(r["ap"]-average_precision_at_k(truth[r["user_id"]],rankings[r["method"]][r["user_id"]][:5])) for r in sql)
    require(len(sql)==len(checked_methods)*len(users) and max_error<1e-12,"Frequency MAP SQL verification failed")
    return {"fold":fold["name"],"maps":maps,"rankings":rankings,"truth":truth,"proxy":proxy,
        "scores":scores,"personal_rows":per_user,"comparison":comparison,
        "candidate_set":eligible,"candidate_array":candidate_ids_array,"popularity":global_popularity,
        "pop_vector":pop_vector,"train_users":set(r[0] for r in con.execute(f"SELECT DISTINCT user_id FROM {train}").fetchall()),
        "training_rows":con.execute(f"SELECT count(*) FROM {train}").fetchone()[0],
        "training_pairs":con.execute(f"SELECT count(*) FROM (SELECT DISTINCT user_id,tv_show_id FROM {train})").fetchone()[0],
        "cold_users":len(users)-con.execute(f"SELECT count(DISTINCT user_id) FROM {train}").fetchone()[0],
        "strict_pairs":sum(map(len,truth.values())),"proxy_pairs":sum(map(len,proxy.values())),
        "strict_empty_users":sum(not truth[u] for u in users),"sql_checks":len(sql),"max_sql_error":max_error,
        "formula_checks":score_checks,"runtime_seconds":round(time.perf_counter()-start,3)}
