"""Re-evaluate both chronological holdouts using per-airing watch fractions.

Ground-truth label: one usable viewing row covers strictly >80% of one airing.
Training signal: one usable viewing row covers >50% of one airing. Sessions are
never combined across records, programs, channels, or airings. All joins reuse the
cached interval edges and the established equal-numeric-channel/timezone mapping.
"""

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np

from advanced_common import ensure_metadata
from advanced_experiment import collect_rankings, input_matrices
from baseline_experiment import BEGIN, FINAL, raw_integrity, require
from common import RAW, RESULTS, SCHEDULES, TIMEZONE, connect, literal, one, rows, save_csv, save_json
from metrics import average_precision_at_k
from recommendation_methods import (content_features, content_score_blocks, mine_association_rules,
                                    reciprocal_rank_fusion, rule_score_blocks)
from validate_submission import candidate_ids, read_template, validate

TRAIN_SIGNAL_FRACTION = 0.50
GROUND_TRUTH_FRACTION = 0.80
RULES = {"min_pair_count": 25, "min_lift": 1.0, "min_confidence": 0.1,
         "min_item_viewers": 20, "max_rules": 100}
HALF_LIFE_DAYS = 14
FOLDS = [
    {"name": "inner_weeks15_16", "start": "2020-04-06 00:00:00", "end": "2020-04-20 00:00:00",
     "schedule_sources": SCHEDULES[:1]},
    {"name": "outer_weeks21_30", "start": "2020-05-18 00:00:00", "end": "2020-07-27 00:00:00",
     "schedule_sources": SCHEDULES[:2]},
]
METHODS = ("popularity", "repeat_recency", "content_based", "association_rules", "hybrid_rrf60")


def training_signal(overlap_seconds, airing_seconds):
    return airing_seconds > 0 and overlap_seconds > TRAIN_SIGNAL_FRACTION * airing_seconds


def ground_truth_positive(overlap_seconds, airing_seconds):
    return airing_seconds > 0 and overlap_seconds > GROUND_TRUTH_FRACTION * airing_seconds


def prepare_cached_edges(con):
    con.execute("""CREATE OR REPLACE TEMP TABLE strict_edges AS SELECT
        e.viewing_id,v.user_id,v.channel_id,e.schedule_id,s.source AS schedule_source,
        s.source_row AS schedule_source_row,e.tv_show_id,v.start_time AS session_start,
        v.end_time AS session_end,e.overlap_start,e.overlap_end,e.overlap_seconds,
        s.duration AS airing_seconds,e.overlap_seconds::DOUBLE/s.duration AS coverage_fraction
        FROM match_edges e JOIN viewing v USING(viewing_id)
        JOIN session_quality q USING(viewing_id) JOIN schedule s USING(schedule_id)
        WHERE q.usable AND e.tv_show_id>0 AND s.duration>0 AND e.overlap_seconds>0""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE strict_metadata AS
        SELECT * FROM advanced_metadata
        UNION ALL
        SELECT {literal(SCHEDULES[2])} AS source,cast(channel_id AS BIGINT),
          strptime(start_time,'%d.%m.%Y %H:%M:%S') AS start_time,cast(duration AS BIGINT),
          cast(tv_show_id AS BIGINT),tv_show_title,tv_show_category,tv_show_genre_1,
          tv_show_genre_2,tv_show_genre_3,year_of_production,director,actors
        FROM read_csv({literal(RAW / SCHEDULES[2])},header=true,all_varchar=true,
          strict_mode=true,ignore_errors=false) WHERE try_cast(tv_show_id AS BIGINT)>0""")


def strict_history(con, fold, target=False):
    name = fold["name"]
    lower, upper = (fold["start"], fold["end"]) if target else (BEGIN, fold["start"])
    table = f"{name}_{'strict_labels' if target else 'train50'}"
    fraction_op = ">"
    fraction = GROUND_TRUTH_FRACTION if target else TRAIN_SIGNAL_FRACTION
    con.execute(f"""CREATE OR REPLACE TEMP TABLE {table} AS SELECT * FROM strict_edges
        WHERE session_start>=TIMESTAMP '{lower}' AND session_end<=TIMESTAMP '{upper}'
          AND coverage_fraction {fraction_op} {fraction}""")
    return table


def candidates_for_fold(con, fold):
    candidate_set = candidate_ids(con, fold["start"], fold["end"], fold["schedule_sources"])
    require(len(candidate_set) >= 5 and all(int(item) > 0 for item in candidate_set),
            f"No valid candidates for {fold['name']}")
    return np.array(sorted(candidate_set), dtype=np.int64), candidate_set


def fit_fold(con, users, fold):
    tick = time.perf_counter()
    train_table = strict_history(con, fold, target=False)
    label_edges = strict_history(con, fold, target=True)
    label_table = f"{fold['name']}_truth"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE {label_table} AS
        SELECT DISTINCT v.user_id,v.tv_show_id FROM {label_edges} v JOIN adv_users USING(user_id)""")
    candidate_ids_array, candidate_set = candidates_for_fold(con, fold)
    con.execute("CREATE OR REPLACE TEMP TABLE strict_candidates(tv_show_id BIGINT)")
    con.executemany("INSERT INTO strict_candidates VALUES (?)", [(int(x),) for x in candidate_ids_array])
    con.execute(f"CREATE OR REPLACE TEMP TABLE {fold['name']}_candidates(tv_show_id BIGINT)")
    con.executemany(f"INSERT INTO {fold['name']}_candidates VALUES (?)", [(int(x),) for x in candidate_ids_array])
    popularity_rows = rows(con, f"""SELECT c.tv_show_id,coalesce(p.viewers,0) AS viewers
        FROM {fold['name']}_candidates c LEFT JOIN (
          SELECT tv_show_id,count(DISTINCT user_id) AS viewers FROM {train_table} GROUP BY tv_show_id
        ) p USING(tv_show_id) ORDER BY viewers DESC,c.tv_show_id""")
    popularity_lookup = {row["tv_show_id"]: row["viewers"] for row in popularity_rows}
    global_ranking = [row["tv_show_id"] for row in popularity_rows]
    popularity_scores = np.array([popularity_lookup[int(x)] for x in candidate_ids_array], dtype=np.int64)
    train_users = {row[0] for row in con.execute(f"SELECT DISTINCT user_id FROM {train_table}").fetchall()}
    cold_users = set(users)-train_users
    binary, personal, item_ids, candidate_columns, matrix_users = input_matrices(
        con, users, candidate_ids_array, history_table=train_table, recency_cutoff=fold["start"])
    require(matrix_users == train_users, "Strict training user matrix differs from training edge set")

    rankings = {"popularity": {u: global_ranking[:100] for u in users}}
    repeat, repeat_coverage = collect_rankings(((0, personal[:, candidate_columns].toarray()),), users,
        candidate_ids_array, popularity_scores, cold_users, global_ranking)
    rankings["repeat_recency"] = repeat
    metadata = canonical_metadata(con, fold["start"], fold["end"], fold["schedule_sources"])
    item_features, feature_info = content_features(metadata, item_ids, include_title=True)
    content, content_coverage = collect_rankings(content_score_blocks(personal, item_features, candidate_columns),
        users, candidate_ids_array, popularity_scores, cold_users, global_ranking)
    rankings["content_based"] = content
    rules, rule_info = mine_association_rules(binary, item_ids, candidate_columns, **RULES)
    association, association_coverage = collect_rankings(rule_score_blocks(personal, rules, candidate_columns),
        users, candidate_ids_array, popularity_scores, cold_users, global_ranking)
    rankings["association_rules"] = association
    rankings["hybrid_rrf60"] = {u: reciprocal_rank_fusion(
        [association[u][:100],repeat[u][:100],global_ranking[:100]],candidate_ids_array,popularity_scores,constant=60)
        for u in users}

    truth = {u:set() for u in users}
    for user,item in con.execute(f"SELECT user_id,tv_show_id FROM {label_table}").fetchall():
        truth[user].add(item)
    proxy = {u:set() for u in users}
    for user,item in con.execute(f"""SELECT DISTINCT i.user_id,i.tv_show_id FROM interactions i JOIN adv_users u USING(user_id)
        WHERE i.session_start>=TIMESTAMP '{fold['start']}' AND i.session_end<=TIMESTAMP '{fold['end']}'""").fetchall():
        proxy[user].add(item)
    require(all(items <= candidate_set for items in truth.values()), "Strict labels escape target schedule candidates")
    comparison, per_user, metrics = [], [], {}
    aps = {}
    for method in METHODS:
        user_aps = []
        for index,user in enumerate(users,1):
            top = rankings[method][user][:5]
            require(len(top)==len(set(top))==5 and set(top)<=candidate_set,
                    f"Invalid strict recommendation list: {fold['name']} / {method}")
            if user in cold_users:
                require(top==global_ranking[:5],"Cold start does not use common popularity fallback")
            score=average_precision_at_k(truth[user],top)
            user_aps.append(score)
            per_user.append({"fold":fold["name"],"method":method,"template_row":index,
                "relevant_items":len(truth[user]),"proxy_60s_items":len(proxy[user]),"ap_at_5":score,
                "hits_at_5":len(set(top)&truth[user]),"predicted_ids":" ".join(map(str,top))})
        aps[method]=np.array(user_aps)
        metrics[method]={"map_at_5":float(np.mean(user_aps)),
            "nonempty_truth_map_at_5":float(np.mean([user_aps[i] for i,u in enumerate(users) if truth[u]])) if any(truth.values()) else 0.0,
            "cold_user_map_at_5":float(np.mean([user_aps[i] for i,u in enumerate(users) if u in cold_users])) if cold_users else None}
        positive_candidate_count=len(set(int(x) for x in candidate_ids_array if popularity_lookup[int(x)]>0))
        comparison.append({"fold":fold["name"],"method":method,"map_at_5":float(np.mean(user_aps)),
            "users":len(users),"excluded_users":0,"empty_truth_users":sum(not truth[u] for u in users),
            "candidates":len(candidate_set),"strict_relevant_user_item_pairs":sum(len(truth[u]) for u in users),
            "strict_distinct_relevant_items":len(set().union(*truth.values())),
            "proxy_60s_relevant_user_item_pairs":sum(len(proxy[u]) for u in users),
            "strict_only_pairs":sum(len(truth[u]-proxy[u]) for u in users),
            "proxy_only_pairs":sum(len(proxy[u]-truth[u]) for u in users),
            "train50_event_airing_signals":con.execute(f"SELECT count(*) FROM {train_table}").fetchone()[0],
            "train50_distinct_user_item_pairs":con.execute(f"SELECT count(*) FROM (SELECT DISTINCT user_id,tv_show_id FROM {train_table})").fetchone()[0],
            "cold_users":len(cold_users),
            "cold_user_map_at_5":float(np.mean([user_aps[i] for i,u in enumerate(users) if u in cold_users])) if cold_users else None,
            "unique_recommended_items":len({x for u in users for x in rankings[method][u][:5]}),
            "candidates_with_training_popularity":positive_candidate_count,
            "content_feature_info":feature_info if method=="content_based" else None,
            "association_rules_retained":rule_info["retained_rules"] if method in ("association_rules","hybrid_rrf60") else None})
    strict_events=con.execute(f"SELECT count(*) FROM {label_edges}").fetchone()[0]
    proxy_pairs=sum(len(proxy[u]) for u in users)
    strict_pairs=sum(len(truth[u]) for u in users)
    con.execute(f"CREATE OR REPLACE TEMP TABLE {fold['name']}_predictions(method VARCHAR,user_id VARCHAR,rank INTEGER,tv_show_id BIGINT)")
    con.executemany(f"INSERT INTO {fold['name']}_predictions VALUES (?,?,?,?)",[
        (method,user,rank,item) for method in METHODS for user in users
        for rank,item in enumerate(rankings[method][user][:5],1)])
    sql_ap=rows(con,f"""WITH hits AS (
      SELECT p.*,CASE WHEN t.tv_show_id IS NULL THEN 0 ELSE 1 END hit
      FROM {fold['name']}_predictions p LEFT JOIN {label_table} t USING(user_id,tv_show_id)),
      cumulative AS (SELECT *,sum(hit) OVER(PARTITION BY method,user_id ORDER BY rank) cum FROM hits),
      denom AS (SELECT user_id,count(*) n FROM {label_table} GROUP BY user_id)
      SELECT c.method,c.user_id,CASE WHEN coalesce(max(d.n),0)=0 THEN 0.0 ELSE
        sum(c.hit*c.cum::DOUBLE/c.rank)/least(5,max(d.n)) END ap
      FROM cumulative c LEFT JOIN denom d USING(user_id) GROUP BY c.method,c.user_id""")
    max_ap_error=max(abs(r["ap"]-average_precision_at_k(
        truth[r["user_id"]],rankings[r["method"]][r["user_id"]][:5])) for r in sql_ap)
    require(len(sql_ap)==len(METHODS)*len(users) and max_ap_error<1e-12,
            f"Independent SQL AP verification failed for {fold['name']}")
    label_summary={"usable_validation_event_airing_pairs_over_80":strict_events,
        "strict_distinct_user_item_pairs":strict_pairs,"proxy_60s_distinct_user_item_pairs":proxy_pairs,
        "strict_only_user_item_pairs":sum(len(truth[u]-proxy[u]) for u in users),
        "proxy_only_user_item_pairs":sum(len(proxy[u]-truth[u]) for u in users),
        "strict_empty_truth_users":sum(not truth[u] for u in users),
        "strict_label_quantiles":one(con,f"""SELECT quantile_cont(n,[0.1,0.25,0.5,0.75,0.9]) AS q FROM
          (SELECT u.user_id,count(t.tv_show_id) n FROM adv_users u LEFT JOIN {label_table} t USING(user_id) GROUP BY u.user_id)""")["q"]}
    # No future session enters features; whole sessions that cross either boundary are omitted.
    checks={"training_after_cutoff":con.execute(f"SELECT count(*) FROM {train_table} WHERE session_end>TIMESTAMP '{fold['start']}'").fetchone()[0],
        "validation_outside_window":con.execute(f"SELECT count(*) FROM {label_edges} WHERE session_start<TIMESTAMP '{fold['start']}' OR session_end>TIMESTAMP '{fold['end']}'").fetchone()[0],
        "validation_label_outside_candidate":con.execute(f"SELECT count(*) FROM {label_table} WHERE tv_show_id NOT IN (SELECT tv_show_id FROM {fold['name']}_candidates)").fetchone()[0]}
    require(all(v==0 for v in checks.values()),f"Temporal/eligibility check failed: {checks}")
    summary={"fold":fold["name"],"start":fold["start"],"end_exclusive":fold["end"],
        "candidate_count":len(candidate_set),"train_users":len(train_users),"template_users":len(users),
        "train_last_session_end":str(con.execute(f"SELECT max(session_end) FROM {train_table}").fetchone()[0]),
        "training_signal_threshold":">50% of this individual airing in one usable viewing row",
        "label_threshold":">80% of this individual airing in one usable viewing row",
        "no_session_or_airing_aggregation":True,"checks":checks,"labels":label_summary,
        "comparison":comparison,"comparison_rows":comparison,"metrics":metrics,
        "independent_sql_ap":{"values_checked":len(sql_ap),"maximum_absolute_error":max_ap_error},
        "rankings":rankings,"truth":truth,"aps":aps,
        "cold_user_count":len(cold_users),"cold_user_rows":[i for i,u in enumerate(users,1) if u in cold_users],
        "global_ranking":global_ranking,"popularity_scores":popularity_scores,
        "candidate_ids":candidate_ids_array,"item_ids":item_ids,"candidate_columns":candidate_columns,
        "train_table":train_table,"label_edges_table":label_edges,"label_table":label_table,
        "training_rows":con.execute(f"SELECT count(*) FROM {train_table}").fetchone()[0],
        "training_user_item_matrix":binary,"training_recency_matrix":personal,
        "content_features":item_features,"rules_matrix":rules}
    print(f"{fold['name']}: strict labels={strict_pairs:,}, proxy={proxy_pairs:,}, candidates={len(candidate_set):,}",flush=True)
    return summary,per_user


def canonical_metadata(con,cutoff,target_end,sources):
    records=rows(con,f"""WITH available AS (
        SELECT *,CASE WHEN start_time<TIMESTAMP '{cutoff}' THEN 0 ELSE 1 END AS phase
        FROM strict_metadata WHERE duration>0 AND start_time<TIMESTAMP '{target_end}'
          AND source IN ({','.join(literal(s) for s in sources)})), prioritized AS (
        SELECT *,min(phase) OVER (PARTITION BY tv_show_id) AS best_phase FROM available)
        SELECT tv_show_id,min(phase) metadata_period,
          first(source ORDER BY start_time,source,channel_id) source,
          first(title ORDER BY start_time,title,channel_id) title,
          CASE WHEN count(DISTINCT category)=1 THEN min(category) END category,
          CASE WHEN count(DISTINCT (genre1,genre2,genre3))=1 THEN min(genre1) END genre1,
          CASE WHEN count(DISTINCT (genre1,genre2,genre3))=1 THEN min(genre2) END genre2,
          CASE WHEN count(DISTINCT (genre1,genre2,genre3))=1 THEN min(genre3) END genre3,
          CASE WHEN count(DISTINCT production_year)<=1 THEN min(production_year) END production_year,
          CASE WHEN count(DISTINCT director)<=1 THEN min(director) END director
        FROM prioritized WHERE phase=best_phase GROUP BY tv_show_id""")
    return {record["tv_show_id"]:record for record in records}


def direct_raw_label_audit(samples,users,folds,sql_truth):
    """Independently reconstruct full labels for sampled users from original CSVs."""
    from collections import defaultdict
    zone=ZoneInfo(TIMEZONE)
    fold_by_name={fold["name"]:fold for fold in folds}
    fold_users=defaultdict(set)
    for sample in samples:
        fold_users[sample["fold"]].add(sample["user_id"])
    sessions={}
    seen=set()
    with (RAW/"dataset11-30.csv").open(encoding="utf-8-sig",newline="") as handle:
        for rownum,record in enumerate(csv.DictReader(handle),1):
            user=record["user_id"]
            active=[name for name,subjects in fold_users.items() if user in subjects]
            if not active: continue
            for name in active:
                fold=fold_by_name[name]
                if not (fold["start"]<=record["start_time"]<fold["end"] and record["stop_time"]<=fold["end"]): continue
                key=(user,record["vsetv_id"],record["start_time"],record["stop_time"],record["duraton"])
                if key in seen: continue
                seen.add(key)
                start=datetime.strptime(record["start_time"],"%Y-%m-%d %H:%M:%S").replace(tzinfo=zone).astimezone(timezone.utc)
                end=datetime.strptime(record["stop_time"],"%Y-%m-%d %H:%M:%S").replace(tzinfo=zone).astimezone(timezone.utc)
                duration=int(record["duraton"])
                if duration<=0 or duration>86400 or end<=start or int((end-start).total_seconds())!=duration: continue
                sessions[(name,rownum)]={"user":user,"channel":int(record["vsetv_id"]),"start":start,"end":end,"edges":[]}
    channels_by_fold={name:{session["channel"] for (f,_),session in sessions.items() if f==name}
                      for name in fold_by_name}
    buckets=defaultdict(list)
    for key,session in sessions.items():
        day=session["start"].astimezone(zone).date()
        last=(session["end"]-timedelta(seconds=1)).astimezone(zone).date()
        while day<=last:
            buckets[(key[0],session["channel"],day)].append(key)
            day+=timedelta(days=1)
    schedule_files=sorted(set(s for fold in folds for s in fold["schedule_sources"]))
    for filename in schedule_files:
        folds_using=[f for f in folds if filename in f["schedule_sources"]]
        if not folds_using: continue
        active_channels=set().union(*(channels_by_fold[f["name"]] for f in folds_using))
        with (RAW/filename).open(encoding="utf-8-sig",newline="") as handle:
            for source_row,row in enumerate(csv.DictReader(handle),1):
                channel=int(row["channel_id"])
                if channel not in active_channels: continue
                schedule_start=datetime.strptime(row["start_time"],"%d.%m.%Y %H:%M:%S").replace(tzinfo=zone).astimezone(timezone.utc)
                duration=int(row["duration"])
                if duration<=0: continue
                schedule_end=schedule_start+timedelta(seconds=duration)
                local_schedule_start=schedule_start.astimezone(zone).replace(tzinfo=None)
                local_schedule_end=schedule_end.astimezone(zone).replace(tzinfo=None)
                local_day=schedule_start.astimezone(zone).date()
                last_day=(schedule_end-timedelta(seconds=1)).astimezone(zone).date()
                possible=set()
                for fold in folds_using:
                    if local_schedule_start>=datetime.fromisoformat(fold["end"]): continue
                    if local_schedule_end<=datetime.fromisoformat(fold["start"]): continue
                    day=local_day
                    while day<=last_day:
                        possible.update(buckets.get((fold["name"],channel,day),[]))
                        day+=timedelta(days=1)
                for key in possible:
                    session=sessions[key]
                    low=max(session["start"],schedule_start); high=min(session["end"],schedule_end)
                    if low<high:
                        seconds=int((high-low).total_seconds())
                        session["edges"].append({"lower":low,"upper":high,"tv_show_id":int(row["tv_show_id"]),
                            "airing_seconds":duration,"overlap_seconds":seconds,"source":filename,
                            "schedule_source_row":source_row,"channel_id":channel})
    raw_truth={name:{user:set() for user in subjects} for name,subjects in fold_users.items()}
    for (name,rownum),session in sessions.items():
        edges=sorted(session["edges"],key=lambda e:(e["lower"],e["schedule_source_row"]))
        previous_end=None; ambiguous=False
        for edge in edges:
            if previous_end is not None and edge["lower"]<previous_end: ambiguous=True
            previous_end=max(previous_end,edge["upper"]) if previous_end is not None else edge["upper"]
        if not ambiguous:
            for edge in edges:
                if edge["tv_show_id"]>0 and edge["overlap_seconds"]>GROUND_TRUTH_FRACTION*edge["airing_seconds"]:
                    raw_truth[name][session["user"]].add(edge["tv_show_id"])
    checks=[]
    for name,subjects in fold_users.items():
        for user in subjects:
            require(raw_truth[name][user]==sql_truth[name][user],
                f"Independent source-CSV strict labels differ for {name} template user")
    for sample in samples:
        key=(sample["fold"],sample["viewing_id"])
        matching=[edge for edge in sessions[key]["edges"] if edge["source"]==sample["schedule_source"]
                  and edge["schedule_source_row"]==sample["schedule_source_row"]]
        require(len(matching)==1,"Sample airing not uniquely found during independent raw scan")
        edge=matching[0]
        positive=edge["overlap_seconds"]>GROUND_TRUTH_FRACTION*edge["airing_seconds"]
        require(positive==sample["event_positive"],"Raw source event coverage classification differs")
        checks.append({k:v for k,v in sample.items() if k!="user_id"} | {"direct_overlap_seconds":edge["overlap_seconds"],
            "direct_airing_seconds":edge["airing_seconds"],
            "direct_fraction":edge["overlap_seconds"]/edge["airing_seconds"],
            "direct_event_positive":positive,"direct_full_user_label_set_match":True})
    return {"method":"Python csv.DictReader, zoneinfo Europe/Kyiv, exact source-row interval reconstruction",
        "template_users_reconstructed":sum(len(v) for v in fold_users.values()),
        "example_event_checks":checks,"all_sampled_users_full_label_sets_match":True}


def choose_independent_examples(con,folds,template_user_ids):
    samples=[]
    for fold in folds:
        name=fold["name"]
        # One positive airing row per user; one strongest sub-threshold edge per
        # user/item where that pair has no strict-positive airing in the fold.
        positive_rows=rows(con,f"""SELECT e.viewing_id,e.user_id,e.tv_show_id,e.schedule_id,e.schedule_source,
          e.schedule_source_row,e.overlap_seconds,e.airing_seconds,e.coverage_fraction
          FROM strict_edges e JOIN adv_users u USING(user_id)
          WHERE e.session_start>=TIMESTAMP '{fold['start']}' AND e.session_end<=TIMESTAMP '{fold['end']}'
            AND e.coverage_fraction>{GROUND_TRUTH_FRACTION}
          QUALIFY row_number() OVER(PARTITION BY e.user_id ORDER BY e.coverage_fraction DESC,e.viewing_id,e.schedule_id)=1
          ORDER BY e.user_id LIMIT 3""")
        negative_rows=rows(con,f"""SELECT e.viewing_id,e.user_id,e.tv_show_id,e.schedule_id,e.schedule_source,
          e.schedule_source_row,e.overlap_seconds,e.airing_seconds,e.coverage_fraction
          FROM strict_edges e JOIN adv_users u USING(user_id)
          WHERE e.session_start>=TIMESTAMP '{fold['start']}' AND e.session_end<=TIMESTAMP '{fold['end']}'
            AND e.coverage_fraction<= {GROUND_TRUTH_FRACTION}
            AND NOT EXISTS(SELECT 1 FROM {name}_truth t WHERE t.user_id=e.user_id AND t.tv_show_id=e.tv_show_id)
          QUALIFY row_number() OVER(PARTITION BY e.user_id,e.tv_show_id ORDER BY e.coverage_fraction DESC,e.viewing_id,e.schedule_id)=1
          ORDER BY e.coverage_fraction DESC,e.user_id LIMIT 3""")
        for label,collection in ((True,positive_rows),(False,negative_rows)):
            for record in collection:
                if record["user_id"] not in template_user_ids: continue
                samples.append({"fold":name,"template_row":template_user_ids[record["user_id"]],
                    "user_id":record["user_id"],"tv_show_id":record["tv_show_id"],
                    "viewing_id":record["viewing_id"],"schedule_source":record["schedule_source"],
                    "schedule_source_row":record["schedule_source_row"],"overlap_seconds":record["overlap_seconds"],
                    "airing_seconds":record["airing_seconds"],"cached_fraction":record["coverage_fraction"],
                    "event_positive":label,"user_item_label_positive":label})
    require(len(samples)>=8,"Could not find enough positive/negative source rows for independent label checks")
    return samples


def run():
    started=time.perf_counter()
    input_hashes=raw_integrity()
    con=connect()
    ensure_metadata(con)
    prepare_cached_edges(con)
    users=read_template()
    template_user_ids={u:i for i,u in enumerate(users,1)}
    con.execute("CREATE OR REPLACE TEMP TABLE adv_users(user_id VARCHAR,template_row INTEGER)")
    con.executemany("INSERT INTO adv_users VALUES (?,?)",[(u,i) for i,u in enumerate(users,1)])
    fold_outputs=[]; all_per_user=[]; all_examples=[]
    sql_truth={}
    for fold in FOLDS:
        output,per_user=fit_fold(con,users,fold)
        fold_outputs.append(output); all_per_user.extend(per_user)
        sql_truth[fold["name"]]=output["truth"]
    samples=choose_independent_examples(con,FOLDS,template_user_ids)
    direct=direct_raw_label_audit(samples,users,FOLDS,sql_truth)
    # Choose the non-repeat alternative with the strongest worst-fold MAP, fixed rule before submission refit.
    by_fold={o["fold"]:o for o in fold_outputs}
    alternatives=("popularity","content_based","association_rules","hybrid_rrf60")
    alternative=sorted(alternatives,key=lambda m:(-min(float(by_fold[f]["metrics"][m]["map_at_5"]) for f in by_fold),
        -np.mean([by_fold[f]["metrics"][m]["map_at_5"] for f in by_fold]),m))[0]
    con.execute("CHECKPOINT")
    con.close()
    # Final refit uses all complete pre-final sessions; only per-session >=50%-airing
    # coverage events enter train signals. No final viewing labels exist or are read.
    final_fold={"name":"final_refit","start":FINAL,"end":"2020-10-19 00:00:00","schedule_sources":SCHEDULES[1:]}
    con=connect(); prepare_cached_edges(con)
    con.execute("CREATE OR REPLACE TEMP TABLE adv_users(user_id VARCHAR,template_row INTEGER)")
    con.executemany("INSERT INTO adv_users VALUES (?,?)",[(u,i) for i,u in enumerate(users,1)])
    final_train_table=strict_history(con,final_fold,target=False)
    final_candidate_array,final_candidates=candidates_for_fold(con,final_fold)
    con.execute("CREATE OR REPLACE TEMP TABLE strict_candidates(tv_show_id BIGINT)")
    con.executemany("INSERT INTO strict_candidates VALUES (?)",[(int(x),) for x in final_candidate_array])
    final_predictions,final_info=fit_final(con,users,final_fold,final_train_table,
        final_candidate_array,final_candidates)
    repeat_path=RESULTS/"submission_strict80_repeat_recency.csv"
    alternative_path=RESULTS/f"submission_strict80_alternative_{alternative}.csv"
    write_submission(repeat_path,users,final_predictions["repeat_recency"],final_candidates)
    write_submission(alternative_path,users,final_predictions[alternative],final_candidates)
    repeat_check=validate(repeat_path,users,final_candidates)
    alternative_check=validate(alternative_path,users,final_candidates)
    repeat_check["path"]=str(repeat_path.relative_to(RESULTS.parent))
    alternative_check["path"]=str(alternative_path.relative_to(RESULTS.parent))
    repeat_exposure={}
    for method,path in (("repeat_recency",repeat_path),(alternative,alternative_path)):
        with path.open(encoding="utf-8",newline="") as h:
            records=list(csv.DictReader(h))
        pairs=[(record["user_id"],int(item)) for record in records for item in record["tv_show_id"].split()]
        con.execute("CREATE OR REPLACE TEMP TABLE strict_final_predictions(user_id VARCHAR,tv_show_id BIGINT)")
        con.executemany("INSERT INTO strict_final_predictions VALUES (?,?)",pairs)
        repeat_exposure[method]=one(con,f"""SELECT count(*) AS total_slots,
          count(*) FILTER(WHERE EXISTS(SELECT 1 FROM {final_train_table} t WHERE t.user_id=p.user_id AND t.tv_show_id=p.tv_show_id)) AS seen_slots
          FROM strict_final_predictions p""")
    con.close()
    require(raw_integrity()==input_hashes,"Raw CSV changed")
    result={"source_protocol":{"participant_article_url":"https://medium.com/data-science/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156",
        "participant_claim":"A program is relevant after >80% of one airing is watched in one sitting without changing channel.",
        "official_status":"Kaggle task page is identified, but detailed scoring/rules page could not be retrieved; the 80% threshold is not confirmed from official rules.",
        "participant_training_method":"The author separately describes >50% of program duration in one sitting for a basic training recommender.",
        "ground_truth_rule":">80% of individual airing duration, strict inequality, for one usable viewing row on one channel; never sum sessions or airings.",
        "training_signal_rule":">50% of individual airing duration per usable viewing row; separate from relevance labels.",
        "channel_switch_interpretation":"one source viewing row has one vsetv_id; no sessions are merged. Intra-row channel-change telemetry is unavailable."},
        "folds":[{k:v for k,v in o.items() if k not in ("rankings","truth","aps","candidate_ids","popularity_scores","training_user_item_matrix","training_recency_matrix","content_features","rules_matrix")} for o in fold_outputs],
        "independent_raw_label_verification":direct,
        "submission_selection":{"repeat_strategy":"repeat_recency","alternative_strategy":alternative,
            "alternative_selection":"best worst-fold strict-80 MAP@5 among popularity/content/association/hybrid; no new hyperparameter tuning",
            "final_training_threshold":">50% per single airing/session","candidate_count":len(final_candidates),
             "final_training_signals":final_info,"candidate_files":{
                "repeat":{"path":str(repeat_path.relative_to(RESULTS.parent)),"validation":repeat_check},
                "alternative":{"path":str(alternative_path.relative_to(RESULTS.parent)),"method":alternative,"validation":alternative_check},
                "previously_seen_prediction_slots":repeat_exposure},
            "uploaded_to_kaggle":False},
        "unchanged_raw_hashes":True,"runtime_seconds":round(time.perf_counter()-started,3)}
    save_csv("strict80_comparison.csv",[row for fold in fold_outputs for row in fold["comparison_rows"]])
    save_csv("strict80_per_user.csv",all_per_user)
    save_json("strict80_label_examples.json",direct)
    save_json("strict80_evaluation.json",result)
    print_summary(result)


def canonical_metadata(con,cutoff,target_end,sources):
    records=rows(con,f"""WITH available AS (
      SELECT *,CASE WHEN start_time<TIMESTAMP '{cutoff}' THEN 0 ELSE 1 END phase
      FROM strict_metadata WHERE source IN ({','.join(literal(s) for s in sources)})
        AND duration>0 AND start_time<TIMESTAMP '{target_end}'), prioritized AS (
      SELECT *,min(phase) OVER(PARTITION BY tv_show_id) best_phase FROM available)
      SELECT tv_show_id,min(phase) metadata_period,
       CASE WHEN count(DISTINCT category)=1 THEN min(category) END category,
       CASE WHEN count(DISTINCT (genre1,genre2,genre3))=1 THEN min(genre1) END genre1,
       CASE WHEN count(DISTINCT (genre1,genre2,genre3))=1 THEN min(genre2) END genre2,
       CASE WHEN count(DISTINCT (genre1,genre2,genre3))=1 THEN min(genre3) END genre3,
       CASE WHEN count(DISTINCT production_year)<=1 THEN min(production_year) END production_year,
       CASE WHEN count(DISTINCT director)<=1 THEN min(director) END director,
       first(title ORDER BY start_time,title,channel_id) title
      FROM prioritized WHERE phase=best_phase GROUP BY tv_show_id""")
    return {r["tv_show_id"]:r for r in records}


def fit_final(con,users,fold,train_table,candidate_array,candidate_set):
    return fit_model_set(con,users,fold,train_table,candidate_array,candidate_set)


def write_submission(path,users,rankings,candidates):
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n"); writer.writerow(["user_id","tv_show_id"])
        for user in users:
            items=rankings[user][:5]
            require(len(items)==len(set(items))==5 and set(items)<=candidates and min(items)>0,
                "Invalid strict-label candidate prediction")
            writer.writerow([user," ".join(map(str,items))])


def print_summary(result):
    import json
    print(json.dumps({"participant_rule":result["source_protocol"],"folds":result["folds"],
        "independent_raw_examples":len(result["independent_raw_label_verification"]["example_event_checks"]),
        "submissions":result["submission_selection"],"runtime_seconds":result["runtime_seconds"]},indent=2,default=str),flush=True)


def fit_model_set(con,users,fold,train_table,candidate_array,candidate_set):
    # Strict train interaction schema remains compatible with the shared sparse builder.
    poprows=rows(con,f"""SELECT c.tv_show_id,coalesce(p.n,0) viewers FROM strict_candidates c LEFT JOIN
      (SELECT tv_show_id,count(DISTINCT user_id) n FROM {train_table} GROUP BY tv_show_id) p USING(tv_show_id)
      ORDER BY viewers DESC,c.tv_show_id""")
    pop={r["tv_show_id"]:r["viewers"] for r in poprows}; global_rank=[r["tv_show_id"] for r in poprows]
    popscores=np.array([pop[int(i)] for i in candidate_array],dtype=np.int64)
    binary,personal,item_ids,candidate_cols,train_users=input_matrices(con,users,candidate_array,
        history_table=train_table,recency_cutoff=fold["start"])
    cold=set(users)-train_users
    repeat,repeat_cov=collect_rankings(((0,personal[:,candidate_cols].toarray()),),users,candidate_array,popscores,cold,global_rank)
    metadata=canonical_metadata(con,fold["start"],fold["end"],fold["schedule_sources"])
    feats,feature_info=content_features(metadata,item_ids,include_title=True)
    content,content_cov=collect_rankings(content_score_blocks(personal,feats,candidate_cols),users,candidate_array,popscores,cold,global_rank)
    rules,rule_info=mine_association_rules(binary,item_ids,candidate_cols,min_pair_count=25,min_lift=1.0,
        min_confidence=0.1,min_item_viewers=20,max_rules=100)
    assoc,assoc_cov=collect_rankings(rule_score_blocks(personal,rules,candidate_cols),users,candidate_array,popscores,cold,global_rank)
    hybrid={u:reciprocal_rank_fusion([assoc[u][:100],repeat[u][:100],global_rank[:100]],candidate_array,popscores,constant=60)
        for u in users}
    rankings={"popularity":{u:global_rank[:100] for u in users},"repeat_recency":repeat,
        "content_based":content,"association_rules":assoc,"hybrid_rrf60":hybrid}
    measures={"popularity":sum(n>0 for n in pop.values()),"repeat_recency":repeat_cov["candidates_with_positive_evidence"],
        "content_based":content_cov["candidates_with_positive_evidence"],"association_rules":assoc_cov["candidates_with_positive_evidence"],
        "hybrid_rrf60":len({item for m in (repeat,content,assoc) for u in users for item in m[u][:100]})}
    return rankings, {"train_signal_rows":con.execute(f"SELECT count(*) FROM {train_table}").fetchone()[0],
        "train_distinct_user_item_pairs":binary.nnz,"train_users":len(train_users),"cold_template_users":len(cold),
        "rules_retained":rule_info["retained_rules"],"candidate_evidence_counts":measures}


if __name__=="__main__":
    run()
