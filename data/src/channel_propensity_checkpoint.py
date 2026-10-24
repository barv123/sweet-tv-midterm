"""Fixed channel-affinity and historical-opportunity experiments; cached inputs only."""
import argparse
import csv
import json
import time
from datetime import datetime

import numpy as np

from advanced_common import load_json
from baseline_experiment import FINAL, require
from common import RESULTS, SCHEDULES, TIMEZONE, connect, literal, save_csv, save_json
from final_competitive_sprint import load_state, fuse, features
from frequency_conservative_checkpoint import digest, final_association, summarize
from personal_frequency_no_decay import cached_events, train_table, predict_no_decay, write_new_submission
from recommendation_methods import ranked_indices
from validate_submission import candidate_ids, read_template, validate

METRICS = "channel_propensity_checkpoint.json"
SETTINGS = {"channel_005": ("channel", .05), "channel_010": ("channel", .10),
            "propensity_005": ("propensity", .05), "propensity_010": ("propensity", .10)}


def install(con, users):
    cached_events(con)
    con.execute("CREATE OR REPLACE TEMP TABLE adv_users(user_id VARCHAR,template_row INTEGER)")
    con.executemany("INSERT INTO adv_users VALUES (?,?)", [(u,i) for i,u in enumerate(users,1)])


def new_features(con, ctx):
    tick=time.perf_counter(); fold=ctx["fold"]; cutoff=literal(fold["start"])
    users=ctx["users"]; ids=ctx["ids"]
    up={u:i for i,u in enumerate(users)}; ip={int(v):i for i,v in enumerate(ids)}
    con.execute(f"""CREATE OR REPLACE TEMP TABLE cp_train AS
        SELECT e.* FROM freq_edges e JOIN adv_users USING(user_id)
        WHERE session_start>='2020-03-09 00:00:00' AND session_end<=TIMESTAMP {cutoff}
        AND coverage_fraction>.50""")
    mismatches=con.execute("SELECT count(*) FROM cp_train e JOIN schedule s USING(schedule_id) WHERE e.channel_id<>s.channel_id OR e.channel_id IS NULL OR s.channel_id IS NULL").fetchone()[0]
    require(mismatches==0,"Unreliable numeric channel mapping; stop channel feature")
    latest=con.execute("SELECT max(session_end) FROM cp_train").fetchone()[0]
    require(str(latest)<=fold["start"],"Future viewing in features")
    channel_rows=con.execute("SELECT user_id,channel_id,count(DISTINCT schedule_id) FROM cp_train GROUP BY user_id,channel_id").fetchall()
    target_rows=con.execute(f"""SELECT tv_show_id,channel_id,count(DISTINCT (start_time,duration)) FROM schedule
        WHERE tv_show_id>0 AND duration>0 AND source IN ({','.join(literal(s) for s in fold['sources'])})
        AND start_time<TIMESTAMP {literal(fold['end'])}
        AND timezone('{TIMEZONE}',timezone('{TIMEZONE}',start_time)+duration*INTERVAL '1 second')>TIMESTAMP {cutoff}
        GROUP BY tv_show_id,channel_id""").fetchall()
    require({r[0] for r in target_rows}==set(ids),"Target metadata candidate mismatch")
    channels=sorted({r[1] for r in channel_rows}|{r[1] for r in target_rows})
    cp={c:i for i,c in enumerate(channels)}
    history=np.zeros((len(users),len(channels)))
    mapping=np.zeros((len(channels),len(ids)))
    for u,c,n in channel_rows: history[up[u],cp[c]]=n
    for item,c,n in target_rows: mapping[cp[c],ip[item]]=n
    mapping/=np.maximum(mapping.sum(axis=0),1)
    affinity=(history/np.maximum(history.sum(axis=1,keepdims=True),1)) @ mapping
    affinity/=np.maximum(affinity.max(axis=1,keepdims=True),1e-12)
    # Opportunity denominators contain only fully completed historical airings.
    con.execute(f"""CREATE OR REPLACE TEMP TABLE cp_schedule AS
        SELECT * FROM schedule WHERE tv_show_id>0 AND duration>0
        AND source IN ({','.join(literal(s) for s in SCHEDULES[:2])})
        AND start_time>=TIMESTAMP '2020-03-09 00:00:00'
        AND timezone('{TIMEZONE}',timezone('{TIMEZONE}',start_time)+duration*INTERVAL '1 second')<=TIMESTAMP {cutoff}""")
    days=(datetime.fromisoformat(fold["start"])-datetime(2020,3,9)).days
    coverage=dict(con.execute("SELECT channel_id,count(DISTINCT CAST(start_time AS DATE)) FROM cp_schedule GROUP BY channel_id").fetchall())
    reliable_channels={c for c,n in coverage.items() if n>=.90*days}
    schedule_stats=con.execute("""SELECT tv_show_id,count(DISTINCT (channel_id,start_time,duration)),list(DISTINCT channel_id)
        FROM cp_schedule GROUP BY tv_show_id""").fetchall()
    opportunities={item:n for item,n,ch in schedule_stats if item in ip and n>=10 and set(ch)<=reliable_channels}
    watched=np.zeros_like(ctx["frequency"],dtype=float)
    for u,item,n in con.execute("""SELECT e.user_id,e.tv_show_id,count(DISTINCT (s.channel_id,s.start_time,s.duration))
        FROM cp_train e JOIN cp_schedule s USING(schedule_id) GROUP BY e.user_id,e.tv_show_id""").fetchall():
        if item in opportunities: watched[up[u],ip[item]]=n
    propensity=np.zeros_like(watched)
    for item,n in opportunities.items():
        j=ip[item]
        require(np.all(watched[:,j]<=n),"Watch count exceeds historical opportunities")
        # Beta(1,19) prior; only personal repeat candidates receive this bonus.
        propensity[:,j]=np.where(ctx["frequency"][:,j]>0,(watched[:,j]+1)/(n+20),0)
    propensity/=np.maximum(propensity.max(axis=1,keepdims=True),1e-12)
    evidence={"channel_id_mismatches":mismatches,"last_training_session_end":str(latest),
        "cutoff":fold["start"],"historical_days":days,"channels":len(channels),
        "reliable_historical_channels":len(reliable_channels),"reliable_propensity_candidates":len(opportunities),
        "eligible_candidates":len(ids),"users_with_channel_signal":int(np.sum(history.sum(axis=1)>0)),
        "users_with_propensity_signal":int(np.sum(propensity.max(axis=1)>0)),
        "feature_seconds":round(time.perf_counter()-tick,3)}
    return {"channel":affinity,"propensity":propensity},evidence


def predictions(ctx, feature, weight):
    output={}
    for i,u in enumerate(ctx["users"]):
        f=ctx["frequency"][i]
        # Both hypotheses retain frequency as the dominant personal signal.
        scores=f+weight*float(f.max())*feature[i]
        ranking=ctx["ids"][ranked_indices(scores,ctx["ids"],ctx["pop"],100)].tolist()
        output[u]=fuse(ranking,ctx["association"][u],ctx["ids"],ctx["pop"])
    return output


def evaluate():
    started=time.perf_counter(); contexts=load_state(); users=read_template()
    require(all(ctx["users"]==users for ctx in contexts),"Template cohort differs from cache")
    expected=load_json("frequency_conservative_checkpoint.json")
    saved={ctx["fold"]["name"]:{} for ctx in contexts}
    with (RESULTS/"frequency_conservative_per_user.csv").open(encoding="utf-8",newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"]=="B_rules_003": saved[row["fold"]][users[int(row["template_row"])-1]]=list(map(int,row["predicted_ids"].split()))
    result={"settings":SETTINGS,"formula":"F + weight*max_user_eligible_F*normalized_feature; unchanged association RRF60 .03",
        "channel_feature":"pre-cutoff >50% distinct-airing channel shares mapped via prediction-period schedule channel proportions; per-user max normalization",
        "propensity_feature":"(watched completed historical airings+1)/(completed historical opportunities+20), normalized per user; only seen programs; >=10 opportunities, >=90% historical-day channel coverage",
        "selection":"inner gain >=.002, outer gain >=.005; both folds >=50 gainers, gainers>losers, positive mean excluding top10 gainers",
        "limitations":["Historical channel-day presence is a completeness proxy, not proof of complete EPG coverage.",
            "Schedule availability does not prove each user had channel access or was available to watch.",
            "Prediction-period channel mappings use exogenous schedule metadata, never future viewing.",
            "Program/channel metadata inherits the established numeric-ID mapping; unseen channels have zero personal affinity."],
        "protected_hashes":{p.name:digest(p) for p in RESULTS.glob("submission*.csv")},"control":[],"features":[],"experiments":{}}
    con=connect(); install(con,users); arrays=[]
    for ctx,old in zip(contexts,expected["folds"]):
        require(candidate_ids(con,ctx["fold"]["start"],ctx["fold"]["end"],ctx["fold"]["sources"])==set(ctx["ids"]),"Cached candidates changed")
        control=predictions(ctx,np.zeros_like(ctx["frequency"]),0)
        require(control==ctx["champion"]==saved[ctx["fold"]["name"]],"Champion rankings not reproduced")
        metric,_,_=summarize(users,ctx["truth"],control,control)
        require(abs(metric["map_at_5"]-old["metrics"]["B_rules_003"]["map_at_5"])<1e-12,"Champion MAP mismatch")
        result["control"].append({"fold":ctx["fold"]["name"],"map_at_5":metric["map_at_5"],"exact_predictions_reproduced":True})
        feats,evidence=new_features(con,ctx); arrays.append(feats)
        result["features"].append({"fold":ctx["fold"]["name"],**evidence})
        save_json(METRICS,result)
        print(json.dumps({"control":result["control"][-1],"features":result["features"][-1]}),flush=True)
    con.close()
    for name,(kind,weight) in SETTINGS.items():
        tick=time.perf_counter(); records=[]; paired=[]
        for ctx,feats in zip(contexts,arrays):
            pred=predictions(ctx,feats[kind],weight)
            require(all(len(v)==len(set(v))==5 and min(v)>0 and set(v)<=set(ctx["ids"]) for v in pred.values()),"Malformed predictions")
            metric,before,after=summarize(users,ctx["truth"],ctx["champion"],pred)
            metric["fold"]=ctx["fold"]["name"]
            metric["membership_change_percent"]=100*metric["top5_membership_changed_users"]/len(users)
            metric["reorder_only_users"]=metric["top5_changed_users"]-metric["top5_membership_changed_users"]
            records.append(metric)
            paired.extend({"fold":metric["fold"],"template_row":i+1,"champion_ap":float(before[i]),
                "ap":float(after[i]),"delta":float(after[i]-before[i]),"predicted_ids":" ".join(map(str,pred[u]))} for i,u in enumerate(users))
        result["experiments"][name]={"folds":records,"scoring_seconds":round(time.perf_counter()-tick,3)}
        save_csv(f"channel_propensity_{name}_per_user.csv",paired)
        save_json(METRICS,result)
        print(json.dumps({"method":name,**result["experiments"][name]}),flush=True)
    result["evaluation_seconds"]=round(time.perf_counter()-started,3)
    save_json(METRICS,result)


def select():
    tick=time.perf_counter(); result=load_json(METRICS); qualified=[]; table=[]
    for name,experiment in result["experiments"].items():
        inner,outer=experiment["folds"]
        table.extend({"method":name,**{k:v for k,v in m.items() if not isinstance(v,dict)}} for m in (inner,outer))
        if inner["paired_mean_delta"]>=.002 and outer["paired_mean_delta"]>=.005 and all(
            m["improved_users"]>=50 and m["improved_users"]>m["worsened_users"] and m["mean_delta_excluding_top10_gainers"]>0 for m in (inner,outer)):
            qualified.append(name)
    winner=max(qualified,key=lambda n:result["experiments"][n]["folds"][1]["map_at_5"]) if qualified else None
    result["winner"]=winner; result["new_submission"]=None
    save_csv("channel_propensity_comparison.csv",table); save_json(METRICS,result)
    if winner:
        con=connect(); users=read_template(); install(con,users)
        fold={"name":"channel_propensity_final","start":FINAL,"end":"2020-10-19 00:00:00","sources":SCHEDULES[1:]}
        training=train_table(con,fold); eligible=candidate_ids(con,FINAL,fold["end"],fold["sources"])
        base,_,pop,_=predict_no_decay(con,users,training,eligible)
        association,_=final_association(con,users,fold,np.array(sorted(eligible)),pop)
        ctx=features(con,users,fold,{"eligible":eligible,"rankings":base,"popularity":pop},association)
        with (RESULTS/"submission_frequency_conservative_B_rules_003_strict80.csv").open(encoding="utf-8",newline="") as handle:
            champion={r["user_id"]:list(map(int,r["tv_show_id"].split())) for r in csv.DictReader(handle)}
        require(ctx["champion"]==champion,"Final champion reproduction failed")
        feats,evidence=new_features(con,ctx); kind,weight=SETTINGS[winner]
        pred=predictions(ctx,feats[kind],weight)
        path=RESULTS/f"submission_channel_propensity_{winner}_strict80.csv"
        require(not path.exists(),"Refusing to overwrite an existing CSV")
        write_new_submission(path,users,pred,eligible)
        result["new_submission"]=validate(path,users,eligible); result["final_features"]=evidence
        con.close()
    require(all(digest(RESULTS/name)==value for name,value in result["protected_hashes"].items()),"Prior submission modified")
    result["prior_submissions_preserved"]=True
    result["selection_refit_seconds"]=round(time.perf_counter()-tick,3)
    result["runtime_seconds"]=result["evaluation_seconds"]+result["selection_refit_seconds"]
    save_json(METRICS,result)
    print(json.dumps({k:result[k] for k in ("winner","new_submission","runtime_seconds","prior_submissions_preserved")}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=("evaluate","select"))
    stage=parser.parse_args().stage
    if stage=="evaluate": evaluate()
    else: select()
