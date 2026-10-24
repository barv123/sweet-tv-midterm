"""Package existing measured evidence for the report; no model fitting/search."""
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil

from common import RESULTS, SCHEDULES, connect, literal, save_json
from final_competitive_sprint import load_state
from metrics import average_precision_at_k
from validate_submission import candidate_ids, read_template, validate

ROOT = RESULTS.parent
ctx = next(c for c in load_state() if c["fold"]["name"] == "outer_weeks21_30")
users = read_template()
# Existing illustrative repeat viewer and the first cold template row; no label-based selection.
selected = [2, next(i for i, n in enumerate(ctx["total"]) if n == 0)]
methods = ("popularity", "repeat_recency", "content_based", "association_rules", "personal_frequency", "champion")
predictions = {m: {} for m in methods}
with (RESULTS / "strict80_per_user.csv").open(encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        i = int(row["template_row"])-1
        if row["fold"] == ctx["fold"]["name"] and i in selected and row["method"] in predictions:
            predictions[row["method"]][i] = list(map(int, row["predicted_ids"].split()))
for i in selected:
    predictions["personal_frequency"][i] = ctx["base"][users[i]][:5]
    predictions["champion"][i] = ctx["champion"][users[i]]
items = sorted({item for p in predictions.values() for v in p.values() for item in v})
con = connect()
metadata = {r[0]: {"title": r[1], "category": r[2]} for r in con.execute(f"""SELECT tv_show_id,
    first(title ORDER BY CASE WHEN start_time<'2020-05-18' THEN 0 ELSE 1 END,start_time,source,channel_id),
    first(category ORDER BY CASE WHEN start_time<'2020-05-18' THEN 0 ELSE 1 END,start_time,source,channel_id)
    FROM advanced_metadata WHERE tv_show_id IN ({','.join(map(str, items))})
    AND start_time<'2020-07-27' GROUP BY tv_show_id""").fetchall()}
examples = []
for i in selected:
    user = users[i]; truth = ctx["truth"][user]
    record = {"template_row": i+1, "selection": "existing repeat-viewer example" if i==2 else "first cold user by template order",
        "training_airings": int(ctx["total"][i]), "relevant_programs": len(truth), "methods": {}}
    for method in methods:
        top = predictions[method][i]
        hits = 0; numerator = Fraction(0)
        for k, item in enumerate(top, 1):
            if item in truth:
                hits += 1; numerator += Fraction(hits, k)
        ap = numerator / min(5, len(truth)) if truth else Fraction(0)
        assert abs(float(ap)-average_precision_at_k(truth, top)) < 1e-12
        record["methods"][method] = {"ap_at_5": float(ap), "exact_ap": str(ap), "hits": hits,
            "recommendations": [{"rank":k,"tv_show_id":item,**metadata[item],"relevant":item in truth,
                "seen_in_training": bool(ctx["frequency"][i, list(ctx["ids"]).index(item)]>0)} for k,item in enumerate(top,1)]}
    examples.append(record)
winner = RESULTS / "submission_frequency_conservative_B_rules_003_strict80.csv"
validation = validate(winner, users, candidate_ids(con, "2020-07-27", "2020-10-19", SCHEDULES[1:]))
con.close()
validation["path"] = winner.relative_to(ROOT).as_posix()
source = Path.home() / "Pictures" / "Screenshots" / "Снимок экрана 2026-10-24 173140.png"
destination = ROOT / "reports" / "kaggle_verified_scores.png"
shutil.copyfile(source, destination)
assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(destination.read_bytes()).digest()
save_json("exam_report_evidence.json", {"fold":"outer_weeks21_30","examples":examples,
    "submission_validation":validation,"screenshot":{"path":destination.relative_to(ROOT).as_posix(),
    "original_filename":source.name,"sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),"unmodified":True},
    "kaggle_submissions":[
        {"model":"Repeat/recency","private":.31000,"public":.33374,"file":"submission_strict80_repeat_recency.csv"},
        {"model":"Frequency, no decay","private":.40179,"public":.41695,"file":"submission_personal_frequency_no_decay_strict80.csv"},
        {"model":"Distinct viewing days","private":.39486,"public":.41768,"file":"submission_frequency_targeted_distinct_days_strict80.csv"},
        {"model":"Frequency + association .03","private":.40943,"public":.42309,"file":winner.name}]})
print(json.dumps({"examples":[{"row":e["template_row"],"training_airings":e["training_airings"],"truth":e["relevant_programs"],
    "AP":{m:v["ap_at_5"] for m,v in e["methods"].items()}} for e in examples],"validation":validation},indent=2))
