"""Audit the high baseline score without rebuilding or overwriting baseline artifacts."""

import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import json
import re
import time
from zoneinfo import ZoneInfo

from advanced_common import ensure_metadata, prepare, protected_hashes
from baseline_experiment import BEGIN, CUT, FINAL, raw_integrity, require
from common import RAW, RESULTS, SCHEDULES, TIMEZONE, connect, literal, one, rows, save_csv, save_json
from metrics import average_precision_at_k, mean_average_precision_at_k


def recompute_raw_users(users, selected, truth, predictions):
    """Independent CSV-only label reconstruction and exact rational AP for real users."""
    selected_users = {users[index - 1] for index in selected}
    zone = ZoneInfo(TIMEZONE)
    records, buckets, duplicates = {}, defaultdict(list), set()
    per_user_records = defaultdict(int)
    with (RAW / "dataset11-30.csv").open(encoding="utf-8-sig", newline="") as handle:
        for source_row, row in enumerate(csv.DictReader(handle), 1):
            user = row["user_id"]
            if user not in selected_users or not (CUT <= row["start_time"] < FINAL) or row["stop_time"] > FINAL:
                continue
            key = tuple(row.values())
            if key in duplicates:
                continue
            duplicates.add(key)
            start = datetime.fromisoformat(row["start_time"]).replace(tzinfo=zone).astimezone(timezone.utc)
            end = datetime.fromisoformat(row["stop_time"]).replace(tzinfo=zone).astimezone(timezone.utc)
            duration = int(row["duraton"])
            if not 0 < duration <= 86400 or (end-start).total_seconds() != duration:
                continue
            record = {"user": user, "channel": int(row["vsetv_id"]), "start": start, "end": end, "edges": []}
            records[source_row] = record
            per_user_records[user] += 1
            day = start.astimezone(zone).date()
            last = (end-timedelta(seconds=1)).astimezone(zone).date()
            while day <= last:
                buckets[(record["channel"], day)].append(source_row)
                day += timedelta(days=1)
    channels = {key[0] for key in buckets}
    for filename in SCHEDULES[:2]:
        with (RAW / filename).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                channel = int(row["channel_id"])
                if channel not in channels or int(row["duration"]) <= 0:
                    continue
                start = datetime.strptime(row["start_time"], "%d.%m.%Y %H:%M:%S").replace(tzinfo=zone).astimezone(timezone.utc)
                end = start + timedelta(seconds=int(row["duration"]))
                day = start.astimezone(zone).date()
                last = (end-timedelta(seconds=1)).astimezone(zone).date()
                candidate_rows = set()
                while day <= last:
                    candidate_rows.update(buckets.get((channel, day), []))
                    day += timedelta(days=1)
                for source_row in candidate_rows:
                    record = records[source_row]
                    lower, upper = max(start, record["start"]), min(end, record["end"])
                    if lower < upper:
                        record["edges"].append((lower, upper, int(row["tv_show_id"])))
    labels = {user: set() for user in selected_users}
    for record in records.values():
        edges = sorted(record["edges"])
        latest_end = None
        ambiguous = False
        for lower, upper, _ in edges:
            if latest_end is not None and lower < latest_end:
                ambiguous = True
            latest_end = max(latest_end, upper) if latest_end else upper
        if not ambiguous:
            labels[record["user"]].update(item for lower, upper, item in edges if item > 0 and (upper-lower).total_seconds() >= 60)
    output = []
    for index in selected:
        user = users[index - 1]
        require(labels[user] == truth[user], f"Raw CSV reconstruction differs for template row {index}")
        for method, by_user in predictions.items():
            items = by_user[user]
            flags = [int(item in labels[user]) for item in items]
            numerator = sum((Fraction(sum(flags[:rank]), rank) for rank, flag in enumerate(flags, 1) if flag), Fraction(0))
            score = numerator / min(5, len(labels[user])) if labels[user] else Fraction(0)
            require(abs(float(score) - average_precision_at_k(truth[user], items)) < 1e-12, "Independent real-user AP failed")
            output.append({"template_row": index, "method": method, "raw_valid_sessions": per_user_records[user],
                "relevant_items": len(labels[user]), "predictions": items, "rank_hit_flags": flags,
                "precision_numerator_fraction": str(numerator), "denominator": min(5, len(labels[user])),
                "ap_fraction": str(score), "ap_at_5": float(score), "raw_csv_labels_exactly_match": True})
    return output


def run():
    started = time.perf_counter()
    protected = protected_hashes()
    raw_integrity()
    con = connect()
    users, truth, popularity, baseline = prepare(con)
    ensure_metadata(con)
    checks_sql = {
        "training_after_cutoff": f"SELECT count(*) FROM adv_train WHERE session_end>TIMESTAMP '{CUT}' OR overlap_end>TIMESTAMP '{CUT}' OR session_start<TIMESTAMP '{BEGIN}'",
        "heldout_outside_period": f"SELECT count(*) FROM adv_val WHERE session_start<TIMESTAMP '{CUT}' OR session_end>TIMESTAMP '{FINAL}'",
        "shared_session_ids": "SELECT count(*) FROM (SELECT viewing_id FROM adv_train INTERSECT SELECT viewing_id FROM adv_val)",
        "label_user_or_time_differs_from_viewing": """SELECT count(*) FROM adv_val i JOIN viewing v USING(viewing_id)
            WHERE i.user_id<>v.user_id OR i.session_start<>v.start_time OR i.session_end<>v.end_time OR i.channel_id<>v.channel_id""",
        "label_item_differs_from_airing": "SELECT count(*) FROM adv_val i JOIN schedule s USING(schedule_id) WHERE i.tv_show_id<>s.tv_show_id OR i.tv_show_id<=0 OR i.overlap_seconds<60",
        "training_schedule_from_validation": f"SELECT count(*) FROM adv_train i JOIN schedule s USING(schedule_id) WHERE s.source<>{literal(SCHEDULES[0])}",
        "label_items_outside_candidates": "SELECT count(*) FROM adv_truth WHERE tv_show_id NOT IN (SELECT tv_show_id FROM adv_candidates)",
    }
    checks = {key: con.execute(sql).fetchone()[0] for key, sql in checks_sql.items()}
    con.execute(f"""CREATE OR REPLACE TEMP TABLE reconstructed_truth AS
        SELECT DISTINCT v.user_id,e.tv_show_id FROM viewing v JOIN adv_users u USING(user_id)
        JOIN match_edges e USING(viewing_id) JOIN session_quality q USING(viewing_id)
        WHERE v.start_time>=TIMESTAMP '{CUT}' AND v.end_time<=TIMESTAMP '{FINAL}'
          AND q.usable AND e.tv_show_id>0 AND e.overlap_seconds>=60""")
    checks["reconstructed_label_symmetric_difference"] = con.execute("""SELECT count(*) FROM (
        (SELECT * FROM reconstructed_truth EXCEPT SELECT * FROM adv_truth)
        UNION ALL (SELECT * FROM adv_truth EXCEPT SELECT * FROM reconstructed_truth))""").fetchone()[0]
    require(all(value == 0 for value in checks.values()), "Baseline provenance audit failed")
    top5 = [row["tv_show_id"] for row in popularity[:5]]
    require(all(baseline["popularity"][user] == top5 for user in users), "Archived popularity differs from training-only counts")
    # Recompute the exact original personal feature formula from the training snapshot only.
    personal = rows(con, f"""WITH daily AS (
        SELECT DISTINCT user_id,tv_show_id,cast(overlap_start AS DATE) AS d FROM adv_train
        JOIN adv_users USING(user_id) JOIN adv_candidates USING(tv_show_id)), scores AS (
        SELECT user_id,tv_show_id,round(sum(pow(2.0,-date_diff('day',d,DATE '{CUT[:10]}')/14.0)),12) score
        FROM daily GROUP BY user_id,tv_show_id), pop AS (
        SELECT tv_show_id,count(DISTINCT user_id) n FROM adv_train GROUP BY tv_show_id)
        SELECT user_id,tv_show_id,row_number() OVER (PARTITION BY user_id ORDER BY score DESC,n DESC,tv_show_id ASC) rank
        FROM scores JOIN pop USING(tv_show_id) QUALIFY rank<=5 ORDER BY user_id,rank""")
    regenerated = {user: [] for user in users}
    for row in personal:
        regenerated[row["user_id"]].append(row["tv_show_id"])
    global_order = [row["tv_show_id"] for row in popularity]
    for user in users:
        for item in global_order:
            if len(regenerated[user]) >= 5:
                break
            if item not in regenerated[user]:
                regenerated[user].append(item)
    require(regenerated == baseline["repeat_recency_heuristic"], "Archived personal rankings differ from training-only features")
    scores = {method: mean_average_precision_at_k(truth, pred, users) for method, pred in baseline.items()}
    expected = {"popularity": 0.7947830687830687, "repeat_recency_heuristic": 0.796042328042328}
    require(all(abs(scores[key] - expected[key]) < 1e-12 for key in expected), "Baseline scores changed")
    distribution = one(con, """SELECT min(n) AS minimum,max(n) AS maximum,avg(n) AS mean,
        quantile_cont(n,[0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]) AS quantiles,
        count(*) FILTER (WHERE n=0) AS empty_users,count(*) FILTER (WHERE n<5) AS fewer_than_5,
        sum(n) AS relevant_user_item_pairs FROM (SELECT u.user_id,count(t.tv_show_id) n
        FROM adv_users u LEFT JOIN adv_truth t USING(user_id) GROUP BY u.user_id)""")
    rank_hits = []
    cumulative = set()
    for rank, item in enumerate(top5, 1):
        hit_users = {user for user in users if item in truth[user]}
        cumulative |= hit_users
        rank_hits.append({"rank": rank, "tv_show_id": item, "training_viewers": popularity[rank - 1]["viewers"],
            "validation_users_with_item": len(hit_users), "hit_rate": len(hit_users)/len(users),
            "cumulative_any_hit_users": len(cumulative)})
    histogram = [{"hits_among_popularity_top5": number,
        "users": sum(len(set(top5) & truth[user]) == number for user in users)} for number in range(6)]
    repeat = one(con, """SELECT count(*) AS repeated_user_item_pairs FROM adv_truth t
        JOIN (SELECT DISTINCT user_id,tv_show_id FROM adv_train) h USING(user_id,tv_show_id)""")
    # Full identity checks use metadata rather than pretending title strings are canonical IDs.
    identity = rows(con, """SELECT tv_show_id,count(*) AS airings,count(DISTINCT title) AS title_variants,
        count(DISTINCT category) AS category_variants,count(DISTINCT production_year) AS year_variants,
        count(DISTINCT director) AS director_variants,
        count(DISTINCT (genre1,genre2,genre3)) AS genre_tuple_variants
        FROM advanced_metadata GROUP BY tv_show_id ORDER BY tv_show_id""")
    save_csv("advanced_item_identity.csv", identity)
    top_metadata = rows(con, f"""SELECT tv_show_id,title,category,genre1,count(*) AS airings
        FROM advanced_metadata WHERE tv_show_id IN ({','.join(map(str, top5))})
        GROUP BY tv_show_id,title,category,genre1 ORDER BY tv_show_id,airings DESC""")
    save_csv("advanced_top5_identity_evidence.csv", top_metadata)
    identity_summary = {
        "positive_ids_checked": len(identity),
        "ids_with_multiple_titles": sum(row["title_variants"] > 1 for row in identity),
        "ids_with_multiple_categories": sum(row["category_variants"] > 1 for row in identity),
        "ids_with_multiple_year_values": sum(row["year_variants"] > 1 for row in identity),
        "ids_with_multiple_directors": sum(row["director_variants"] > 1 for row in identity),
        "ids_with_multiple_genre_tuples": sum(row["genre_tuple_variants"] > 1 for row in identity),
        "top5_checks": [row for row in identity if row["tv_show_id"] in top5],
        "meaning": "Multiple titles can be episodes/editions or translations; metadata agreement is evidence, not external identity certification.",
    }
    # Select examples spanning complete misses, partial hits, perfect AP, empty truth, and cold history.
    train_users = {row[0] for row in con.execute("SELECT DISTINCT user_id FROM adv_train").fetchall()}
    predicates = [lambda u: bool(truth[u]) and average_precision_at_k(truth[u], top5) == 0,
                  lambda u: 0 < average_precision_at_k(truth[u], top5) < 0.6,
                  lambda u: 0.6 <= average_precision_at_k(truth[u], top5) < 1,
                  lambda u: average_precision_at_k(truth[u], top5) == 1,
                  lambda u: not truth[u], lambda u: u not in train_users and bool(truth[u])]
    selected = sorted({next(index for index, user in enumerate(users, 1) if predicate(user)) for predicate in predicates})
    print(f"Reconstructing held-out labels directly from CSV for {len(selected)} users", flush=True)
    real_examples = recompute_raw_users(users, selected, truth, baseline)
    result = {"provenance_checks": checks, "archived_rankings_reproduced_from_training_only": True,
        "baseline_map_at_5": scores, "relevant_count_distribution": distribution, "top5_rank_hit_rates": rank_hits,
        "top5_hit_histogram": histogram, "top5_total_hits": sum(row["validation_users_with_item"] for row in rank_hits),
        "top5_micro_recall": sum(row["validation_users_with_item"] for row in rank_hits)/distribution["relevant_user_item_pairs"],
        "repeat_labels": repeat, "item_identity": identity_summary, "independent_real_user_examples": real_examples,
        "protocol_changed": False, "baseline_artifacts_unchanged": protected_hashes() == protected,
        "runtime_seconds": round(time.perf_counter() - started, 3)}
    require(result["baseline_artifacts_unchanged"], "Baseline artifacts changed")
    save_json("advanced_validation_audit.json", result)
    save_csv("advanced_top5_coverage.csv", rank_hits)
    con.close()
    print(json.dumps({key: result[key] for key in ("baseline_map_at_5", "relevant_count_distribution", "top5_hit_histogram", "item_identity", "runtime_seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    run()
