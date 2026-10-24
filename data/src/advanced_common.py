"""Shared frozen protocol and evidence helpers for the advanced experiments."""

import csv
import hashlib
import json

from baseline_experiment import BEGIN, CUT, FINAL, require
from common import RAW, RESULTS, SCHEDULES, TIMEZONE, literal, rows
from validate_submission import read_template


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def protected_hashes():
    paths = list(RESULTS.glob("baseline_*.csv")) + list(RESULTS.glob("baseline_*.json"))
    paths += list(RESULTS.glob("popularity_*.csv")) + list(RESULTS.glob("submission_baseline*"))
    return {path.name: file_hash(path) for path in paths}


def prepare(con):
    users = read_template()
    values = ",".join(f"({literal(user)},{index})" for index, user in enumerate(users, 1))
    con.execute(f"CREATE OR REPLACE TEMP TABLE adv_users AS SELECT * FROM (VALUES {values}) t(user_id,template_row)")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE adv_train AS SELECT * FROM interactions
        WHERE session_start>=TIMESTAMP '{BEGIN}' AND session_end<=TIMESTAMP '{CUT}'""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE adv_val AS SELECT * FROM interactions
        WHERE session_start>=TIMESTAMP '{CUT}' AND session_end<=TIMESTAMP '{FINAL}'""")
    con.execute("""CREATE OR REPLACE TEMP TABLE adv_truth AS SELECT DISTINCT user_id,tv_show_id
        FROM adv_val JOIN adv_users USING(user_id)""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE adv_candidates AS SELECT DISTINCT tv_show_id FROM schedule
        WHERE tv_show_id>0 AND duration>0 AND source IN ({','.join(literal(s) for s in SCHEDULES[:2])})
          AND start_time<TIMESTAMP '{FINAL}'
          AND timezone('{TIMEZONE}',timezone('{TIMEZONE}',start_time)+duration*INTERVAL '1 second')>TIMESTAMP '{CUT}'""")
    truth = {user: set() for user in users}
    for user, item in con.execute("SELECT user_id,tv_show_id FROM adv_truth").fetchall():
        truth[user].add(item)
    popularity = rows(con, """SELECT c.tv_show_id,coalesce(p.viewers,0) AS viewers FROM adv_candidates c
        LEFT JOIN (SELECT tv_show_id,count(DISTINCT user_id) AS viewers FROM adv_train GROUP BY tv_show_id) p USING(tv_show_id)
        ORDER BY viewers DESC,c.tv_show_id""")
    baseline = {"popularity": {}, "repeat_recency_heuristic": {}}
    with (RESULTS / "baseline_per_user_evaluation.csv").open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            baseline[record["method"]][users[int(record["template_row"]) - 1]] = list(map(int, record["predicted_ids"].split()))
    return users, truth, popularity, baseline


def ensure_metadata(con):
    """Add only the metadata omitted from the original slim cache, once."""
    con.execute("""CREATE TABLE IF NOT EXISTS advanced_metadata (
        source VARCHAR,channel_id BIGINT,start_time TIMESTAMP,duration BIGINT,tv_show_id BIGINT,
        title VARCHAR,category VARCHAR,genre1 VARCHAR,genre2 VARCHAR,genre3 VARCHAR,
        production_year VARCHAR,director VARCHAR,actors VARCHAR)""")
    present = {row[0] for row in con.execute("SELECT DISTINCT source FROM advanced_metadata").fetchall()}
    for filename in SCHEDULES[:2]:
        if filename not in present:
            con.execute(f"""INSERT INTO advanced_metadata SELECT {literal(filename)},cast(channel_id AS BIGINT),
                strptime(start_time,'%d.%m.%Y %H:%M:%S'),cast(duration AS BIGINT),cast(tv_show_id AS BIGINT),
                tv_show_title,tv_show_category,tv_show_genre_1,tv_show_genre_2,tv_show_genre_3,
                year_of_production,director,actors FROM read_csv({literal(RAW / filename)},
                    header=true,all_varchar=true,strict_mode=true,ignore_errors=false)
                WHERE cast(tv_show_id AS BIGINT)>0""")
    expected = con.execute(f"""SELECT count(*) FROM schedule WHERE tv_show_id>0
        AND source IN ({','.join(literal(s) for s in SCHEDULES[:2])})""").fetchone()[0]
    require(con.execute("SELECT count(*) FROM advanced_metadata").fetchone()[0] == expected, "Incomplete metadata cache")


def load_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))
