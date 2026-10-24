"""Interval matching, quality gates, temporal probes, and independent CSV checks.

Run audit.py first. Tables remain in the configured local cache; only aggregate
results and a small set of source-row verification examples are published.
Matching uses shared numeric channel IDs as an explicitly provisional hypothesis.
"""

import csv
from datetime import datetime, timedelta, timezone
import time
from zoneinfo import ZoneInfo

from common import CACHE, RAW, SCHEDULES, TIMEZONE, connect, elapsed_sql, literal, one, rows, save_csv, save_json

CUT = "2020-05-18 00:00:00"
END = "2020-07-27 00:00:00"


def independent_check(con):
    """Re-read CSVs with stdlib and compare complete sets of overlapping airings."""
    choices = [
        ("single_identified", "identified_edges=1 AND all_edges=1"),
        ("multiple_identified", "identified_edges>1 AND NOT ambiguous"),
        ("unknown_only", "all_edges>0 AND identified_edges=0"),
        ("unmatched_channel", "all_edges=0 AND channel_id NOT IN (SELECT channel_id FROM schedule)"),
        ("schedule_gap", "all_edges=0 AND channel_id IN (SELECT channel_id FROM schedule)"),
        ("ambiguous", "ambiguous"),
        ("duration_mismatch", "duration<>date_diff('second',start_time,end_time)"),
        ("long_session", "duration>86400"),
        ("split_crossing", f"start_time<TIMESTAMP '{CUT}' AND end_time>TIMESTAMP '{CUT}'"),
        ("validation", f"start_time>=TIMESTAMP '{CUT}' AND identified_edges>0"),
        ("late_validation", "start_time>=TIMESTAMP '2020-07-20' AND identified_edges>0"),
        ("dst_day", "cast(start_time AS DATE)=DATE '2020-03-29' AND identified_edges>0"),
        ("dst_crossing_identified", "start_time<TIMESTAMP '2020-03-29 03:00:00' AND end_time>=TIMESTAMP '2020-03-29 04:00:00' AND identified_edges>0"),
        ("exact_boundary", "viewing_id IN (SELECT e.viewing_id FROM match_edges e JOIN schedule s USING(schedule_id) JOIN viewing v USING(viewing_id) WHERE v.start_time=s.start_time OR v.end_time=s.end_time)"),
    ]
    selected = {}
    for label, predicate in choices:
        found = con.execute(f"SELECT min(viewing_id) FROM session_quality WHERE {predicate}").fetchone()[0]
        if found is not None:
            selected.setdefault(found, []).append(label)
    originals = {}
    zone = ZoneInfo(TIMEZONE)
    with (RAW / "dataset11-30.csv").open(encoding="utf-8-sig", newline="") as handle:
        for index, record in enumerate(csv.DictReader(handle), 1):
            if index in selected:
                originals[index] = {"channel": int(record["vsetv_id"]),
                    "start": datetime.strptime(record["start_time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=zone).astimezone(timezone.utc),
                    "end": datetime.strptime(record["stop_time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=zone).astimezone(timezone.utc),
                    "duration": int(record["duraton"]), "edges": []}
    by_channel = {}
    for index, record in originals.items():
        by_channel.setdefault(record["channel"], []).append((index, record))
    # Independent full scan: no DuckDB time parser, joins, or derived schedule tables.
    for filename in SCHEDULES[:2]:
        with (RAW / filename).open(encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle), 1):
                channel = int(row["channel_id"])
                if channel not in by_channel:
                    continue
                start = datetime.strptime(row["start_time"], "%d.%m.%Y %H:%M:%S").replace(tzinfo=zone).astimezone(timezone.utc)
                end = start + timedelta(seconds=int(row["duration"]))
                if end <= start:
                    continue
                for _, record in by_channel[channel]:
                    seconds = int((min(record["end"], end) - max(record["start"], start)).total_seconds())
                    if seconds > 0:
                        record["edges"].append((filename, index, int(row["tv_show_id"]), seconds))
    checked = []
    for index, record in originals.items():
        expected = sorted(record["edges"])
        actual = sorted(con.execute("""SELECT s.source,s.source_row,s.tv_show_id,e.overlap_seconds
            FROM match_edges e JOIN schedule s USING(schedule_id) WHERE e.viewing_id=?""", [index]).fetchall())
        assert actual == expected, (index, actual, expected)
        checked.append({"viewing_source_row": index, "cases": selected[index],
            "channel_id": record["channel"], "start_time": record["start"].astimezone(zone), "stop_time": record["end"].astimezone(zone),
            "reported_duration": record["duration"], "independently_verified": True,
            "schedule_overlaps": [{"file": edge[0], "source_row": edge[1], "tv_show_id": edge[2],
                                   "overlap_seconds": edge[3]} for edge in expected]})
    save_json("verified_examples.json", checked)
    return {"examples": len(checked), "passed": len(checked), "method": "Independent csv.DictReader + zoneinfo UTC interval scan over both historical schedules"}


def temporal_probe(con):
    """Coverage cannot determine timezone; report boundary alignment and controls."""
    con.execute("""CREATE OR REPLACE TEMP TABLE probe AS SELECT * FROM viewing
        WHERE viewing_id%100=0 AND start_time>=TIMESTAMP '2020-03-10'
        AND end_time<TIMESTAMP '2020-05-18'""")
    con.execute("""CREATE OR REPLACE TEMP TABLE channel_control AS
        SELECT channel_id,coalesce(lead(channel_id) OVER (ORDER BY channel_id),min(channel_id) OVER ()) AS control_channel
        FROM (SELECT DISTINCT channel_id FROM viewing WHERE channel_id IN (SELECT channel_id FROM historical_schedule))""")
    sample_count = con.execute("SELECT count(*) FROM probe").fetchone()[0]
    output = []
    for control in (False, True):
        for shift in ((0,) if control else (-3, -2, -1, 0, 1, 2, 3)):
            record = one(con, f"""WITH shifted AS (
                SELECT v.viewing_id,{'c.control_channel' if control else 'v.channel_id'} AS channel_id,
                    v.start_time+INTERVAL '{shift} hour' AS start_time,
                    v.end_time+INTERVAL '{shift} hour' AS end_time,
                    CASE WHEN v.start_time<TIMESTAMP '2020-03-29 03:00:00' THEN 'before_DST' ELSE 'after_DST' END AS era
                FROM probe v LEFT JOIN channel_control c USING(channel_id)), candidates AS (
                SELECT v.viewing_id,v.era,
                    bool_or(v.start_time<s.end_time AND v.end_time>s.start_time) AS any_overlap,
                    bool_or(v.start_time<s.end_time AND v.end_time>s.start_time AND s.tv_show_id>0) AS identified_overlap,
                    min(abs(date_diff('second',v.end_time,s.start_time))) AS stop_boundary_distance,
                    min(abs(date_diff('second',v.start_time,s.start_time))) AS start_boundary_distance
                FROM shifted v LEFT JOIN historical_schedule s ON v.channel_id=s.channel_id
                  AND s.start_time>=date_trunc('day',v.start_time)-INTERVAL '1 day'
                  AND s.start_time<date_trunc('day',v.end_time)+INTERVAL '2 day'
                GROUP BY v.viewing_id,v.era)
                SELECT count(*) FILTER (WHERE any_overlap) AS any_schedule_matches,
                  count(*) FILTER (WHERE identified_overlap) AS identified_matches,
                  count(*) FILTER (WHERE stop_boundary_distance<=120) AS stops_within_120s_of_boundary,
                  count(*) FILTER (WHERE start_boundary_distance<=120) AS starts_within_120s_of_boundary,
                  count(*) FILTER (WHERE era='before_DST') AS before_dst_rows,
                  count(*) FILTER (WHERE era='before_DST' AND stop_boundary_distance<=120) AS before_dst_stop_aligned,
                  count(*) FILTER (WHERE era='after_DST') AS after_dst_rows,
                  count(*) FILTER (WHERE era='after_DST' AND stop_boundary_distance<=120) AS after_dst_stop_aligned
                FROM candidates""")
            output.append({"channel_mapping": "cyclic_permutation_control" if control else "equal_numeric_ids",
                "viewing_shift_hours": shift, "sample_rows": sample_count, **record})
    save_csv("timezone_channel_probes.csv", output)
    return output


def run():
    started = time.perf_counter()
    con = connect()
    report = {"assumptions": {"channel_mapping": "equal numeric vsetv_id and channel_id, conditional on source confirmation",
        "timezone": "Europe/Kyiv-compatible local clocks; elapsed durations computed in UTC; exact timezone not encoded by source",
        "intervals": "half-open [start,end); retain strictly positive overlap",
        "unknown_items": "tv_show_id=0 participates in coverage diagnostics but is never an interaction"}}
    # Final-period schedule is intentionally unavailable to the historical join.
    con.execute(f"""CREATE OR REPLACE VIEW historical_schedule AS SELECT * EXCLUDE(end_time),
        timezone('{TIMEZONE}',timezone('{TIMEZONE}',start_time)+duration*INTERVAL '1 second') AS end_time FROM schedule
        WHERE source IN ({','.join(literal(s) for s in SCHEDULES[:2])}) AND duration>0""")
    con.execute("""CREATE OR REPLACE TABLE schedule_days AS SELECT s.schedule_id,s.channel_id,
        s.start_time,s.end_time,s.tv_show_id,cast(d.day AS DATE) AS day FROM historical_schedule s,
        LATERAL generate_series(date_trunc('day',s.start_time),
            date_trunc('day',s.end_time-INTERVAL '1 second'),INTERVAL '1 day') d(day)""")
    con.execute("""CREATE OR REPLACE TABLE match_edges (
        viewing_id BIGINT,schedule_id BIGINT,tv_show_id BIGINT,
        overlap_start TIMESTAMP,overlap_end TIMESTAMP,overlap_seconds BIGINT)""")
    weeks = con.execute("SELECT DISTINCT date_trunc('week',start_time) AS week FROM viewing ORDER BY week").fetchall()
    for (week,) in weeks:
        tick = time.perf_counter()
        week_end = week + timedelta(days=7)
        con.execute(f"""CREATE OR REPLACE TEMP TABLE week_viewing AS SELECT * FROM viewing
            WHERE start_time>=TIMESTAMP {literal(week)} AND start_time<TIMESTAMP {literal(week_end)}
              AND end_time>start_time""")
        con.execute(f"""INSERT INTO match_edges WITH v AS (
            SELECT v.*,cast(d.day AS DATE) AS day FROM week_viewing v,
            LATERAL generate_series(date_trunc('day',v.start_time),
                date_trunc('day',v.end_time-INTERVAL '1 second'),INTERVAL '1 day') d(day)
            )
            SELECT v.viewing_id,s.schedule_id,s.tv_show_id,
                greatest(v.start_time,s.start_time),least(v.end_time,s.end_time),
                {elapsed_sql('greatest(v.start_time,s.start_time)', 'least(v.end_time,s.end_time)')}
            FROM v JOIN schedule_days s ON v.channel_id=s.channel_id AND v.day=s.day
            WHERE v.start_time<s.end_time AND v.end_time>s.start_time
              AND v.day=cast(greatest(v.start_time,s.start_time) AS DATE)""")
        print(f"Matched week {week.isocalendar().week}: {time.perf_counter()-tick:.1f}s", flush=True)
    con.execute("""CREATE OR REPLACE TABLE edge_summary AS WITH ordered AS (
        SELECT *,max(overlap_end) OVER (PARTITION BY viewing_id ORDER BY overlap_start,schedule_id
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS previous_end FROM match_edges)
        SELECT viewing_id,count(*) AS all_edges,count(*) FILTER (WHERE tv_show_id>0) AS identified_edges,
          count(DISTINCT tv_show_id) FILTER (WHERE tv_show_id>0) AS identified_items,
          sum(overlap_seconds) AS schedule_seconds,
          sum(overlap_seconds) FILTER (WHERE tv_show_id>0) AS identified_seconds,
          coalesce(bool_or(overlap_start<previous_end),false) AS ambiguous
        FROM ordered GROUP BY viewing_id""")
    con.execute(f"""CREATE OR REPLACE TABLE session_quality AS WITH numbered AS (
        SELECT *,row_number() OVER (PARTITION BY user_id,channel_id,start_time,end_time,duration ORDER BY viewing_id) AS duplicate_rank
        FROM viewing)
        SELECT v.*,coalesce(e.all_edges,0) AS all_edges,coalesce(e.identified_edges,0) AS identified_edges,
          coalesce(e.identified_items,0) AS identified_items,coalesce(e.schedule_seconds,0) AS schedule_seconds,
          coalesce(e.identified_seconds,0) AS identified_seconds,coalesce(e.ambiguous,false) AS ambiguous,
          {elapsed_sql('start_time', 'end_time')} AS elapsed_seconds,
          (duplicate_rank=1 AND duration>0 AND duration<=86400 AND end_time>start_time
            AND duration={elapsed_sql('start_time', 'end_time')} AND NOT coalesce(e.ambiguous,false)) AS usable
        FROM numbered v LEFT JOIN edge_summary e USING(viewing_id)""")
    report["coverage"] = one(con, """SELECT count(*) AS raw_viewing_rows,
        count(*) FILTER (WHERE all_edges>0) AS any_schedule_matched_rows,
        count(*) FILTER (WHERE identified_edges>0) AS identified_matched_rows,
        100.0*count(*) FILTER (WHERE identified_edges>0)/count(*) AS identified_matched_percent,
        count(*) FILTER (WHERE all_edges>0 AND identified_edges=0) AS only_unknown_program_rows,
        count(*) FILTER (WHERE all_edges=0) AS no_schedule_overlap_rows,
        count(*) FILTER (WHERE all_edges=0 AND channel_id NOT IN (SELECT channel_id FROM schedule)) AS missing_channel_rows,
        count(*) FILTER (WHERE all_edges=0 AND channel_id IN (SELECT channel_id FROM schedule)) AS uncovered_time_rows,
        count(*) FILTER (WHERE identified_edges>0 AND identified_seconds<elapsed_seconds) AS partially_identified_rows,
        count(*) FILTER (WHERE schedule_seconds<elapsed_seconds) AS incompletely_schedule_covered_rows,
        count(*) FILTER (WHERE all_edges>1) AS multi_airing_rows,
        count(*) FILTER (WHERE identified_items>1) AS multi_identified_item_rows,
        count(*) FILTER (WHERE ambiguous) AS ambiguous_rows,
        count(*) FILTER (WHERE duplicate_rank>1) AS duplicate_rows_removed,
        count(*) FILTER (WHERE NOT usable) AS excluded_quality_rows,
        count(*) FILTER (WHERE duration<>elapsed_seconds) AS timezone_adjusted_duration_mismatches,
        count(*) FILTER (WHERE usable AND identified_edges>0) AS usable_matched_sessions,
        count(DISTINCT user_id) FILTER (WHERE usable AND identified_edges>0) AS usable_matched_users,
        sum(identified_edges) AS raw_identified_edges,
        sum(identified_edges) FILTER (WHERE usable) AS usable_identified_edges
        FROM session_quality""")
    report["duration_discrepancies"] = rows(con, """SELECT duration-date_diff('second',start_time,end_time) AS reported_minus_elapsed,
        cast(start_time AS DATE) AS start_date,count(*) AS rows FROM viewing
        WHERE duration<>date_diff('second',start_time,end_time) GROUP BY ALL ORDER BY start_date,reported_minus_elapsed""")
    save_csv("duration_discrepancies.csv", report["duration_discrepancies"])
    save_csv("match_by_channel.csv", rows(con, """SELECT channel_id,count(*) AS viewing_rows,
        count(*) FILTER (WHERE all_edges>0) AS any_schedule_rows,
        count(*) FILTER (WHERE identified_edges>0) AS identified_rows,
        count(*) FILTER (WHERE usable AND identified_edges>0) AS usable_identified_rows,
        count(*) FILTER (WHERE ambiguous) AS ambiguous_rows FROM session_quality GROUP BY channel_id ORDER BY viewing_rows DESC"""))
    # Conservative label candidate: >=60 seconds on an identified airing, not an inferred item for unknown programs.
    con.execute("""CREATE OR REPLACE TABLE interactions AS SELECT v.viewing_id,v.user_id,v.channel_id,
        e.schedule_id,e.tv_show_id,v.start_time AS session_start,v.end_time AS session_end,
        e.overlap_start,e.overlap_end,e.overlap_seconds
        FROM session_quality v JOIN match_edges e USING(viewing_id)
        WHERE v.usable AND e.tv_show_id>0 AND e.overlap_seconds>=60""")
    report["interaction_thresholds"] = []
    for threshold in (1, 30, 60, 300, 600):
        report["interaction_thresholds"].append({"minimum_overlap_seconds": threshold, **one(con, f"""SELECT
            count(*) AS airing_interactions,count(DISTINCT v.viewing_id) AS sessions,
            count(DISTINCT v.user_id) AS users,count(DISTINCT e.tv_show_id) AS items,
            count(DISTINCT (v.user_id,e.tv_show_id)) AS user_item_pairs
            FROM session_quality v JOIN match_edges e USING(viewing_id)
            WHERE v.usable AND e.tv_show_id>0 AND e.overlap_seconds>={threshold}""")})
    save_csv("interaction_thresholds.csv", report["interaction_thresholds"])
    report["split"] = []
    periods = [("train", "2020-03-09 00:00:00", CUT), ("validation", CUT, END),
               ("refit", "2020-03-09 00:00:00", END)]
    for name, lower, upper in periods:
        record = {"period": name, "start_inclusive": lower, "end_exclusive": upper}
        record.update(one(con, f"""SELECT count(*) AS start_assigned_raw_sessions,
            count(DISTINCT user_id) AS raw_users,
            count(*) FILTER (WHERE end_time>TIMESTAMP '{upper}') AS cutoff_crossing_sessions
            FROM viewing WHERE start_time>=TIMESTAMP '{lower}' AND start_time<TIMESTAMP '{upper}'"""))
        record.update(one(con, f"""SELECT count(*) AS usable_interactions,count(DISTINCT viewing_id) AS usable_sessions,
            count(DISTINCT user_id) AS usable_users,count(DISTINCT tv_show_id) AS usable_items,
            count(DISTINCT (user_id,tv_show_id)) AS user_item_pairs,
            count(DISTINCT user_id) FILTER (WHERE user_id IN (SELECT user_id FROM submission)) AS submission_users_with_history
            FROM interactions WHERE session_start>=TIMESTAMP '{lower}' AND session_end<=TIMESTAMP '{upper}'"""))
        report["split"].append(record)
    save_csv("split_summary.csv", report["split"])
    con.execute(f"""CREATE OR REPLACE VIEW train_interactions AS SELECT * FROM interactions
        WHERE session_start>=TIMESTAMP '2020-03-09' AND session_end<=TIMESTAMP '{CUT}'""")
    con.execute(f"""CREATE OR REPLACE VIEW validation_interactions AS SELECT * FROM interactions
        WHERE session_start>=TIMESTAMP '{CUT}' AND session_end<=TIMESTAMP '{END}'""")
    report["validation_cohorts"] = one(con, """SELECT count(DISTINCT user_id) AS validation_users,
        count(DISTINCT user_id) FILTER (WHERE user_id NOT IN (SELECT user_id FROM train_interactions)) AS cold_users,
        count(DISTINCT tv_show_id) FILTER (WHERE tv_show_id NOT IN (SELECT tv_show_id FROM train_interactions)) AS cold_items,
        count(DISTINCT (user_id,tv_show_id)) FILTER (WHERE tv_show_id NOT IN (SELECT tv_show_id FROM train_interactions)) AS cold_item_user_pairs,
        count(DISTINCT user_id) FILTER (WHERE user_id IN (SELECT user_id FROM submission)) AS submission_users_with_labels
        FROM validation_interactions""")
    report["validation_cohorts"]["users_with_train_but_no_validation_labels"] = con.execute("""SELECT count(DISTINCT user_id)
        FROM train_interactions WHERE user_id NOT IN (SELECT user_id FROM validation_interactions)""").fetchone()[0]
    report["validation_cohorts"]["repeated_train_user_item_pairs"] = con.execute("""SELECT count(*) FROM (
        SELECT DISTINCT user_id,tv_show_id FROM validation_interactions INTERSECT
        SELECT DISTINCT user_id,tv_show_id FROM train_interactions)""").fetchone()[0]
    report["submission_history"] = one(con, """SELECT count(*) AS submission_users,
        count(*) FILTER (WHERE user_id NOT IN (SELECT user_id FROM interactions)) AS without_usable_history,
        count(*) FILTER (WHERE user_id NOT IN (SELECT user_id FROM train_interactions)) AS without_train_history,
        count(*) FILTER (WHERE user_id NOT IN (SELECT user_id FROM validation_interactions)) AS without_validation_labels
        FROM submission""")
    save_csv("user_history_distribution.csv", rows(con, """WITH per_user AS (
        SELECT user_id,count(*) AS interactions,count(DISTINCT tv_show_id) AS items,sum(overlap_seconds) AS seconds
        FROM interactions GROUP BY user_id)
        SELECT count(*) AS users,min(interactions) AS min_interactions,
        quantile_cont(interactions,[0.1,0.25,0.5,0.75,0.9,0.99]) AS interaction_quantiles,
        max(interactions) AS max_interactions,min(items) AS min_items,
        quantile_cont(items,[0.1,0.25,0.5,0.75,0.9,0.99]) AS item_quantiles,max(items) AS max_items FROM per_user"""))
    report["candidate_catalogs"] = []
    for name, lower, upper, files in [
        ("train", "2020-03-09", CUT, SCHEDULES[:1]),
        ("validation", CUT, END, SCHEDULES[:2]),
        ("final", END, "2020-10-19 00:00:00", SCHEDULES[1:])]:
        report["candidate_catalogs"].append({"period": name, **one(con, f"""SELECT count(*) AS airings,
            count(DISTINCT tv_show_id) AS identified_items,
            count(DISTINCT tv_show_id) FILTER (WHERE channel_id IN (SELECT DISTINCT channel_id FROM viewing WHERE start_time<TIMESTAMP '{lower}')) AS items_on_previously_viewed_channels
            FROM schedule WHERE source IN ({','.join(literal(s) for s in files)}) AND tv_show_id>0 AND duration>0
              AND start_time<TIMESTAMP '{upper}' AND end_time>TIMESTAMP '{lower}'""")})
    report["schedule_file_boundary_rows"] = rows(con, """SELECT source,
        count(*) FILTER (WHERE start_time<CASE WHEN source LIKE '%11-20%' THEN TIMESTAMP '2020-03-09'
          WHEN source LIKE '%21-30%' THEN TIMESTAMP '2020-05-18' ELSE TIMESTAMP '2020-07-27' END) AS before_named_period,
        count(*) FILTER (WHERE start_time>=CASE WHEN source LIKE '%11-20%' THEN TIMESTAMP '2020-05-18'
          WHEN source LIKE '%21-30%' THEN TIMESTAMP '2020-07-27' ELSE TIMESTAMP '2020-10-19' END) AS after_named_period
        FROM schedule GROUP BY source ORDER BY source""")
    report["timezone_probes"] = temporal_probe(con)
    print("Running independent CSV verification", flush=True)
    report["independent_verification"] = independent_check(con)
    # Full-set invariants exercise cross-midnight expansion, unknown IDs, and quality gates.
    checks = {
        "duplicate_edges": "SELECT count(*) FROM (SELECT viewing_id,schedule_id FROM match_edges GROUP BY ALL HAVING count(*)>1)",
        "nonpositive_overlap": "SELECT count(*) FROM match_edges WHERE overlap_seconds<=0 OR overlap_start>=overlap_end",
        "invalid_interaction_items": "SELECT count(*) FROM interactions WHERE tv_show_id IS NULL OR tv_show_id<=0",
        "bad_quality_interactions": "SELECT count(*) FROM interactions i JOIN session_quality q USING(viewing_id) WHERE NOT q.usable",
        "train_cutoff_violations": f"SELECT count(*) FROM train_interactions WHERE overlap_end>TIMESTAMP '{CUT}' OR session_end>TIMESTAMP '{CUT}'",
        "validation_cutoff_violations": f"SELECT count(*) FROM validation_interactions WHERE overlap_start<TIMESTAMP '{CUT}' OR overlap_end>TIMESTAMP '{END}'",
        "future_schedule_used": f"SELECT count(*) FROM match_edges e JOIN schedule s USING(schedule_id) WHERE s.source={literal(SCHEDULES[2])}",
        "invalid_overlap_arithmetic": f"""SELECT count(*) FROM match_edges e JOIN viewing v USING(viewing_id) JOIN historical_schedule s USING(schedule_id)
            WHERE e.overlap_start<>greatest(v.start_time,s.start_time) OR e.overlap_end<>least(v.end_time,s.end_time)
              OR e.overlap_seconds<>{elapsed_sql('e.overlap_start', 'e.overlap_end')} OR v.channel_id<>s.channel_id""",
        "unambiguous_overcount": "SELECT count(*) FROM session_quality WHERE NOT ambiguous AND schedule_seconds>elapsed_seconds",
    }
    report["invariants"] = {name: con.execute(sql).fetchone()[0] for name, sql in checks.items()}
    assert all(value == 0 for value in report["invariants"].values()), report["invariants"]
    # Persist ready-to-use labels in the private cache, without creating a public user history file.
    report["local_interaction_table"] = str(CACHE / "audit.duckdb") + ":interactions"
    report["runtime_seconds"] = round(time.perf_counter() - started, 3)
    save_json("matching.json", report)
    con.execute("CHECKPOINT")
    con.close()
    print(f"Matching complete: {report['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    run()
