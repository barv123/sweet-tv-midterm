"""Verify input integrity, timezone evidence, candidate coverage, and artifacts."""

import hashlib
import json
import time

from common import RAW, RESULTS, SCHEDULES, connect, literal, one, rows, save_csv, save_json


def run():
    started = time.perf_counter()
    audit = json.loads((RESULTS / "audit.json").read_text(encoding="utf-8"))
    matching = json.loads((RESULTS / "matching.json").read_text(encoding="utf-8"))
    con = connect()
    result = {"input_integrity": {}}
    for filename, info in audit["datasets"].items():
        with (RAW / filename).open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        assert digest == info["sha256"], f"Input changed: {filename}"
        result["input_integrity"][filename] = "SHA-256 unchanged"
    result["kyiv_dst_check"] = one(con, """SELECT count(*) AS discrepant_sessions,
        count(*) FILTER (WHERE duration=date_diff('second',
            timezone('Europe/Kyiv',start_time),timezone('Europe/Kyiv',end_time))) AS resolved_as_kyiv_elapsed_seconds,
        count(*) FILTER (WHERE timezone('Europe/Kyiv',timezone('Europe/Kyiv',start_time))<>start_time
            OR timezone('Europe/Kyiv',timezone('Europe/Kyiv',end_time))<>end_time) AS nonexistent_endpoint_times
        FROM viewing WHERE duration<>date_diff('second',start_time,end_time)""")
    result["dst_window"] = one(con, """SELECT count(*) AS sessions_starting_on_dst_day,
        count(*) FILTER (WHERE start_time>=TIMESTAMP '2020-03-29 03:00:00' AND start_time<TIMESTAMP '2020-03-29 04:00:00') AS starts_in_kyiv_missing_hour,
        count(*) FILTER (WHERE end_time>=TIMESTAMP '2020-03-29 03:00:00' AND end_time<TIMESTAMP '2020-03-29 04:00:00') AS stops_in_kyiv_missing_hour
        FROM viewing WHERE start_time>=TIMESTAMP '2020-03-28' AND start_time<TIMESTAMP '2020-03-30'""")
    # Name explicitly distinguishes the 48-hour endpoint check from one-day counts.
    result["dst_window"]["sessions_starting_on_dst_day"] = con.execute("""SELECT count(*) FROM viewing
        WHERE start_time>=TIMESTAMP '2020-03-29' AND start_time<TIMESTAMP '2020-03-30'""").fetchone()[0]
    result["schedule_dst_check"] = one(con, """SELECT count(*) AS rows_starting_in_kyiv_missing_hour,
        count(DISTINCT channel_id) AS channels FROM schedule
        WHERE start_time>=TIMESTAMP '2020-03-29 03:00:00' AND start_time<TIMESTAMP '2020-03-29 04:00:00'""")
    result["schedule_overlap_details"] = rows(con, """SELECT source,source_row,channel_id,start_time,end_time,tv_show_id,previous_end,
        date_diff('second',start_time,least(end_time,previous_end)) AS overlapping_seconds FROM (
        SELECT *,max(end_time) OVER (PARTITION BY channel_id ORDER BY start_time,schedule_id
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS previous_end FROM schedule WHERE duration>0)
        WHERE start_time<previous_end ORDER BY start_time,channel_id""")
    save_csv("schedule_overlaps.csv", result["schedule_overlap_details"])
    result["schedule_dst_adjacency"] = rows(con, """WITH ordered AS (
        SELECT channel_id,start_time,end_time,lead(start_time) OVER (PARTITION BY channel_id ORDER BY start_time,schedule_id) AS next_start
        FROM schedule WHERE duration>0)
        SELECT date_diff('second',end_time,next_start) AS next_start_minus_computed_end_seconds,count(*) AS rows
        FROM ordered WHERE start_time<TIMESTAMP '2020-03-29 04:00:00' AND end_time>=TIMESTAMP '2020-03-29 03:00:00'
        GROUP BY ALL ORDER BY 1""")
    result["same_user_overlaps"] = one(con, """WITH ordered AS (
        SELECT *,max(end_time) OVER (PARTITION BY user_id,channel_id ORDER BY start_time,viewing_id
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS previous_end
        FROM session_quality WHERE duplicate_rank=1)
        SELECT count(*) FILTER (WHERE start_time<previous_end) AS same_user_channel_overlapping_rows,
            count(DISTINCT user_id) FILTER (WHERE start_time<previous_end) AS affected_users FROM ordered""")
    result["unmatched_time_breakdown"] = one(con, """WITH bounds AS (
        SELECT channel_id,min(start_time) AS first_start,max(end_time) AS last_end FROM historical_schedule GROUP BY channel_id)
        SELECT count(*) AS shared_channel_unmatched_rows,
          count(*) FILTER (WHERE v.end_time<=b.first_start) AS before_channel_schedule_begins,
          count(*) FILTER (WHERE v.start_time>=b.last_end) AS after_channel_schedule_ends,
          count(*) FILTER (WHERE v.end_time>b.first_start AND v.start_time<b.last_end) AS inside_channel_coverage_gaps
        FROM session_quality v JOIN bounds b USING(channel_id) WHERE v.all_edges=0""")
    save_csv("unmatched_by_channel.csv", rows(con, """SELECT v.channel_id,count(*) AS unmatched_rows,
        min(v.start_time) AS first_unmatched_view,max(v.end_time) AS last_unmatched_view,
        min(s.first_schedule) AS first_schedule,max(s.last_schedule) AS last_schedule
        FROM session_quality v LEFT JOIN (
          SELECT channel_id,min(start_time) AS first_schedule,max(end_time) AS last_schedule FROM historical_schedule GROUP BY channel_id
        ) s USING(channel_id) WHERE v.all_edges=0 GROUP BY v.channel_id ORDER BY unmatched_rows DESC"""))
    result["corrected_dst_adjacency"] = rows(con, """WITH ordered AS (
        SELECT channel_id,start_time,end_time,lead(start_time) OVER (PARTITION BY channel_id ORDER BY start_time,schedule_id) AS next_start
        FROM historical_schedule)
        SELECT date_diff('second',end_time,next_start) AS next_start_minus_corrected_end_seconds,count(*) AS rows
        FROM ordered WHERE start_time<TIMESTAMP '2020-03-29 04:00:00' AND end_time>=TIMESTAMP '2020-03-29 04:00:00'
        GROUP BY ALL ORDER BY 1""")
    result["conflicting_session_keys"] = one(con, """WITH conflicts AS (
        SELECT user_id,channel_id,start_time FROM viewing GROUP BY ALL HAVING count(DISTINCT (end_time,duration))>1)
        SELECT count(*) AS affected_rows,count(DISTINCT v.user_id) AS affected_users FROM viewing v JOIN conflicts c
        USING(user_id,channel_id,start_time)""")
    result["zero_id_summary"] = one(con, """SELECT count(*) FILTER (WHERE tv_show_id=0) AS zero_id_rows,
        100.0*count(*) FILTER (WHERE tv_show_id=0)/count(*) AS zero_id_percent FROM schedule""")
    result["candidate_coverage"] = []
    for period, lower, upper, source_files, history in [
        ("validation", "2020-05-18", "2020-07-27", SCHEDULES[:2], "train_interactions"),
        ("final", "2020-07-27", "2020-10-19", SCHEDULES[1:], "interactions")]:
        con.execute(f"""CREATE OR REPLACE TEMP VIEW target_catalog AS SELECT DISTINCT tv_show_id FROM schedule
            WHERE source IN ({','.join(literal(file) for file in source_files)}) AND duration>0 AND tv_show_id>0
            AND start_time<TIMESTAMP '{upper}' AND end_time>TIMESTAMP '{lower}'""")
        info = {"period": period, **one(con, f"""SELECT count(*) AS candidates,
            count(*) FILTER (WHERE tv_show_id NOT IN (SELECT tv_show_id FROM {history})) AS candidates_unseen_in_history
            FROM target_catalog""")}
        if period == "validation":
            info["label_items_outside_catalog"] = con.execute("""SELECT count(DISTINCT tv_show_id) FROM validation_interactions
                WHERE tv_show_id NOT IN (SELECT tv_show_id FROM target_catalog)""").fetchone()[0]
            assert info["label_items_outside_catalog"] == 0
        result["candidate_coverage"].append(info)
    result["train_schedule_source_leaks"] = con.execute(f"""SELECT count(*) FROM train_interactions i JOIN schedule s USING(schedule_id)
        WHERE s.source<>{literal(SCHEDULES[0])}""").fetchone()[0]
    assert result["train_schedule_source_leaks"] == 0
    result["validation_relevance_distribution"] = one(con, """SELECT min(n) AS min_relevant_items,
        quantile_cont(n,[0.25,0.5,0.75,0.9]) AS relevant_item_quantiles,max(n) AS max_relevant_items,
        count(*) FILTER (WHERE n<5) AS users_with_fewer_than_5_items
        FROM (SELECT user_id,count(DISTINCT tv_show_id) n FROM validation_interactions GROUP BY user_id)""")
    # Verify all reported scalar counts against persistent artifacts.
    result["interaction_table_rows"] = con.execute("SELECT count(*) FROM interactions").fetchone()[0]
    assert result["interaction_table_rows"] == matching["split"][2]["usable_interactions"]
    assert matching["coverage"]["raw_viewing_rows"] == audit["datasets"]["dataset11-30.csv"]["rows"]
    assert (matching["coverage"]["identified_matched_rows"] + matching["coverage"]["only_unknown_program_rows"]
            + matching["coverage"]["no_schedule_overlap_rows"]) == matching["coverage"]["raw_viewing_rows"]
    result["runtime_seconds"] = round(time.perf_counter() - started, 3)
    save_json("verification.json", result)
    con.close()
    print(f"Verification complete: {result['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    run()
