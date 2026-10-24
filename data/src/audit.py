"""Audit every CSV without loading the full collection into Python memory.

Run from the project root: python data/src/audit.py
Raw files are opened read-only; all working tables are stored under .cache/.
"""

import csv
import hashlib
import time

from common import RAW, SCHEDULES, connect, literal, one, rows, save_csv, save_json


def duration_summary(con, table, column):
    return one(con, f"""
        SELECT count(*) AS rows, count({column}) AS nonmissing,
          min({column}) AS minimum, max({column}) AS maximum,
          avg({column}) AS mean, stddev_pop({column}) AS stddev,
          quantile_cont({column}, [0.01,0.05,0.25,0.5,0.75,0.9,0.95,0.99,0.999]) AS quantiles,
          count(*) FILTER (WHERE {column}<0) AS negative,
          count(*) FILTER (WHERE {column}=0) AS zero,
          count(*) FILTER (WHERE {column}>0 AND {column}<10) AS under_10_seconds,
          count(*) FILTER (WHERE {column}>0 AND {column}<60) AS under_60_seconds,
          count(*) FILTER (WHERE {column}>86400) AS over_24_hours,
          count(*) FILTER (WHERE {column}>604800) AS over_7_days
        FROM {table}
    """)


def run():
    started = time.perf_counter()
    con = connect()
    report = {"duckdb_version": __import__("duckdb").__version__, "memory_limit": "768MB", "threads": 2,
              "datasets": {}, "timestamp_note": "Both timestamp formats are timezone-naive; no UTC offset is encoded."}
    con.execute("DROP TABLE IF EXISTS schedule")
    con.execute("""CREATE TABLE schedule (
        schedule_id BIGINT, source VARCHAR, source_row BIGINT, channel_id BIGINT,
        channel_title VARCHAR, start_time TIMESTAMP, end_time TIMESTAMP,
        duration BIGINT, tv_show_id BIGINT, tv_show_title VARCHAR)""")
    schedule_offset = 0
    for filename in ["dataset11-30.csv", *SCHEDULES, "submission.csv"]:
        tick = time.perf_counter()
        path = RAW / filename
        print(f"Auditing {filename}", flush=True)
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
        # Single-threaded import preserves source row numbers for independent checks.
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=true")
        con.execute(f"""CREATE OR REPLACE TABLE raw_stage AS
            SELECT row_number() OVER () AS source_row, * FROM read_csv(
              {literal(path)}, header=true, all_varchar=true, strict_mode=true,
              parallel=false, null_padding=false, ignore_errors=false)""")
        con.execute("SET threads=2")
        con.execute("SET preserve_insertion_order=false")
        count = con.execute("SELECT count(*) FROM raw_stage").fetchone()[0]
        info = {"rows": count, "bytes": path.stat().st_size, "sha256": digest, "columns": header,
                "storage_types": {column: "VARCHAR (lossless CSV text)" for column in header}}
        info["missing"] = one(con, "SELECT " + ", ".join(
            f'count(*) FILTER (WHERE "{column}" IS NULL OR trim("{column}")=\'\') AS "{column}"'
            for column in header) + " FROM raw_stage")
        group = ", ".join('"' + column + '"' for column in header)
        info["duplicate_rows_excess"] = con.execute(
            f"SELECT coalesce(sum(n-1),0) FROM (SELECT count(*) n FROM raw_stage GROUP BY {group} HAVING count(*)>1)"
        ).fetchone()[0]
        if filename.startswith("dataset"):
            info["logical_types"] = {"user_id": "decimal unsigned 64-bit identifier stored as VARCHAR",
                "vsetv_id": "BIGINT", "start_time": "TIMESTAMP", "stop_time": "TIMESTAMP", "duraton": "BIGINT seconds (source spelling)"}
            info["timestamp_format"] = "%Y-%m-%d %H:%M:%S"
            con.execute("""CREATE OR REPLACE TABLE viewing AS SELECT source_row AS viewing_id,
                user_id, try_cast(vsetv_id AS BIGINT) AS channel_id,
                try_strptime(start_time, '%Y-%m-%d %H:%M:%S') AS start_time,
                try_strptime(stop_time, '%Y-%m-%d %H:%M:%S') AS end_time,
                try_cast(duraton AS BIGINT) AS duration FROM raw_stage""")
            info["identifier_checks"] = one(con, """SELECT
                count(*) FILTER (WHERE try_cast(user_id AS UBIGINT) IS NULL) AS invalid_user_ids,
                count(*) FILTER (WHERE try_cast(user_id AS HUGEINT)>9223372036854775807) AS users_above_signed_int64_rows,
                count(*) FILTER (WHERE user_id<>cast(try_cast(user_id AS UBIGINT) AS VARCHAR)) AS noncanonical_user_ids,
                count(*) FILTER (WHERE try_cast(vsetv_id AS BIGINT) IS NULL) AS invalid_channel_ids
                FROM raw_stage""")
            info["coverage"] = one(con, """SELECT count(DISTINCT user_id) AS unique_users,
                count(DISTINCT channel_id) AS unique_channels, min(start_time) AS first_start,
                max(start_time) AS last_start, min(end_time) AS first_stop, max(end_time) AS last_stop,
                count(*) FILTER (WHERE start_time IS NULL) AS invalid_start,
                count(*) FILTER (WHERE end_time IS NULL) AS invalid_stop,
                count(*) FILTER (WHERE duration IS NULL) AS invalid_duration,
                count(*) FILTER (WHERE end_time<=start_time) AS nonpositive_intervals,
                count(*) FILTER (WHERE duration<>date_diff('second',start_time,end_time)) AS duration_mismatch,
                count(*) FILTER (WHERE channel_id<=0) AS nonpositive_channel_ids
                FROM viewing""")
            info["duration"] = duration_summary(con, "viewing", "duration")
            info["elapsed_duration"] = duration_summary(con, "viewing", "date_diff('second',start_time,end_time)")
            info["conflicting_session_keys"] = con.execute("""SELECT count(*) FROM (
                SELECT user_id,channel_id,start_time FROM viewing GROUP BY ALL
                HAVING count(DISTINCT (end_time,duration))>1)""").fetchone()[0]
            info["user_overlapping_records"] = one(con, """SELECT
                count(*) FILTER (WHERE start_time<previous_end) AS rows_overlapping_previous_running_end,
                count(DISTINCT user_id) FILTER (WHERE start_time<previous_end) AS users_with_overlap
                FROM (SELECT *,max(end_time) OVER (PARTITION BY user_id ORDER BY start_time,viewing_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) previous_end FROM viewing)""")
            save_csv("viewing_by_week.csv", rows(con, """SELECT year(start_time) AS year,week(start_time) AS iso_week,
                count(*) AS viewing_rows,count(DISTINCT user_id) AS users,count(DISTINCT channel_id) AS channels,
                min(start_time) AS first_start,max(end_time) AS last_stop,sum(duration) AS seconds
                FROM viewing GROUP BY ALL ORDER BY year,iso_week"""))
        elif filename.startswith("export"):
            info["logical_types"] = {column: ("BIGINT" if column in ("channel_id", "duration", "tv_show_id")
                else "TIMESTAMP" if column == "start_time" else "VARCHAR") for column in header}
            info["timestamp_format"] = "%d.%m.%Y %H:%M:%S"
            con.execute(f"""INSERT INTO schedule SELECT {schedule_offset}+source_row,
                {literal(filename)},source_row,try_cast(channel_id AS BIGINT),channel_title,
                try_strptime(start_time, '%d.%m.%Y %H:%M:%S'),
                try_strptime(start_time, '%d.%m.%Y %H:%M:%S')+try_cast(duration AS BIGINT)*INTERVAL '1 second',
                try_cast(duration AS BIGINT),try_cast(tv_show_id AS BIGINT),tv_show_title FROM raw_stage""")
            con.execute(f"CREATE OR REPLACE TEMP VIEW current_schedule AS SELECT * FROM schedule WHERE source={literal(filename)}")
            info["coverage"] = one(con, """SELECT count(DISTINCT channel_id) AS unique_channels,
                count(DISTINCT tv_show_id) FILTER (WHERE tv_show_id>0) AS identified_programs,
                count(DISTINCT tv_show_title) AS unique_program_titles,
                min(start_time) AS first_start,max(start_time) AS last_start,max(end_time) AS last_end,
                count(*) FILTER (WHERE start_time IS NULL) AS invalid_start,
                count(*) FILTER (WHERE duration IS NULL) AS invalid_duration,
                count(*) FILTER (WHERE channel_id IS NULL OR channel_id<=0) AS invalid_channel_ids,
                count(*) FILTER (WHERE tv_show_id IS NULL OR tv_show_id<0) AS invalid_program_ids,
                count(*) FILTER (WHERE tv_show_id=0) AS zero_program_rows,
                100.0*count(*) FILTER (WHERE tv_show_id=0)/count(*) AS zero_program_percent
                FROM current_schedule""")
            info["duration"] = duration_summary(con, "current_schedule", "duration")
            schedule_offset += count
        else:
            info["logical_types"] = {"user_id": "decimal unsigned 64-bit identifier stored as VARCHAR", "tv_show_id": "space-separated list of five item IDs"}
            con.execute("CREATE OR REPLACE TABLE submission AS SELECT * FROM raw_stage")
            info["format"] = one(con, """SELECT count(DISTINCT user_id) AS unique_users,
                count(*) FILTER (WHERE tv_show_id='0 0 0 0 0') AS all_zero_placeholder_rows,
                count(*) FILTER (WHERE NOT regexp_full_match(tv_show_id,'[0-9]+ [0-9]+ [0-9]+ [0-9]+ [0-9]+')) AS invalid_list_rows,
                count(*) FILTER (WHERE try_cast(user_id AS UBIGINT) IS NULL) AS invalid_user_ids,
                min(length(user_id)) AS min_user_id_digits,max(length(user_id)) AS max_user_id_digits
                FROM submission""")
            with path.open("rb") as handle:
                prefix = handle.read(65536)
            info["format"].update({"delimiter": ",", "encoding": "UTF-8 compatible ASCII",
                "has_utf8_bom": prefix.startswith(b'\xef\xbb\xbf'),
                "line_endings": "CRLF" if b'\r\n' in prefix else "LF", "header": ",".join(header)})
        info["runtime_seconds"] = round(time.perf_counter() - tick, 3)
        report["datasets"][filename] = info
        save_json("audit.json", report)
        print(f"  {count:,} rows; {info['runtime_seconds']:.1f}s", flush=True)
    con.execute("DROP TABLE raw_stage")
    report["schedule_integrity"] = one(con, """SELECT count(*) AS rows,
        count(DISTINCT channel_id) AS channels,
        count(DISTINCT tv_show_id) FILTER (WHERE tv_show_id>0) AS identified_programs,
        count(*) FILTER (WHERE duration<=0) AS nonpositive_durations
        FROM schedule""")
    report["schedule_integrity"]["semantic_duplicate_rows_excess"] = con.execute("""SELECT sum(n-1) FROM (
        SELECT count(*) n FROM schedule GROUP BY channel_id,channel_title,start_time,end_time,duration,tv_show_id,tv_show_title)""").fetchone()[0]
    report["schedule_integrity"]["conflicting_channel_start_keys"] = con.execute("""SELECT count(*) FROM (
        SELECT channel_id,start_time FROM schedule GROUP BY ALL HAVING count(DISTINCT (end_time,tv_show_id))>1)""").fetchone()[0]
    report["schedule_integrity"]["overlap"] = one(con, """SELECT
        count(*) FILTER (WHERE start_time<previous_end) AS overlapping_rows,
        count(*) FILTER (WHERE start_time>previous_end) AS gap_rows,
        max(date_diff('second',previous_end,start_time)) AS largest_gap_seconds
        FROM (SELECT *,max(end_time) OVER (PARTITION BY channel_id ORDER BY start_time,schedule_id
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS previous_end FROM schedule)""")
    report["schedule_integrity"]["channel_ids_with_multiple_titles"] = con.execute("""SELECT count(*) FROM (
        SELECT channel_id FROM schedule GROUP BY channel_id HAVING count(DISTINCT channel_title)>1)""").fetchone()[0]
    report["schedule_integrity"]["channel_titles_with_multiple_ids"] = con.execute("""SELECT count(*) FROM (
        SELECT channel_title FROM schedule GROUP BY channel_title HAVING count(DISTINCT channel_id)>1)""").fetchone()[0]
    report["schedule_integrity"]["program_ids_with_multiple_titles"] = con.execute("""SELECT count(*) FROM (
        SELECT tv_show_id FROM schedule WHERE tv_show_id>0 GROUP BY tv_show_id HAVING count(DISTINCT tv_show_title)>1)""").fetchone()[0]
    report["schedule_integrity"]["program_titles_with_multiple_positive_ids"] = con.execute("""SELECT count(*) FROM (
        SELECT tv_show_title FROM schedule WHERE tv_show_id>0 GROUP BY tv_show_title HAVING count(DISTINCT tv_show_id)>1)""").fetchone()[0]
    save_csv("schedule_by_week.csv", rows(con, """SELECT source,year(start_time) AS year,week(start_time) AS iso_week,
        count(*) AS schedule_rows,count(DISTINCT channel_id) AS channels,
        count(DISTINCT tv_show_id) FILTER (WHERE tv_show_id>0) AS identified_programs,
        count(*) FILTER (WHERE tv_show_id=0) AS zero_program_rows,
        min(start_time) AS first_start,max(end_time) AS last_end FROM schedule GROUP BY ALL ORDER BY source,year,iso_week"""))
    save_csv("channel_coverage.csv", rows(con, """WITH v AS (
        SELECT channel_id,count(*) viewing_rows,count(DISTINCT user_id) users,min(start_time) first_view,max(end_time) last_view
        FROM viewing GROUP BY channel_id), s AS (
        SELECT channel_id,count(*) schedule_rows,count(DISTINCT channel_title) title_count,
        min(channel_title) example_title,min(start_time) first_schedule,max(end_time) last_schedule
        FROM schedule GROUP BY channel_id)
        SELECT coalesce(v.channel_id,s.channel_id) channel_id,v.* EXCLUDE(channel_id),s.* EXCLUDE(channel_id)
        FROM v FULL JOIN s USING(channel_id) ORDER BY channel_id"""))
    report["channel_id_evidence"] = one(con, """SELECT count(*) AS viewing_rows,
        count(DISTINCT channel_id) AS viewing_channels,
        count(*) FILTER (WHERE channel_id IN (SELECT channel_id FROM schedule)) AS rows_with_shared_numeric_channel_id,
        count(DISTINCT channel_id) FILTER (WHERE channel_id IN (SELECT channel_id FROM schedule)) AS shared_numeric_channel_ids,
        list(DISTINCT channel_id ORDER BY channel_id) FILTER (WHERE channel_id NOT IN (SELECT channel_id FROM schedule)) AS viewing_only_channel_ids
        FROM viewing""")
    report["users"] = one(con, """SELECT count(DISTINCT user_id) AS all_viewing_users,
        count(DISTINCT user_id) FILTER (WHERE channel_id>0 AND end_time>start_time AND duration>0
          AND duration=date_diff('second',start_time,end_time)) AS valid_interval_users,
        count(DISTINCT user_id) FILTER (WHERE user_id IN (SELECT user_id FROM submission)) AS viewing_users_in_submission
        FROM viewing""")
    report["users"]["submission_users_without_viewing"] = con.execute("""SELECT count(*) FROM submission
        WHERE user_id NOT IN (SELECT user_id FROM viewing)""").fetchone()[0]
    report["runtime_seconds"] = round(time.perf_counter() - started, 3)
    save_json("audit.json", report)
    con.execute("CHECKPOINT")
    con.close()
    print(f"Audit complete: {report['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    run()
