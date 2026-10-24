"""Shared disk-backed analysis utilities."""

import csv
import json
import os
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
CACHE = Path(os.environ.get("SWEET_TV_CACHE", ROOT / ".cache" / "audit"))
SCHEDULES = [f"export_arh_{period}-final.csv" for period in ("11-20", "21-30", "31-42")]
TIMEZONE = "Europe/Kyiv"


def elapsed_sql(start, end):
    return f"date_diff('second',timezone('{TIMEZONE}',{start}),timezone('{TIMEZONE}',{end}))"


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def connect():
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    con = duckdb.connect(str(CACHE / "audit.duckdb"))
    con.execute("SET memory_limit='768MB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory={literal(CACHE / 'spill')}")
    return con


def rows(con, sql):
    result = con.execute(sql)
    names = [item[0] for item in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


def one(con, sql):
    return rows(con, sql)[0]


def save_json(name, value):
    (RESULTS / name).write_text(json.dumps(value, indent=2, default=str, ensure_ascii=True) + "\n", encoding="utf-8")


def save_csv(name, records):
    if not records:
        return
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
