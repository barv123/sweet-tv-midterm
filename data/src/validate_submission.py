"""Validate the saved CSV against the original template and final schedule.

Run: python data/src/validate_submission.py [path/to/submission.csv]
Uses the prepared audit database; never uploads a submission.
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from common import RAW, RESULTS, SCHEDULES, TIMEZONE, connect, literal


def candidate_ids(con, lower, upper, files):
    return {row[0] for row in con.execute(f"""SELECT DISTINCT tv_show_id FROM schedule
        WHERE tv_show_id>0 AND duration>0 AND source IN ({','.join(literal(file) for file in files)})
          AND start_time<TIMESTAMP {literal(upper)}
          AND timezone('{TIMEZONE}',timezone('{TIMEZONE}',start_time)+duration*INTERVAL '1 second')
            >TIMESTAMP {literal(lower)}""").fetchall()}


def read_template():
    with (RAW / "submission.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["user_id", "tv_show_id"]:
            raise ValueError("Unexpected original template header")
        users = [row["user_id"] for row in reader]
    if len(users) != 1260 or len(set(users)) != len(users):
        raise ValueError("Unexpected template user count or duplicate users")
    return users


def validate(path, expected_users, eligible):
    path = Path(path)
    content = path.read_bytes()
    errors = []
    if content.startswith(b"\xef\xbb\xbf"):
        errors.append("UTF-8 BOM present")
    if b"\r" in content:
        errors.append("Expected LF line endings")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["user_id", "tv_show_id"]:
            errors.append("Header must be user_id,tv_show_id")
        records = list(reader)
    users = [record.get("user_id") for record in records]
    if users != expected_users:
        errors.append("Users or row order differ from the original template")
    if len(set(users)) != len(users):
        errors.append("Duplicate user IDs")
    invalid_format = invalid_items = repeated_items = 0
    for record in records:
        value = record.get("tv_show_id")
        if set(record) != {"user_id", "tv_show_id"} or value is None or not re.fullmatch(r"[1-9][0-9]*( [1-9][0-9]*){4}", value):
            invalid_format += 1
            continue
        items = [int(token) for token in value.split(" ")]
        repeated_items += len(set(items)) != 5
        invalid_items += any(item not in eligible for item in items)
    if invalid_format or invalid_items or repeated_items:
        errors.append("Malformed, ineligible, nonpositive, or repeated predictions")
    result = {
        "path": str(path), "rows": len(records), "unique_users": len(set(users)),
        "template_order_exact": users == expected_users, "eligible_candidates": len(eligible),
        "prediction_slots": len(records) * 5, "invalid_format_rows": invalid_format,
        "ineligible_item_rows": invalid_items, "repeated_item_rows": repeated_items,
        "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
        "header": "user_id,tv_show_id", "line_endings": "LF", "errors": errors, "passed": not errors,
    }
    if errors:
        raise ValueError(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=RESULTS / "submission_baseline.csv")
    args = parser.parse_args()
    con = connect()
    eligible = candidate_ids(con, "2020-07-27", "2020-10-19", SCHEDULES[1:])
    con.close()
    print(json.dumps(validate(args.path, read_template(), eligible), indent=2))


if __name__ == "__main__":
    main()
