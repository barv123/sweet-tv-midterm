"""Recheck final-period submission constraints without rerunning any model."""

import csv
import hashlib

from advanced_common import load_json
from baseline_experiment import FINAL, TARGET_END, raw_integrity, require
from common import RESULTS, SCHEDULES, TIMEZONE, connect, literal, one, save_json
from validate_submission import candidate_ids, read_template, validate


def run():
    con = connect()
    path = RESULTS / "advanced_candidate_submission.csv"
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    with path.open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    template = read_template()
    expected = load_json("advanced_submission_validation.json")
    require(digest == expected["submission_validation"]["sha256"], "Candidate file changed")
    eligible = candidate_ids(con, FINAL, TARGET_END, SCHEDULES[1:])
    csv_validation = validate(path, template, eligible)
    csv_validation["path"] = "results/advanced_candidate_submission.csv"
    con.execute("CREATE OR REPLACE TEMP TABLE final_prediction_pairs(user_id VARCHAR,tv_show_id BIGINT)")
    pairs = [(row["user_id"], int(item)) for row in predictions for item in row["tv_show_id"].split()]
    con.executemany("INSERT INTO final_prediction_pairs VALUES (?,?)", pairs)
    final_end = one(con, f"""SELECT count(*) AS training_interactions,
        count(DISTINCT viewing_id) AS training_sessions,count(DISTINCT user_id) AS historical_users,
        max(session_end) AS last_training_session_end FROM interactions
        WHERE session_start>='2020-03-09 00:00:00' AND session_end<=TIMESTAMP '{FINAL}'""")
    seen = one(con, f"""SELECT count(*) AS prediction_slots,
        count(DISTINCT (p.user_id,p.tv_show_id)) AS unique_user_item_predictions,
        count(*) FILTER (WHERE EXISTS (SELECT 1 FROM interactions i WHERE i.user_id=p.user_id
          AND i.tv_show_id=p.tv_show_id AND i.session_end<=TIMESTAMP '{FINAL}')) AS previously_seen_slots,
        count(DISTINCT (p.user_id,p.tv_show_id)) FILTER (WHERE EXISTS (SELECT 1 FROM interactions i
          WHERE i.user_id=p.user_id AND i.tv_show_id=p.tv_show_id AND i.session_end<=TIMESTAMP '{FINAL}')) AS previously_seen_user_item_pairs
        FROM final_prediction_pairs p""")
    schedule = one(con, f"""SELECT count(*) FILTER (WHERE tv_show_id>0 AND duration>0) AS positive_nonzero_airings_in_source_files,
        count(*) FILTER (WHERE tv_show_id>0 AND duration>0 AND start_time<TIMESTAMP '{TARGET_END}'
          AND timezone('{TIMEZONE}',timezone('{TIMEZONE}',start_time)+duration*INTERVAL '1 second')>TIMESTAMP '{FINAL}') AS airings_intersecting_target,
        count(DISTINCT tv_show_id) FILTER (WHERE tv_show_id>0 AND duration>0 AND start_time<TIMESTAMP '{TARGET_END}'
          AND timezone('{TIMEZONE}',timezone('{TIMEZONE}',start_time)+duration*INTERVAL '1 second')>TIMESTAMP '{FINAL}') AS target_candidates,
        count(*) FILTER (WHERE source={literal(SCHEDULES[1])} AND start_time<TIMESTAMP '{FINAL}'
          AND timezone('{TIMEZONE}',timezone('{TIMEZONE}',start_time)+duration*INTERVAL '1 second')>TIMESTAMP '{FINAL}'
          AND tv_show_id>0 AND duration>0) AS previous_file_boundary_airings,
        count(*) FILTER (WHERE start_time>=TIMESTAMP '{TARGET_END}' AND tv_show_id>0 AND duration>0) AS starts_at_or_after_exclusive_end
        FROM schedule WHERE source IN ({','.join(literal(s) for s in SCHEDULES[1:])})""")
    require(csv_validation["passed"], "Final candidate file structure invalid")
    require(seen["prediction_slots"] == 6300 and seen["unique_user_item_predictions"] == 6300,
            "Final file does not contain 6300 unique user-item slots")
    require(schedule["target_candidates"] == len(eligible) == 6649, "Candidate boundary count changed")
    require(final_end["last_training_session_end"].isoformat(sep=" ") < FINAL, "Final model uses future behavior")
    result = {"submission_validation": csv_validation, "final_training_history": final_end,
        "final_schedule_and_boundaries": schedule, "previously_seen_recommendations": seen,
        "candidate_window": {"start_inclusive": FINAL, "end_exclusive": TARGET_END,
            "eligible_program_ids": len(eligible), "uses_positive_tv_show_ids_only": True},
        "raw_inputs_match_audit": False,
        "official_kaggle_rules_found_locally": False,
        "official_rule_limitations": ["No organizer evaluator, scoring source, or candidate-policy document exists in the local project.",
            "The template confirms columns, row order, and five space-separated slots; it does not define MAP@5 empty-truth handling or repeat eligibility.",
            "Previously watched programs and the project's >=60-second relevance rule are local protocol assumptions, not independently confirmed Kaggle rules."],
        "kaggle_uploaded": False}
    # Explicitly verify every raw source checksum rather than relying on dict truthiness.
    audit_hashes = {name: info["sha256"] for name, info in load_json("audit.json")["datasets"].items()}
    result["raw_inputs_match_audit"] = raw_integrity() == audit_hashes
    require(result["raw_inputs_match_audit"], "A raw CSV changed")
    save_json("final_protocol_check.json", result)
    con.close()
    print(result)


if __name__ == "__main__":
    run()
