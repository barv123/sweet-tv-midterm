"""Write the focused pre-Kaggle reliability addendum from measured artifacts."""

import csv

from advanced_common import load_json
from common import RESULTS, ROOT


def markdown_table(headers, records):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(str(value) for value in row) + " |" for row in records])


def run():
    inner = load_json("pre_kaggle_reliability.json")
    final = load_json("final_protocol_check.json")
    scores = inner["comparison"]
    methods = ["popularity", "repeat_recency", "hybrid_rrf60"]
    comparison = [[row["method"], f"{row['map_at_5']:.8f}", row["evaluated_users"], row["excluded_users"],
                   row["candidate_count"], f"{row['distinct_relevant_user_item_pairs']:,}",
                   row["empty_truth_users"], f"{row['runtime_seconds']:.3f}"]
                  for row in scores if row["method"] in methods]
    evidence_rows = []
    with (RESULTS / "validation_user_top5_evidence.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            evidence_rows.append([row["template_row"], row["relevant_items_for_user"], row["method"], row["rank"],
                row["tv_show_id"], row["representative_schedule_title"], row["category"],
                "Yes" if row["heldout_relevant"] == "True" else "No"])
    local_rules = "No official Kaggle rules/evaluator were found in the local project."
    report = f"""# Pre-Kaggle Reliability Check

## Decision

**NO-GO for uploading the current hybrid candidate yet.** The final CSV is structurally
valid, but one fixed earlier holdout produces a materially different model ordering,
and the local project contains no official evaluator or candidate/repeat policy. The
current hybrid validation gain is therefore not sufficiently reliable to choose it for
the first competition upload.

## One Additional Chronological Holdout

The existing audited cache was reused; neither the initial audit nor the advanced full
validation experiment was rerun. Training uses completed sessions from weeks 11-14
([2020-03-09, 2020-04-06)); held-out viewing is weeks 15-16
([2020-04-06, 2020-04-20)). The 11-20 schedule defines 2,143 positive-ID programs
whose broadcast intervals intersect the holdout. There are 75,493 distinct relevant
user-program pairs across all 1,260 original template users; no users were excluded,
including 39 with empty observed validation truth. Each method uses the same list of
eligible items, the same labels, the same AP denominator `min(5, relevant count)`, and
zero AP for empty truth. Previously seen programs remain allowed, as in the declared
project protocol. The fixed RRF60 rule settings were carried over without tuning.

{markdown_table(['Method','MAP@5','Users','Excluded','Candidates','Relevant pairs','Empty truth','Runtime'], comparison)}

Hybrid minus popularity is **{inner['hybrid_minus_popularity']['absolute_map_delta']:+.8f}**. Per user,
hybrid beat popularity for {inner['hybrid_minus_popularity']['users_hybrid_better']} users, tied
{inner['hybrid_minus_popularity']['users_tied']}, and lost for {inner['hybrid_minus_popularity']['users_hybrid_worse']}.
However, repeat/recency outscored hybrid by **{scores[1]['map_at_5']-scores[2]['map_at_5']:.8f}** on this holdout.
The inner hybrid score is **{scores[2]['map_at_5']:.8f}**, versus **0.81696561** on the later
weeks 21-30 validation. This one check supports some hybrid improvement over popularity,
but not the previous conclusion that hybrid is the strongest strategy. It is a single
holdout, not a statistically independent final benchmark.

### Leakage and calculation checks

Training stops at **{inner['training_data']['last_session_end']}**. The following checks all returned zero:

{markdown_table(['Check','Violations'], [[name, value] for name, value in inner['temporal_checks'].items()])}

Training produced {inner['training_data']['interactions']:,} training interactions from
{inner['training_data']['users']:,} users; the held-out partition has
{inner['validation_data']['interactions']:,} interaction rows. The candidate pool has
{inner['candidate_schedule']['intersecting_positive_airings']:,} intersecting airings.
The 14-day repeat profile, min-pair-count-25/lift-1 association rules, and RRF constant
60 were fixed from the existing experiment. The 18 users without training history use
the same global popularity fallback for all methods.

Python MAP@5 was compared with an independent SQL window-function implementation for
all {inner['independent_sql_ap']['values_checked']:,} method-user scores; maximum difference
was {inner['independent_sql_ap']['max_absolute_error']:.1f}. All top-five lists were checked
for five unique positive IDs in the holdout candidate set. Per-method runtimes exclude
{inner['shared_matrix_preparation_seconds']:.3f} s of shared sparse-history preparation;
hybrid runtime includes popularity, repeat scoring, rule fitting/scoring, and fusion.

## Two Held-Out User Case Studies (Weeks 21-30)

These are two rows from the existing later validation, not the new inner holdout.
The 40-row evidence CSV records the five recommendations from popularity,
content-based filtering (metadata + title), association rules, and Hybrid RRF60;
each row includes the held-out relevance flag. Template row 2 has 19 relevant programs;
row 44 has 87. Program titles/categories are the most frequent source-schedule values
with lexical tie-breaking, retained verbatim (the source metadata is Russian).

{markdown_table(['Template row','Relevant count','Method','Rank','Program ID','Representative title','Category','Held-out relevant'], evidence_rows)}

The case study demonstrates both a favorable user (row 2: popularity AP@5 0.55,
hybrid 1.00) and a difficult user (row 44: popularity, content, association, and
hybrid each AP@5 0.00). It is descriptive evidence, not another tuning set.

## Local Competition-Rule and Final-File Check

{local_rules} The project template and reports document the agreed local AP@5 convention
and submission shape, but cannot establish the organizer's actual evaluator. In
particular, empty-truth handling, minimum viewing duration, recommendation eligibility,
target-schedule availability, and whether repeats are accepted remain unverified against
Kaggle's rules.

The local final candidate rule is interval intersection with the final target window
**[{final['candidate_window']['start_inclusive']}, {final['candidate_window']['end_exclusive']})**,
using positive program IDs from schedule files weeks 21-30 and 31-42. It yields
{final['candidate_window']['eligible_program_ids']:,} IDs. The training history ends at
**{final['final_training_history']['last_training_session_end']}**, before the target starts.
Boundary checks counted {final['final_schedule_and_boundaries']['previous_file_boundary_airings']} prior-file
airings intersecting the target start and excluded
{final['final_schedule_and_boundaries']['starts_at_or_after_exclusive_end']} airings starting at/after the
exclusive target end. The schedule exports do not prove when those schedules became
available to competitors; their prediction-time availability cannot be confirmed.

Independent revalidation of `results/advanced_candidate_submission.csv` passed:

- Exact original header and all **1,260** users in template order.
- **6,300** prediction slots, five distinct positive eligible IDs per user.
- **0** malformed rows, invalid candidate IDs, or repeated IDs within a user's list.
- **{final['previously_seen_recommendations']['previously_seen_slots']:,} of 6,300** slots recommend a program
  that user had already watched in the available training history. This follows the local
  protocol, but is a material unresolved rule risk: repeat eligibility is not specified
  by the template.
- SHA-256 `{final['submission_validation']['sha256']}`; unchanged from the previously verified candidate.
- Raw CSV checksums still match the original audit. No Kaggle upload was performed.

## Evidence and Reproduction

The inner holdout is reproducible with `data/src/pre_kaggle_reliability.py`; final-file
checks without model refitting use `data/src/final_protocol_check.py`. Evidence is in
`results/pre_kaggle_reliability.json`, `results/inner_holdout_evaluation.csv`,
`results/validation_user_top5_evidence.csv`, and `results/final_protocol_check.json`.
The saved advanced candidate, baseline results, raw inputs, and advanced model results
were not overwritten.

**Remaining pre-upload requirements:** obtain the official metric/candidate/repeat rules,
resolve the conflicting chronological model ranking with another untouched time split
or a rule-driven strategy decision, and only then request approval to upload.
"""
    (ROOT / "reports" / "pre_kaggle_reliability.md").write_text(report, encoding="utf-8")
    existing = (ROOT / "reports" / "advanced_recommendation_experiment.md").read_text(encoding="utf-8")
    marker = "## VIII. Pre-Kaggle Reliability Follow-up"
    if marker not in existing:
        existing += f"""

{marker}

One fixed weeks 15-16 chronological inner holdout and a local candidate-file rules
check were completed after this report. See [the reliability addendum](pre_kaggle_reliability.md)
for measured scores, per-user cases, boundary validation, and the upload recommendation.
"""
        (ROOT / "reports" / "advanced_recommendation_experiment.md").write_text(existing, encoding="utf-8")
    print("Wrote reports/pre_kaggle_reliability.md and linked it from the advanced report.")


if __name__ == "__main__":
    run()
