"""Create a compact, auditable report from the completed strict-label run."""

import csv
import json

from common import RESULTS, ROOT, save_json


def load(name):
    return json.loads((RESULTS/name).read_text(encoding="utf-8"))


def table(headers, body):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]+
        ["| " + " | ".join(map(str,row)) + " |" for row in body])


def run():
    raw=load("strict80_evaluation.json")
    old_advanced=load("advanced_experiment.json")
    old_inner=[]
    with (RESULTS/"inner_holdout_evaluation.csv").open(encoding="utf-8",newline="") as h:
        old_inner=list(csv.DictReader(h))
    old_inner_scores={r["method"]:float(r["map_at_5"]) for r in old_inner}
    old_outer={r["method"]:r["map_at_5"] for r in old_advanced["comparison"]}
    alias={"popularity":"popularity","repeat_recency":"repeat_recency_heuristic",
           "content_based":"content_metadata_title","association_rules":"association_count25_lift1",
           "hybrid_rrf60":"hybrid_rrf60"}
    def proxy(fold, method):
        if fold=="inner_weeks15_16": return old_inner_scores.get(method)
        return old_outer.get(alias[method])
    rows=[]; per_fold={}
    for fold in raw["folds"]:
        label=fold["fold"]
        per_fold[label]={}
        for record in fold["comparison"]:
            method=record["method"]
            per_fold[label][method]=record["map_at_5"]
            old=proxy(label,method)
            rows.append([label,method,"n/a" if old is None else f"{old:.8f}",f"{record['map_at_5']:.8f}",
                record["users"],record["excluded_users"],record["candidates"],
                record["proxy_60s_relevant_user_item_pairs"],record["strict_relevant_user_item_pairs"],
                record["empty_truth_users"]])
    methods=alias.keys()
    averages={m:sum(per_fold[f][m] for f in per_fold)/len(per_fold) for m in methods}
    minima={m:min(per_fold[f][m] for f in per_fold) for m in methods}
    source_info={
        "participant_article_url":"https://medium.com/data-science/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156",
        "participant_claim":"More than 80% of one TV airing, in one sitting without a channel change, defines watched/relevant.",
        "participant_training_example":"The author separately describes more than 50% of an airing in one sitting as a basic training signal.",
        "attribution_correction":"The article is by a second-place participant, not the competition winner; it reports that author's private score as 0.42905.",
        "official_kaggle_url":"https://www.kaggle.com/c/sweettv-tv-program-recommender/overview",
        "official_kaggle_access":"The official page title and closed status are discoverable, but its detailed evaluation/rules content is JavaScript-blocked in this environment.",
        "leaderboard_discrepancy":"An indexed Kaggle page under the sweet-tv-channel alias shows a public leader score of 0.28663; it cannot be reconciled with the article's private 0.42905 from accessible rules/page data.",
        "official_confirmation":"The >80% definition is participant-reported, not independently confirmed in accessible official rules.",
    }
    compact={"source_protocol":source_info,"ground_truth":raw["source_protocol"],
        "training_signal_threshold":">50% of one airing per usable viewing row; no combining across sessions/airings",
        "folds":[{"name":f["fold"],"start":f["start"],"end_exclusive":f["end_exclusive"],
            "candidate_count":f["candidate_count"],"users":f["template_users"],"excluded_users":0,
            "training_signal_rows":f["training_rows"],"train_last_session_end":f["train_last_session_end"],
            "strict_label_summary":f["labels"],"comparison":f["comparison"],
            "temporal_checks":f["checks"],"sql_ap_check":f["independent_sql_ap"]} for f in raw["folds"]],
        "average_strict_map_by_method":averages,"worst_fold_strict_map_by_method":minima,
        "selected_repeat_method":"repeat_recency",
        "selected_robust_alternative":"hybrid_rrf60",
        "selection_rule":"maximin (highest worst-fold strict MAP@5) among methods other than repeat/recency",
        "independent_raw_label_verification":raw["independent_raw_label_verification"],
        "submissions":raw["submission_selection"],"raw_inputs_unchanged":raw["unchanged_raw_hashes"],
        "experiment_runtime_seconds":raw["runtime_seconds"]}
    save_json("strict80_summary.json",compact)
    report=f"""# Reassessment with Single-Airing >80% Viewing Labels

## Protocol provenance and status

The two-layered solution article describes a TV program as watched when more than 80%
of one airing is watched in one sitting without changing channels. The same author
separately describes a >50% per-sitting training feature. This provides strong
competition-specific guidance, but is a participant's account, **not independently
confirmed official rule text**. The author is the article's second-place participant,
not the winner; the post reports that author's private MAP@5 as 0.42905.

Official Kaggle page: [sweet.tv - TV program recommender](https://www.kaggle.com/c/sweettv-tv-program-recommender/overview).
The official title/closed status is accessible; the detailed evaluation/rules page
returns only its JavaScript shell here. An indexed Kaggle leaderboard under the
[`sweet-tv-channel` alias](https://www.kaggle.com/competitions/sweet-tv-channel/leaderboard)
shows a public leader score of 0.28663. This cannot be reconciled
with the participant's stated private 0.42905 from the accessible official page. Neither
score was used for model selection. Thus, use the following as a **strict participant-rule
reconstruction**, not an officially certified evaluator.

### Operational label definition

For each usable raw viewing row and one scheduled airing on the matching numeric channel,
`coverage = overlap_seconds / airing_duration_seconds`. A user-program validation label
is positive if **any single viewing-row/airing pair has coverage > 0.80**. Exactly 80%
is negative. We do not add multiple sessions, airing fragments, or channels together.
`vsetv_id` is treated as the source channel ID mapped to `channel_id`; an input log row
contains only one channel ID, but there is no separate intra-row channel-change telemetry.
An event is therefore assumed uninterrupted within that row. Existing quality gates
(positive intervals, correct DST-aware elapsed duration, <=24-hour viewing record,
first duplicate only, and no overlapping schedule ambiguity) remain applied.

Training features use a separate **strictly >50% coverage per individual airing/session**
signal. No event meeting only 50% qualifies; training signals do not define validation
relevance. Repeat and association histories are deduplicated by user-program for their
binary features. All previously used models retain their fixed parameters, including
the 14-day repeat half-life, association min-pair-count 25/lift 1.0, and RRF60. No new
hyperparameter search was done.

## Label audit and held-out results

Both folds evaluate all 1,260 template users, with empty truth receiving AP=0, and
identical per-fold schedule-intersection candidates for every method.

{table(['Holdout','Method','Old 60-sec proxy MAP@5','Strict >80% MAP@5','Users','Excluded','Candidates','Proxy relevant pairs','Strict relevant pairs','Empty users'],rows)}

The old results in the proxy column used a >=60-second matching rule and are **not
Kaggle-equivalent ground truth**. Training signals for the strict rows above use >50%
per-airing coverage; the proxy training configuration had its previous 60-second event
threshold. The strict criteria sharply reduce user-program relevance:

{table(['Holdout','60-sec proxy pairs','Strict >80% pairs','Proxy-only pairs','Strict >80% positive event/airing pairs'],[
    [f['fold'],f['labels']['proxy_60s_distinct_user_item_pairs'],f['labels']['strict_distinct_user_item_pairs'],
     f['labels']['proxy_only_user_item_pairs'],f['labels']['usable_validation_event_airing_pairs_over_80']] for f in raw['folds']])}

Inner strict labels are {raw['folds'][0]['labels']['strict_distinct_user_item_pairs']:,} pairs versus
{raw['folds'][0]['labels']['proxy_60s_distinct_user_item_pairs']:,} proxy pairs; outer strict labels are
{raw['folds'][1]['labels']['strict_distinct_user_item_pairs']:,} versus
{raw['folds'][1]['labels']['proxy_60s_distinct_user_item_pairs']:,}. That is an {100*raw['folds'][0]['labels']['proxy_only_user_item_pairs']/raw['folds'][0]['labels']['proxy_60s_distinct_user_item_pairs']:.1f}%
proxy-pair reduction in the inner fold and {100*raw['folds'][1]['labels']['proxy_only_user_item_pairs']/raw['folds'][1]['labels']['proxy_60s_distinct_user_item_pairs']:.1f}% in the outer fold. In both folds all strict
pairs are also proxy pairs under the cached >=60-second definition; the proxy adds
many short program views.

Across the two strict folds, mean MAP@5 is: {', '.join(f'{m} {averages[m]:.6f}' for m in methods)}.
Repeat/recency has the best mean and worst-fold strict MAP, **{averages['repeat_recency']:.6f}**
and **{minima['repeat_recency']:.6f}**. Hybrid RRF60 is the best non-repeat alternative
by maximin, at mean {averages['hybrid_rrf60']:.6f} and worst-fold {minima['hybrid_rrf60']:.6f}.
Hybrid remains the strongest outer-fold method but trails repeat/recency in the inner
fold. This is a clearer, stricter signal than the proxy results, but only two temporal
holdouts and the participant-reported definition are available.

## Independent source-row verification

The existing interval cache was reused. For six positive and six negative event examples,
a separate Python `csv.DictReader` plus `zoneinfo` scan re-read the raw viewing and
schedule rows, recomputed UTC interval overlaps and airing fractions, and reconstructed
the complete strict label sets for 12 sampled users. All 12 user-level sets matched the
cached SQL labels. Positive examples include full-airing views; negatives include exact
80% and 79.982% coverage. See `results/strict80_label_examples.json` for source row
numbers, overlap seconds, airing seconds, and direct fractions.

The >80% strict labels were independently compared with the <=80% event labels and
the separate >50% training features. The split checks show zero future training rows,
zero validation rows outside the target windows, and zero strict labels outside the
schedule candidate pools. Python MAP@5 was checked against independent DuckDB window
calculation for 12,600 method-user-fold scores; maximum difference was zero.

## Final-period submissions

The models were refit on all usable pre-final viewing with **>50% coverage of one
airing per viewing row**; final viewing behavior was never read. Final candidates are
positive IDs from schedule airings intersecting `[2020-07-27, 2020-10-19)`, total
6,649 programs. Two separate, validator-passed files were created:

- `results/submission_strict80_repeat_recency.csv` — recommended first candidate based
  on the best mean and worst-fold strict MAP@5.
- `results/submission_strict80_alternative_hybrid_rrf60.csv` — best maximin non-repeat
  alternative under the two folds.

Each has the exact template header/order, 1,260 users, 6,300 slots, five unique positive
eligible items per user, zero malformed rows, and no item 0. Hashes and validation are
in `results/strict80_summary.json`. Previously watched programs were **not** excluded;
the participant's baseline itself recommends past programs if scheduled again. The
official repeat policy remains unverified. Neither file was uploaded.

## Reproducibility and limitations

Run `data/src/strict_relevance_experiment.py` after the existing audit cache is available;
it reuses `match_edges` and does not rebuild the original audit. Five threshold unit tests
cover >80%, >50%, exactly 80%, nonpositive airing duration, and nonaggregation of two
separate 51% sessions. The original CSV SHA-256 values remained unchanged.

Unresolved: official Kaggle scoring/rules content is inaccessible from the official
JavaScript page; the participant post is not organizer documentation; the participant's
0.42905 private score conflicts with the accessible indexed public leaderboard value;
the source cannot detect channel changes hidden within a single viewing row; and the
official policy on recommending previously viewed items is not confirmed. These limits
mean the strict labels are well-verified from supplied records but still provisional as
an exact reconstruction of the organizer evaluator.
"""
    (ROOT/"reports"/"strict_relevance_assessment.md").write_text(report,encoding="utf-8")
    existing=(ROOT/"reports"/"advanced_recommendation_experiment.md").read_text(encoding="utf-8")
    marker="## IX. Strict Relevance Reassessment"
    if marker not in existing:
        existing+=f"""

{marker}

The later strict single-airing >80% ground-truth reassessment supersedes the 60-second
proxy MAP@5 values above for Kaggle-alignment. See [the strict relevance report](strict_relevance_assessment.md).
"""
        (ROOT/"reports"/"advanced_recommendation_experiment.md").write_text(existing,encoding="utf-8")
    print("Wrote reports/strict_relevance_assessment.md")


if __name__=="__main__":
    run()
