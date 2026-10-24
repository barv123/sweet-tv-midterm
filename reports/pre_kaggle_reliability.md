# Pre-Kaggle Reliability Check

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

| Method | MAP@5 | Users | Excluded | Candidates | Relevant pairs | Empty truth | Runtime |
| --- | --- | --- | --- | --- | --- | --- | --- |
| popularity | 0.63760317 | 1260 | 0 | 2143 | 75,493 | 39 | 0.135 |
| repeat_recency | 0.72389418 | 1260 | 0 | 2143 | 75,493 | 39 | 0.266 |
| hybrid_rrf60 | 0.67445966 | 1260 | 0 | 2143 | 75,493 | 39 | 2.984 |

Hybrid minus popularity is **+0.03685648**. Per user,
hybrid beat popularity for 427 users, tied
595, and lost for 238.
However, repeat/recency outscored hybrid by **0.04943452** on this holdout.
The inner hybrid score is **0.67445966**, versus **0.81696561** on the later
weeks 21-30 validation. This one check supports some hybrid improvement over popularity,
but not the previous conclusion that hybrid is the strongest strategy. It is a single
holdout, not a statistically independent final benchmark.

### Leakage and calculation checks

Training stops at **2020-04-05 23:59:59**. The following checks all returned zero:

| Check | Violations |
| --- | --- |
| training_rows_after_cutoff | 0 |
| validation_rows_outside_window | 0 |
| shared_viewing_ids | 0 |
| training_interactions_using_later_schedule_file | 0 |
| labels_outside_candidate_set | 0 |

Training produced 945,701 training interactions from
4,779 users; the held-out partition has
461,908 interaction rows. The candidate pool has
23,362 intersecting airings.
The 14-day repeat profile, min-pair-count-25/lift-1 association rules, and RRF constant
60 were fixed from the existing experiment. The 18 users without training history use
the same global popularity fallback for all methods.

Python MAP@5 was compared with an independent SQL window-function implementation for
all 3,780 method-user scores; maximum difference
was 0.0. All top-five lists were checked
for five unique positive IDs in the holdout candidate set. Per-method runtimes exclude
1.282 s of shared sparse-history preparation;
hybrid runtime includes popularity, repeat scoring, rule fitting/scoring, and fusion.

## Two Held-Out User Case Studies (Weeks 21-30)

These are two rows from the existing later validation, not the new inner holdout.
The 40-row evidence CSV records the five recommendations from popularity,
content-based filtering (metadata + title), association rules, and Hybrid RRF60;
each row includes the held-out relevance flag. Template row 2 has 19 relevant programs;
row 44 has 87. Program titles/categories are the most frequent source-schedule values
with lexical tie-breaking, retained verbatim (the source metadata is Russian).

| Template row | Relevant count | Method | Rank | Program ID | Representative title | Category | Held-out relevant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 19 | popularity | 1 | 2400480 | ТСН. | Инфо | Yes |
| 2 | 19 | popularity | 2 | 240081 | Факты. | Инфо | Yes |
| 2 | 19 | popularity | 3 | 2400508 | Сегодня. | Инфо | No |
| 2 | 19 | popularity | 4 | 6500479 | 'Мир наизнанку'. | Познавательное | Yes |
| 2 | 19 | popularity | 5 | 2400467 | 'Викна-Новости'. | Инфо | No |
| 2 | 19 | content_metadata_title | 1 | 2400480 | ТСН. | Инфо | Yes |
| 2 | 19 | content_metadata_title | 2 | 240081 | Факты. | Инфо | Yes |
| 2 | 19 | content_metadata_title | 3 | 5900347 | 'Деньги'. | Инфо | No |
| 2 | 19 | content_metadata_title | 4 | 6700482 | Секретные материалы. | Другое | Yes |
| 2 | 19 | content_metadata_title | 5 | 500368 | Ток-шоу '60 минут' с Ольгой Скабеевой и Евгением Поповым. | Развлечения | No |
| 2 | 19 | association_count25_lift1 | 1 | 2400508 | Сегодня. | Инфо | No |
| 2 | 19 | association_count25_lift1 | 2 | 6500479 | 'Мир наизнанку'. | Познавательное | Yes |
| 2 | 19 | association_count25_lift1 | 3 | 2400467 | 'Викна-Новости'. | Инфо | No |
| 2 | 19 | association_count25_lift1 | 4 | 240081 | Факты. | Инфо | Yes |
| 2 | 19 | association_count25_lift1 | 5 | 10002514 | Т/с 'Кухня'. | Сериалы | Yes |
| 2 | 19 | hybrid_rrf60 | 1 | 2400480 | ТСН. | Инфо | Yes |
| 2 | 19 | hybrid_rrf60 | 2 | 240081 | Факты. | Инфо | Yes |
| 2 | 19 | hybrid_rrf60 | 3 | 6500479 | 'Мир наизнанку'. | Познавательное | Yes |
| 2 | 19 | hybrid_rrf60 | 4 | 200352 | 'Вечерний квартал'. | Развлечения | Yes |
| 2 | 19 | hybrid_rrf60 | 5 | 6700482 | Секретные материалы. | Другое | Yes |
| 44 | 87 | popularity | 1 | 2400480 | ТСН. | Инфо | No |
| 44 | 87 | popularity | 2 | 240081 | Факты. | Инфо | No |
| 44 | 87 | popularity | 3 | 2400508 | Сегодня. | Инфо | No |
| 44 | 87 | popularity | 4 | 6500479 | 'Мир наизнанку'. | Познавательное | No |
| 44 | 87 | popularity | 5 | 2400467 | 'Викна-Новости'. | Инфо | No |
| 44 | 87 | content_metadata_title | 1 | 90054150 | Х/ф 'Даже не думай!'. | Фильмы | No |
| 44 | 87 | content_metadata_title | 2 | 90058273 | Х/ф 'Турбофорсаж'. | Фильмы | No |
| 44 | 87 | content_metadata_title | 3 | 90076564 | Х/ф 'Бесплатные образцы'. | Фильмы | No |
| 44 | 87 | content_metadata_title | 4 | 90076595 | Х/ф 'Гуманитарные науки'. | Фильмы | No |
| 44 | 87 | content_metadata_title | 5 | 90079285 | Х/ф 'Тронутые'. | Фильмы | No |
| 44 | 87 | association_count25_lift1 | 1 | 240081 | Факты. | Инфо | No |
| 44 | 87 | association_count25_lift1 | 2 | 10002514 | Т/с 'Кухня'. | Сериалы | No |
| 44 | 87 | association_count25_lift1 | 3 | 700389 | Кто сверху? | Развлечения | No |
| 44 | 87 | association_count25_lift1 | 4 | 200352 | 'Вечерний квартал'. | Развлечения | No |
| 44 | 87 | association_count25_lift1 | 5 | 2400480 | ТСН. | Инфо | No |
| 44 | 87 | hybrid_rrf60 | 1 | 240081 | Факты. | Инфо | No |
| 44 | 87 | hybrid_rrf60 | 2 | 2400480 | ТСН. | Инфо | No |
| 44 | 87 | hybrid_rrf60 | 3 | 10002514 | Т/с 'Кухня'. | Сериалы | No |
| 44 | 87 | hybrid_rrf60 | 4 | 6500479 | 'Мир наизнанку'. | Познавательное | No |
| 44 | 87 | hybrid_rrf60 | 5 | 200352 | 'Вечерний квартал'. | Развлечения | No |

The case study demonstrates both a favorable user (row 2: popularity AP@5 0.55,
hybrid 1.00) and a difficult user (row 44: popularity, content, association, and
hybrid each AP@5 0.00). It is descriptive evidence, not another tuning set.

## Local Competition-Rule and Final-File Check

No official Kaggle rules/evaluator were found in the local project. The project template and reports document the agreed local AP@5 convention
and submission shape, but cannot establish the organizer's actual evaluator. In
particular, empty-truth handling, minimum viewing duration, recommendation eligibility,
target-schedule availability, and whether repeats are accepted remain unverified against
Kaggle's rules.

The local final candidate rule is interval intersection with the final target window
**[2020-07-27 00:00:00, 2020-10-19 00:00:00)**,
using positive program IDs from schedule files weeks 21-30 and 31-42. It yields
6,649 IDs. The training history ends at
**2020-07-26 23:59:58**, before the target starts.
Boundary checks counted 58 prior-file
airings intersecting the target start and excluded
257 airings starting at/after the
exclusive target end. The schedule exports do not prove when those schedules became
available to competitors; their prediction-time availability cannot be confirmed.

Independent revalidation of `results/advanced_candidate_submission.csv` passed:

- Exact original header and all **1,260** users in template order.
- **6,300** prediction slots, five distinct positive eligible IDs per user.
- **0** malformed rows, invalid candidate IDs, or repeated IDs within a user's list.
- **6,189 of 6,300** slots recommend a program
  that user had already watched in the available training history. This follows the local
  protocol, but is a material unresolved rule risk: repeat eligibility is not specified
  by the template.
- SHA-256 `4abb82f4dd24b0c8f199ea33262f2c2ec474e42cd161f8b8083f3d92d85248ee`; unchanged from the previously verified candidate.
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
