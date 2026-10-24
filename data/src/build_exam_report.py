"""Build a six-page IEEE-aligned report solely from frozen evidence and authored text.

Five content pages plus references. Does not train models or reconstruct sessions.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import json
import re
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle, Spacer, Frame, Image
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]; R=ROOT/"results"; OUT=ROOT/"reports"
def load(name): return json.loads((R/name).read_text(encoding="utf-8"))
evidence=load("exam_report_evidence.json"); strict=load("strict80_summary.json")
conservative=load("frequency_conservative_checkpoint.json"); targeted=load("frequency_targeted_checkpoint.json")
frequency=load("frequency_no_decay_results.json"); sprint=load("final_competitive_sprint.json")
channel=load("channel_propensity_checkpoint.json")
for name,file in (("TR","times.ttf"),("TB","timesbd.ttf"),("TI","timesi.ttf"),("TBI","timesbi.ttf")):
    pdfmetrics.registerFont(TTFont(name,str(Path("C:/Windows/Fonts")/file)))
pdfmetrics.registerFontFamily("TR",normal="TR",bold="TB",italic="TI",boldItalic="TBI")
body=ParagraphStyle("body",fontName="TR",fontSize=9.2,leading=11.2,spaceAfter=5)
small=ParagraphStyle("small",parent=body,fontSize=8,leading=9.5,spaceAfter=3)
head=ParagraphStyle("head",parent=body,fontName="TB",fontSize=11,leading=13,spaceBefore=6,spaceAfter=5)
title=ParagraphStyle("title",parent=head,fontSize=17,leading=19,alignment=1)
center=ParagraphStyle("center",parent=body,alignment=1)
source=[]
def p(text,style=body):
    source.append(re.sub("<[^>]+>","",text))
    return Paragraph(text,style)
def h(text): return p(text,head)
def table(headers,rows,widths):
    source.append(" | ".join(headers)); source.extend(" | ".join(map(str,r)) for r in rows)
    data=[[Paragraph(escape(str(x)),small) for x in headers]]+[[Paragraph(escape(str(x)),small) for x in row] for row in rows]
    t=Table(data,colWidths=widths,hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e8edf3")),("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LINEBELOW",(0,0),(-1,0),.5,colors.black),("LINEBELOW",(0,-1),(-1,-1),.3,colors.grey),
        ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    return t
W,H=A4; margin=38; gap=18; full=W-2*margin; col=(full-gap)/2
path=OUT/"amina_bayandinova_exam_report.pdf"
c=canvas.Canvas(str(path),pagesize=A4,pageCompression=1)
c.setTitle("Temporal TV Recommendation: Frequency, Content, and Association Evidence")
c.setAuthor("Amina Bayandinova")
def frame(flow,x,y,w,height):
    pending=list(flow); Frame(x,y,w,height,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0).addFromList(pending,c)
    if pending: raise RuntimeError(f"Report overflow on page {c.getPageNumber()}: {len(pending)} blocks")
def finish():
    c.setFont("TR",8); c.drawString(margin,23,"Amina Bayandinova | SWEET.TV final exam | 24 October 2026")
    c.drawRightString(W-margin,23,str(c.getPageNumber())); c.showPage()

# PAGE 1
frame([p("Temporal TV Recommendation:<br/>Frequency, Content, and Association Evidence",title),
    p("Amina Bayandinova · Individual · BASB252 · HSE University<br/>akbayandinova@edu.hse.ru · 2026-10-24<br/>LLM4Rec — Final Exam Quiz (GSB)",center),
    p('<b>Repository:</b> <link href="https://github.com/barv123/sweet-tv-midterm">https://github.com/barv123/sweet-tv-midterm</link>',center)],margin,H-150,full,115)
left=[p("<b>Abstract—</b>Recommending five future TV programs requires converting channel-viewing intervals into sparse and incomplete program evidence. This study implements content-based filtering and directional association rules, compares them with popularity and personal-frequency baselines on two chronological holdouts, and evaluates a conservative rank-fusion strategy. The designated frequency-plus-association submission achieved Kaggle Private MAP@5 of 0.40943 and Public MAP@5 of 0.42309, documented by the original score screenshot. Repeated local improvements did not reliably transfer to Kaggle, motivating explicit relevance rules, reproducible checks, and rejection of unsupported replacements."),
 p("<i>Index Terms—</i>recommender systems, implicit feedback, association rules, temporal validation, MAP@5.",small),
 h("I. Introduction"),p("The task is to predict five program IDs for each of 1,260 template users during ISO weeks 31–42 of 2020. A service needs both familiar programs and plausible alternatives, but the data contain viewing intervals rather than explicit ratings or satisfaction. For example, a repeat cartoon viewer may benefit from familiar episodes; a popular news show is not necessarily a good substitute."),
 p("The contributions are an audited single-airing interaction definition, two independently implemented required methods, and an evidence-backed decision to retain a modest association supplement to personal frequency. No online engagement or revenue improvement is claimed."),
 h("II. Related Work"),p("Content and collaborative approaches use different evidence and have different cold-start limits [1]. Directional association rules summarize item co-occurrence using support, confidence, and lift [2]; this project computes pairwise rules directly from sparse user transactions, rather than using a third-party recommender. Reciprocal-rank fusion combines rankings without treating scores as calibrated probabilities [3]."),
 p("Melentev's competition write-up [4] motivated the separate &gt;50% training and &gt;80% relevance thresholds. It is participant methodology, not organizer scoring code. Only its prose was consulted; no linked competitor repository or notebook was opened or adapted. The project uses its own implementation. Matrix factorization and personalized PageRank were considered but not fitted; sparse item-CF was tested later and rejected on temporal trade-offs."),
 h("Dataset"),p("The log contains 5,821,987 intervals from 4,914 users and 202 viewing channels. The three schedule files contain 1,074,235, 1,074,394, and 1,295,378 rows; their union has 405 channels and 9,671 positive program IDs. Only 115,613/1,074,235 (10.76%) early schedule rows carry a nonzero program ID. This is a fraction of rows, not a measured duration-weighted airtime fraction. User IDs remain strings: many exceed signed 64-bit range.")]
right=[h("III. Method"),p("<b>Pipeline.</b> Parse timestamps → apply quality gates → match channel/time intervals → construct per-airing signals → fit on pre-cutoff history → restrict to scheduled candidates → rank five unique positive IDs. The audit cache stores matches; experiments reuse it rather than rebuilding interval joins."),
 p("<b>Matching.</b> Viewing vsetv_id equals schedule channel_id. Overlap is max(0, min(view end, airing end) − max(view start, airing start)), measured in elapsed seconds with Europe/Kyiv DST handling. Divide by that airing's duration. A usable single row must cover strictly &gt;50% for training and strictly &gt;80% for relevance. Exactly 80% is negative. Never add separate sessions/channels/airings to cross a threshold."),
 p("Discard nonpositive IDs, unusable durations, exact duplicate copies, &gt;24-hour viewing rows, and ambiguous simultaneous scheduled airings. Program ID 0 is unlabelled context, never a recommendation. Candidate airings intersect the prediction window; repeats are allowed identically across methods. Future viewing is never used; future schedule metadata is exogenous."),
 p("<b>Popularity and frequency.</b> Popularity counts distinct training users per program. Frequency counts distinct qualifying schedule IDs for each user-program pair, without decay. Ties use popularity then ascending numeric program ID. Cold users receive the same popularity list; zero-score slots also use this fallback. Repeat/recency instead sums distinct watched dates with a 14-day exponential half-life."),
 p("<b>Required method 1: content.</b> Weighted tokens represent category (1), genre (1), director (0.5), production decade (0.3), title words (1), and missingness (0.1). Vocabulary and IDF are fitted on pre-cutoff metadata; IDF = 1 + ln((1+n)/(1+df)). Item vectors and weighted user profiles are L2-normalized, then ranked by cosine similarity. User profiles use 14-day recency weights. Sparse metadata limits discrimination."),
 p("<b>Required method 2: association.</b> One binary transaction per training user. For A→B: support=nAB/N, confidence=nAB/nA, lift=confidence/(nB/N). Keep pair count ≥25, item viewers ≥20, confidence ≥0.1, lift ≥1.0, and at most 100 outgoing rules per antecedent. Weight = confidence × ln(1+lift); aggregate with the user's normalized 14-day history. Self-rules are excluded; already-watched candidates remain eligible."),
 p("<b>Designated hybrid.</b> Score(i)=I(i∈F100)/(60+rF(i)) + 0.03 I(i∈A5)/(60+rA(i)). Here F100 is frequency's Top-100 and A5 is association's Top-5. Absent components contribute zero; ties follow the common policy. Two predeclared fusion weights (0.03/0.06) were evaluated; 0.03 improved both folds. The association component retains recency, even though the dominant frequency component does not.")]
frame(left,margin,44,col,H-204); frame(right,margin+col+gap,44,col,H-204); finish()

# PAGE 2
rows=[]
names={"popularity":"Popularity", "repeat_recency":"Repeat/recency", "content_based":"Content (required)","association_rules":"Association (required)","hybrid_rrf60":"Equal RRF60 (earlier)"}
for key,label in names.items():
    vals=[next(r["map_at_5"] for r in f["comparison"] if r["method"]==key) for f in strict["folds"]]
    rows.append([label,*[f"{v:.8f}" for v in vals]])
rows.append(["Frequency, no decay",*[f"{frequency['no_decay_MAP_at_5']['by_fold'][f['name']]:.8f}" for f in strict['folds']]])
rows.append(["Distinct days",*[f"{f['metrics']['distinct_days']['map_at_5']:.8f}" for f in targeted['folds']]])
rows.append(["Frequency + association .03",*[f"{f['metrics']['B_rules_003']['map_at_5']:.8f}" for f in conservative['folds']]])
flow=[h("IV. Experiments: Identical Temporal Protocol"),
 p("All methods evaluate the same 1,260 users, including empty truth with AP=0. Inner training: [9 March, 6 April); validation: [6 April, 20 April), 2020. Outer training: [9 March, 18 May); validation: [18 May, 27 July). Whole sessions crossing a cutoff are omitted. Inner/outer pools contain 2,143/6,384 candidates and 8,326/39,698 relevant user-program pairs; 104/24 users have empty truth. Final training ends 27 July; the candidate window is [27 July, 19 October), with 6,649 eligible programs."),
 p("AP@5(u) = Σ(k=1..5) [hit(k) × cumulative_hits(k)/k] / min(5, |relevant(u)|), or zero for empty truth; MAP@5 is the mean over all users. The denominator convention and label reconstruction are explicitly stated, not asserted to be the hidden Kaggle evaluator."),
 table(["Method", "Inner MAP@5", "Outer MAP@5"],rows,[full*.52,full*.24,full*.24]),Spacer(1,6),
 p("The hybrid improves no-decay frequency by +0.00344577 inner and +0.01016997 outer. Improved/worsened users: 277/187 inner and 300/181 outer. Removing the ten largest positive user changes still leaves mean gains of +0.00117733/+0.00847267. Membership changed for 415/407 users; another 598/621 changed order only. These diagnostics describe development evidence, not formal generalization guarantees."),
 h("Four Actual Kaggle Submissions"),
 table(["Submission (chronological)","Public", "Private"],[[r["model"],f"{r['public']:.5f}",f"{r['private']:.5f}"] for r in evidence['kaggle_submissions']],[full*.6,full*.2,full*.2]),Spacer(1,6),
 Image(str(OUT/"kaggle_verified_scores.png"),width=full,height=full*783/1557),
 p("Fig. 1. Genuine, unmodified local Kaggle screenshot (24 October 2026, 17:31:40); display-scaled only. The first row is the designated submission. Scores were transcribed and checked against this image; no live API response or cohort rank is claimed.",small),
 p("The instructor's private-score exam version grades returned Private score, whereas another local version says Public. Both are shown. Late submissions are accepted even though the original leaderboard is frozen. Distinct days improved both local folds but reduced Private MAP@5 from 0.40179 to 0.39486: an observed warning against equating local gains with Kaggle gains.")]
frame(flow,margin,44,full,H-85); finish()

# PAGE 3
labels={"popularity":"Popularity","repeat_recency":"Repeat/recency","content_based":"Content","association_rules":"Association","personal_frequency":"Frequency","champion":"Designated hybrid"}
def example_column(example):
    blocks=[h(f"User {example['template_row']}: {'repeat viewer' if example['training_airings'] else 'cold start'}"),
        p(f"Template row {example['template_row']} (pseudonymous), {example['training_airings']} qualifying training airings and {example['relevant_programs']} held-out relevant programs. Selection: {example['selection']}; the cold case is chosen using training history only. Same user across every core method.")]
    methodrows=[]
    for method,val in example["methods"].items():
        listing=" → ".join(str(r['tv_show_id'])+("*" if r['relevant'] else "") for r in val['recommendations'])
        methodrows.append([labels[method],listing,f"{val['ap_at_5']:.4f}"])
    blocks.append(table(["Method","Ordered Top-5 IDs (*=hit)","AP@5"],methodrows,[col*.23,col*.60,col*.17]))
    blocks.append(p("Titles below are exact schedule strings (one earliest available representative per ID); episode/title aliases exist. Categories are translated: Children, News, Entertainment, Series. This is descriptive lookup, not a new model feature.",small))
    unique={r['tv_show_id']:r for v in example['methods'].values() for r in v['recommendations']}
    translations={"Детям":"Children","Инфо":"News","Развлечения":"Entertainment","Сериалы":"Series"}
    for item,r in sorted(unique.items()):
        blocks.append(p(f"<b>{item}</b> — {escape(r['title'])} [{escape(translations.get(r['category'],r['category']))}]",small))
    return blocks
left=example_column(evidence['examples'][0]); right=example_column(evidence['examples'][1])
right += [h("What these users actually show"),p("User 3's frequency list hits three children's programs at ranks 1–3 (AP=0.60). The hybrid promotes a nonrelevant news program to rank 3, moving a hit to rank 4 (AP=0.55). Its aggregate gain therefore does not imply every user's gain. Association alone leans toward popular news and obtains only one hit; content proposes four unseen children's films, none relevant in this holdout. Semantic novelty is not automatically useful discovery."),
 p("For user 114, all six core methods return the identical popularity fallback. Only program 240081 at rank 2 is relevant, so AP=(1/2)/5=0.10. The absence of pre-cutoff qualifying records does not establish that the person never watched TV or that the account was newly created. Cold personalization cannot be inferred from these observations."),
 p("For a platform, retain familiar programs as the backbone and use a small co-view correction rather than allowing broad category similarity to replace strong repeat evidence. The ranking should be presented as a predicted watch list, not evidence of enjoyment. Inspect cold-user and children's-content performance separately; these examples do not identify a user's age or household composition."),
 p("The early equal-weight hybrid is an historical comparison; later feature variants are ablations rather than the two independently required methods. Complete paired ablation outputs and configurations are retained separately. No matrix-factorization or PageRank result is claimed.",small)]
frame([h("IV-B. Same Two Users Across the Core Methods (Outer Holdout)")],margin,H-70,full,30)
frame(left,margin,44,col,H-120); frame(right,margin+col+gap,44,col,H-120); finish()

# PAGE 4
ablation=[]
ablation.append(["Distinct days",*[f"{f['metrics']['distinct_days']['map_at_5']:.5f}" for f in targeted['folds']]])
ablation.append(["Future exposure (cap 15%)",*[f"{f['metrics']['schedule_exposure']['map_at_5']:.5f}" for f in targeted['folds']]])
for key in ("A_days_005","A_days_010","B_rules_006"):
    ablation.append([key,*[f"{f['metrics'][key]['map_at_5']:.5f}" for f in conservative['folds']]])
for result in (sprint,channel):
    for name,v in result['experiments'].items(): ablation.append([name,*[f"{m['map_at_5']:.5f}" for m in v['folds']]])
left=[h("V. Discussion and Rejected Improvements"),table(["Completed setting", "Inner", "Outer"],ablation,[col*.55,col*.225,col*.225]),
 p("Days bonuses used normalized frequency plus 0.05/0.10 normalized days. Future exposure was a capped schedule-frequency boost. A_low025 uses F&gt;50 + 0.25F30–50; A_low050_high025 adds 0.50F30–50 + 0.25F&gt;80. B_cf003/006 add item-CF RRF weights 0.03/0.06. CF uses shrunk cosine co-view weights, ≥5 joint users, ≥10 viewers and at most 40 neighbors." ,small),
 p("C_weak0035 increases association weight to 0.06 when max count&lt;3 or Top-5 share&lt;0.35; C_weak0050 uses &lt;5 or &lt;0.50 and weight 0.10. Strong users retain 0.03 and cold users popularity. Channel and historical-propensity bonuses use 0.05/0.10 × maximum personal frequency × normalized feature. Historical propensity is (watched+1)/(completed opportunities+20), requiring ≥10 airings and ≥90% channel-day coverage.",small),
 p("After the champion, a candidate needed ≥0.002 inner and ≥0.005 outer gain, ≥50 improving users on each fold, more gainers than losers, and positive mean gain after excluding the ten largest gains. No candidate passed. C_weak0050 reached 0.55011618 outer, but inner gain was only 0.00037037 and became −0.00230267 after removing ten gains. No fifth Kaggle submission was forced.")]
right=[h("Failure → Cause → Fix → Verification"),p("A real execution failure was <i>AssertionError: No-decay leakage check failed</i>. The check joined each validation user-program label back to every historical event for that pair, so legitimate earlier observations were falsely counted outside the fold. The fix tested whether each label had a qualifying in-window event, while retaining a separate training cutoff assertion. The corrected run reproduced both folds; independent SQL/Python AP checks agreed. This fixed a diagnostic query, not a hidden-label leak."),
 h("What worked; what failed"),p("No-decay frequency outperformed the 14-day recency heuristic on the long horizon, and a small association correction improved both local folds and returned Kaggle scores. In contrast, replacing frequency by distinct days lost Private performance despite local gains. Channel affinity and historical exposure normalization lost on both folds; they were not silently retuned."),
 p("Sparse fields plausibly constrain content discrimination: first genre is missing in roughly 57% of schedule rows and director in roughly 90%. This is a measured data property, not proof of the causal reason for the content model's lower MAP. User 3 provides a concrete case where broad children's metadata yields irrelevant alternatives."),
 h("Limits of validation and inference"),p("The two folds have different horizon lengths, activity distributions and candidate pools. Higher outer MAP is not directly a better model: longer truth windows supply more relevant programs. All 1,260 users remain in the denominator. The same folds were repeatedly reused for development; neither is an untouched final test, and approximate paired intervals after selection are not confirmatory tests."),
 p("Kaggle scores were not fitting labels, but observing four submissions and then choosing a strategy introduces adaptive selection. The Private score is therefore reported as an external returned score, not an unbiased estimate of deployment performance. The strict &gt;80% reconstruction is participant-informed; the instructor permits a justified threshold, but this does not certify the hidden evaluator."),
 p("The log supplies channel intervals, not intra-session channel-change telemetry, enjoyment, device identity, or entitlement. Schedule completeness and program aliases remain imperfect. In the latest check, only 282/2,143 inner and 648/6,384 outer candidates passed historical-exposure reliability filters. Day coverage does not prove user opportunity to watch."),
 h("Business decision and next step"),p("Select the frequency-plus-association champion: cheap aggregation, transparent components, valid candidate coverage, and the strongest actual score among the four submissions. It mainly reranks repeat preferences; on the outer fold, 407 inserted Top-5 slots included only 25 programs absent from &gt;50% training history. Do not market it as broad discovery. With more time, reserve a genuinely untouched later block and audit timestamp/channel completeness before any new tuning. A future controlled online study would be a proposal, not a completed experiment.")]
frame(left,margin,44,col,H-85);frame(right,margin+col+gap,44,col,H-85);finish()

# PAGE 5
left=[h("IV-C. Reproducible Numerical Verification"),p("For template row 3 on the outer fold, the frequency recommendations are [12001682, 12002856, 12001732, 2400467, 10003159]. Their qualifying training-airing counts are [29, 12, 10, 9, 8], and relevance flags are [1,1,1,0,0]. With 21 relevant programs, AP@5 = (1/1 + 2/2 + 3/3)/min(5,21) = 3/5 = 0.600000."),
 p("The champion changes the list to [12001682, 12002856, 2400467, 12001732, 2400480], with flags [1,1,0,1,0]. AP@5 = (1 + 1 + 3/4)/5 = 11/20 = 0.550000. This explicit counterexample prevents claiming universal hybrid improvement. Exact Fraction arithmetic and the metric function were cross-checked again during report preparation; this is a tool-executed verification, not an invented claim of personal hand calculation."),
 p("Previously saved evidence reconstructs 12 sampled users' strict labels directly from raw CSV rows, including exact-80% negatives. Across the strict comparison, 12,600 method-user-fold AP values matched SQL/Python calculations with maximum error zero. The current preparation reran 25 unit tests: all passed. It also revalidated the exact champion file, rather than training a new candidate."),
 h("Reproducibility and resources"),p("Windows; AMD Ryzen 5 3500U; 7,448,702,976 bytes installed usable memory reported by the system. Python 3.12; NumPy 2.4.4; SciPy 1.17.1; DuckDB 1.4.3; tzdata 2025.3 [5]. DuckDB uses two threads and a 768 MB managed-memory limit with disk spilling. Algorithms are deterministic: no stochastic training seed is needed; ties use fixed numeric order."),
 p("Saved successful cached-run times: no-decay evaluation/refit 42.367 s; distinct-days/exposure checkpoint 40.377 s; conservative four-setting checkpoint/refit 44.821 s; later six-setting sprint 69.782 s; channel/propensity checkpoint 39.486 s. These are different scopes, exclude failed attempts/setup, and are not per-model speed benchmarks."),
 p("From a fresh authorized dataset: install requirements.txt, run data/src/run_pipeline.py, then the documented strict evaluation and frozen winner reproduction. The final reproduction command is data/src/reproduce_winning_submission.py; it refuses to overwrite the champion and requires a byte-identical SHA-256. Report construction uses requirements-report.txt and data/src/build_exam_report.py; it reads small saved evidence, not raw logs."),
 p("Final CSV: <b>submission_frequency_conservative_B_rules_003_strict80.csv</b>. Header user_id,tv_show_id; 1,260 exact-order users; 6,300 prediction slots; five unique positive eligible IDs each; zero format/eligibility errors. SHA-256: <font size='7'>b9830a42c341a02caf2557f1469653b482a39ca277af225eabfa7d2557d25fbb</font>.")]
right=[h("VI. AI Usage Disclosure"),p("OpenCode 1.18.31 was used for code generation, debugging, cached-data analysis, tests, model comparisons, Git operations, reference checking, and report drafting. The original tool-native export records model identifiers glm-5.3, gpt-5.6-terra, gpt-6-astra, gpt-6-luna, gpt-6.1-sol, and nemotron-3.5-super-vl-free. These are recorded identifiers, not claims that every model performed every task."),
 p("<b>Student-directed decisions evidenced in the session:</b> requesting strict relevance checks, protecting the best CSV, limiting experiments and runtimes, questioning local-to-Kaggle transfer, requiring negative results, and stopping optimization. The supplied screenshot corroborates the four returned Kaggle scores. No externally sourced competitor implementation was used; NumPy/SciPy/DuckDB are legitimate dependencies."),
 p("<b>Verification attribution.</b> Automated actions include unit tests, interval/label checks, byte hashes, independent SQL/Python AP agreement, exact arithmetic, and saved-prediction reproduction. The assistant checked reference metadata against Crossref and inspected the participant prose and library documentation. These checks do not establish that the student personally read every paper, manually recomputed every result, or reviewed every code line. No such claim is made."),
 p("<b>Trusted or not fully verified:</b> hidden Kaggle labels and evaluation implementation; whether a source viewing row is truly uninterrupted; complete schedule/user-entitlement coverage; and the interpretation of anonymous users. Bibliographic metadata validation is not a claim of reading all cited full texts. The student must personally review this draft, redo the displayed arithmetic, and explain the decision in their own words before submission; the session does not establish that these final personal actions are complete."),
 p("<b>Original audit trail.</b> Submit amina_bayandinova_exam_session.json exported directly from session ses_e9311914effezXO4RVA3gLTl5K. The exporter captures native stdout bytes without reconstruction, sanitization, truncation, or reserialization. The export is a snapshot of a continuing session; export again after final work if later messages must be included. It contains more than 10 MB, so the inherited guidelines require contacting the instructor about upload size. Do not silently replace it with a summary."),
 h("Delivery and provenance"),p("The private-score exam brief specifies 24 October 2026, 19:00 MSK and exactly two LMS files: this PDF and the original session JSON. The exact designated CSV is in GitHub. The other local exam version uses Public score, so Fig. 1 supplies both. The inherited guideline has conflicting five-/six-page statements; this report conservatively uses five content pages plus references."),
 p("The repository resolves through authenticated Git; anonymous access returned 404 during preparation. The instructor must have repository access. Raw datasets, caches, full user histories and the session export are not published. Only the explicitly requested final template prediction CSV is excepted from the submission ignore rule. A screenshot and a report do not themselves grant dataset redistribution rights.")]
frame(left,margin,44,col,H-85);frame(right,margin+col+gap,44,col,H-85);finish()

# PAGE 6: references only.
refs=[
 '[1] G. Adomavicius and A. Tuzhilin, “Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions,” <i>IEEE Transactions on Knowledge and Data Engineering</i>, vol. 17, no. 6, pp. 734–749, 2005. DOI: 10.1109/TKDE.2005.99. <link href="https://doi.org/10.1109/TKDE.2005.99">https://doi.org/10.1109/TKDE.2005.99</link>. Metadata verified through <link href="https://api.crossref.org/works/10.1109/TKDE.2005.99">Crossref</link>, accessed 2026-10-24.',
 '[2] R. Agrawal, T. Imieliński, and A. Swami, “Mining association rules between sets of items in large databases,” in <i>Proc. ACM SIGMOD International Conference on Management of Data</i>, 1993, pp. 207–216. DOI: 10.1145/170035.170072. <link href="https://doi.org/10.1145/170035.170072">https://doi.org/10.1145/170035.170072</link>. Metadata verified through <link href="https://api.crossref.org/works/10.1145%2F170035.170072">Crossref</link>, accessed 2026-10-24.',
 '[3] G. V. Cormack, C. L. A. Clarke, and S. Buettcher, “Reciprocal rank fusion outperforms condorcet and individual rank learning methods,” in <i>Proc. 32nd International ACM SIGIR Conference</i>, 2009, pp. 758–759. DOI: 10.1145/1571941.1572114. <link href="https://doi.org/10.1145/1571941.1572114">https://doi.org/10.1145/1571941.1572114</link>. Metadata verified through <link href="https://api.crossref.org/works/10.1145/1571941.1572114">Crossref</link>, accessed 2026-10-24.',
 '[4] N. Melentev, “Two-layered Recommender System Methodology: A Prize-Winning Solution,” <i>Towards Data Science / Medium</i>, Mar. 20, 2021. <link href="https://medium.com/data-science/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156">medium.com/data-science/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156</link>. Accessed 2026-10-24. Participant-methodology prose only; not a scholarly paper or organizer rule specification. Linked competitor code was not consulted.',
 '[5] NumPy Developers, “NumPy documentation”; SciPy Community, “Sparse arrays (scipy.sparse)”; DuckDB Foundation, “DuckDB documentation.” <link href="https://numpy.org/doc/stable/">https://numpy.org/doc/stable/</link>; <link href="https://docs.scipy.org/doc/scipy/reference/sparse.html">https://docs.scipy.org/doc/scipy/reference/sparse.html</link>; <link href="https://duckdb.org/docs/stable/">https://duckdb.org/docs/stable/</link>. Accessed 2026-10-24. Current documentation versions differ from the pinned execution versions listed in the report.',
 '[6] LLM4Rec course materials, HSE University, “Final Exam Quiz (GSB): TV Program Recommendation on sweet.tv,” local files exam_quiz_GSB.md and exam_quiz_GSB (2).md; S. Jin, “LLM4Rec — Homework Guidelines,” local homework_guidelines.pdf. Inspected 2026-10-24. The two exam files differ on Public versus Private grading; both specify the same deadline and two-file submission. No invented public URL is supplied for private local materials.',
 '[7] A. Bayandinova, “SWEET.TV Midterm,” project source and measured evidence, 2026. <link href="https://github.com/barv123/sweet-tv-midterm">https://github.com/barv123/sweet-tv-midterm</link>. Authenticated Git remote verified; reviewer authorization may be required. Competition: <link href="https://www.kaggle.com/competitions/sweettv-tv-program-recommender">https://www.kaggle.com/competitions/sweettv-tv-program-recommender</link>. Returned scores are evidenced by Fig. 1 rather than a claim of live API verification.'
]
frame([h("References")]+[p(r) for r in refs],margin,44,full,H-85);finish();c.save()
reader=PdfReader(path)
assert len(reader.pages)==6
text="\n".join(page.extract_text() for page in reader.pages)
for token in ("0.40943","0.42309","0.42616645","0.54430666","12001682","0.550000","AI Usage Disclosure"):
    assert token in text,token
assert len(reader.pages[1].images)>=1
assert all(len(page.extract_text())>200 for page in reader.pages)
(OUT/"exam_report_source.md").write_text("\n\n".join(source)+"\n",encoding="utf-8")
(R/"exam_pdf_verification.json").write_text(json.dumps({"pages":6,"body_pages":5,"references_pages":1,
    "opened_with_pypdf":True,"headline_numbers_found":True,"screenshot_embedded":True,
    "all_frames_fit":True,"page_text_characters":[len(p.extract_text()) for p in reader.pages]},indent=2)+"\n",encoding="utf-8")
print(str(path));print("PDF parsed successfully: five content pages + one references page; screenshot embedded.")
