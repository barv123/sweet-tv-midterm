"""Reproduce the frozen winner from the audit cache, without any model search."""
import argparse
from pathlib import Path
import numpy as np
from baseline_experiment import FINAL, require
from common import RESULTS, SCHEDULES, connect, save_json
from final_competitive_sprint import fuse
from frequency_conservative_checkpoint import digest, final_association
from personal_frequency_no_decay import cached_events, train_table, predict_no_decay, write_new_submission
from validate_submission import candidate_ids, read_template, validate

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",type=Path,default=RESULTS/"submission_reproduced_winner.csv")
args=parser.parse_args()
original=RESULTS/"submission_frequency_conservative_B_rules_003_strict80.csv"
require(args.output.resolve()!=original.resolve(),"Never overwrite the submitted champion")
require(not args.output.exists(),"Output already exists; choose a new filename")
users=read_template(); con=connect(); cached_events(con)
con.execute("CREATE OR REPLACE TEMP TABLE adv_users(user_id VARCHAR,template_row INTEGER)")
con.executemany("INSERT INTO adv_users VALUES (?,?)",[(u,i) for i,u in enumerate(users,1)])
fold={"name":"reproduce_winner","start":FINAL,"end":"2020-10-19 00:00:00","sources":SCHEDULES[1:]}
training=train_table(con,fold); eligible=candidate_ids(con,FINAL,fold["end"],fold["sources"])
rankings,_,pop,stats=predict_no_decay(con,users,training,eligible)
ids=np.array(sorted(eligible)); popvec=np.array([pop[int(i)] for i in ids])
association,nrules=final_association(con,users,fold,ids,pop)
forecast={u:fuse(rankings[u],association[u],ids,popvec) for u in users}
write_new_submission(args.output,users,forecast,eligible)
check=validate(args.output,users,eligible)
require(digest(args.output)==digest(original),"Reproduced bytes differ from Kaggle champion")
con.close()
save_json("winner_reproduction_check.json",{"sha256":check["sha256"],"byte_identical":True,
    "rows":check["rows"],"passed":check["passed"],"rules_retained":nrules,"training":stats})
print("Winner reproduced byte-for-byte: "+check["sha256"])
