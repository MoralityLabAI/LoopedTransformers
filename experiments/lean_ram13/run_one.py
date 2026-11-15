import argparse,json,sys
from pathlib import Path
sys.path.insert(0,'/mnt/data/ram13_local')
import harness
p=argparse.ArgumentParser()
p.add_argument('--case',type=int,required=True)
p.add_argument('--policy',required=True)
p.add_argument('--phase',choices=('primary','secondary'),default='primary')
a=p.parse_args()
orig=harness.module_sources()
cases=[(m,t) for m,(_,names) in harness.LOCKED.items() for t in names]
if not(0<=a.case<len(cases)):p.error('case out of bounds')
allowed=harness.POLICIES if a.phase=='primary' else harness.SECOND
if a.policy not in allowed:p.error('policy not in registered suite')
mod,theorem=cases[a.case]
out=harness.OUT/a.phase/f'{a.case:02d}'/a.policy
if (out/'case.json').is_file():
 r=json.loads((out/'case.json').read_text())
 print('ALREADY_SCORED',a.case,mod,theorem,a.policy,r['accepted'],r['reason'],flush=True)
 sys.exit(0)
if out.exists():p.error('incomplete existing result output requires explicit quarantine; no automatic overwrite')
ranker=harness.Rankers()
budget=harness.PRIMARY if a.phase=='primary' else harness.SECONDARY
r=harness.search(orig[mod],theorem,ranker,a.policy,budget,out)
print('RAM13',a.phase,a.case,mod,theorem,a.policy,r['accepted'],r['reason'],r['logical_evaluations'],round(r['elapsed_seconds'],3),flush=True)
