import argparse,json,sys
from pathlib import Path
sys.path.insert(0,'/mnt/data/ram13_local')
import harness
p=argparse.ArgumentParser();p.add_argument('--start',type=int,required=True);p.add_argument('--stop',type=int,required=True);args=p.parse_args()
assert 0<=args.start<args.stop<=12
cases=[(m,t) for m,(_,ts) in harness.LOCKED.items() for t in ts]
sources=harness.module_sources()
rank=harness.Rankers()
for i in range(args.start,args.stop):
    mod,theorem=cases[i]
    names=harness.SECOND[i%len(harness.SECOND):]+harness.SECOND[:i%len(harness.SECOND)]
    for name in names:
        out=harness.OUT/'secondary'/f'{i:02d}'/name
        if (out/'case.json').exists():
            r=json.loads((out/'case.json').read_text());print('PREVIOUS',i,name,r['accepted'],flush=True);continue
        if out.exists():raise ValueError('incomplete, quarantine before full restart: '+str(out))
        r=harness.search(sources[mod],theorem,rank,name,harness.SECONDARY,out)
        print('RAM13_SECONDARY',i,mod,theorem,name,r['accepted'],r['reason'],r['logical_evaluations'],round(r['elapsed_seconds'],2),flush=True)
