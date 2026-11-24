"""Postprocess bounded teacher pilot on CPU; one package per independently downloadable arm.

No teacher inputs or future demand enter x. Oracle metrics use future labels only as
hindsight ceilings. Whole-turn measures and short continuations are marked as such.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
source_dir=Path(__file__).resolve().parent
if not (source_dir/'gym_data.py').is_file():
    source_dir=source_dir.parent/'LoopedTransformers'/'InkyNano'
sys.path.insert(0,str(source_dir))
from gym_data import build
from routing import parse_rows, simulate


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def average(values):
    return float(np.mean(values)) if values else None


def overlap(rows, next_token=False):
    by_root=defaultdict(list)
    visits=defaultdict(dict)
    for (root,turn,t,l,p),w in rows:
        visits[root,turn,t,l][p]=w
    for (root,turn,t,l),d in visits.items():
        if next_token:
            other=visits.get((root,turn,t+1,l),{})
            pairs=[(d[pass_id],other[pass_id]) for pass_id in d if pass_id in other]
            if not pairs:continue
        else:
            if l not in range(6,18):continue
            pairs=[(d[0],d[1]),(d[1],d[2])]
        for a,b in pairs:
            by_root[root].extend([len(a.keys()&b.keys())/8])
    return {'root_mean_top8_overlap':average([average(v) for v in by_root.values()]),
            'n_roots':len(by_root),'pair_counts':sum(map(len,by_root.values()))}


def persistent_rolling(rows,prompt_rows,h=32,oracle=False):
    """Six fixed IDs from turn hindsight or causal prompt; two replace per segment."""
    groups=defaultdict(list);prior=defaultdict(lambda:defaultdict(float))
    for (root,turn,t,l,p),w in prompt_rows:
        for e,v in w.items():prior[root,turn,l][e]+=v
    for (root,turn,t,l,p),w in rows:groups[root,turn,l,t//h].append(w)
    by_root=defaultdict(lambda:[0.,0,0]);fixed={};previous={};last={}
    for (root,turn,l,seg),vs in sorted(groups.items()):
        key=root,turn,l
        if key not in fixed:
            scores=defaultdict(float)
            if oracle:
                for (r,u,layer,s),vv in groups.items():
                    if (r,u,layer)==key:
                        for w in vv:
                            for e,v in w.items():scores[e]+=v
            else:scores.update(prior[key])
            fixed[key]=frozenset(sorted(range(128),key=lambda e:(-scores[e],e))[:6])
        target=defaultdict(float)
        for w in vs:
            for e,v in w.items():target[e]+=v
        roll_scores=target if oracle else previous.get(key,{})
        rolling=frozenset(sorted(set(range(128))-fixed[key],
            key=lambda e:(-roll_scores.get(e,0.),e))[:2])
        active=fixed[key]|rolling
        by_root[root][0]+=sum(sum(w.get(e,0.) for e in active) for w in vs)
        by_root[root][1]+=len(vs)
        by_root[root][2]+=len(active-last.get(key,frozenset()))
        previous[key]=target;last[key]=active
    return {'root_mean_retained_mass':average([v[0]/v[1] for v in by_root.values()]),
            'total_admissions':sum(v[2] for v in by_root.values()),
            'n_roots':len(by_root),'horizon':h,
            'method':'hindsight six fixed + current-segment two' if oracle else 'prompt six fixed + previous-segment two'}


def analyse_arm(folder,out):
    rpath=folder/'roster.jsonl';tpath=folder/'trace.jsonl'
    if not rpath.is_file() or not tpath.is_file():return {'status':'missing'}
    roster=[json.loads(s) for s in rpath.read_text().splitlines() if s.strip()]
    if not roster:return {'status':'empty'}
    meta={(r['root'],r['turn']):r for r in roster}
    data=[json.loads(s) for s in tpath.read_text().splitlines() if s.strip()]
    full=parse_rows((json.dumps(r) for r in data),strict=True)
    cont=[];prompt=[]
    for r in data:
        key=r['root'],r['turn'];p=meta[key]['prompt_len']
        if r['phase']=='continuation':
            r=dict(r);r['token']-=p;cont.append(r)
        elif r['phase']=='prompt':prompt.append(r)
        else:raise ValueError('unknown phase')
    if not cont:raise ValueError('missing continuation visits')
    future=parse_rows((json.dumps(r) for r in cont),strict=True)
    prefix=parse_rows((json.dumps(r) for r in prompt),strict=True)
    for h in (0,32,64,128):
        d=build(roster,full,h)
        np.savez_compressed(out/f'demand_H{h}.npz',**d)
    oracles={str(h):{str(k):simulate(future,k,h,1,'oracle')['root_mean_retained_mass']
                      for k in (8,16,32)} for h in (0,32,64,128)}
    lengths=[len(r['token_ids'])-r['prompt_len'] for r in roster]
    summary={'roots':len(roster),'train_roots':len({r['root'] for r in roster if r['split']=='train'}),
        'dev_roots':len({r['root'] for r in roster if r['split']=='dev'}),
        'n_continuation_tokens':sum(lengths),'continuation_lengths':{'min':min(lengths),'max':max(lengths)},
        'n_visits':len(future),'all_48_visits':True,'oracle_root_mean_mass':oracles,
        'cross_pass':overlap(future),'adjacent_token':overlap(future,True),
        'six_plus_two':{str(h):{'causal':persistent_rolling(future,prefix,h,False),
                                'hindsight':persistent_rolling(future,prefix,h,True)}
                        for h in (32,64,128)},
        'capacity16_active8_note':'top-8 teacher choices retained by a 16-slot resident set; restricted-router task quality not tested',
        'window_limit':'when a continuation fits a single segment, horizon coverage equals whole-turn coverage',
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)}
                 for p in sorted(out.iterdir()) if p.is_file()}}
    (out/'routing_stats.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


def main():
    a=argparse.ArgumentParser();a.add_argument('--pilot-root',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True);args=a.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    report={}
    for arm in ('reference_forced','generated'):
        # The original exporter nests routing/; the cached incremental pilot
        # writes its two arms directly under the output root.
        src=args.pilot_root/arm
        if not (src/'roster.jsonl').is_file():src=args.pilot_root/'routing'/arm
        out=args.out/arm;out.mkdir(exist_ok=True)
        for n in ('roster.jsonl','trace.jsonl'):
            if (src/n).is_file(): (out/n).write_bytes((src/n).read_bytes())
        report[arm]=analyse_arm(out,out)
    (args.out/'analysis_receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({arm:{k:r.get(k) for k in ('roots','n_continuation_tokens','n_visits','status')}
                      for arm,r in report.items()}))
if __name__=='__main__':main()
