"""Finite metamorphic test of induced-coordinate equivariance and zero BP fixed point."""
import json
from pathlib import Path
import numpy as np
from pilot import Frame,lift,sha
H=Path(__file__).resolve().parent

def run():
    worlds=[w for fn in ('receipt.json','scale.json') for w in json.loads((H/fn).read_text())['worlds']];checks=0;transforms=0;zero=0;uniform=0;by_width={}
    for w in worlds:
        d=w['d'];rng=np.random.default_rng(w['seed']+79117);original=Frame(w['induced']);order=sorted(range(48),key=lambda j:(w['costs'][j],j));masks=set(w['initial']+w['pool']+w['targets']+[q^a for q in w['targets'] for a in w['pool']]);
        for _ in range(4):
            g=[]
            while len(g)<d:
                q=int(rng.integers(1,1<<d,dtype=np.uint64))
                if Frame(g).coord(q) is None:g.append(q)
            transform=lambda q:lift(g,q)
            frame=Frame([transform(q) for q in w['initial']]+[transform(w['pool'][j]) for j in order]);assert frame.rows==[transform(q) for q in original.rows];transforms+=1
            for q in masks:assert frame.coord(transform(q))==original.coord(q);checks+=1
        for a in w['runs']:
            if a['arm']!='raw_bp':continue
            rows=w['initial']+[t['query'] for t in a['trace']];is_zero=all(q.bit_count()>=2 for q in rows);is_uniform=all(p==.5 for p in a['final']['prob']);assert not is_zero or is_uniform;zero+=is_zero;uniform+=is_uniform;rec=by_width.setdefault(d,[0,0,0]);rec[0]+=1;rec[1]+=is_zero;rec[2]+=is_uniform
    result=dict(status='PASS',worlds=len(worlds),invertible_transformations=transforms,coordinate_equalities=checks,no_singleton_raw_trajectories=zero,uniform_raw_trajectories=uniform,by_width=by_width,columns=['raw_trajectories','no_singleton','uniform_predictions'],input_hashes={fn:sha(H/fn) for fn in ('receipt.json','scale.json')},source_sha256=sha(__file__))
    (H/'gauge.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':run()
