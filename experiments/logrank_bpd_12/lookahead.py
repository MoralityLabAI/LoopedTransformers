"""12B: exact adaptive two-step acquisition on exposed development worlds."""
import json,time
from pathlib import Path
import numpy as np
from pilot import HERE, B, basis, member, dot, walsh, gains, score, sha

def two_step(mu,qs,pool,weights,noise):
    q=np.array(qs);a=np.array(pool);rho=1-2*noise;one=gains(lambda x:mu[x],q,a,weights,noise);value=one.copy()
    # First observation a; conditional on each outcome choose the best remaining b.
    for i,v in enumerate(a):
        for z in (-1,1):
            mass=(1+rho*z*mu[v])/2
            def moment(x):return (mu[x]+rho*z*mu[x^v])/(2*mass)
            g=gains(moment,q,a,weights,noise);g[i]=-np.inf
            value[i]+=mass*max(g)
    return value

def run():
    inputs=sorted(HERE.glob('receipt_*.json'));worlds=[w for p in inputs for w in json.loads(p.read_text())['worlds']];rows=[]
    for w in worlds:
        for noise in (.15,.35):
            states=np.arange(1<<w['d']);parity=np.array([int(i).bit_count()%2 for i in states]);rs=list(w['initial']);obs=[dot(a,w['hidden'])^int(u<noise) for a,u in zip(rs,w['uniforms'])];p=np.ones(len(states))/len(states)
            def upd(p,a,b):
                p=p*np.where(parity[states&a]==b,1-noise,noise);return p/p.sum()
            for a,b in zip(rs,obs):p=upd(p,a,b)
            mu=walsh(p);pool=w['pool'].copy();trace=[];steps=[];sec=0.;weights=np.array(w['weights'])
            for t in range(B+1):
                sc=score(mu,w);sc['rank']=len(basis(rs));steps.append(sc)
                if t==B:break
                start=time.perf_counter();v=two_step(mu,w['targets'],pool,weights,noise) if t<B-1 else gains(lambda x:mu[x],w['targets'],pool,weights,noise);i=int(v.argmax());sec+=time.perf_counter()-start;a=pool.pop(i);b=dot(a,w['hidden'])^int(w['uniforms'][len(w['initial'])+t]<noise);trace.append(dict(query=a,obs=b,expands=not member(a,basis(rs)),expected_gain=float(v[i])));rs.append(a);p=upd(p,a,b);mu=walsh(p)
            rows.append(dict(seed=w['seed'],r=w['r'],noise=noise,arm='two_step',initial_brier=steps[0]['brier'],brier=sc['brier'],posterior_risk=float(weights@(np.array(sc['prob'])*(1-np.array(sc['prob'])))),old=sc['correct_old'],new=sc['correct_new'],coverage=sc['coverage'],covered_correct=sc['covered_correct'],rank_gain=sc['rank']-w['r'],selection_seconds=sec,trace=trace,steps=steps))
        print(w['seed'],flush=True)
    (HERE/'lookahead.json').write_text(json.dumps(dict(evidence='adaptive exposed-root diagnostic',source_sha256=sha(__file__),inputs={p.name:sha(p) for p in inputs},rows=rows),separators=(',',':'))+'\n')
if __name__=='__main__':run()
