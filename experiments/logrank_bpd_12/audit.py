"""Independent integer-likelihood replay and same-source learned-policy replay."""
import json,math,hashlib
from pathlib import Path
import numpy as np
from pilot import features
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dot(a,b):return (a&b).bit_count()%2
def span(rows):
    s={0}
    for a in rows:
        if a not in s:s.update(a^x for x in tuple(s))
    return s

def integer_moments(errors,m,u):
    weights=[u**e*(20-u)**(m-e) for e in range(m+1)]
    a=np.array([weights[e] for e in errors],dtype=object);n=len(a);h=1
    while h<n:
        for i in range(0,n,2*h):
            l=a[i:i+h].copy();r=a[i+h:i+2*h].copy();a[i:i+h]=l+r;a[i+h:i+2*h]=l-r
        h*=2
    den=a[0];assert den>0
    return np.array([x/den for x in a])

def audit(paths):
    inputs=[json.loads(p.read_text()) for p in paths];rec=inputs[0];assert all(x['model']==rec['model'] and x['training']==rec['training'] for x in inputs)
    assert rec['source_sha256']==sha(HERE/'pilot.py') and rec['protocol_sha256']==sha(HERE/'PROTOCOL.md') and rec['bp_sha256']==sha(HERE/'bp.py')
    training={x['seed'] for x in rec['training']};worlds=[w for x in inputs for w in x['worlds']];seeds={w['seed'] for w in worlds};assert len(training)==16 and len(seeds)==len(worlds)==24 and not training&seeds
    model={k:np.asarray(rec['model'][k]) for k in ('mean','std','coef')};maxerr=0.;checks=0;direct=0;choices=0;bp_choices=0;rows=[]
    for w in worlds:
        r=w['r'];d=w['d'];n=1<<d;qs=np.asarray(w['targets']);wt=np.asarray(w['weights']);y=np.array([dot(int(q),w['hidden']) for q in qs]);assert len(set(qs))==32 and len(set(w['pool']))==48 and not set(qs)&(set(w['pool'])|set(w['initial']));assert len(span(w['initial']))==1<<r and len(w['initial'])==4*r
        states=np.arange(n);table=np.array([int(i).bit_count()%2 for i in states]);assert len(w['runs'])==14
        for run in w['runs']:
            p=run['noise'];u=round(20*p);rho=1-2*p;arm=run['arm'];obs=run['initial_obs'].copy();rr=w['initial'].copy();pool=w['pool'].copy();err=np.zeros(n,dtype=int)
            assert obs==[dot(q,w['hidden'])^int(t<p) for q,t in zip(rr,w['uniforms'])]
            for q,b in zip(rr,obs):err+=table[states&q]!=b
            for step,sc in enumerate(run['steps']):
                mu=integer_moments(err,len(rr),u);s=span(rr);pr=(1-mu[qs])/2;assert sc['rank']==len(s).bit_length()-1;assert all(mu[q]==0 for q in range(n) if q not in s)
                maxerr=max(maxerr,float(np.max(abs(pr-np.array(sc['prob'])))));assert np.allclose(pr,sc['prob'],atol=2e-12,rtol=0);checks+=len(qs)
                pred=pr>.5+1e-12;ok=pred==y;take=np.maximum(pr,1-pr)>=.9-1e-12;assert sc['correct_old']==sum(ok[:8]) and sc['correct_new']==sum(ok[8:]);assert abs(sc['brier']-wt@(pr-y)**2)<2e-12;assert abs(sc['coverage']-wt@take)<2e-12;assert abs(sc['covered_correct']-wt@(take&ok))<2e-12
                if step==12:break
                a=run['trace'][step]['query'];b=run['trace'][step]['obs'];assert a in pool and run['trace'][step]['expands']==(a not in s);assert b==dot(a,w['hidden'])^int(w['uniforms'][4*r+step]<p)
                # Compute hypothetical posterior means using exact moments and both observation values.
                def gain(a):
                    ans=0.
                    for z in (-1,1):
                        mass=(1+z*rho*mu[a])/2
                        after=(mu[qs]+z*rho*mu[qs^a])/(2*mass)
                        ans+=mass*float(wt@(after-mu[qs])**2)/4
                    return ans
                g=gain(a);direct+=1
                if arm in ('exact_gain','rank_first','old_only'):
                    elig=pool if arm=='exact_gain' else ([q for q in pool if q<1<<r] if arm=='old_only' else [q for q in pool if q not in s] or pool)
                    assert a in elig and g>=max(gain(q) for q in elig)-2e-12;choices+=1
                elif arm=='entropy':assert abs(mu[a])<=min(abs(mu[q]) for q in pool)+2e-12;choices+=1
                elif arm=='random':assert a==next(q for q in w['random_order'] if q in pool);choices+=1
                else:
                    x,_=features(w,rr,obs,pool,p,step);value=x[:,0] if arm=='bp_gain' else ((x-model['mean'])/model['std'])@model['coef'];assert a==pool[int(np.argmax(value))];bp_choices+=1
                pool.remove(a);rr.append(a);obs.append(b);err+=table[states&a]!=b
            rows.append(dict(seed=w['seed'],r=r,noise=p,arm=arm,initial_brier=run['steps'][0]['brier'],brier=sc['brier'],posterior_risk=float(wt@(pr*(1-pr))),old=sc['correct_old'],new=sc['correct_new'],coverage=sc['coverage'],covered_correct=sc['covered_correct'],rank_gain=sc['rank']-r,selection_seconds=run['selection_seconds']))
        print('AUDIT',w['seed'],flush=True)
    return dict(status='PASS',worlds=len(worlds),training_worlds=len(training),runs=len(rows),probabilities=checks,conditional_gain_checks=direct,independent_choices=choices,same_source_bp_choices=bp_choices,max_probability_error=maxerr,inputs={p.name:sha(p) for p in paths},rows=rows)
if __name__=='__main__':
    paths=sorted(HERE.glob('receipt_*.json'));r=audit(paths);(HERE/'audit.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='rows'})
