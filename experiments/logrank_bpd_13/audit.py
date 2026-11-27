"""Independent integer posterior, edge-list BP, budget and gauge replay."""
import json, hashlib, math
from fractions import Fraction
from pathlib import Path
from collections import Counter
import numpy as np
H=Path(__file__).resolve().parent
B=24
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dot(a,b):return (int(a)&int(b)).bit_count()%2
class Basis:
    # LSB pivots, independently implemented coordinate tracking.
    def __init__(self,rows):
        self.piv={};self.rows=[]
        for a in rows:
            a=int(a);v=a;c=1<<len(self.rows)
            while v:
                i=(v&-v).bit_length()-1
                if i not in self.piv:self.piv[i]=(v,c);self.rows.append(a);break
                b,t=self.piv[i];v^=b;c^=t
    def coord(self,a):
        a=int(a);c=0
        while a:
            i=(a&-a).bit_length()-1
            if i not in self.piv:return None
            v,t=self.piv[i];a^=v;c^=t
        return c

def edge_bp(rows,obs,p,d):
    ll={};scale=math.log((1-p)/p)
    for q,b in zip(rows,obs):ll[q]=ll.get(q,0.)+(1-2*b)*scale
    keys=sorted(ll);fi=[];vi=[]
    for j,q in enumerate(keys):
        for i in range(d):
            if q>>i&1:fi.append(j);vi.append(i)
    fi=np.array(fi);vi=np.array(vi);f=np.zeros(len(fi));v=f.copy();n=len(keys);sg=np.tanh(np.array([ll[q] for q in keys])/2);iters=40
    for t in range(40):
        a=np.tanh(v/2);zero=a==0;prod=np.ones(n);np.multiply.at(prod,fi,np.where(zero,1.,a));zc=np.bincount(fi,weights=zero,minlength=n);z=zc[fi]-zero
        exc=np.divide(prod[fi],a,out=prod[fi].copy(),where=~zero);exc=np.where(z==0,exc,0.);new=2*np.arctanh(np.clip(sg[fi]*exc,-1+1e-12,1-1e-12));delta=np.max(abs(new-f));f=(f+new)/2;total=np.bincount(vi,weights=f,minlength=d);v=total[vi]-f
        if delta<1e-6:iters=t+1;break
    return np.tanh(total/2),iters

def bp_prob(w,rows,obs,p,kind):
    d=w['d'];f=Basis([1<<i for i in range(d)] if kind=='raw' else w[kind]);s=Basis(rows);mu,it=edge_bp([f.coord(q) for q in rows],obs,p,d)
    cache={}
    def moment(q):
        q=int(q)
        if q not in cache:
            c=f.coord(q);cache[q]=0. if s.coord(q) is None else math.prod(mu[i] for i in range(d) if c>>i&1)
        return cache[q]
    return moment,it

def exact_moment(rows,obs,p,d):
    n=1<<d;states=np.arange(n);par=np.fromiter((i.bit_count()%2 for i in range(n)),np.int8,n);errors=np.zeros(n,dtype=int)
    for a,b in zip(rows,obs):errors+=(par[states&a]!=b)
    u=Fraction(str(p));m=len(rows);weights=[u.numerator**i*(u.denominator-u.numerator)**(m-i) for i in range(m+1)];f=np.bincount(errors,minlength=m+1);den=sum(int(c)*v for c,v in zip(f,weights));assert den>0
    # Independent physical-coordinate integer Walsh transform.
    z=np.array([weights[e] for e in errors],dtype=object);h=1
    while h<n:
        for i in range(0,n,2*h):
            a=z[i:i+h].copy();b=z[i+h:i+2*h].copy();z[i:i+h]=a+b;z[i+h:i+2*h]=a-b
        h*=2
    assert z[0]==den
    return lambda q:float(Fraction(z[int(q)],den))

def check_score(moment,w,s):
    p=np.clip(np.array([(1-moment(q))/2 for q in w['targets']]),0,1);y=np.array([dot(q,w['hidden']) for q in w['targets']]);wt=np.array(w['weights']);saved=np.array(s['prob']);delta=float(np.max(abs(p-saved)));assert delta<2e-8,delta
    pred=saved>.5+1e-12;ok=pred==y;take=np.maximum(saved,1-saved)>=.9-1e-12
    for k,v in dict(brier=wt@(saved-y)**2,risk=wt@(saved*(1-saved)),accuracy=wt@ok,old_correct=sum(ok[:8]),new_correct=sum(ok[8:]),coverage=wt@take,covered_correct=wt@(take&ok)).items():assert abs(s[k]-v)<2e-12,(k,s[k],v)
    return delta

def run():
    counts=Counter();max_bp=0.;max_exact=0.;rows_out=[];model=json.loads((H/'model.json').read_text());assert model['source_sha256']==sha(H/'pilot.py') and model['bp_sha256']==sha(H/'bp.py') and model['protocol_sha256']==sha(H/'PROTOCOL.md');seeds=set()
    for file in ('receipt.json','scale.json'):
        r=json.loads((H/file).read_text());assert r['source_sha256']==sha(H/'pilot.py') and r['protocol_sha256']==sha(H/'PROTOCOL.md') and r['model_sha256']==sha(H/'model.json')
        for w in r['worlds']:
            seed=w['seed'];assert seed not in seeds and seed not in model['train_seeds'];seeds.add(seed);d=w['d'];rank=w['r'];init=w['initial'];pool=w['pool'];cost=w['costs'];assert len(set(pool))==48 and len(set(w['targets']))==32 and not set(w['targets'])&(set(pool)|set(init));assert len(init)==4*rank and len(Basis(init).rows)==rank
            f=Basis(init+[pool[j] for j in sorted(range(48),key=lambda j:(cost[j],j))]);assert f.rows==w['induced'] and len(f.rows)==d and len(Basis(w['planted']).rows)==d;assert all(1<=x<=3 for x in cost)
            counts['worlds']+=1;assert len(w['runs'])==(16 if file=='receipt.json' else 8)
            for a in w['runs']:
                arm=a['arm'];p=a['noise'];rows=init.copy();obs=[dot(q,w['hidden'])^int(u<p) for q,u in zip(rows,w['uniforms'])];assert obs==a['initial_obs'];spent=0;remain=set(range(48));fixed=arm.endswith('_fixed');kind='raw' if arm=='raw_bp' else 'planted' if arm=='planted_bp' else 'induced'
                for j,t in enumerate(a['trace']):
                    i=t['j'];q=t['query'];assert i in remain and q==pool[i] and t['cost']==cost[i] and spent+cost[i]<=B;assert t['expands']==(Basis(rows).coord(q) is None);assert t['obs']==dot(q,w['hidden'])^int(w['uniforms'][len(init)+j]<p);assert fixed or t['coverage']<.8-1e-12
                    if arm=='random_induced':assert i==next(x for x in w['random_order'] if x in remain and spent+cost[x]<=B);counts['random_choices']+=1
                    remain.remove(i);spent+=cost[i];rows.append(q);obs.append(t['obs']);counts['budget_steps']+=1
                assert spent==a['spent'] and len(Basis(rows).rows)==a['rank'];affordable=any(spent+cost[i]<=B for i in remain);assert a['reason']==('confidence' if affordable else 'budget');assert not affordable or (not fixed and a['final']['coverage']>=.8-1e-12)
                if 'reference'in a:
                    em=exact_moment(rows,obs,p,d);max_exact=max(max_exact,check_score(em,w,a['reference']));counts['exact_probabilities']+=32
                    assert abs(a['posterior_regret']-np.array(w['weights'])@((np.array(a['final']['prob'])-np.array(a['reference']['prob']))**2))<2e-12
                    assert all(abs(em(q))<1e-12 for q in w['targets'] if Basis(rows).coord(q) is None)
                if arm.startswith('exact'):
                    max_exact=max(max_exact,check_score(em,w,a['final']));counts['exact_final_probabilities']+=32
                else:
                    bm,_=bp_prob(w,rows,obs,p,kind);max_bp=max(max_bp,check_score(bm,w,a['final']));counts['bp_final_probabilities']+=32
                for k,s in a.get('same_trace',{}).items():
                    bm,_=bp_prob(w,rows,obs,p,k);max_bp=max(max_bp,check_score(bm,w,s));counts['same_trace_bp_probabilities']+=32
                counts['trajectories']+=1
                rows_out.append(dict(seed=seed,r=rank,d=d,arm=arm,noise=p,spent=spent,rank_gain=a['rank']-rank,reason=a['reason'],**{k:v for k,v in a['final'].items() if k!='prob'},reference_brier=a.get('reference',{}).get('brier'),posterior_regret=a.get('posterior_regret'),same_trace={k:{j:v for j,v in s.items() if j!='prob'} for k,s in a.get('same_trace',{}).items()}))
            print('AUDIT',seed,flush=True)
    assert len(seeds)==32 and counts['trajectories']==448
    out=dict(status='PASS',**counts,max_bp_probability_difference=max_bp,max_exact_probability_difference=max_exact,training_seeds=model['train_seeds'],source_hashes={name:sha(H/name) for name in ('pilot.py','bp.py','PROTOCOL.md','model.json','receipt.json','scale.json','audit.py')},rows=rows_out,scope='independent final posterior in physical coordinates, edge-list BP final and same-trace probabilities, scores, budgets, rank, noise draws, random choices; no ridge refit or nonrandom acquisition-choice replay')
    (H/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='rows'})
if __name__=='__main__':run()
