"""Independent integer-likelihood, edge-list BP and expanded-state replay."""
import json,hashlib,math
from fractions import Fraction
from pathlib import Path
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def parity(a,b):return (a&b).bit_count()%2

def rank(rows):
    piv={}
    for q in rows:
        while q and q.bit_length()-1 in piv:q^=piv[q.bit_length()-1]
        if q:piv[q.bit_length()-1]=q
    return len(piv)

def verify_score(pr,y,row):
    pr=np.asarray(pr);y=np.asarray(y);err=float(np.max(abs(pr-row['prob'])));assert err<2e-7,err
    p=np.asarray(row['prob']);pred=(p>.5+1e-12).astype(int);take=np.maximum(p,1-p)>=.9-1e-12
    assert row['correct']==int((pred==y).sum()) and row['covered']==int(take.sum()) and row['covered_correct']==int(((pred==y)&take).sum())
    assert abs(row['brier']-np.mean((p-y)**2))<1e-12
    return len(y)
def posterior(rows,obs,p,r,queries):
    n=1<<r;state=np.arange(n);lut=np.fromiter((i.bit_count()%2 for i in range(n)),np.int8,n);errors=np.zeros(n,dtype=int)
    for a,b in zip(rows,obs):errors+=(lut[state&a]!=b)
    u=Fraction(str(p));m=len(rows);weights=[u.numerator**i*(u.denominator-u.numerator)**(m-i) for i in range(m+1)];freq=np.bincount(errors,minlength=m+1);den=sum(int(c)*v for c,v in zip(freq,weights));assert den>0
    def prob(q):
        counts=np.bincount(errors[lut[state&q]==1],minlength=m+1);return float(Fraction(sum(int(c)*v for c,v in zip(counts,weights)),den))
    return np.array([prob(q) for q in queries]),int(np.argmin(errors))
def edge_bp(rows,obs,p,r):
    weights={};ll=30 if p==0 else math.log((1-p)/p)
    for q,b in zip(rows,obs):weights[q]=weights.get(q,0)+(1-2*b)*ll
    keys=sorted(weights);fi=[];vi=[]
    for j,q in enumerate(keys):
        for i in range(r):
            if q>>i&1:fi.append(j);vi.append(i)
    fi=np.array(fi);vi=np.array(vi);f=np.zeros(len(fi));v=f.copy();n=len(keys);sg=np.tanh(np.array([weights[q] for q in keys])/2);iters=60
    for t in range(60):
        a=np.tanh(v/2);zero=a==0;prod=np.ones(n);np.multiply.at(prod,fi,np.where(zero,1.,a));zc=np.bincount(fi,weights=zero,minlength=n);z=zc[fi]-zero
        exc=np.divide(prod[fi],a,out=prod[fi].copy(),where=~zero);exc=np.where(z==0,exc,0.);new=2*np.arctanh(np.clip(sg[fi]*exc,-1+1e-12,1-1e-12));delta=np.max(abs(new-f));f=(f+new)/2;total=np.bincount(vi,weights=f,minlength=r);v=total[vi]-f
        if delta<1e-6:iters=t+1;break
    return np.tanh(total/2),iters

def audit():
    data=json.load(open('receipt.json'));assert data['source_sha256']==sha('pilot.py');counts={'worlds':0,'conditions':0,'exact_query_probabilities':0,'bp_query_probabilities':0,'bp_marginal_replay':0,'bp_numeric_mismatch_conditions':0,'expansion_probabilities':0,'target_choices':0};maxdelta=0.;bad=[]
    for w in data['worlds']:
        k,r,s=w['k'],w['r'],w['state'];counts['worlds']+=1;assert rank(w['basis'])==r+3 and all(a.bit_count()%2==0 for a in w['basis']);assert s==sum(parity(a,w['hidden'])<<i for i,a in enumerate(w['basis']));test=w['test'];assert len(set(test))==32 and all(q.bit_count()>=2 for q in test);assert w['labels']==[parity(q,s) for q in test]
        for arm,a in w['rows'].items():
            assert len(a)==32*r and a[:r]==[1<<i for i in range(r)] and not set(a)&set(test);assert all(0<q<1<<r for q in a)
            if arm=='sparse':assert all(q.bit_count() in (2,3) for q in a[r:])
            if arm=='repeat':assert all(q.bit_count()==1 for q in a)
        for a in w['results']:
            counts['conditions']+=1;p=a['noise'];m=a['budget']*r;rows=w['rows'][a['arm']][:m];obs=[parity(q,s)^int(u<p) for q,u in zip(rows,w['uniforms'])];mu,it=edge_bp(rows,obs,p,r);delta=float(np.max(abs(mu-a['bp_mu'])));maxdelta=max(maxdelta,delta);counts['bp_marginal_replay']+=r
            if delta>2e-7 or it!=a['bp_iterations']:
                counts['bp_numeric_mismatch_conditions']+=1;bad.append(dict(seed=w['seed'],noise=p,arm=a['arm'],budget=a['budget'],delta=delta,iterations=[it,a['bp_iterations']]))
            saved=a['bp_mu'];pr=[(1-math.prod(saved[i] for i in range(r) if q>>i&1))/2 for q in test];counts['bp_query_probabilities']+=verify_score(pr,w['labels'],a['bp']);state=sum(int(v< -2e-12)<<i for i,v in enumerate(saved));assert state==a['bp_state'] and a['bp_block']==(state==s%(1<<r))
            if 'exact' in a:
                pr,z=posterior(rows,obs,p,r,test);counts['exact_query_probabilities']+=verify_score(pr,w['labels'],a['exact']);assert z==a['exact_state'] and a['exact_block']==(z==s%(1<<r))
        for e in w['expansion']:
            p=e['noise'];rows=w['rows']['dense'][:4*r];obs=[parity(q,s)^int(u<p) for q,u in zip(rows,w['uniforms'])];targets=[(1<<(r+j))^q for j,ts in enumerate(w['targets']) for q in ts];assert e['truth']==[parity(q,s) for q in targets]
            for a in e['policies']:
                calls=a['calls'];assert not set(calls)&set(test+targets+rows);assert a['obs']==[parity(q,s)^int(u<p) for q,u in zip(calls,w['extra_uniforms'])];assert rank(rows+calls)==a['rank_after'];pr,_=posterior(rows+calls,obs+a['obs'],p,r+3,test+targets);counts['expansion_probabilities']+=verify_score(pr[:32],w['labels'],a['inspan'])+verify_score(pr[32:],e['truth'],a['outside'])
                if a['policy']=='expand_target':
                    for j,ts in enumerate(w['targets']):
                        pool=w['pool'][j];q=[t^b for b in pool for t in ts];prob,_=posterior(rows,obs,p,r,q);gains=((1-2*prob)**2).reshape(-1,8).mean(1);idx=pool.index(calls[j]^(1<<(r+j)));assert gains.max()-gains[idx]<2e-12;counts['target_choices']+=1
    print(json.dumps(dict(**counts,max_bp_marginal_delta=maxdelta,bp_mismatch_details=bad,source_sha256=data['source_sha256'],receipt_sha256=sha('receipt.json'),status='PASS_WITH_BP_PORTABILITY_DIFFERENCES' if bad else 'PASS'),indent=2))
if __name__=='__main__':audit()
