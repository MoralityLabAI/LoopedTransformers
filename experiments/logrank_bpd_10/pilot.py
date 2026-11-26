"""LOGRANK-BPD-10: controlled parity identifiability and noisy decoding."""
import hashlib, json, math
from pathlib import Path
import numpy as np


def dot(a,b): return (int(a)&int(b)).bit_count()&1

def span(basis):
    s=[0]
    for a in basis: s += [x^a for x in s]
    return s

def random_even(g,k):
    x=int(g.integers(0,1<<k,dtype=np.uint64))
    return x^((x.bit_count()&1))

def world(k,r,seed):
    g=np.random.default_rng(seed); basis=[]; masks=[0]
    while len(basis)<r:
        a=random_even(g,k)
        if a not in masks: basis.append(a); masks += [x^a for x in masks]
    n=1<<r; h=int(g.integers(0,1<<k,dtype=np.uint64)); coeff=list(1<<i for i in range(r))+g.integers(1,n,3*r).tolist()
    avail=np.array(sorted(set(range(1,n))-set(coeff))); g.shuffle(avail); test=avail[:32].tolist(); pool=avail[32:80].tolist(); outside=[]; ms=set(masks)
    while len(outside)<32:
        a=random_even(g,k)
        if a not in ms and a not in outside: outside.append(a)
    return dict(k=k,r=r,seed=seed,basis=basis,coeff=coeff,test=test,pool=pool,outside=outside,h=h),masks

def posterior(table,coeff,obs,noise):
    mism=(table[coeff]!=np.array(obs)[:,None]).sum(0)
    if noise==0:
        p=(mism==0).astype(float); assert p.sum()>0
    else: p=np.exp((mism-mism.min())*math.log(noise/(1-noise)))
    return p/p.sum()

def probs(table,queries,p): return table[queries]@p

def score(prob,y):
    pred=(prob>.5).astype(int); take=np.maximum(prob,1-prob)>=.9-1e-12
    return dict(correct=int((pred==y).sum()),covered=int(take.sum()),covered_correct=int(((pred==y)&take).sum()),brier=float(np.mean((prob-y)**2)),pred=pred.tolist(),prob=prob.tolist())

def hard_fit(A,y,seed):
    # Train hard signed component reads with straight-through tanh gradients.
    g=np.random.default_rng(seed); z=g.normal(0,.25,(4,A.shape[1])); m=np.zeros_like(z); v=np.zeros_like(z)
    target=1-2*y
    for t in range(1,201):
        s=np.where(z>=0,1.,-1.); p=1-2*((s<0).astype(np.int32)@A.T%2); err=p-target
        grad=((err*p)@A)*s*(1-np.tanh(z)**2)/len(y); m=.9*m+.1*grad; v=.999*v+.001*grad**2
        z-=.05*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
    s=(z<0).astype(np.int32); err=((s@A.T%2)!=y).sum(1); j=int(err.argmin()); h=sum(int(b)<<i for i,b in enumerate(s[j]))
    return h,err.tolist()

def run():
    rows=[]
    for k in (16,32):
        for r in (8,12):
            n=1<<r; parity=np.array([i.bit_count()&1 for i in range(n)],dtype=np.uint8); table=parity[np.arange(n)[:,None]&np.arange(n)]
            for s in range(6):
                seed=3100+100*k+10*r+s; w,masks=world(k,r,seed); h=w['h']; coeff=w['coeff']; test=w['test']; outside=w['outside']; pool=w['pool']; y=np.array([dot(masks[q],h) for q in test]); yo=np.array([dot(q,h) for q in outside]); A=np.array([[(masks[a]>>j)&1 for j in range(k)] for a in coeff],dtype=np.int32)
                for noise in (0.,.15,.35):
                    g=np.random.default_rng(seed+50000); flips=(g.random(len(coeff))<noise).astype(int); obs=np.array([dot(masks[a],h) for a in coeff])^flips; basis_state=sum(int(v)<<i for i,v in enumerate(obs[:r])); base=np.array([dot(a,basis_state) for a in test]); bh,errs=hard_fit(A,obs,seed+60000); hard=np.array([dot(masks[a],bh) for a in test]); ho=np.array([dot(a,bh) for a in outside]); p=posterior(table,coeff,obs,noise); initial=score(probs(table,test,p),y); active={}; uniforms=g.random(4); random_order=g.permutation(pool).tolist()
                    for method in ('entropy','random'):
                        c=list(coeff); o=obs.tolist(); remain=list(pool); trace=[]; post=p
                        for step in range(4):
                            if method=='entropy': a=remain[int(np.argmin(abs(probs(table,remain,post)-.5)))]
                            else: a=random_order[step]
                            remain.remove(a); b=dot(masks[a],h)^int(uniforms[step]<noise); c.append(a); o.append(b); post=posterior(table,c,o,noise); trace.append(dict(query=a,obs=b,score=score(probs(table,test,post),y)))
                        active[method]=trace
                    rows.append(dict(**w,noise=noise,obs=obs.tolist(),flips=flips.tolist(),test_y=y.tolist(),outside_y=yo.tolist(),basis_correct=int((base==y).sum()),hard_h=bh,hard_train_errors=errs,hard_correct=int((hard==y).sum()),hard_outside_correct=int((ho==yo).sum()),posterior=initial,active=active))
                print(k,r,seed,flush=True)
    return dict(evidence='development only; known coordinate map; no singleton supervision',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows)

if __name__=='__main__':
    Path('receipt.json').write_text(json.dumps(run(),indent=2)+'\n')
