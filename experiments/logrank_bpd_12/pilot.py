"""Rank-adaptive noisy verifier acquisition: development only."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from bp import bp
HERE=Path(__file__).resolve().parent
ARMS=('random','old_only','rank_first','entropy','exact_gain','bp_gain','learned')
NOISE=(.15,.35); B=12
NAMES=('bp_gain','probe_abs','probe_uncertainty','expands','new_mass','xor_mass','degree','new_degree','count','rank','step','noise','old_weight','target_uncertainty','rho2_new_mass','overlap')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dot(a,b):return (int(a)&int(b)).bit_count()%2
def basis(rows):
    piv={}
    for a in rows:
        a=int(a)
        while a and a.bit_length()-1 in piv:a^=piv[a.bit_length()-1]
        if a:piv[a.bit_length()-1]=a
    return piv
def member(a,piv):
    a=int(a)
    while a:
        b=piv.get(a.bit_length()-1)
        if b is None:return False
        a^=b
    return True
def walsh(p):
    x=p.copy();h=1
    while h<len(x):
        a=x.reshape(-1,2*h);l=a[:,:h].copy();r=a[:,h:].copy();a[:,:h]=l+r;a[:,h:]=l-r;h*=2
    return x
def world(r,i):
    seed=120000+100*r+i;g=np.random.default_rng(seed);d=r+3;n=1<<d;old=1<<r
    target=g.choice([q for q in range(1,old) if q.bit_count()>1],8,replace=False).tolist();pool=g.choice([q for q in range(1,old) if q not in target],16,replace=False).tolist()
    for j in range(3):
        off=g.choice(old,16,replace=False).tolist();target += [(1<<(r+j))^q for q in off[:8]];pool += [(1<<(r+j))^q for q in off[8:]]
    pool+=g.choice([q for q in range(old,n) if (q>>r).bit_count()>=2],8,replace=False).tolist();g.shuffle(pool)
    initial=[1<<j for j in range(r)]+g.choice([q for q in range(1,old) if q not in target],3*r,replace=True).tolist();oldmass=.25 if i%2==0 else .75
    assert len(set(pool))==48 and not set(pool)&set(target) and not set(initial)&set(target)
    return dict(r=r,d=d,seed=seed,hidden=int(g.integers(n)),targets=target,pool=pool,initial=initial,weights=[oldmass/8]*8+[(1-oldmass)/24]*24,oldmass=oldmass,uniforms=g.random(len(initial)+B).tolist(),random_order=g.permutation(pool).tolist())
def gains(moment,qs,pool,weights,noise):
    q=np.array(qs);a=np.array(pool);rho=1-2*noise;ma=moment(a);cov=moment(q[:,None]^a)-moment(q)[:,None]*ma
    return rho**2*(weights@cov**2)/(4*(1-(rho*ma)**2))
def features(w,rows,obs,pool,noise,step):
    piv=basis(rows);mu,it=bp(rows,obs,noise,w['d']);r=w['r'];qs=w['targets'];weights=np.array(w['weights']);cache={};inside={}
    def inspan(q):
        q=int(q)
        if q not in inside:inside[q]=member(q,piv)
        return inside[q]
    def mom(q):
        q=int(q)
        if q not in cache:cache[q]=float(np.prod([mu[i] for i in range(w['d']) if q>>i&1])) if inspan(q) else 0.
        return cache[q]
    def moment(a):return np.array([mom(q) for q in a.flat]).reshape(a.shape)
    g=gains(moment,qs,pool,weights,noise);unc=float(weights@(1-moment(np.array(qs))**2));x=[]
    for a,v in zip(pool,g):
        ma=abs(mom(a));newmass=sum(p for q,p in zip(qs,weights) if not inspan(q) and inspan(q^a));xormass=sum(p for q,p in zip(qs,weights) if inspan(q^a));overlap=sum(p*(q&a).bit_count()/q.bit_count() for q,p in zip(qs,weights));x.append([v,ma,1-ma**2,int(not inspan(a)),newmass,xormass,a.bit_count()/w['d'],(a>>r).bit_count()/3,rows.count(a)/4,len(piv)/w['d'],step/B,noise,w['oldmass'],unc,(1-2*noise)**2*newmass,overlap])
    return np.array(x),it
def score(mu,w):
    p=np.clip((1-mu[w['targets']])/2,0,1);y=np.array([dot(q,w['hidden']) for q in w['targets']]);ok=(p>.5+1e-12)==y;wt=np.array(w['weights']);take=np.maximum(p,1-p)>=.9-1e-12
    return dict(prob=p.tolist(),brier=float(wt@(p-y)**2),correct_old=int(ok[:8].sum()),correct_new=int(ok[8:].sum()),coverage=float(wt@take),covered_correct=float(wt@(take&ok)))
def trajectory(w,noise,arm,model=None,collect=False):
    states=np.arange(1<<w['d']);parity=np.array([int(i).bit_count()%2 for i in states]);rows=list(w['initial']);obs=[dot(a,w['hidden'])^int(u<noise) for a,u in zip(rows,w['uniforms'])];p=np.ones(len(states))/len(states)
    def upd(p,a,b):
        p=p*np.where(parity[states&a]==b,1-noise,noise);return p/p.sum()
    for a,b in zip(rows,obs):p=upd(p,a,b)
    mu=walsh(p);pool=list(w['pool']);trace=[];scored=[];training=[];its=0;sec=0.
    for step in range(B+1):
        piv=basis(rows);sc=score(mu,w);sc['rank']=len(piv);scored.append(sc)
        if step==B:break
        start=time.perf_counter()
        if collect or arm in ('old_only','rank_first','exact_gain'):g=gains(lambda a:mu[a],w['targets'],pool,np.array(w['weights']),noise)
        if collect or arm in ('bp_gain','learned'):x,it=features(w,rows,obs,pool,noise,step);its+=it
        if collect:training.append((x,g))
        if arm=='random':j=pool.index(next(a for a in w['random_order'] if a in pool))
        elif arm=='old_only':j=max((j for j,a in enumerate(pool) if a<1<<w['r']),key=lambda j:g[j])
        elif arm=='rank_first':j=max([j for j,a in enumerate(pool) if not member(a,piv)] or list(range(len(pool))),key=lambda j:g[j])
        elif arm=='entropy':j=int(np.argmin(abs(mu[pool])))
        elif arm=='exact_gain':j=int(g.argmax())
        elif arm=='bp_gain':j=int(x[:,0].argmax())
        elif arm=='learned':j=int((((x-model['mean'])/model['std'])@model['coef']).argmax())
        else:raise ValueError(arm)
        sec+=time.perf_counter()-start;a=pool.pop(j);b=dot(a,w['hidden'])^int(w['uniforms'][len(w['initial'])+step]<noise);trace.append(dict(query=a,obs=b,expands=not member(a,piv)));rows.append(a);obs.append(b);p=upd(p,a,b);mu=walsh(p)
    return dict(arm=arm,noise=noise,initial_obs=obs[:len(w['initial'])],trace=trace,steps=scored,selection_seconds=sec,bp_iterations=its),training
def train():
    X=[];Y=[];training=[]
    for r in (5,8):
        for i in range(8):
            w=world(r,i)
            for noise in NOISE:
                for arm in ('exact_gain','random'):
                    row,data=trajectory(w,noise,arm,collect=True);X.extend(a for a,b in data);Y.extend(b for a,b in data);training.append(dict(seed=w['seed'],noise=noise,arm=arm,trace=row['trace']))
    x=np.concatenate(X);y=np.concatenate(Y);mean=x.mean(0);std=x.std(0);std[std<1e-12]=1;xx=(x-mean)/std;coef=np.linalg.solve(xx.T@xx+10*np.eye(xx.shape[1]),xx.T@(y-y.mean()));model=dict(mean=mean.tolist(),std=std.tolist(),coef=coef.tolist(),intercept=float(y.mean()),samples=len(y),rmse=float(np.sqrt(np.mean((xx@coef+y.mean()-y)**2))),features=NAMES)
    (HERE/'model.json').write_text(json.dumps(dict(model=model,training=training,protocol_sha256=sha(HERE/'PROTOCOL.md'),source_sha256=sha(__file__),bp_sha256=sha(HERE/'bp.py')),indent=2)+'\n');print('TRAIN',len(y),model['rmse'],flush=True)
def evaluate(start,end):
    rec=json.loads((HERE/'model.json').read_text());model={k:np.array(rec['model'][k]) for k in ('mean','std','coef')};assert rec['source_sha256']==sha(__file__);result=[]
    for r in (5,8):
        for i in range(start,end):
            w=world(r,i);w['runs']=[]
            for noise in NOISE:
                for arm in ARMS:w['runs'].append(trajectory(w,noise,arm,model)[0])
            result.append(w);print('EVAL',r,w['seed'],flush=True)
    (HERE/f'receipt_{start}_{end}.json').write_text(json.dumps(dict(**rec,worlds=result),separators=(',',':'))+'\n')
if __name__=='__main__':
    import sys
    if sys.argv[1]=='train':train()
    else:evaluate(int(sys.argv[1]),int(sys.argv[2]))
