"""Resident and retention-aware separator caching, development only."""
import itertools,json,random,sys,time,resource,math
from collections import Counter
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from pilot21 import Tree,Graph,Memo,plan,mask,sha,encode,direct_demand
H=Path(__file__).resolve().parent
FAMILIES=('chain','hub','dense'); REGIMES=tuple(itertools.product(('stable','shift'),('local','uniform')))
POLICIES=('lru','planned_cold','marginal','resident','retention'); BUDGETS=(16,64)
def world(n,f,i):
    seed=220000+1000*n+100*FAMILIES.index(f)+i;g=random.Random(seed);order=list(range(n));g.shuffle(order)
    rows=[1<<j for j in range(n)]*2
    if f=='chain':triples=[mask(order[j:j+3]) for j in range(n-2)]
    elif f=='hub':triples=[mask((order[j%2],a,b)) for j,(a,b) in enumerate(zip(order[2:],order[3:]))]
    else:triples=[]
    rows+=triples
    while len(rows)<5*n:rows.append(g.choice(triples) if triples else mask(g.sample(range(n),3)))
    hidden=g.getrandbits(n);uniforms=[g.random() for _ in rows]
    return dict(seed=seed,n=n,family=f,rows=rows,hidden=hidden,uniforms=uniforms,offline_order=order)
def workload(w,pop,locality):
    n=w['n'];g=random.Random(w['seed']+22100);pairs=list(itertools.combinations(range(n),2));hot=g.sample(range(n),2);batches=[];sources=[]
    for epoch in range(4):
        if epoch and pop=='shift':hot=g.sample([j for j in range(n) if j not in hot],2)
        old=g.sample(batches[-1],8) if epoch else [];exclude=set(batches[-1]) if epoch else set()
        pool=[q for q in pairs if q not in exclude and any(s in q for s in hot)];need=16-len(old);new=g.sample(pool,min(need,len(pool)))
        if len(new)<need:new+=g.sample([q for q in pairs if q not in exclude and q not in new],need-len(new))
        batch=old+new;g.shuffle(batch);batches.append(batch);sources.append(hot.copy())
    u=random.Random(w['seed']+22200);site=u.randrange(n);draws=[u.randrange(n) for _ in range(3)];uniforms=[u.random() for _ in range(3)]
    updates=list(zip([site]*3 if locality=='local' else draws,uniforms))
    v=random.Random(w['seed']+22300);scenarios=[[site] if locality=='local' else [v.randrange(n) for _ in range(4)] for _ in range(3)]+[[]]
    return dict(batches=batches,sources=sources,updates=updates,scenario_sites=scenarios,local_site=site)
class Planner:
    def __init__(self,g,t):
        self.g=g;self.edges=g.order;self.ix={e:i for i,e in enumerate(self.edges)};self.E=len(self.edges);self.deps=[[self.ix[d] for d in g.dep[e]] for e in self.edges];self.local=np.array([g.local[e] for e in self.edges],dtype=np.int64);self.charges=np.array([g.charge[e] for e in self.edges],dtype=np.int64)
        self.survive={}
        for owner in t.nbr:
            marked=set();todo=[(owner,None)]
            for j,prev in todo:
                for k in t.nbr[j]:
                    if k!=prev:marked.add((j,k));todo.append((k,j))
            self.survive[owner]=np.array([e not in marked for e in self.edges],dtype=bool)
    def array(self,C):return np.array([e in C for e in self.edges],dtype=bool)
    def flow(self,demand,target,resident):
        """Vector rows are target sets, top-down demand with exact lazy fills."""
        D=np.broadcast_to(demand,target.shape).copy();filled=target & resident;work=np.zeros(len(target),dtype=np.int64)
        for i in range(self.E-1,-1,-1):
            d=D[:,i];m=np.where(filled[:,i],0,np.where(target[:,i],np.minimum(d,1),d));filled[:,i]|=target[:,i] & (d>0);work+=m*self.local[i]
            for j in self.deps[i]:D[:,j]+=m
        return work,filled
    def score(self,demand,target,resident,owners):
        now,filled=self.flow(demand,target,resident);N=len(owners);val=now*max(1,N)
        if N:
            surv=np.stack([self.survive[o] for o in owners]);t=np.repeat(target,N,axis=0);r=(filled[:,None,:]&surv[None,:,:]).reshape(t.shape);f,_=self.flow(demand,t,r);val+=f.reshape(len(target),N).sum(axis=1)
        return val,now,filled
    def greedy(self,d,C,cap,owners):
        demand=np.array([d.get(e,0) for e in self.edges],dtype=np.int64);resident=self.array(C)[None,:];target=np.zeros((1,self.E),dtype=bool);left=cap;steps=[];order=[];evals=0
        while True:
            elig=np.flatnonzero(~target[0] & (self.charges<=left));
            if not len(elig):break
            candidates=np.repeat(target,len(elig)+1,axis=0);candidates[np.arange(1,len(elig)+1),elig]=True
            vals,now,_=self.score(demand,candidates,resident,owners);evals+=len(vals);base=int(vals[0]);best=None
            for j,v in zip(elig,vals[1:]):
                gain=base-int(v)
                if gain<=0:continue
                j=int(j);rank=(F(gain,int(self.charges[j])),tuple(-x for x in self.edges[j]))
                if best is None or rank>best[0]:best=(rank,j,gain)
            if best is None:break
            _,j,gain=best;target[0,j]=True;left-=int(self.charges[j]);order.append(self.edges[j]);steps.append([*self.edges[j],base,base-gain]);
        vals,now,filled=self.score(demand,target,resident,owners)
        return order,dict(objective=int(vals[0]),predicted_now=int(now[0]),predicted_filled=[list(e) for i,e in enumerate(self.edges) if filled[0,i]],steps=steps,candidate_evals=evals,scenario_owners=owners)
def run_arm(w,p,wl,compiled,arm):
    policy,kib=arm;start=time.perf_counter();t=Tree(w,p,compiled);g=Graph(t) if kib else None;memo=Memo(t,g,'marginal' if policy in ('resident','retention') else policy,kib*1024) if kib else None;planner=Planner(g,t) if policy in ('resident','retention') else None
    if memo:t.cache=memo
    owner={i:j for j,ids in t.local.items() for i in ids};epochs=[]
    for epoch,qs in enumerate(wl['batches']):
        begin=time.perf_counter();t.reset();update=None;pr=None;plansec=0.
        if memo:memo.reset()
        if epoch:
            ix,u=wl['updates'][epoch-1];row=t.rows[ix];y=((row&w['hidden']).bit_count()&1)^int(u<p);j=owner[ix]
            for x in t.states[j]:t.base[j][x]*=t.u if ((x&row).bit_count()&1)!=y else t.v
            update=dict(index=ix,obs=y,owner=j)
            if memo:update['counts']=memo.invalidate(j)
        if memo:
            if policy!='lru':
                tick=time.perf_counter();d=direct_demand(t,qs);res=set(memo.data);owners=[owner[i] for i in wl['scenario_sites'][epoch]] if policy=='retention' else []
                if planner:target,stats=planner.greedy(d,res,memo.cap,owners)
                else:target,stats=g.greedy(d,memo.cap,policy)
                memo.retarget(target);pr=dict(target=[list(e) for e in target],resident=[list(e) for e in sorted(res)],demand=encode(d),**stats);plansec=time.perf_counter()-tick
            z=t.finish(t.n,[(a,memo[a,t.n]) for a in t.nbr[t.n]]);signed=t.batch(qs)
            if pr:assert memo.external==d
            if planner:assert set(map(tuple,pr['predicted_filled']))==set(memo.data)
        else:z=t.direct();signed=[t.direct((1<<a)|(1<<b)) for a,b in qs]
        assert z>0 and all(-z<=x<=z for x in signed)
        rec=dict(epoch=epoch,update=update,partition=str(z),signed=[str(x) for x in signed],counts=t.counts(),planning_seconds=plansec,seconds=time.perf_counter()-begin,plan=pr)
        if memo:rec.update(cache=dict(memo.counts),events=memo.events,bytes=memo.bytes,peak_bytes=memo.peak,external=encode(memo.external))
        epochs.append(rec)
    return dict(policy=policy,kib=kib,epochs=epochs,seconds=time.perf_counter()-start,charges=encode(g.charge) if g else [],peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
def evaluate(w,p):
    compiled=plan(w['n'],w['rows']);peak=max(a.bit_count() for a in compiled[1].values());r=dict(seed=w['seed'],noise=p,max_scope=peak,status='refused' if peak>6 else 'solved',regimes=[])
    if peak>6:return r
    arms=[('direct',0)]+[(a,k) for k in BUDGETS for a in POLICIES]
    for pop,loc in REGIMES:
        wl=workload(w,pop,loc);offset=(w['seed']+round(p*100)+REGIMES.index((pop,loc)))%len(arms);out=[run_arm(w,p,wl,compiled,a) for a in arms[offset:]+arms[:offset]];ref=next(a for a in out if a['policy']=='direct')['epochs']
        for a in out:
            for x,y in zip(a['epochs'],ref):assert (x['partition'],x['signed'])==(y['partition'],y['signed'])
        r['regimes'].append(dict(pop=pop,locality=loc,workload=wl,arms=out))
    return r
def main():
    n=int(sys.argv[1]);worlds=[world(n,f,i) for f in FAMILIES for i in range(3)];out=[]
    for w in worlds:
        for p in (.15,.35):
            out.append(evaluate(w,p));print(w['seed'],p,out[-1]['status'],flush=True)
            (H/f'receipt{n}.json').write_text(json.dumps(dict(source_sha256=sha(__file__),protocol_sha256=sha(H/'PROTOCOL.md'),worlds=worlds,results=out),separators=(',',':'))+'\n')
if __name__=='__main__':main()
