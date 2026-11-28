"""Marginal separator-cache planning under changing evidence (development)."""
import itertools,json,random,sys,time,heapq,resource
from collections import Counter,OrderedDict
from fractions import Fraction as F
from pathlib import Path
from base19 import Tree,plan,mask,sha,subs
from base20 import Cache
H=Path(__file__).resolve().parent
FAMILIES=('chain','hub','dense'); REGIMES=('stable_local','shift_distributed')
POLICIES=('lru','online_cold','planned_cold','marginal','lagged'); BUDGETS=(16,64)

def world(n,f,i):
    seed=210000+1000*n+100*FAMILIES.index(f)+i;g=random.Random(seed);order=list(range(n));g.shuffle(order)
    rows=[1<<j for j in range(n)]*2
    if f=='chain':triples=[mask(order[j:j+3]) for j in range(n-2)]
    elif f=='hub':triples=[mask((order[j%2],a,b)) for j,(a,b) in enumerate(zip(order[2:],order[3:]))]
    else:triples=[]
    rows+=triples
    while len(rows)<5*n:rows.append(g.choice(triples) if triples else mask(g.sample(range(n),3)))
    hidden=g.getrandbits(n);uniforms=[g.random() for _ in rows]
    return dict(seed=seed,n=n,family=f,rows=rows,hidden=hidden,uniforms=uniforms,offline_order=order)

def workload(w,regime):
    n=w['n'];g=random.Random(w['seed']+21100+REGIMES.index(regime));pairs=list(itertools.combinations(range(n),2));hot=g.sample(range(n),2);first_hot=hot.copy();batches=[];sources=[]
    for epoch in range(4):
        if epoch and regime=='shift_distributed':hot=g.sample([j for j in range(n) if j not in hot],2)
        old=g.sample(batches[-1],8) if epoch else [];exclude=set(batches[-1]) if epoch else set()
        pool=[q for q in pairs if q not in exclude and any(s in q for s in hot)];need=16-len(old);new=g.sample(pool,min(need,len(pool)))
        if len(new)<need:new+=g.sample([q for q in pairs if q not in exclude and q not in new],need-len(new))
        batch=old+new;g.shuffle(batch);batches.append(batch);sources.append(hot.copy())
    sites=[first_hot[0]]*3 if regime=='stable_local' else g.sample(range(n),3)
    updates=[(j,g.random()) for j in sites]
    return dict(batches=batches,sources=sources,updates=updates)

class Graph:
    def __init__(self,t):
        self.edges=sorted((j,k) for j in t.nbr for k in t.nbr[j]);self.dep={e:tuple((a,e[0]) for a in t.nbr[e[0]] if a!=e[1]) for e in self.edges};self.parents={e:[] for e in self.edges}
        for e,ds in self.dep.items():
            for d in ds:self.parents[d].append(e)
        self.local={e:len(t.states[e[0]])*len(t.nbr[e[0]]) for e in self.edges};self.size={};self.factors={}
        def count(e):
            if e not in self.size:
                self.size[e]=1+sum(count(d) for d in self.dep[e]);self.factors[e]=len(t.local[e[0]])+sum(self.factors[d] for d in self.dep[e])
            return self.size[e]
        for e in self.edges:count(e)
        self.order=sorted(self.edges,key=lambda e:(self.size[e],e));self.charge={}
        for e in self.edges:
            big=(max(t.u,t.v)**(self.factors[e]+3))<<self.size[e]
            self.charge[e]=Cache.charge(e,{x:big for x in subs(t.separator(*e))})
        self.cold=self.residual(set())
    def residual(self,C):
        r={}
        for e in self.order:r[e]=0 if e in C else self.local[e]+sum(r[d] for d in self.dep[e])
        return r
    def flow(self,demand,C):
        d={e:demand.get(e,0) for e in self.edges}
        for e in reversed(self.order):
            if e in C:d[e]=0
            else:
                for k in self.dep[e]:d[k]+=d[e]
        return d
    def work(self,demand,C):return sum(demand.get(e,0)*r for e,r in self.residual(C).items())
    def greedy(self,demand,cap,kind):
        if kind=='planned_cold':
            order=sorted(self.edges,key=lambda e:(-F(demand.get(e,0)*self.cold[e],self.charge[e]),e));C=[];left=cap
            for e in order:
                if demand.get(e,0)>0 and self.charge[e]<=left:C.append(e);left-=self.charge[e]
            return C,dict(heap_pops=0,updates=0)
        C=set();chosen=[];left=cap;D=self.flow(demand,C);R=self.cold.copy();heap=[(-F(D[e]*R[e],self.charge[e]),e) for e in self.edges];heapq.heapify(heap);pops=updates=0
        while heap:
            old,e=heapq.heappop(heap);pops+=1
            if self.charge[e]>left:continue
            v=-F(D[e]*R[e],self.charge[e]);assert old<=v
            if v!=old:heapq.heappush(heap,(v,e));continue
            if not v:break
            chosen.append(e);C.add(e);left-=self.charge[e];de,re=D[e],R[e];D[e]=R[e]=0
            todo=list(self.dep[e])
            for k in todo:
                if k in C:continue
                D[k]-=de;assert D[k]>=0;updates+=1;todo.extend(self.dep[k])
            todo=list(self.parents[e])
            for k in todo:
                if k in C:continue
                R[k]-=re;assert R[k]>=0;updates+=1;todo.extend(self.parents[k])
        return chosen,dict(heap_pops=pops,updates=updates)

def direct_demand(t,queries):
    d=Counter((a,t.n) for a in t.nbr[t.n]);remain=set(map(tuple,queries))
    while remain:
        freq=Counter(v for q in remain for v in q);s=min(freq,key=lambda j:(-freq[j],j));chosen={q for q in remain if s in q};targets={q[0] if q[1]==s else q[1] for q in chosen};edges=set();prev={s:None}
        for k in targets:
            path=t.path(s,k)
            for a,b in zip(path,path[1:]):edges.add((a,b));prev[b]=a
        for j,k in edges:d.update((a,j) for a in t.nbr[j] if a!=k and a!=prev[j])
        for j in targets:d.update((a,j) for a in t.nbr[j] if a!=prev[j])
        remain-=chosen
    return d

class Memo:
    def __init__(self,t,g,policy,budget):
        self.t=t;self.g=g;self.policy=policy;self.cap=budget;self.data=OrderedDict();self.req=Counter();self.target=set();self.bytes=0;self.peak=0;self.reset()
    def reset(self):self.events=[];self.counts=Counter();self.external=Counter()
    def remove(self,e,why):
        del self.data[e];self.bytes-=self.g.charge[e];self.events.append([why,*e,self.g.charge[e]]);self.counts[why]+=1
    def admit(self,e,msg):
        c=self.g.charge[e];assert Cache.charge(e,msg)<=c
        if c>self.cap or (self.policy.startswith('planned') or self.policy in ('marginal','lagged')) and e not in self.target:self.counts['rejected']+=1;return
        if self.bytes+c>self.cap:
            assert self.policy in ('lru','online_cold')
            order=list(self.data) if self.policy=='lru' else sorted(self.data,key=lambda k:(self.req[k]*self.g.cold[k]/self.g.charge[k],k))
            victims=[];free=self.cap-self.bytes;rank=lambda k:(self.req[k]*self.g.cold[k]/self.g.charge[k],k)
            for k in order:
                if self.policy=='online_cold' and rank(e)<=rank(k):self.counts['rejected']+=1;return
                victims.append(k);free+=self.g.charge[k]
                if free>=c:break
            for k in victims:self.remove(k,'evict')
        self.data[e]=msg;self.bytes+=c;self.peak=max(self.peak,self.bytes);self.events.append(['insert',*e,c]);self.counts['insert']+=1;assert self.bytes<=self.cap
    def __getitem__(self,e):self.external[e]+=1;return self.get(e)
    def get(self,e):
        self.req[e]+=1;self.counts['requests']+=1
        if e in self.data:self.counts['hits']+=1;self.data.move_to_end(e);return self.data[e]
        self.counts['misses']+=1;j,k=e;msg=self.t.message(j,k,[(a,self.get((a,j))) for a in self.t.nbr[j] if a!=k]);self.admit(e,msg);return msg
    def invalidate(self,owner):
        marked=set();todo=[(owner,None)]
        for j,prev in todo:
            for k in self.t.nbr[j]:
                if k!=prev:marked.add((j,k));todo.append((k,j))
        before=len(self.data)
        for e in list(self.data):
            if e in marked:self.remove(e,'invalidate')
        return [before,len(self.data)]
    def retarget(self,target):
        self.target=set(target);assert sum(self.g.charge[e] for e in target)<=self.cap
        for e in list(self.data):
            if e not in self.target:self.remove(e,'retarget')

def encode(d):return [[*e,v] for e,v in sorted(d.items()) if v]
def run_arm(w,p,regime,wl,compiled,arm):
    policy,kib=arm;start=time.perf_counter();t=Tree(w,p,compiled);g=Graph(t) if kib else None;memo=Memo(t,g,policy,kib*1024) if kib else None
    if memo:t.cache=memo
    owner={i:j for j,ids in t.local.items() for i in ids};prev=None;epochs=[]
    for epoch,qs in enumerate(wl['batches']):
        begin=time.perf_counter();t.reset();update=None;planrec=None;plansec=0.
        if memo:memo.reset()
        if epoch:
            ix,u=wl['updates'][epoch-1];row=t.rows[ix];y=((row&w['hidden']).bit_count()&1)^int(u<p);j=owner[ix]
            for x in t.states[j]:t.base[j][x]*=t.u if ((x&row).bit_count()&1)!=y else t.v
            update=dict(index=ix,obs=y,owner=j)
            if memo:update['counts']=memo.invalidate(j)
        if memo:
            if policy in ('planned_cold','marginal','lagged'):
                tick=time.perf_counter();d=direct_demand(t,qs);forecast=prev if policy=='lagged' and prev is not None else d;target,stats=g.greedy(forecast,memo.cap,policy);memo.retarget(target);planrec=dict(target=[list(e) for e in target],forecast=encode(forecast),current=encode(d),warm_work=g.work(forecast,set(target)),uncached_work=g.work(forecast,set()),**stats);prev=d;plansec=time.perf_counter()-tick
            z=t.finish(t.n,[(a,memo[a,t.n]) for a in t.nbr[t.n]]);signed=t.batch(qs)
            if planrec:assert memo.external==d
        else:z=t.direct();signed=[t.direct((1<<a)|(1<<b)) for a,b in qs]
        assert z>0 and all(-z<=x<=z for x in signed)
        rec=dict(epoch=epoch,update=update,partition=str(z),signed=[str(x) for x in signed],counts=t.counts(),planning_seconds=plansec,seconds=time.perf_counter()-begin,plan=planrec)
        if memo:rec.update(cache=dict(memo.counts),events=memo.events,bytes=memo.bytes,peak_bytes=memo.peak,external=encode(memo.external))
        epochs.append(rec)
    return dict(policy=policy,kib=kib,epochs=epochs,seconds=time.perf_counter()-start,charges=encode(g.charge) if g else [],peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def evaluate(w,p):
    compiled=plan(w['n'],w['rows']);peak=max(a.bit_count() for a in compiled[1].values());r=dict(seed=w['seed'],noise=p,max_scope=peak,status='refused' if peak>6 else 'solved',regimes=[])
    if peak>6:return r
    arms=[('direct',0)]+[(a,k) for k in BUDGETS for a in POLICIES]
    for regime in REGIMES:
        wl=workload(w,regime);offset=(w['seed']+round(p*100)+REGIMES.index(regime))%len(arms);order=arms[offset:]+arms[:offset];out=[run_arm(w,p,regime,wl,compiled,a) for a in order];ref=next(a for a in out if a['policy']=='direct')['epochs']
        for a in out:
            for x,y in zip(a['epochs'],ref):assert (x['partition'],x['signed'])==(y['partition'],y['signed'])
        r['regimes'].append(dict(regime=regime,workload=wl,arms=out))
    return r

def main():
    n=int(sys.argv[1]);worlds=[world(n,f,i) for f in FAMILIES for i in range(3)];out=[]
    for w in worlds:
        for p in (.15,.35):out.append(evaluate(w,p));print(w['seed'],p,out[-1]['status'],flush=True)
    (H/f'receipt{n}.json').write_text(json.dumps(dict(source_sha256=sha(__file__),protocol_sha256=sha(H/'PROTOCOL.md'),base_sha256={a:sha(H/a) for a in ('base19.py','base20.py','reference19.py')},worlds=worlds,results=out),separators=(',',':'))+'\n')
if __name__=='__main__':main()
