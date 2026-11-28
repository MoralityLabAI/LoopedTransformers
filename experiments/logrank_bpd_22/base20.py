"""Online partial separator cache with selective evidence invalidation."""
import itertools, json, random, sys, time, resource
from collections import Counter, OrderedDict
from functools import lru_cache
from fractions import Fraction
from pathlib import Path
from base19 import Tree, plan, mask, bits, sha

H=Path(__file__).resolve().parent
ARMS=[('direct',0,False)]+[(p,k,False) for k in (16,64,256) for p in ('lru','value')]+[('lru',1024,False),('lru',64,True),('lru',1024,True)]
FAMILIES=('chain','hub','dense')

def world(n,family,i):
    seed=200000+1000*n+100*FAMILIES.index(family)+i; g=random.Random(seed)
    order=list(range(n));g.shuffle(order);rows=[1<<j for j in range(n)]*2
    if family=='chain': triples=[mask(order[j:j+3]) for j in range(n-2)]
    elif family=='hub':
        hubs=order[:2];leaf=order[2:];triples=[mask((hubs[j%2],a,b)) for j,(a,b) in enumerate(zip(leaf,leaf[1:]))]
    else:triples=[]
    rows+=triples
    while len(rows)<5*n:rows.append(g.choice(triples) if triples else mask(g.sample(range(n),3)))
    hidden=g.getrandbits(n);uniforms=[g.random() for _ in rows]; updates=[(g.randrange(len(rows)),g.random()) for _ in range(3)]
    pairs=list(itertools.combinations(range(n),2));qs=[]
    for s in g.sample(range(n),2):
        a=[q for q in pairs if s in q and q not in qs];qs+=g.sample(a,min(8,len(a)))
    qs+=g.sample([q for q in pairs if q not in qs],16-len(qs));batches=[qs]
    for _ in range(3):
        old=g.sample(batches[-1],8);new=g.sample([q for q in pairs if q not in batches[-1]],8);batch=old+new;g.shuffle(batch);batches.append(batch)
    return dict(n=n,family=family,seed=seed,rows=rows,hidden=hidden,uniforms=uniforms,updates=updates,batches=batches,offline_order=order)

class Cache:
    def __init__(self,tree,policy,budget):
        self.tree=tree;self.policy=policy;self.cap=budget;self.data=OrderedDict();self.charges={};self.counts=Counter();self.req=Counter();self.bytes=0;self.peak=0;self.events=[]
    @lru_cache(None)
    def cold(self,j,k):
        t=self.tree
        return len(t.states[j]) * len(t.nbr[j]) + sum(self.cold(a,j) for a in t.nbr[j] if a!=k)
    @staticmethod
    def charge(key,table):
        return 256+sys.getsizeof(key)+sum(sys.getsizeof(x) for x in key)+sys.getsizeof(table)+sum(sys.getsizeof(x)+sys.getsizeof(y) for x,y in table.items())
    def rank(self,key,charge):return self.req[key]*self.cold(*key)/charge,key
    def remove(self,key,reason):
        c=self.charges.pop(key);del self.data[key];self.bytes-=c;self.counts[reason]+=1;self.events.append([reason,*key,c]);assert self.bytes>=0
    def admit(self,key,table):
        c=self.charge(key,table)
        if c>self.cap:self.counts['rejected']+=1;return
        victims=[];free=self.cap-self.bytes
        if c>free:
            order=list(self.data) if self.policy=='lru' else sorted(self.data,key=lambda k:self.rank(k,self.charges[k]))
            for k in order:
                if self.policy=='value' and self.rank(key,c)<=self.rank(k,self.charges[k]):self.counts['rejected']+=1;return
                victims.append(k);free+=self.charges[k]
                if free>=c:break
        assert free>=c
        for k in victims:self.remove(k,'evict')
        self.data[key]=table;self.charges[key]=c;self.bytes+=c;self.peak=max(self.peak,self.bytes);self.events.append(['insert',*key,c]);self.counts['insert']+=1;assert self.bytes<=self.cap
    def __getitem__(self,key):
        self.req[key]+=1;self.counts['requests']+=1
        if key in self.data:
            self.counts['hits']+=1;self.data.move_to_end(key);return self.data[key]
        self.counts['misses']+=1;j,k=key;t=self.tree;msg=t.message(j,k,[(a,self[a,j]) for a in t.nbr[j] if a!=k]);self.admit(key,msg);return msg
    def invalidate(self,owner,flush):
        affected=set();stack=[(owner,None)]
        while stack:
            j,prev=stack.pop()
            for k in self.tree.nbr[j]:
                if k!=prev:affected.add((j,k));stack.append((k,j))
        assert len(affected)==self.tree.n
        before=len(self.data)
        for key in list(self.data):
            if flush or key in affected:self.remove(key,'invalidate')
        return dict(owner=owner,before=before,after=len(self.data),affected_directions=len(affected),flush=flush)
    def clear_stats(self):self.counts=Counter();self.events=[]

def evaluate_arm(w,p,compiled,arm):
    policy,kib,flush=arm;start=time.perf_counter();t=Tree(w,p,compiled);memo=None
    owner={i:j for j,ids in t.local.items() for i in ids}
    if policy!='direct':memo=Cache(t,policy,kib*1024);t.cache=memo
    epochs=[]
    for i,qs in enumerate(w['batches']):
        begin=time.perf_counter();t.reset();update=None
        if memo:memo.clear_stats()
        if i:
            ix,u=w['updates'][i-1];row=t.rows[ix];y=((row&w['hidden']).bit_count()&1)^int(u<p);j=owner[ix]
            # The graph and ownership are unchanged because this is a repeated observation mask.
            for x in t.states[j]:t.base[j][x]*=t.u if ((x&row).bit_count()&1)!=y else t.v
            update=dict(index=ix,obs=y,owner=j)
            if memo:update.update(memo.invalidate(j,flush))
        if memo:
            z=t.finish(t.n,[(a,memo[a,t.n]) for a in t.nbr[t.n]]);signed=t.batch(qs)
        else:z=t.direct();signed=[t.direct((1<<a)|(1<<b)) for a,b in qs]
        assert z>0 and all(-z<=v<=z for v in signed)
        sec=time.perf_counter()-begin
        rec=dict(epoch=i,update=update,partition=str(z),signed=[str(v) for v in signed],counts=t.counts(),seconds=sec)
        if memo:rec.update(cache=dict(memo.counts),events=memo.events,bytes=memo.bytes,peak_bytes=memo.peak,entries=len(memo.data))
        epochs.append(rec)
    if memo:memo.cold.cache_clear()
    return dict(policy=policy,kib=kib,flush=flush,epochs=epochs,seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def evaluate(w,p):
    start=time.perf_counter();compiled=plan(w['n'],w['rows']);sec=time.perf_counter()-start;peak=max(a.bit_count() for a in compiled[1].values());result=dict(seed=w['seed'],noise=p,max_scope=peak,planning_seconds=sec,status='refused' if peak>6 else 'solved')
    if peak>6:return result
    result['arms']=[evaluate_arm(w,p,compiled,a) for a in ARMS]
    direct=result['arms'][0]['epochs']
    for a in result['arms']:
        for x,y in zip(a['epochs'],direct):assert (x['partition'],x['signed'])==(y['partition'],y['signed'])
    return result

def main():
    sizes=[int(sys.argv[1])] if len(sys.argv)>1 else [12,48,96]
    worlds=[world(n,f,i) for n in sizes for f in FAMILIES for i in range(3)];results=[]
    for w in worlds:
        for p in (.15,.35):results.append(evaluate(w,p))
        print(w['n'],w['family'],w['seed'],results[-1]['status'],flush=True)
    (H/('receipt'+str(sizes[0])+'.json' if len(sizes)==1 else 'receipt.json')).write_text(json.dumps(dict(evidence='development only',source_sha256=sha(__file__),base_sha256=sha(H/'base19.py'),protocol_sha256=sha(H/'PROTOCOL.md'),worlds=worlds,results=results),separators=(',',':'))+'\n')
if __name__=='__main__':main()
