"""Exact reusable separator messages for parity queries (development only)."""
import hashlib, itertools, json, math, random, resource, struct, sys, time
from fractions import Fraction
from pathlib import Path
H=Path(__file__).resolve().parent
SIZES=(12,48,96); FAMILIES=('chain','hub','dense'); CAP=6; PREFIXES=(4,16,64); CACHE_KIB=(16,64,256,1024)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bits(a): return [i for i in range(a.bit_length()) if a>>i&1]
def mask(s): return sum(1<<i for i in s)
def subs(a):
    x=a
    while True:
        yield x
        if not x: break
        x=(x-1)&a

def world(n,family,i):
    seed=190000+1000*n+100*FAMILIES.index(family)+i; g=random.Random(seed)
    order=list(range(n)); g.shuffle(order); rows=[1<<j for j in range(n)]*2
    if family=='chain':
        triples=[mask(order[j:j+3]) for j in range(n-2)]
    elif family=='hub':
        hub=order[:2]; leaf=order[2:]; triples=[mask((hub[k%2],a,b)) for k,(a,b) in enumerate(zip(leaf,leaf[1:]))]
    else: triples=[]
    rows+=triples
    while len(rows)<5*n:
        rows.append(g.choice(triples) if triples else mask(g.sample(range(n),3)))
    h=g.getrandbits(n); uniforms=[g.random() for _ in rows]
    pairs=list(itertools.combinations(range(n),2)); g.shuffle(pairs); queries=[]
    sources=g.sample(range(n),min(4,n))
    for s in sources:
        targets=[(min(s,t),max(s,t)) for t in range(n) if t!=s];g.shuffle(targets)
        queries.extend(q for q in targets[:8] if q not in queries)
    queries.extend(q for q in pairs if q not in queries); queries=queries[:64]
    assert len(queries)==64
    return dict(n=n,family=family,seed=seed,rows=rows,hidden=h,uniforms=uniforms,queries=queries,offline_order=order)

def plan(n,rows):
    adj=[set() for _ in range(n)]
    for a in rows:
        vs=bits(a)
        for j in vs:adj[j].update(set(vs)-{j})
    remaining=set(range(n)); order=[]; bags={}
    while remaining:
        j=min(remaining,key=lambda j:(len(adj[j]),j)); nb=adj[j].copy();bags[j]=mask(nb|{j});order.append(j);remaining.remove(j)
        for t in nb:adj[t].discard(j);adj[t].update(nb-{t})
    pos={j:i for i,j in enumerate(order)};parent={};local={j:[] for j in range(n+1)};bags[n]=0
    for j in order:
        nb=bits(bags[j]^(1<<j));parent[j]=min(nb,key=pos.get) if nb else n
    for k,a in enumerate(rows):local[min(bits(a),key=pos.get)].append(k)
    nbr={j:[] for j in bags}
    for j,t in parent.items():nbr[j].append(t);nbr[t].append(j)
    return order,bags,parent,local,nbr

class Tree:
    def __init__(self,w,p,compiled):
        self.n=w['n'];self.order,self.bags,self.parent,self.local,self.nbr=compiled;self.rows=w['rows'];self.obs=[((a&w['hidden']).bit_count()&1)^int(u<p) for a,u in zip(self.rows,w['uniforms'])];f=Fraction(str(p));self.u=f.numerator;self.v=f.denominator-self.u
        self.states={j:list(subs(a)) for j,a in self.bags.items()};self.base={j:{x:math.prod(self.u if ((x&self.rows[i]).bit_count()&1)!=self.obs[i] else self.v for i in self.local[j]) for x in self.states[j]} for j in self.bags}
        self.cache={};self.work=0;self.adds=0;self.calls=0
    def reset(self):self.work=self.adds=self.calls=0
    def counts(self):return [self.work,self.adds,self.calls]
    def separator(self,j,k):return self.bags[j]&self.bags[k]
    def message(self,j,k,incoming,sign=0):
        sep=self.separator(j,k);out={x:0 for x in subs(sep)};self.calls+=1
        for x in self.states[j]:
            val=self.base[j][x]; val=-val if (x&sign).bit_count()&1 else val
            for t,tab in incoming:val*=tab[x&self.separator(j,t)];self.work+=1
            out[x&sep]+=val;self.adds+=1
        return out
    def finish(self,j,incoming,sign=0):
        z=0;self.calls+=1
        for x in self.states[j]:
            val=self.base[j][x];val=-val if (x&sign).bit_count()&1 else val
            for t,tab in incoming:val*=tab[x&self.separator(j,t)];self.work+=1
            z+=val;self.adds+=1
        return z
    def direct(self,sign=0):
        msgs={}
        for j in self.order:
            t=self.parent[j];msgs[j,t]=self.message(j,t,[(a,msgs.pop((a,j))) for a in self.nbr[j] if a!=t],sign&(1<<j))
        return self.finish(self.n,[(a,msgs.pop((a,self.n))) for a in self.nbr[self.n]])
    def calibrate(self):
        for j in self.order:
            t=self.parent[j];self.cache[j,t]=self.message(j,t,[(a,self.cache[a,j]) for a in self.nbr[j] if a!=t])
        for j in [self.n]+list(reversed(self.order)):
            for t in self.nbr[j]:
                if self.parent.get(t)==j:self.cache[j,t]=self.message(j,t,[(a,self.cache[a,j]) for a in self.nbr[j] if a!=t])
        return self.finish(self.n,[(a,self.cache[a,self.n]) for a in self.nbr[self.n]])
    def cache_preflight(self):
        # Upper bound on retained dicts and their entries, counting repeated objects pessimistically.
        # Local factors, temporary contractions, planning, and receipts are outside this cache cap.
        big=sys.getsizeof((max(self.u,self.v)**len(self.rows))<<self.n); total=sys.getsizeof({(j,k):None for j in self.nbr for k in self.nbr[j]})
        for j in self.nbr:
            for k in self.nbr[j]:
                d={x:0 for x in subs(self.separator(j,k))};total+=sys.getsizeof((j,k))+2*sys.getsizeof(j)+sys.getsizeof(d)+sum(sys.getsizeof(x)+big for x in d)
        return total
    def cache_bytes(self):
        return sys.getsizeof(self.cache)+sum(sys.getsizeof(key)+sum(sys.getsizeof(x) for x in key)+sys.getsizeof(d)+sum(sys.getsizeof(x)+sys.getsizeof(v) for x,v in d.items()) for key,d in self.cache.items())
    def path(self,s,t):
        seen={s:None};queue=[s]
        for j in queue:
            if j==t:break
            for k in self.nbr[j]:
                if k not in seen:seen[k]=j;queue.append(k)
        a=[t]
        while a[-1]!=s:a.append(seen[a[-1]])
        return a[::-1]
    def pair(self,s,t):
        path=self.path(s,t);prev=None;msg=None
        for i,j in enumerate(path):
            nxt=path[i+1] if i+1<len(path) else None
            incoming=[(a,msg if a==prev else self.cache[a,j]) for a in self.nbr[j] if a!=nxt]
            if nxt is None:return self.finish(j,incoming,1<<t)
            msg=self.message(j,nxt,incoming,1<<s if i==0 else 0);prev=j
    def batch(self,queries):
        # Greedy public source orientation: group the largest number of unassigned pairs at a vertex.
        remain=set(map(tuple,queries));values={}
        while remain:
            s=min(set(itertools.chain.from_iterable(remain)),key=lambda j:(-sum(j in q for q in remain),j));chosen=[q for q in sorted(remain) if s in q];targets={q[0] if q[1]==s else q[1]:q for q in chosen};edges={}
            for t in targets:
                path=self.path(s,t)
                for a,b in zip(path,path[1:]):edges.setdefault(a,set()).add(b)
            def visit(j,prev,msg):
                if j in targets:
                    inc=[(a,msg if a==prev else self.cache[a,j]) for a in self.nbr[j]];values[targets[j]]=self.finish(j,inc,1<<j)
                for k in sorted(edges.get(j,())):
                    inc=[(a,msg if a==prev else self.cache[a,j]) for a in self.nbr[j] if a!=k];visit(k,j,self.message(j,k,inc,1<<s if j==s else 0))
            visit(s,None,None);remain.difference_update(chosen)
        return [values[tuple(q)] for q in queries]

def evaluate(w,p):
    t=time.perf_counter();compiled=plan(w['n'],w['rows']);peak=max(a.bit_count() for a in compiled[1].values());plan_sec=time.perf_counter()-t
    r=dict(seed=w['seed'],noise=p,max_scope=peak,plan_seconds=plan_sec,status='refused' if peak>CAP else 'solved')
    if peak>CAP:return r
    t=time.perf_counter();tree=Tree(w,p,compiled);setup=time.perf_counter()-t; t=time.perf_counter();pre=tree.cache_preflight();preflight_seconds=time.perf_counter()-t;assert pre<=1024*1024;qs=w['queries'];t=time.perf_counter();base=tree.direct();root_seconds=time.perf_counter()-t;root_counts=tree.counts();assert base>0;tree.reset()
    t=time.perf_counter();direct=[];prefix=[]
    for i,(a,b) in enumerate(qs,1):
        direct.append(tree.direct((1<<a)|(1<<b)))
        if i in PREFIXES:prefix.append(dict(q=i,seconds=time.perf_counter()-t,counts=tree.counts()))
    tree.reset();t=time.perf_counter();z=tree.calibrate();cal_time=time.perf_counter()-t;cal_counts=tree.counts();actual=tree.cache_bytes();assert z==base and actual<=pre
    tree.reset();t=time.perf_counter();path=[];path_prefix=[]
    for i,(a,b) in enumerate(qs,1):
        path.append(tree.pair(a,b))
        if i in PREFIXES:path_prefix.append(dict(q=i,seconds=time.perf_counter()-t,counts=tree.counts()))
    batch=[]
    for k in PREFIXES:
        tree.reset();t=time.perf_counter();v=tree.batch(qs[:k]);sec=time.perf_counter()-t;assert v==direct[:k];batch.append(dict(q=k,seconds=sec,counts=tree.counts()))
    assert path==direct
    r.update(setup_seconds=setup,root_seconds=root_seconds,root_counts=root_counts,preflight_seconds=preflight_seconds,calibration_seconds=cal_time,calibration_counts=cal_counts,cache_preflight_bytes=pre,cache_estimated_bytes=actual,cache_admission={k:pre<=1024*k for k in CACHE_KIB},direct=prefix,path=path_prefix,batch=batch,partition=str(z),signed=[str(x) for x in direct],probabilities=[str(Fraction(z-x,2*z)) for x in direct],peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return r

def main():
    worlds=[world(n,f,i) for n in SIZES for f in FAMILIES for i in range(3)];results=[]
    for w in worlds:
        for p in (.15,.35):results.append(evaluate(w,p))
        print(w['seed'],results[-1]['status'],results[-1]['max_scope'],flush=True)
    (H/'receipt.json').write_text(json.dumps(dict(evidence='development',source_sha256=sha(__file__),protocol_sha256=sha(H/'PROTOCOL.md'),worlds=worlds,results=results),separators=(',',':'))+'\n')
if __name__=='__main__':main()
