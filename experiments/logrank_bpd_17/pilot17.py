"""Cutset-conditioned integer posterior intervals; development-only."""
import json, random, time, resource, math
from fractions import Fraction as F
from pathlib import Path
from pilot16 import bits, mask, par, sha, condition, bound
H=Path(__file__).resolve().parent
ARMS=('none3','none5','prefix2','random2','degree2','scope2')

def world(n,family,i):
    seed=170000+1000*n+100*('hub','dense').index(family)+i
    g=random.Random(seed); order=list(range(n));g.shuffle(order);hubs=order[:2];leaf=order[2:]
    rows=[1<<j for j in range(n)]*2
    if family=='hub':
        rows += [mask((hubs[j%2],a,b)) for j,(a,b) in enumerate(zip(leaf,leaf[1:]))]
        while len(rows)<5*n:
            j=g.randrange(len(leaf)-1);rows.append(mask((g.choice(hubs),leaf[j],leaf[j+1])))
    else:rows += [mask(g.sample(range(n),3)) for _ in range(3*n)]
    h=g.getrandbits(n);u=[g.random() for _ in rows];qs=[]
    while len(qs)<4:
        q=mask(g.sample(range(n),2))
        if q not in qs:qs.append(q)
    return dict(seed=seed,n=n,family=family,rows=rows,hidden=h,uniforms=u,queries=qs,offline_order=order)

def graph(rows,active):
    adj={i:0 for i in bits(active)}
    for a in set(rows):
        for i in bits(a):adj[i]|=a^(1<<i)
    return adj

def simulate(rows,active):
    adj=graph(rows,active);peak=excess=0
    while adj:
        i=min(adj,key=lambda v:(adj[v].bit_count(),v));nb=adj.pop(i);s=1+nb.bit_count();peak=max(peak,s);excess+=max(0,s-3)
        for j in bits(nb):adj[j]=(adj[j]|nb)&~((1<<i)|(1<<j))
    return peak,excess

def structural_score(rows,active,queries):
    res=[simulate(rows,active)]
    for q in queries:
        if q:
            z=q&-q
            res.append(simulate([a^q if a&z else a for a in rows],active^z))
        else:res.append(res[0])
    return max(a for a,b in res),sum(b for a,b in res)

def choose(w,arm):
    n=w['n'];rows=w['rows'];active=(1<<n)-1;qs=w['queries']
    if arm.startswith('none'):return 0
    if arm=='prefix2':return 3
    if arm=='random2':return mask(random.Random(w['seed']+17017).sample(range(n),2))
    cut=0
    for _ in range(2):
        adj=graph([a&active for a in rows],active)
        if arm=='degree2':v=min(adj,key=lambda j:(-adj[j].bit_count(),j))
        else:
            def score(j):
                a=active^(1<<j);return (*structural_score([r&a for r in rows],a,[q&a for q in qs]),-adj[j].bit_count(),j)
            v=min(adj,key=score)
        cut|=1<<v;active^=1<<v
    return cut

def assignments(cut):
    x=cut
    while True:
        yield x
        if not x:break
        x=(x-1)&cut

def evaluate(w,p,arm):
    start=time.perf_counter();n=w['n'];rows=w['rows'];obs=[par(a&w['hidden'])^int(u<p) for a,u in zip(rows,w['uniforms'])];cut=choose(w,arm);active=((1<<n)-1)^cut;cap=5 if arm=='none5' else 3
    select_seconds=time.perf_counter()-start;cells=[];calls=0;entries=0;slots=0
    totals=[[0,0] for _ in range(2*len(w['queries']))];root=[0,0]
    def kernel(rr,oo,aa):
        nonlocal calls,entries,slots
        b=bound(rr,oo,aa,p,'mb'+str(cap));calls+=1;entries+=b['stats']['entries'];slots=max(slots,b['stats']['peak_table_slots']);return b
    def add(dest,b):dest[0]+=int(b['lo']);dest[1]+=int(b['hi'])
    for x in assignments(cut):
        rr=[a&active for a in rows];oo=[y^par(a&x) for a,y in zip(rows,obs)];z=kernel(rr,oo,active);add(root,z);events=[]
        for j,q in enumerate(w['queries']):
            f=par(q&x);r=q&active
            if not r:
                add(totals[2*j+f],z);events.append(None);continue
            t=[]
            for b in (0,1):
                s=kernel(*condition(rr,oo,active,r,b^f));add(totals[2*j+b],s);t.append(s)
            events.append(t)
        cells.append(dict(assignment=x,root=z,events=events))
    queries=[]
    for j,q in enumerate(w['queries']):
        (l0,u0),(l1,u1)=totals[2*j:2*j+2]
        lo=F(l1,l1+u0);hi=F(u1,u1+l0);assert 0<=lo<=hi<=1
        cert=1 if lo>=F(9,10) else 0 if hi<=F(1,10) else None
        queries.append(dict(q=q,lo=str(lo),hi=str(hi),cert=cert,truth=par(q&w['hidden']),weights=[[str(l0),str(u0)],[str(l1),str(u1)]]))
    return dict(seed=w['seed'],noise=p,arm=arm,obs=obs,cut=cut,scope=structural_score([a&active for a in rows],active,[q&active for q in w['queries']]),root=list(map(str,root)),queries=queries,cells=cells,kernel_calls=calls,entries=entries,peak_table_slots=slots,select_seconds=select_seconds,seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def task(t):return evaluate(*t)

def main():
    import multiprocessing as mp
    worlds=[world(n,f,i) for n in (12,24,48) for f in ('hub','dense') for i in range(3)];tasks=[(w,p,a) for w in worlds for p in (.15,.35) for a in ARMS]
    with mp.get_context('forkserver').Pool(4,maxtasksperchild=1) as pool:
        out=[]
        for r in pool.imap(task,tasks,chunksize=1):
            out.append(r)
            if len(out)%12==0:print(len(out),r['seed'],r['arm'],flush=True)
    (H/'receipt.json').write_text(json.dumps(dict(source_sha256=sha(__file__),kernel_sha256=sha(H/'pilot16.py'),protocol_sha256=sha(H/'PROTOCOL.md'),worlds=worlds,results=out),separators=(',',':'))+'\n')
if __name__=='__main__':main()
