"""Cost-gated disjoint cutset frontier, integer likelihood certificates."""
import json, math, random, time, resource, sys, struct
from fractions import Fraction as F
from pathlib import Path
from dataclasses import dataclass, field
from pilot16 import bits, mask, par, sha, condition, bound
H=Path(__file__).resolve().parent
ARMS=('scope3','scope5','degree2','random2','adaptive')
FAMILIES=('chain','hub','dense')

def world(n,family,i):
    seed=180000+1000*n+100*FAMILIES.index(family)+i;g=random.Random(seed);order=list(range(n));g.shuffle(order)
    k=i+1 if family=='hub' else 0;hubs=order[:k];leaf=order[k:];rows=[1<<j for j in range(n)]*2
    if family=='hub':
        # Variable number of planted separators, ensure every leaf edge occurs.
        rows += [mask((hubs[j%k],a,b)) for j,(a,b) in enumerate(zip(leaf,leaf[1:]))]
        while len(rows)<5*n:
            j=g.randrange(len(leaf)-1);rows.append(mask((g.choice(hubs),leaf[j],leaf[j+1])))
    elif family=='chain':
        rows += [mask(order[j:j+3]) for j in range(n-2)]
        while len(rows)<5*n:
            j=g.randrange(n-2);rows.append(mask(order[j:j+3]))
    else:rows += [mask(g.sample(range(n),3)) for _ in range(3*n)]
    h=g.getrandbits(n);u=[g.random() for _ in rows];qs=[]
    while len(qs)<4:
        q=mask(g.sample(range(n),2))
        if q not in qs:qs.append(q)
    return dict(seed=seed,n=n,family=family,rows=rows,hidden=h,uniforms=u,queries=qs,offline_order=order,hub_count=k)

def slots(f):return sum(2<<a.bit_count() for a in f)

def plan(rows,active,cap):
    """Predict exact counted join work and conservative live factor-table slots from scopes alone."""
    f=sorted(set(rows));used=0
    for a in f:used|=a
    f += [1<<i for i in bits(active&~used)];rem=set(bits(active));work=splits=0;peak=slots(f);max_scope=0
    while rem:
        def scope(v):
            z=0
            for a in f:
                if a>>v&1:z|=a
            return z.bit_count(),v
        v=min(rem,key=scope);rem.remove(v);bit=1<<v;bucket=[a for a in f if a&bit];f=[a for a in f if not a&bit];bins=[];sc=[]
        for a in sorted(bucket,key=lambda a:(-a.bit_count(),bits(a))):
            if a.bit_count()>cap:return None
            ids=[j for j,b in enumerate(sc) if (a|b).bit_count()<=cap]
            if ids:j=min(ids,key=lambda j:((a|sc[j]).bit_count(),j));sc[j]|=a;bins[j]+=1
            else:sc.append(a);bins.append(1)
        splits+=len(sc)-1
        for a,k in zip(sc,bins):
            work+=(1<<a.bit_count())*k;max_scope=max(max_scope,a.bit_count());f.append(a^bit)
        peak=max(peak,slots(f)+slots(bucket))
    return work,peak,splits,max_scope

def cut_degree(rows,active,k):
    cut=0
    for _ in range(min(k,active.bit_count())):
        g={i:0 for i in bits(active)}
        for a in set(rows):
            a&=active
            for i in bits(a):g[i]|=a^(1<<i)
        j=min(g,key=lambda j:(-g[j].bit_count(),j));cut|=1<<j;active^=1<<j
    return cut

def assignments(cut):
    a=cut
    while True:
        yield a
        if not a:break
        a=(a-1)&cut

def intersect(a,b):
    z=tuple((max(l,L),min(u,U)) for (l,u),(L,U) in zip(a,b));assert all(l<=u for l,u in z);return z

def total(nodes):return tuple((sum(x.interval()[b][0] for x in nodes),sum(x.interval()[b][1] for x in nodes)) for b in (0,1))

@dataclass
class Node:
    fixed:int
    value:int
    cap:int
    own:tuple
    pressure:int
    children:list=field(default_factory=list)
    def interval(self):return intersect(self.own,total(self.children)) if self.children else self.own
    def leaves(self):
        if not self.children:yield self
        else:
            for child in self.children:yield from child.leaves()

def prob(pair):
    (l0,u0),(l1,u1)=pair;return F(l1,l1+u0),F(u1,u1+l0)

def evaluate(w,p,arm):
    start=time.perf_counter();n=w['n'];rows=w['rows'];obs=[par(a&w['hidden'])^int(u<p) for a,u in zip(rows,w['uniforms'])];qs=w['queries'];full=(1<<n)-1
    caps=dict(entries=900*n,slots=45*n,calls=128);cost=dict(entries=0,slots=0,calls=0);planning=0.;records=[];actions=[];roots=[];ctx=[];plans={}
    # Conservative payload bound: all individual nonnegative tables <= 2^n * max(u,v)^m.
    u,v=F(str(p)).as_integer_ratio();v-=u;entry_bytes=sys.getsizeof((max(u,v)**len(rows))<<n)+struct.calcsize('P')
    for q in qs:
        rr,oo,active=condition(rows,obs,full,q,0);_,oo1,_=condition(rows,obs,full,q,1);ctx.append((rr,(oo,oo1),active))
    def getplan(j,fixed,cap):
        nonlocal planning
        key=(j,fixed,cap)
        if key not in plans:
            t=time.perf_counter();rr,_,active=ctx[j];active^=fixed;plans[key]=plan([a&active for a in rr],active,cap);planning+=time.perf_counter()-t
        return plans[key]
    def kernel(j,fixed,value,cap):
        rr,oo,active=ctx[j];active^=fixed;rr2=[a&active for a in rr];result=[]
        met=getplan(j,fixed,cap) if cap else (0,0,0,0);assert met is not None
        for b in (0,1):
            oo2=[y^par(a&value) for a,y in zip(rr,oo[b])];z=bound(rr2,oo2,active,p,'mb'+str(cap) if cap else 'independent');cost['calls']+=1
            if cap:
                assert z['stats']['entries']==met[0] and z['stats']['peak_table_slots']<=met[1] and z['stats']['splits']==met[2]
                cost['entries']+=met[0];cost['slots']=max(cost['slots'],met[1])
            assert all(cost[k]<=caps[k] for k in caps)
            result.append((int(z['lo']),int(z['hi'])));records.append(dict(query=j,side=b,fixed=fixed,value=value,cap=cap,plan=list(met),result=z))
        return tuple(result)
    for j in range(len(qs)):
        rr,_,_=ctx[j];roots.append(Node(0,0,0,kernel(j,0,0,0),len(set(rr))))
    def feasible(j,node,cut,cap):
        if not cut and cap<=node.cap:return None
        met=getplan(j,node.fixed|cut,cap)
        if met is None:return None
        count=1<<cut.bit_count();work=2*count*met[0];calls=2*count
        if cost['entries']+work>caps['entries'] or cost['calls']+calls>caps['calls'] or met[1]>caps['slots']:return None
        return work,calls,met[2],met[1]
    def snapshot():return [list(map(str,prob(r.interval()))) for r in roots]
    def apply(j,node,cut,cap,est):
        old=snapshot();before=cost.copy();before_pair=node.interval()
        if cut:
            node.children=[Node(node.fixed|cut,node.value|x,cap,kernel(j,node.fixed|cut,node.value|x,cap),est[2]) for x in assignments(cut)]
        else:
            node.own=intersect(node.own,kernel(j,node.fixed,node.value,cap));node.cap=cap;node.pressure=est[2]
        assert cost['entries']-before['entries']==est[0] and cost['calls']-before['calls']==est[1]
        new=snapshot()
        # Intersection at every node gives monotone event intervals, also when a split's raw sum is worse.
        assert all(F(a[0])<=F(b[0])<=F(b[1])<=F(a[1]) for a,b in zip(old,new))
        actions.append(dict(query=j,fixed=node.fixed,value=node.value,cut=cut,cap=cap,estimate=est,before=old,after=new,cost=cost.copy()))
    initial=snapshot()
    if arm!='adaptive':
        for j,node in enumerate(roots):
            rr,_,active=ctx[j];cap=5 if arm=='scope5' else 3;cut=0
            if arm=='degree2':cut=cut_degree(rr,active,2)
            if arm=='random2':cut=mask(random.Random(w['seed']+18018+j).sample(bits(active),2))
            est=feasible(j,node,cut,cap)
            if est is not None and prob(node.interval())[0]!=prob(node.interval())[1]:apply(j,node,cut,cap,est)
    else:
        while True:
            choices=[]
            for j,root in enumerate(roots):
                root_pair=root.interval();l,u0=prob(root_pair);width=float(u0-l)
                if not width:continue
                # Relative contribution of this cell to the maximum possible partition weight.
                root_upper=sum(u for l,u in total(list(root.leaves())))
                rr,_,active=ctx[j]
                for node in root.leaves():
                    if all(l==u for l,u in node.interval()):continue
                    influence=float(F(sum(u-l for l,u in node.interval()),root_upper))
                    kmax=min(2,4-node.fixed.bit_count());cuts=[(0,c) for c in (3,5,7) if c>node.cap]
                    cuts += [(cut_degree(rr,active^node.fixed,k),3) for k in range(1,kmax+1)]
                    for cut,cap in cuts:
                        est=feasible(j,node,cut,cap)
                        if est is None:continue
                        work,calls,splits,peak=est
                        # Exact residual plans first. Otherwise prefer fewer split buckets per unit work.
                        reduction=max(0,node.pressure-splits)/max(1,node.pressure)
                        key=(int(splits==0),width*influence*(1 if splits==0 else .05+reduction)/max(1,work),-work,-j,-node.fixed,-node.value,-cut,-cap)
                        choices.append((key,j,node,cut,cap,est))
            if not choices:break
            _,j,node,cut,cap,est=max(choices,key=lambda x:x[0]);apply(j,node,cut,cap,est)
    queries=[]
    for j,r in enumerate(roots):
        pair=r.interval();l,u0=prob(pair);cert=1 if l>=F(9,10) else 0 if u0<=F(1,10) else None
        queries.append(dict(q=qs[j],lo=str(l),hi=str(u0),weights=[list(map(str,a)) for a in pair],cert=cert,truth=par(qs[j]&w['hidden']),frontier=[(a.fixed,a.value,a.cap) for a in r.leaves()]))
    return dict(seed=w['seed'],noise=p,arm=arm,obs=obs,caps=caps,cost=cost,entry_bytes_upper=entry_bytes,counted_payload_upper_bytes=cost['slots']*entry_bytes,planning_seconds=planning,seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,initial=initial,actions=actions,queries=queries,records=records)

def task(t):return evaluate(*t)
def main():
    import multiprocessing as mp
    worlds=[world(n,f,i) for n in (12,24,48) for f in FAMILIES for i in range(3)];tasks=[(w,p,a) for w in worlds for p in (.15,.35) for a in ARMS];out=[]
    with mp.get_context('forkserver').Pool(4,maxtasksperchild=1) as pool:
        for r in pool.imap(task,tasks,chunksize=1):
            out.append(r)
            if len(out)%10==0:print(len(out),r['seed'],r['arm'],r['cost'],flush=True)
    (H/'receipt.json').write_text(json.dumps(dict(source_sha256=sha(__file__),kernel_sha256=sha(H/'pilot16.py'),protocol_sha256=sha(H/'PROTOCOL.md'),worlds=worlds,results=out),separators=(',',':'))+'\n')
if __name__=='__main__':main()
