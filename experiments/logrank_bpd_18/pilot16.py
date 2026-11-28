"""Integer factor bounds, quotient components, and bounded mini-bucket elimination."""
import itertools as it, json, math, random, time, resource, hashlib, sys
from fractions import Fraction as F
from collections import Counter, defaultdict
from pathlib import Path
H=Path(__file__).resolve().parent
ARMS=('independent','component7','mb3','mb5','mb7')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def par(a):return a.bit_count()&1
def bits(a):return tuple(i for i in range(a.bit_length()) if a>>i&1)
def mask(s):return sum(1<<i for i in s)
def basis(rows):
    piv={};keep=[]
    for a in rows:
        v=a;c=1<<len(keep)
        while v:
            j=v.bit_length()-1
            if j not in piv:piv[j]=(v,c);keep.append(a);break
            x,y=piv[j];v^=x;c^=y
    return keep,piv
def coord(a,piv):
    c=0
    while a:
        x,y=piv[a.bit_length()-1];a^=x;c^=y
    return c
def grouped(rows,obs,p):
    u,v=F(str(p)).as_integer_ratio();v-=u
    g=defaultdict(lambda:[0,0])
    for a,y in zip(rows,obs):g[a][y]+=1
    return {a:(v**n0*u**n1,u**n0*v**n1) for a,(n0,n1) in g.items()}
def condition(rows,obs,n,q,b):
    # Fix q dot h=b by substituting its lowest variable; preserve all likelihood weights.
    if q==0:return rows,obs,n
    z=q&-q
    return [a^q if a&z else a for a in rows],[y^b if a&z else y for a,y in zip(rows,obs)],n^z

def independent(g,active):
    fixed=g.get(0,(1,1))[0];items=[(a,x,y) for a,(x,y) in g.items() if a]
    items.sort(key=lambda t:(F(min(t[1:]),max(t[1:])),t[0]))
    selected=set(basis(a for a,_,_ in items)[0]);lo=hi=fixed
    for a,x,y in items:
        if a in selected:lo*=x+y;hi*=x+y
        else:lo*=min(x,y);hi*=max(x,y)
    k=active.bit_count()-len(selected);return lo<<k,hi<<k

def components(g,active):
    groups=[]
    for a in sorted(g):
        if not a:continue
        hits=[i for i,v in enumerate(groups) if v&a]
        for i in reversed(hits):a|=groups.pop(i)
        groups.append(a)
    used=0
    for a in groups:used|=a
    groups.extend(1<<i for i in bits(active^used))
    return groups

def component_bound(g,active,cap,stats):
    lo=hi=g.get(0,(1,1))[0]
    for a in components(g,active):
        f={k:v for k,v in g.items() if k and k&a};bb,piv=basis(f);r=len(bb)
        if r<=cap:
            pairs=[(coord(q,piv),v) for q,v in f.items()];z=sum(math.prod(v[par(q&h)] for q,v in pairs) for h in range(1<<r))<<(a.bit_count()-r);l=u=z
            stats['integrated_states']+=1<<r;stats['max_scope']=max(stats['max_scope'],r)
        else:l,u=independent(f,a);stats['fallback_components']+=1
        lo*=l;hi*=u
    return lo,hi

# Factor = (sorted variable tuple, lower table, upper table). Bit i indexes scope[i].
def init_factors(g,active):
    factors=[]
    for a,(x,y) in sorted(g.items()):
        s=bits(a);t=tuple((x,y)[i.bit_count()&1] for i in range(1<<len(s)));factors.append((s,t,t))
    # Explicit unit factors ensure free coordinates are counted.
    used=0
    for a in g:used|=a
    for i in bits(active&~used):factors.append(((i,),(1,1),(1,1)))
    return factors

def join_elim(group,var,anchor,stats):
    scope=tuple(sorted(set().union(*(set(f[0]) for f in group))));out=tuple(i for i in scope if i!=var);pos={j:i for i,j in enumerate(scope)};indices=[tuple(pos[j] for j in f[0]) for f in group];k=pos[var];sz=1<<len(out);L=[];U=[]
    for i in range(sz):
        a=(i&((1<<k)-1))|((i>>k)<<(k+1));vals=[]
        for h in (a,a|(1<<k)):
            ix=[sum((h>>j&1)<<b for b,j in enumerate(p)) for p in indices];vals.append((math.prod(f[1][j] for f,j in zip(group,ix)),math.prod(f[2][j] for f,j in zip(group,ix))))
        L.append(sum(t[0] for t in vals) if anchor else min(t[0] for t in vals));U.append(sum(t[1] for t in vals) if anchor else max(t[1] for t in vals))
    stats['entries']+=2*sz*len(group);stats['max_scope']=max(stats['max_scope'],len(scope));return (out,tuple(L),tuple(U))

def minibucket(g,active,cap,stats):
    factors=init_factors(g,active);remaining=set(bits(active));trace=[]
    while remaining:
        def score(v):return (len(set().union(*(set(f[0]) for f in factors if v in f[0]))),v)
        v=min(remaining,key=score);remaining.remove(v);bucket=[f for f in factors if v in f[0]];factors=[f for f in factors if v not in f[0]];assert bucket
        groups=[];scopes=[]
        for f in sorted(bucket,key=lambda f:(-len(f[0]),f[0])):
            s=set(f[0]);assert len(s)<=cap
            ids=[i for i,t in enumerate(scopes) if len(t|s)<=cap]
            if ids:
                i=min(ids,key=lambda i:(len(scopes[i]|s),i));groups[i].append(f);scopes[i]|=s
            else:groups.append([f]);scopes.append(s)
        stats['splits']+=len(groups)-1
        # Sum exactly one group; bound all other groups pointwise by min/max.
        anchor=max(range(len(groups)),key=lambda i:(len(scopes[i]),len(groups[i]),-i));trace.append((v,[len(t) for t in scopes],anchor))
        for i,group in enumerate(groups):factors.append(join_elim(group,v,i==anchor,stats))
        stats['peak_table_slots']=max(stats['peak_table_slots'],sum(len(f[1])+len(f[2]) for f in factors)+sum(len(f[1])+len(f[2]) for f in bucket))
    assert all(not f[0] for f in factors);return math.prod(f[1][0] for f in factors),math.prod(f[2][0] for f in factors),trace

def bound(rows,obs,active,p,arm):
    g=grouped(rows,obs,p);base=independent(g,active);stats=dict(integrated_states=0,fallback_components=0,entries=0,max_scope=0,splits=0,peak_table_slots=0);trace=[]
    if arm=='independent':raw=base
    elif arm=='component7':raw=component_bound(g,active,7,stats)
    else:lo,hi,trace=minibucket(g,active,int(arm[2:]),stats);raw=lo,hi
    lo,hi=max(raw[0],base[0]),min(raw[1],base[1]);assert lo<=hi
    return dict(lo=str(lo),hi=str(hi),raw=list(map(str,raw)),baseline=list(map(str,base)),stats=stats,trace=trace)

def world(n,family,i):
    seed=160000+1000*n+100*('block','chain','dense').index(family)+i;g=random.Random(seed);rows=[1<<j for j in range(n)]*2
    for _ in range(3*n):
        if family=='block':j=g.randrange(n//4)*4;s=g.sample(range(j,j+4),3)
        elif family=='chain':j=g.randrange(n-2);s=range(j,j+3)
        else:s=g.sample(range(n),3)
        rows.append(mask(s))
    h=g.getrandbits(n);uniforms=[g.random() for _ in rows];q=[]
    while len(q)<8:
        if len(q)<4:
            j=g.randrange(n-1);a=mask((j,j+1))
        else:a=mask(g.sample(range(n),2))
        if a not in q:q.append(a)
    return dict(n=n,family=family,seed=seed,rows=rows,hidden=h,uniforms=uniforms,queries=q)

def evaluate(w,p,arm):
    rows=w['rows'];obs=[par(a&w['hidden'])^int(u<p) for a,u in zip(rows,w['uniforms'])];active=(1<<w['n'])-1;start=time.perf_counter();root=bound(rows,obs,active,p,arm);result=[]
    for q in w['queries']:
        b=[bound(*condition(rows,obs,active,q,i),p,arm) for i in (0,1)];l0,l1=[int(x['lo']) for x in b];u0,u1=[int(x['hi']) for x in b];lo=F(l1,l1+u0);hi=F(u1,u1+l0);assert lo<=hi;cert=1 if lo>=F(9,10) else 0 if hi<=F(1,10) else None
        result.append(dict(q=q,lo=str(lo),hi=str(hi),cert=cert,truth=par(q&w['hidden']),weights=b))
    return dict(seed=w['seed'],noise=p,arm=arm,obs=obs,root=root,queries=result,seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def fixture():
    g=random.Random(16016);checks=0
    for n in range(3,9):
        for _ in range(6):
            rows=[mask(g.sample(range(n),g.randint(1,min(3,n)))) for j in range(3*n)];obs=[g.randrange(2) for a in rows]
            for p in (.15,.35):
                u,v=F(str(p)).as_integer_ratio();v-=u
                q=mask(g.sample(range(n),2));active=(1<<n)-1
                for b in (None,0,1):
                    rr,oo,aa=(rows,obs,active) if b is None else condition(rows,obs,active,q,b);z=sum(u**e*v**(len(rows)-e) for h in range(1<<n) if b is None or par(q&h)==b for e in [sum(par(a&h)!=y for a,y in zip(rows,obs))]);
                    for arm in ARMS:
                        r=bound(rr,oo,aa,p,arm);assert int(r['lo'])<=z<=int(r['hi']) and int(r['raw'][0])<=z<=int(r['raw'][1]);checks+=1
    return checks

def main():
    import multiprocessing as mp
    if sys.argv[1]=='fixture':print(fixture());return
    worlds=[world(n,f,i) for n in (12,24,48) for f in ('block','chain','dense') for i in range(4)];tasks=[(w,p,a) for w in worlds for p in (.15,.35) for a in ARMS]
    # One fresh forkserver worker per task; peak RSS includes interpreter + solver, not growing parent aggregation.
    with mp.get_context('forkserver').Pool(4,maxtasksperchild=1) as pool:
        results=[]
        for r in pool.imap(_task,tasks,chunksize=1):
            results.append(r)
            if len(results)%10==0:print(len(results),r['seed'],r['arm'],flush=True)
    (H/'receipt.json').write_text(json.dumps(dict(protocol_sha256=sha(H/'PROTOCOL.md'),source_sha256=sha(__file__),fixture_checks=fixture(),worlds=worlds,results=results),separators=(',',':'))+'\n')
def _task(t):return evaluate(*t)
if __name__=='__main__':main()
