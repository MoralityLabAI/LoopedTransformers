"""Independent absolute-mask arithmetic, disjoint cuts and offline exact reference."""
import json, math, random, hashlib
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from audit16 import group, independent, mb, frontier_exact, par, sha
H=Path(__file__).resolve().parent

def project(mask,order):
    return sum(((mask>>v)&1)<<j for j,v in enumerate(order))

def exact_reference(w,p,obs):
    n=w['n'];rows=w['rows'];qs=w['queries']
    if w['family']=='dense' and n>12:return None
    if n==12:
        u,v=F(str(p)).as_integer_ratio();v-=u;m=len(rows);weight=[u**e*v**(m-e) for e in range(m+1)]
        ws=[weight[sum(par(a&h)!=b for a,b in zip(rows,obs))] for h in range(1<<n)]
        return sum(ws),[sum(v for h,v in enumerate(ws) if par(q&h)) for q in qs]
    order=w['offline_order'];g=group([project(a,order) for a in rows],obs,p);z=frontier_exact(g,n)[0]
    return z,[frontier_exact(g,n,project(q,order))[0] for q in qs]

def conditioned(g,pivot,q,b):
    gg={}
    for a,(x,y) in g.items():
        if a&pivot:
            a^=q
            if b:x,y=y,x
        u,v=gg.get(a,(1,1));gg[a]=(u*x,v*y)
    return gg

def run():
    r=json.loads((H/'receipt.json').read_text());assert r['source_sha256']==sha(H/'pilot17.py') and r['kernel_sha256']==sha(H/'pilot16.py') and r['protocol_sha256']==sha(H/'PROTOCOL.md');assert sha(H/'audit16.py')=='d544e0dfec048d26bf7b47ccd0d14a282a905ba74a286aa97e0a362d64d8fe02'
    ws={w['seed']:w for w in r['worlds']};assert len(ws)==18;counts=Counter();refs={};seen=set();summ=[]
    for a in r['results']:
        w=ws[a['seed']];n=w['n'];p=a['noise'];arm=a['arm'];key=(w['seed'],p,arm);assert key not in seen;seen.add(key);rows=w['rows'];obs=[par(q&w['hidden'])^int(u<p) for q,u in zip(rows,w['uniforms'])];assert obs==a['obs'];assert len(rows)==5*n and all(rows.count(1<<j)==2 for j in range(n));assert all(q.bit_count()==3 for q in rows[2*n:]);assert len(set(w['queries']))==4 and all(q.bit_count()==2 and q not in rows for q in w['queries'])
        cut=a['cut'];active=((1<<n)-1)^cut;cap=5 if arm=='none5' else 3;assert cut.bit_count()==(0 if arm.startswith('none') else 2) and cut>>n==0
        if arm=='prefix2':assert cut==3
        if arm=='random2':assert cut==sum(1<<i for i in random.Random(w['seed']+17017).sample(range(n),2))
        # Independently reconstruct assignment partition.
        expected={sum(b<<j for b,j in zip(bs,[j for j in range(n) if cut>>j&1])) for bs in __import__('itertools').product((0,1),repeat=cut.bit_count())}
        assert len(a['cells'])==len(expected) and {c['assignment'] for c in a['cells']}==expected and len(expected)*(1<<active.bit_count())==1<<n
        k=(w['seed'],p)
        if k not in refs:refs[k]=exact_reference(w,p,obs)
        ref=refs[k];totals=[[0,0] for _ in range(8)];root=[0,0];calls=entries=slots=0
        def check(saved,g,act):
            nonlocal calls,entries,slots
            bas=independent(g,act);raw=mb(g,act,cap,saved['trace']);lo=max(bas[0],raw[0]);hi=min(bas[1],raw[1]);assert 0<lo<=hi and list(map(str,bas))==saved['baseline'] and list(map(str,raw))==saved['raw'] and (str(lo),str(hi))==(saved['lo'],saved['hi']);calls+=1;entries+=saved['stats']['entries'];slots=max(slots,saved['stats']['peak_table_slots']);counts['kernel_weight_replays']+=1;return lo,hi
        def add(dest,b):dest[0]+=b[0];dest[1]+=b[1]
        g0=group(rows,obs,p)
        for c in a['cells']:
            x=c['assignment'];g={}
            for mask,(v,z) in g0.items():
                if par(mask&x):v,z=z,v
                residual=mask&active;u,t=g.get(residual,(1,1));g[residual]=(u*v,t*z)
            z=check(c['root'],g,active);add(root,z);counts['partition_cells']+=1
            for j,q in enumerate(w['queries']):
                f=par(q&x);r0=q&active;es=c['events'][j]
                if not r0:assert es is None;add(totals[2*j+f],z);counts['root_reuses']+=1;continue
                pivot=r0&-r0
                for b in (0,1):add(totals[2*j+b],check(es[b],conditioned(g,pivot,r0,b^f),active^pivot))
        assert list(map(str,root))==a['root'] and calls==a['kernel_calls'] and entries==a['entries'] and slots==a['peak_table_slots']
        if ref:assert root[0]<=ref[0]<=root[1];counts['exact_root_enclosures']+=1
        for j,q in enumerate(w['queries']):
            s=a['queries'][j];(l0,u0),(l1,u1)=totals[2*j:2*j+2];lo,hi=F(l1,l1+u0),F(u1,u1+l0);assert s['weights']==[[str(l0),str(u0)],[str(l1),str(u1)]];assert (str(lo),str(hi))==(s['lo'],s['hi']) and 0<=lo<=hi<=1;cert=1 if lo>=F(9,10) else 0 if hi<=F(1,10) else None;assert cert==s['cert'] and s['truth']==par(q&w['hidden']);counts['intervals']+=1
            if ref:
                z,zs=ref;z1=zs[j];assert l0<=z-z1<=u0 and l1<=z1<=u1 and lo<=F(z1,z)<=hi;counts['exact_interval_enclosures']+=1
        counts['runs']+=1
        if counts['runs']%36==0:print('AUDIT',counts['runs'],flush=True)
    assert counts['runs']==216 and len(seen)==216
    out=dict(status='PASS',**counts,worlds=len(ws),conditions_with_exact_reference=sum(v is not None for v in refs.values()),source_sha256=sha(H/'pilot17.py'),audit_sha256=sha(__file__),receipt_sha256=sha(H/'receipt.json'),scope='Independent integer factor replay via absolute masks and recorded mini-bucket grouping; disjoint cut partitions, query weights, certificates, and offline exact references. Degree/scope selectors, cost counters and RSS are not independently reimplemented; prefix/random cuts verified.')
    (H/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':run()
