"""Independent set-scope and absolute-mask replay, exact reference and budgets."""
import json, math, hashlib, sys, struct, random
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from audit16 import group, independent, mb, frontier_exact, par
H=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def vertices(a):return {i for i in range(a.bit_length()) if a>>i&1}
def integer(s):return sum(1<<i for i in s)
def scope_metrics(gg,active,cap,trace):
    # Independently reconstruct grouping on sets and count every table slot.
    fac=[vertices(a) for a in sorted(gg)];used=set().union(*fac);fac.extend({j} for j in sorted(vertices(active)-used));rem=vertices(active)
    slots=lambda fs:sum(2**(len(s)+1) for s in fs)
    work=0;peak=slots(fac);splits=0;max_scope=0
    assert len(trace)==len(rem)
    for v,sizes,anchor in trace:
        assert v==min(rem,key=lambda j:(len(set().union(*(s for s in fac if j in s))),j));rem.remove(v)
        bucket=[s for s in fac if v in s];fac=[s for s in fac if v not in s];bins=[];sc=[]
        for s in sorted(bucket,key=lambda s:(-len(s),tuple(sorted(s)))):
            assert len(s)<=cap;ids=[j for j,t in enumerate(sc) if len(s|t)<=cap]
            if ids:
                j=min(ids,key=lambda j:(len(sc[j]|s),j));bins[j]+=1;sc[j]|=s
            else:bins.append(1);sc.append(s.copy())
        assert sizes==[len(s) for s in sc] and anchor==max(range(len(bins)),key=lambda j:(len(sc[j]),bins[j],-j))
        splits+=len(sc)-1
        for k,s in zip(bins,sc):work+=k*2**len(s);max_scope=max(max_scope,len(s));fac.append(s-{v})
        peak=max(peak,slots(fac)+slots(bucket))
    assert not rem and all(not s for s in fac)
    return [work,peak,splits,max_scope]

def pair_intersect(a,b):
    r=[(max(l,L),min(u,U)) for (l,u),(L,U) in zip(a,b)];assert all(l<=u for l,u in r);return r

def tree_bounds(node):
    if not node['children']:return node['own']
    bs=[tree_bounds(c) for c in node['children']];return pair_intersect(node['own'],[(sum(b[i][0] for b in bs),sum(b[i][1] for b in bs)) for i in (0,1)])
def leaves(node):
    if not node['children']:return [node]
    return [l for c in node['children'] for l in leaves(c)]
def event(pair):
    (l0,u0),(l1,u1)=pair;return F(l1,l1+u0),F(u1,u1+l0)
def snapshot(roots):return [list(map(str,event(tree_bounds(n)))) for n in roots]
def cut_degree(rows,active,k):
    g=vertices(active);cut=set()
    for _ in range(k):
        adj={i:set() for i in g}
        for row in rows:
            s=vertices(row)&g
            for v in s:adj[v]|=s-{v}
        v=min(g,key=lambda j:(-len(adj[j]),j));cut.add(v);g.remove(v)
    return integer(cut)
def condition_group(g,q,b):
    pivot=q&-q;gg={}
    for a,(x,y) in g.items():
        if a&pivot:
            a^=q
            if b:x,y=y,x
        l,u=gg.get(a,(1,1));gg[a]=(l*x,u*y)
    return gg

def restrict_group(g,fixed,value):
    gg={}
    for a,(x,y) in g.items():
        if par(a&value):x,y=y,x
        a&=~fixed;l,u=gg.get(a,(1,1));gg[a]=(l*x,u*y)
    return gg

def exact_ref(w,obs,p):
    n=w['n'];rows=w['rows'];qs=w['queries']
    if n==12:
        u,v=F(str(p)).as_integer_ratio();v-=u;m=len(obs);wt=[u**e*v**(m-e) for e in range(m+1)]
        ws=[wt[sum(par(a&h)!=b for a,b in zip(rows,obs))] for h in range(1<<n)];return sum(ws),[sum(x for h,x in enumerate(ws) if par(h&q)) for q in qs],ws
    if w['family']=='dense':return None
    order=w['offline_order'];project=lambda a:sum((a>>i&1)<<j for j,i in enumerate(order));g=group([project(a) for a in rows],obs,p);z=frontier_exact(g,n)[0]
    return z,[frontier_exact(g,n,project(q))[0] for q in qs],None

def run():
    r=json.loads((H/'receipt.json').read_text());assert r['source_sha256']==sha(H/'pilot18.py') and r['kernel_sha256']==sha(H/'pilot16.py') and r['protocol_sha256']==sha(H/'PROTOCOL.md');assert sha(H/'audit16.py')=='d544e0dfec048d26bf7b47ccd0d14a282a905ba74a286aa97e0a362d64d8fe02'
    ws={w['seed']:w for w in r['worlds']};assert len(ws)==27;seen=set();counts=Counter();refs={};all_rows=[]
    for a in r['results']:
        w=ws[a['seed']];n=w['n'];p=a['noise'];arm=a['arm'];qs=w['queries'];key=(w['seed'],p,arm);assert key not in seen;seen.add(key)
        assert len(w['rows'])==5*n and all(w['rows'].count(1<<j)==2 for j in range(n)) and all(q.bit_count()==3 for q in w['rows'][2*n:]);assert len(set(qs))==4 and all(q.bit_count()==2 and q not in w['rows'] for q in qs)
        obs=[par(q&w['hidden'])^int(u<p) for q,u in zip(w['rows'],w['uniforms'])];assert a['obs']==obs;g=group(w['rows'],obs,p);ctx=[([condition_group(g,q,b) for b in (0,1)],((1<<n)-1)^(q&-q)) for q in qs]
        u,v=F(str(p)).as_integer_ratio();v-=u;assert a['entry_bytes_upper']==sys.getsizeof((max(u,v)**len(obs))<<n)+struct.calcsize('P')
        keyref=(w['seed'],p)
        if keyref not in refs:refs[keyref]=exact_ref(w,obs,p)
        ref=refs[keyref];caps=dict(entries=900*n,slots=45*n,calls=128);assert caps==a['caps'];cost=dict(entries=0,slots=0,calls=0);i=0
        def kernel(j,fixed,value,cap):
            nonlocal i
            assert fixed&~ctx[j][1]==0 and value&~fixed==0;out=[]
            for b in (0,1):
                rec=a['records'][i];i+=1;assert (rec['query'],rec['side'],rec['fixed'],rec['value'],rec['cap'])==(j,b,fixed,value,cap);z=rec['result'];gg=restrict_group(ctx[j][0][b],fixed,value);active=ctx[j][1]^fixed
                bas=independent(gg,active);assert list(map(str,bas))==z['baseline'];raw=mb(gg,active,cap,z['trace']) if cap else bas;assert list(map(str,raw))==z['raw'];l,u=max(bas[0],raw[0]),min(bas[1],raw[1]);assert 0<l<=u and [str(l),str(u)]==[z['lo'],z['hi']]
                met=scope_metrics(gg,active,cap,z['trace']) if cap else [0,0,0,0];assert rec['plan']==met;assert not cap or (z['stats']['entries']==met[0] and z['stats']['peak_table_slots']<=met[1] and z['stats']['splits']==met[2] and z['stats']['max_scope']==met[3]);cost['entries']+=met[0];cost['slots']=max(cost['slots'],met[1]);cost['calls']+=1;assert all(cost[k]<=caps[k] for k in caps)
                if ref and ref[2] is not None:
                    exact=sum(wt for h,wt in enumerate(ref[2]) if par(h&qs[j])==b and h&fixed==value);assert l<=exact<=u;counts['exhaustive_cell_enclosures']+=1
                counts['weight_bound_replays']+=1;counts['budget_replays']+=1;out.append((l,u))
            return out
        roots=[dict(fixed=0,value=0,cap=0,own=kernel(j,0,0,0),children=[]) for j in range(4)];assert snapshot(roots)==a['initial']
        prev_j=-1
        for step in a['actions']:
            j,fixed,value,cut,cap=[step[x] for x in ('query','fixed','value','cut','cap')];assert 0<=j<4 and cut&fixed==0 and cut&~ctx[j][1]==0 and cap in (3,5,7);root=roots[j];node=next(x for x in leaves(root) if (x['fixed'],x['value'])==(fixed,value));assert step['before']==snapshot(roots);before=cost.copy();remaining=ctx[j][1]^fixed
            if arm!='adaptive':
                assert fixed==value==0 and j>prev_j;prev_j=j;assert cap==(5 if arm=='scope5' else 3)
                assert cut==(0 if arm.startswith('scope') else cut_degree(ctx[j][0][0].keys(),remaining,2) if arm=='degree2' else integer(random.Random(w['seed']+18018+j).sample(sorted(vertices(remaining)),2)))
            elif cut:
                assert cap==3 and 1<=cut.bit_count()<=2 and (cut|fixed).bit_count()<=4 and cut==cut_degree(ctx[j][0][0].keys(),remaining,cut.bit_count())
            else:assert cap>node['cap']
            counts['structural_action_checks']+=1
            if cut:
                # Product assignment enumeration in descending numeric order, independent of pilot's submask loop.
                import itertools
                assignments=sorted([integer(v for v,b in zip(sorted(vertices(cut)),bs) if b) for bs in itertools.product((0,1),repeat=cut.bit_count())],reverse=True)
                assert len(set(assignments))==1<<cut.bit_count();node['children']=[dict(fixed=fixed|cut,value=value|x,cap=cap,own=kernel(j,fixed|cut,value|x,cap),children=[]) for x in assignments];counts['disjoint_splits']+=1
            else:node['own']=pair_intersect(node['own'],kernel(j,fixed,value,cap));node['cap']=cap
            assert step['estimate'][0]==cost['entries']-before['entries'] and step['estimate'][1]==cost['calls']-before['calls'];assert step['estimate'][2:]==a['records'][i-1]['plan'][2:3]+a['records'][i-1]['plan'][1:2];assert step['cost']==cost
            new=snapshot(roots);assert new==step['after'] and all(F(old[0])<=F(cur[0])<=F(cur[1])<=F(old[1]) for old,cur in zip(step['before'],new));counts['monotone_actions']+=1
        assert i==len(a['records']) and cost==a['cost'] and a['counted_payload_upper_bytes']==cost['slots']*a['entry_bytes_upper']
        for j,(root,saved) in enumerate(zip(roots,a['queries'])):
            b=tree_bounds(root);l,u=event(b);assert saved['q']==qs[j] and saved['weights']==[list(map(str,v)) for v in b];assert [str(l),str(u)]==[saved['lo'],saved['hi']];assert saved['cert']==(1 if l>=F(9,10) else 0 if u<=F(1,10) else None) and saved['truth']==par(qs[j]&w['hidden'])
            ls=leaves(root);assert saved['frontier']==[[x['fixed'],x['value'],x['cap']] for x in ls];assert sum(1<<(n-1-x['fixed'].bit_count()) for x in ls)==1<<(n-1)
            assert all((x['value']^y['value'])&x['fixed']&y['fixed'] for ii,x in enumerate(ls) for y in ls[ii+1:]);counts['disjoint_frontier_checks']+=1
            if ref:
                z,zs,_=ref;z1=zs[j];assert b[0][0]<=z-z1<=b[0][1] and b[1][0]<=z1<=b[1][1] and l<=F(z1,z)<=u;counts['exact_interval_enclosures']+=1
            counts['intervals']+=1
        counts['runs']+=1
        if counts['runs']%30==0:print('AUDIT',counts['runs'],flush=True)
    assert counts['runs']==270
    out=dict(status='PASS',**counts,worlds=27,conditions_with_exact_reference=sum(v is not None for v in refs.values()),receipt_sha256=sha(H/'receipt.json'),source_sha256=sha(Path(__file__)),scope='Independent absolute-mask weight bounds, set-scope work/slot plans, structural cut checks, disjoint frontier replay, monotone interval intersection, exact references. Does not independently optimize adaptive priority or prove every rejected candidate infeasible. No independent runtime replay.')
    (H/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':run()
