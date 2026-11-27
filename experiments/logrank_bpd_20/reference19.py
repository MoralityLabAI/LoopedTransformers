"""Independent exact marginal reference and public graph/cost replay."""
import hashlib, json, math, sys
from collections import defaultdict, deque, Counter
from fractions import Fraction
from pathlib import Path
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def vars_(a):return {j for j in range(a.bit_length()) if a>>j&1}
def reference(w,obs,p):
    f=Fraction(str(p));u,v=f.numerator,f.denominator-f.numerator;n=w['n'];rows=w['rows']
    if n==12:
        wt=[u**e*v**(len(obs)-e) for e in range(len(obs)+1)]
        states=[wt[sum(((h&a).bit_count()&1)!=b for a,b in zip(rows,obs))] for h in range(1<<n)]
        z=sum(states);return z,[sum(x for h,x in enumerate(states) if ((h>>a)^(h>>b))&1) for a,b in w['queries']]
    if w['family']=='dense':return None
    pos={j:i for i,j in enumerate(w['offline_order'])};project=lambda a:sum(1<<pos[j] for j in vars_(a));fac={}
    for a,b in zip(rows,obs):
        a=project(a);x,y=fac.get(a,(1,1));fac[a]=(x*(u if b else v),y*(v if b else u))
    def solve(event=None):
        terms=list(fac.items())+([] if event is None else [(event,(0,1))]);dp={0:1}
        for j in range(n):
            b=1<<j;done=[(a,t) for a,t in terms if a.bit_length()==j+1];future=0
            for a,t in terms:
                if a.bit_length()>j+1:future|=a
            future&=2*b-1;out=defaultdict(int)
            for h,x in dp.items():
                for y in (h,h|b):out[y&future]+=x*math.prod(t[(a&y).bit_count()&1] for a,t in done)
            dp=out
        return dp[0]
    z=solve();return z,[solve((1<<pos[a])|(1<<pos[b])) for a,b in w['queries']]

def graph(w):
    n=w['n'];g={i:set() for i in range(n)};bags={};order=[]
    for a in w['rows']:
        v=vars_(a)
        for j in v:g[j]|=v-{j}
    while g:
        i=min(g,key=lambda j:(len(g[j]),j));v=g.pop(i);bags[i]=v|{i};order.append(i)
        for j in v:g[j].update(v-{j});g[j].discard(i)
    bags[n]=set();pos={v:i for i,v in enumerate(order)};adj={j:set() for j in bags};parent={}
    for j in order:
        nb=bags[j]-{j};t=min(nb,key=pos.get) if nb else n;parent[j]=t;adj[j].add(t);adj[t].add(j)
        assert nb<=bags[t]
    for a in w['rows']:assert any(vars_(a)<=s for s in bags.values())
    for v in range(n):
        nodes={j for j in bags if v in bags[j]};seen={v};q=deque([v])
        while q:
            j=q.popleft()
            for t in adj[j]&nodes-seen:seen.add(t);q.append(t)
        assert seen==nodes
    return bags,adj,parent

def costs(w,bags,adj,parent):
    n=w['n'];sz=lambda j:2**len(bags[j]);plus=lambda x,y:tuple(a+b for a,b in zip(x,y));zero=(0,0,0)
    def visit(j,k=None):return (sz(j)*(len(adj[j])-int(k is not None)),sz(j),1)
    direct=zero
    for j in bags:direct=plus(direct,visit(j,parent.get(j)))
    calibration=visit(n)
    for j in bags:
        for k in adj[j]:calibration=plus(calibration,visit(j,k))
    def path(s,t):
        prev={s:None};q=deque([s])
        while q:
            j=q.popleft()
            if j==t:break
            for k in adj[j]-prev.keys():prev[k]=j;q.append(k)
        out=[t]
        while out[-1]!=s:out.append(prev[out[-1]])
        return out[::-1]
    def path_cost(qs):
        c=zero
        for s,t in qs:
            p=path(s,t)
            for i,j in enumerate(p):c=plus(c,visit(j,p[i+1] if i+1<len(p) else None))
        return c
    def batch_cost(qs):
        remain=set(map(tuple,qs));c=zero
        while remain:
            sources=Counter(v for q in remain for v in q);s=min(sources,key=lambda j:(-sources[j],j));chosen={q for q in remain if s in q};edges=set()
            for q in chosen:
                t=q[0] if q[1]==s else q[1];p=path(s,t);edges|=set(zip(p,p[1:]));c=plus(c,visit(t))
            for j,k in edges:c=plus(c,visit(j,k))
            remain-=chosen
        return c
    return direct,calibration,path_cost,batch_cost

def main():
    r=json.loads((H/'receipt.json').read_text());assert r['source_sha256']==sha(H/'pilot19.py') and r['protocol_sha256']==sha(H/'PROTOCOL.md');ws={w['seed']:w for w in r['worlds']};assert len(ws)==27;seen=set();c=Counter()
    for a in r['results']:
        w=ws[a['seed']];n=w['n'];p=a['noise'];assert (w['seed'],p) not in seen;seen.add((w['seed'],p));assert len(w['rows'])==5*n and len(set(map(tuple,w['queries'])))==64 and all(i<j for i,j in w['queries']);assert all(w['rows'].count(1<<j)==2 for j in range(n))
        obs=[((q&w['hidden']).bit_count()&1)^int(t<p) for q,t in zip(w['rows'],w['uniforms'])];bags,adj,parent=graph(w);peak=max(map(len,bags.values()));assert peak==a['max_scope'];c['graph_replays']+=1
        ref=reference(w,obs,p)
        if n==12:c['exhaustive_states']+=1<<n
        if a['status']=='refused':assert peak>6;c['refused']+=1;continue
        assert peak<=6 and ref is not None;z,events=ref;assert z==int(a['partition']);assert [z-2*t for t in events]==list(map(int,a['signed']));assert [str(Fraction(t,z)) for t in events]==a['probabilities'];c['exact_partitions']+=1;c['exact_query_checks']+=len(events)
        direct,cal,path,batch=costs(w,bags,adj,parent);assert list(direct)==a['root_counts'] and list(cal)==a['calibration_counts']
        for x,y,b in zip(a['direct'],a['path'],a['batch']):
            k=x['q'];qs=w['queries'][:k];assert x['counts']==[k*t for t in direct] and y['counts']==list(path(qs)) and b['counts']==list(batch(qs));c['work_ledgers']+=3
        assert a['cache_estimated_bytes']<=a['cache_preflight_bytes']<=1024*1024;assert all(v==(a['cache_preflight_bytes']<=1024*int(k)) for k,v in a['cache_admission'].items());c['cache_admission_checks']+=len(a['cache_admission'])
    assert len(seen)==54
    out=dict(status='PASS',**c,source_sha256=sha(H/'pilot19.py'),receipt_sha256=sha(H/'receipt.json'),audit_sha256=sha(__file__),scope='Independent set-graph running-intersection and cost-ledger replay; direct full enumeration at n12 and chronological frontier DP on structured graphs; admission comparisons, not independent Python object-size or timing replay. Refused dense n12 reference computed but no claimed solver prediction.')
    (H/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
