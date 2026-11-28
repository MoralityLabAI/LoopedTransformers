"""Independent set-graph cache/reference replay and greedy recomputation."""
import json,sys,math,random
from pathlib import Path
from fractions import Fraction as F
from collections import Counter,OrderedDict,deque
from reference19 import graph,reference,costs,sha
H=Path(__file__).resolve().parent

def decode(rows):return {(j,k):v for j,k,v in rows}
def check_graph(w,p):
    bags,adj,parent=graph(w);n=w['n'];edges=sorted((j,k) for j in adj for k in adj[j]);order=list(bags);pos={j:i for i,j in enumerate(order)};owners=[min((j for j in range(n) if a>>j&1),key=pos.get) for a in w['rows']];nbr={j:[] for j in bags}
    for j,k in parent.items():nbr[j].append(k);nbr[k].append(j)
    sides={}
    for j,k in edges:
        seen={j};q=[j]
        for a in q:
            for b in adj[a]-seen:
                if a==j and b==k:continue
                seen.add(b);q.append(b)
        sides[j,k]=seen
    u,v=F(str(p)).as_integer_ratio();v-=u;charges={};local={e:(1<<len(bags[e[0]]))*len(adj[e[0]]) for e in edges};dep={e:[(a,e[0]) for a in nbr[e[0]] if a!=e[1]] for e in edges}
    for e in edges:
        sep=bags[e[0]]&bags[e[1]];m=sum(j in sides[e] for j in owners);maxval=max(u,v)**(m+3)*2**len(sides[e]);states=[sum(1<<j for j,b in zip(sorted(sep),bs) if b) for bs in __import__('itertools').product((0,1),repeat=len(sep))];d={h:maxval for h in states};charges[e]=256+sys.getsizeof(e)+sum(sys.getsizeof(j) for j in e)+sys.getsizeof(d)+sum(sys.getsizeof(h)+sys.getsizeof(maxval) for h in states)
    topo=sorted(edges,key=lambda e:(len(sides[e]),e))
    def residual(C):
        r={}
        for e in topo:r[e]=0 if e in C else local[e]+sum(r[d] for d in dep[e])
        return r
    cold=residual(set())
    return bags,adj,parent,nbr,owners,sides,charges,dep,topo,local,cold,residual

def demand_trace(n,qs,bags,nbr):
    # Independently visit the union of source-rooted tree paths.
    requests=[];base=[0,0,0]
    def op(j,exclude=None,prev=None):
        requests.extend((k,j) for k in nbr[j] if k!=exclude and k!=prev);s=2**len(bags[j]);base[0]+=s*(len(nbr[j])-int(exclude is not None));base[1]+=s;base[2]+=1
    op(n);remain=set(map(tuple,qs))
    while remain:
        counts=Counter(x for q in remain for x in q);s=min(counts,key=lambda j:(-counts[j],j));selected={q for q in remain if s in q};targets={a if b==s else b for a,b in selected};par={s:None};q=deque([s]);children={j:[] for j in bags}
        while q:
            j=q.popleft()
            for k in nbr[j]:
                if k not in par:par[k]=j;q.append(k)
        for t in targets:
            j=t
            while par[j] is not None:
                k=par[j]
                if j not in children[k]:children[k].append(j)
                j=k
        def visit(j):
            if j in targets:op(j,prev=par[j])
            for k in sorted(children[j]):op(j,k,par[j]);visit(k)
        visit(s);remain-=selected
    return requests,base

def recompute_plan(kind,demand,cap,charges,topo,dep,cold,residual,counts):
    C=set();out=[];left=cap
    if kind=='planned_cold':
        for e in sorted(topo,key=lambda e:(-F(demand.get(e,0)*cold[e],charges[e]),e)):
            if demand.get(e,0)>0 and charges[e]<=left:out.append(e);left-=charges[e]
    else:
        while True:
            r=residual(C);d={e:demand.get(e,0) for e in topo}
            for e in reversed(topo):
                if e in C:d[e]=0
                else:
                    for k in dep[e]:d[k]+=d[e]
            elig=[e for e in topo if e not in C and charges[e]<=left and d[e]*r[e]>0]
            if not elig:break
            # Integer ratio comparison, independent of lazy heap implementation.
            e=elig[0]
            for k in elig[1:]:
                a=d[k]*r[k]*charges[e];b=d[e]*r[e]*charges[k]
                if a>b or a==b and k<e:e=k
            out.append(e);C.add(e);left-=charges[e];counts['greedy_steps']+=1
    r=residual(set(out));return out,sum(demand.get(e,0)*r[e] for e in topo)

def audit(n):
    path=H/f'receipt{n}.json';r=json.loads(path.read_text());assert r['source_sha256']==sha(H/'pilot21.py') and r['protocol_sha256']==sha(H/'PROTOCOL.md');assert all(sha(H/k)==v for k,v in r['base_sha256'].items());ws={w['seed']:w for w in r['worlds']};counts=Counter();seen=set();rows=[]
    for a in r['results']:
        w=ws[a['seed']];p=a['noise'];assert (w['seed'],p) not in seen;seen.add((w['seed'],p));bags,_,_=graph(w);peak=max(map(len,bags.values()));assert peak==a['max_scope'];counts['graph_replays']+=1
        if a['status']=='refused':assert peak>6 and not a['regimes'];counts['refusals']+=1;continue
        assert peak<=6;counts['admitted']+=1;assert len(a['regimes'])==2;bags,adj,parent,nbr,owners,sides,charge,dep,topo,local,cold,residual=check_graph(w,p)
        for reg in a['regimes']:
            regime=reg['regime'];wl=reg['workload'];qs=wl['batches'];assert len(qs)==4 and all(len(set(map(tuple,q)))==16 for q in qs);assert all(len(set(map(tuple,x))&set(map(tuple,y)))==8 for x,y in zip(qs,qs[1:]));assert len(wl['updates'])==3
            sites=[ix for ix,u in wl['updates']];assert len(set(sites))==(1 if regime=='stable_local' else 3)
            assert all(s==wl['sources'][0] for s in wl['sources']) if regime=='stable_local' else all(not set(x)&set(y) for x,y in zip(wl['sources'],wl['sources'][1:]))
            obs=[((q&w['hidden']).bit_count()&1)^int(u<p) for q,u in zip(w['rows'],w['uniforms'])];rr=w['rows'].copy();refs=[];updates=[];traces=[]
            for epoch,q in enumerate(qs):
                if epoch:
                    ix,u=wl['updates'][epoch-1];row=w['rows'][ix];y=((row&w['hidden']).bit_count()&1)^int(u<p);rr.append(row);obs.append(y);updates.append((ix,y,owners[ix]))
                z,events=reference(dict(w,rows=rr,queries=q),obs,p);refs.append((str(z),[str(z-2*x) for x in events]));counts['exact_partitions']+=1;counts['reference_queries']+=16
                if n==12:counts['enumerated_states']+=1<<n
                traces.append(demand_trace(n,q,bags,nbr))
            assert len(reg['arms'])==11 and {(b['policy'],b['kib']) for b in reg['arms']}=={('direct',0)}|{(a,k) for k in (16,64) for a in ('lru','online_cold','planned_cold','marginal','lagged')}
            for b in reg['arms']:
                policy=b['policy'];cap=b['kib']*1024;store=OrderedDict();req=Counter();peakb=bytes_=0;prev=None;arith=0;plans=0;plan_time=0.;plan_work=0;plan_empty=0
                if cap:assert decode(b['charges'])==charge
                else:assert not b['charges']
                for epoch,e in enumerate(b['epochs']):
                    assert e['epoch']==epoch and (e['partition'],e['signed'])==refs[epoch];counts['arm_query_checks']+=16;arith+=sum(e['counts'][:2]);requests,base=traces[epoch];stat=Counter();log=[];u=e['update'];assert (u is None)==(epoch==0)
                    if epoch:assert (u['index'],u['obs'],u['owner'])==updates[epoch-1]
                    if not cap:
                        root_cost=costs(w,bags,adj,parent)[0];assert e['counts']==[17*x for x in root_cost];counts['arithmetic_ledgers']+=1;continue
                    assert decode(e['external'])==dict(Counter(requests));counts['demand_replays']+=1
                    def remove(k,reason):
                        nonlocal bytes_
                        del store[k];bytes_-=charge[k];stat[reason]+=1;log.append([reason,*k,charge[k]])
                    if epoch:
                        before=len(store)
                        for k in list(store):
                            if u['owner'] in sides[k]:remove(k,'invalidate')
                        assert u['counts']==[before,len(store)];counts['invalidation_replays']+=1
                    pr=e['plan'];target=None
                    if policy in ('planned_cold','marginal','lagged'):
                        demand=Counter(requests);forecast=prev if policy=='lagged' and prev is not None else demand;prev=demand;assert decode(pr['forecast'])==dict(forecast) and decode(pr['current'])==dict(demand)
                        # Same topology and forecasts at both noise levels; independently recompute every plan.
                        order,warm=recompute_plan(policy,forecast,cap,charge,topo,dep,cold,residual,counts);assert [list(k) for k in order]==pr['target'] and warm==pr['warm_work'];assert pr['uncached_work']==sum(forecast.get(k,0)*cold[k] for k in topo);target=set(order);assert sum(charge[k] for k in target)<=cap
                        for k in list(store):
                            if k not in target:remove(k,'retarget')
                        counts['plan_replays']+=1;plans+=1;plan_work+=warm;plan_empty+=pr['uncached_work'];plan_time+=e['planning_seconds']
                    else:assert pr is None
                    calc=base.copy()
                    def get(k):
                        nonlocal bytes_,peakb
                        req[k]+=1;stat['requests']+=1
                        if k in store:stat['hits']+=1;store.move_to_end(k);return
                        stat['misses']+=1
                        for d in dep[k]:get(d)
                        j,t=k;s=2**len(bags[j]);calc[0]+=s*(len(adj[j])-1);calc[1]+=s;calc[2]+=1;c=charge[k]
                        if c>cap or target is not None and k not in target:stat['rejected']+=1;return
                        if bytes_+c>cap:
                            rank=lambda x:(req[x]*cold[x]/charge[x],x)
                            order=list(store) if policy=='lru' else sorted(store,key=rank);victims=[];free=cap-bytes_
                            for x in order:
                                if policy=='online_cold' and rank(k)<=rank(x):stat['rejected']+=1;return
                                victims.append(x);free+=charge[x]
                                if free>=c:break
                            for x in victims:remove(x,'evict')
                        store[k]=True;bytes_+=c;peakb=max(peakb,bytes_);stat['insert']+=1;log.append(['insert',*k,c]);assert bytes_<=cap
                    for k in requests:get(k)
                    assert calc==e['counts'] and log==e['events'] and dict(stat)==e['cache'] and bytes_==e['bytes'] and peakb==e['peak_bytes'];counts['arithmetic_ledgers']+=1;counts['cache_transitions']+=len(log);counts['epoch_ledgers']+=1
                rows.append(dict(n=n,family=w['family'],seed=w['seed'],noise=p,regime=regime,policy=policy,kib=b['kib'],work=arith,seconds=b['seconds'],peak_bytes=peakb,plan_seconds=plan_time,warm_work=plan_work if plans else None,uncached_work=plan_empty if plans else None));counts['trajectories']+=1
        print('AUDIT',n,w['seed'],p,flush=True)
    assert len(seen)==18;out=dict(status='PASS',counts=dict(counts),source_sha256=sha(__file__),receipt_sha256=sha(path),rows=rows);(H/f'audit{n}.json').write_text(json.dumps(out,indent=2)+'\n');print(out['counts'],flush=True)

def fixtures():
    from pilot21 import Graph,Tree,world,plan
    g=random.Random(212121);checks=greedy=0
    for i in range(12):
        w=world(12,('chain','hub')[i%2],i);t=Tree(w,.15,plan(12,w['rows']));G=Graph(t);es=G.edges
        def brute(d,C):
            def run(e):return 0 if e in C else G.local[e]+sum(run(k) for k in G.dep[e])
            return sum(v*run(e) for e,v in d.items())
        for _ in range(16):
            d={e:g.randrange(4) for e in es};C=set(g.sample(es,g.randrange(10)));D=G.flow(d,C);R=G.residual(C);base=brute(d,C)
            for e in set(es)-C:assert base-brute(d,C|{e})==D[e]*R[e];checks+=1
            # A second cache set containing the first checks diminishing marginal returns.
            T=C|set(g.sample(es,6));large=brute(d,T)
            for e in set(es)-T:assert base-brute(d,C|{e})>=large-brute(d,T|{e});checks+=1
            cap=g.randrange(1000,18000);a,_=G.greedy(d,cap,'marginal');b,_=recompute_plan('marginal',d,cap,G.charge,G.order,G.dep,G.cold,G.residual,Counter());assert a==b;greedy+=1
    out=dict(status='PASS',marginal_and_diminishing_returns_checks=checks,fixture_greedy_plans=greedy);(H/'fixture.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':
    if sys.argv[1]=='fixture':fixtures()
    else:audit(int(sys.argv[1]))
