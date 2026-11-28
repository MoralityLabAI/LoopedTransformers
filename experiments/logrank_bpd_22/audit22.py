"""Independent likelihood, lazy-fill objective and cache ledger replay."""
import json,sys,random,itertools
from pathlib import Path
from collections import Counter,OrderedDict
from fractions import Fraction as F
from audit21 import check_graph,demand_trace,recompute_plan,decode
from reference19 import graph,reference,costs,sha
H=Path(__file__).resolve().parent
POLICIES=('lru','planned_cold','marginal','resident','retention')
def pinned(requests,C,R,dep,local):
    """Sequential recursive reference, independent of vector/topological count planner."""
    filled=set(R)&C;work=0
    def get(e):
        nonlocal work
        if e in filled:return
        for k in dep[e]:get(k)
        work+=local[e]
        if e in C:filled.add(e)
    for e in requests:get(e)
    return work,filled

def counts_work(d,C,R,topo,dep,local):
    # Scalar topological count reference for candidate ranking; compared to recursive replay.
    D=Counter(d);filled=set(R)&C;work=0
    for e in reversed(topo):
        m=D[e]
        if e in filled:m=0
        elif e in C and m:filled.add(e);m=1
        work+=local[e]*m
        if m:
            for k in dep[e]:D[k]+=m
    return work,filled

def objective(d,C,R,owners,sides,topo,dep,local):
    now,filled=counts_work(d,C,R,topo,dep,local)
    return now*max(1,len(owners))+sum(counts_work(d,C,{e for e in filled if o not in sides[e]},topo,dep,local)[0] for o in owners),now,filled

def audit(n):
    path=H/f'receipt{n}.json';r=json.loads(path.read_text());assert r['source_sha256']==sha(H/'pilot22.py') and r['protocol_sha256']==sha(H/'PROTOCOL.md');ws={w['seed']:w for w in r['worlds']};assert len(ws)==9;counts=Counter();seen=set();rows=[]
    for a in r['results']:
        w=ws[a['seed']];p=a['noise'];assert (w['seed'],p) not in seen;seen.add((w['seed'],p));bags,_,_=graph(w);peak=max(map(len,bags.values()));assert peak==a['max_scope'];counts['graph_replays']+=1
        if a['status']=='refused':assert peak>6 and not a['regimes'];counts['refused']+=1;continue
        assert peak<=6;counts['admitted']+=1;assert len(a['regimes'])==4;bags,adj,parent,nbr,owners,sides,charge,dep,topo,local,cold,residual=check_graph(w,p)
        regimes={(x['pop'],x['locality']):x['workload'] for x in a['regimes']};assert len(regimes)==4
        for pop in ('stable','shift'):assert all(regimes[pop,'local'][k]==regimes[pop,'uniform'][k] for k in ('batches','sources'))
        for loc in ('local','uniform'):assert all(regimes['stable',loc][k]==regimes['shift',loc][k] for k in ('updates','scenario_sites','local_site'))
        counts['factorial_checks']+=4
        for reg in a['regimes']:
            pop,loc=reg['pop'],reg['locality'];wl=reg['workload'];qs=wl['batches'];assert len(qs)==4 and all(len(set(map(tuple,q)))==16 for q in qs);assert all(len(set(map(tuple,x))&set(map(tuple,y)))==8 for x,y in zip(qs,qs[1:]));assert len(wl['updates'])==3
            assert all(s==wl['sources'][0] for s in wl['sources']) if pop=='stable' else all(not set(x)&set(y) for x,y in zip(wl['sources'],wl['sources'][1:]))
            # Replay independent update and scenario RNGs, including public local site.
            u=random.Random(w['seed']+22200);site=u.randrange(n);draws=[u.randrange(n) for _ in range(3)];uniforms=[u.random() for _ in range(3)];assert wl['local_site']==site and wl['updates']==[list(z) for z in zip([site]*3 if loc=='local' else draws,uniforms)]
            v=random.Random(w['seed']+22300);assert wl['scenario_sites']==[[site] if loc=='local' else [v.randrange(n) for _ in range(4)] for _ in range(3)]+[[]]
            obs=[((q&w['hidden']).bit_count()&1)^int(u<p) for q,u in zip(w['rows'],w['uniforms'])];rr=w['rows'].copy();refs=[];updates=[];traces=[]
            for epoch,q in enumerate(qs):
                if epoch:
                    ix,u=wl['updates'][epoch-1];row=w['rows'][ix];y=((row&w['hidden']).bit_count()&1)^int(u<p);rr.append(row);obs.append(y);updates.append((ix,y,owners[ix]))
                z,events=reference(dict(w,rows=rr,queries=q),obs,p);refs.append((str(z),[str(z-2*x) for x in events]));counts['exact_partitions']+=1;counts['reference_queries']+=16
                if n==12:counts['enumerated_states']+=1<<n
                traces.append(demand_trace(n,q,bags,nbr))
            assert len(reg['arms'])==11 and {(b['policy'],b['kib']) for b in reg['arms']}=={('direct',0)}|{(a,k) for k in (16,64) for a in POLICIES}
            for b in reg['arms']:
                policy=b['policy'];cap=b['kib']*1024;store=OrderedDict();peakb=bytes_=arith=0;plan_time=0.;invalidations=0;fills=0;retargets=0;unsigned=0;assert len(b['epochs'])==4
                if cap:assert decode(b['charges'])==charge
                else:assert not b['charges']
                for epoch,e in enumerate(b['epochs']):
                    assert e['epoch']==epoch and (e['partition'],e['signed'])==refs[epoch];counts['arm_query_checks']+=16;arith+=sum(e['counts'][:2]);requests,base=traces[epoch];stat=Counter();log=[];u=e['update'];assert (u is None)==(epoch==0)
                    if epoch:assert (u['index'],u['obs'],u['owner'])==updates[epoch-1]
                    if not cap:
                        root_cost=costs(w,bags,adj,parent)[0];assert e['counts']==[17*x for x in root_cost];counts['arithmetic_ledgers']+=1;continue
                    d=Counter(requests);assert decode(e['external'])==dict(d);counts['demand_replays']+=1
                    def remove(k,reason):
                        nonlocal bytes_
                        del store[k];bytes_-=charge[k];stat[reason]+=1;log.append([reason,*k,charge[k]])
                    if epoch:
                        before=len(store)
                        for k in list(store):
                            if u['owner'] in sides[k]:remove(k,'invalidate')
                        assert u['counts']==[before,len(store)];counts['invalidation_replays']+=1
                    pr=e['plan'];target=None;expected_now=None;res=set(store)
                    if policy!='lru':
                        assert decode(pr['demand'])==dict(d) and set(map(tuple,pr['resident']))==res;order=list(map(tuple,pr['target']));target=set(order);assert len(order)==len(target) and sum(charge[k] for k in target)<=cap
                        if policy in ('planned_cold','marginal'):
                            expected,_=recompute_plan(policy,d,cap,charge,topo,dep,cold,residual,counts);assert order==expected;counts['warm_plan_replays']+=1
                        else:
                            os=[owners[i] for i in wl['scenario_sites'][epoch]] if policy=='retention' else [];assert os==pr['scenario_owners'];C=set();left=cap
                            assert len(pr['steps'])==len(order)
                            for k,s in zip(order,pr['steps']):
                                assert tuple(s[:2])==k and k not in C and charge[k]<=left;old=objective(d,C,res,os,sides,topo,dep,local)[0];new=objective(d,C|{k},res,os,sides,topo,dep,local)[0];assert old==s[2] and new==s[3] and new<old
                                if n==12:
                                    elig=[x for x in topo if x not in C and charge[x]<=left];best=max(elig,key=lambda x:(F(old-objective(d,C|{x},res,os,sides,topo,dep,local)[0],charge[x]),tuple(-v for v in x)));assert k==best;counts['full_greedy_steps']+=1
                                C.add(k);left-=charge[k];counts['resident_greedy_steps']+=1
                            if n==12:assert all(objective(d,C|{k},res,os,sides,topo,dep,local)[0]>=objective(d,C,res,os,sides,topo,dep,local)[0] for k in topo if k not in C and charge[k]<=left);counts['full_greedy_stops']+=1
                            obj,now,filled=objective(d,target,res,os,sides,topo,dep,local);assert (obj,now)==(pr['objective'],pr['predicted_now']) and filled==set(map(tuple,pr['predicted_filled']));expected_now,expected_filled=pinned(requests,target,res,dep,local);assert expected_now==now and expected_filled==filled
                            # Independently recompute each forecast scenario by sequential recursion.
                            assert obj==max(1,len(os))*now+sum(pinned(requests,target,{k for k in filled if o not in sides[k]},dep,local)[0] for o in os);counts['resident_plan_replays']+=1
                        for k in list(store):
                            if k not in target:remove(k,'retarget')
                        plan_time+=e['planning_seconds']
                    else:assert pr is None
                    calc=base.copy()
                    def get(k):
                        nonlocal bytes_,peakb
                        stat['requests']+=1
                        if k in store:stat['hits']+=1;store.move_to_end(k);return
                        stat['misses']+=1
                        for d0 in dep[k]:get(d0)
                        j,t=k;s=2**len(bags[j]);calc[0]+=s*(len(adj[j])-1);calc[1]+=s;calc[2]+=1;c=charge[k]
                        if c>cap or target is not None and k not in target:stat['rejected']+=1;return
                        while bytes_+c>cap:
                            assert policy=='lru';remove(next(iter(store)),'evict')
                        store[k]=True;bytes_+=c;peakb=max(peakb,bytes_);stat['insert']+=1;log.append(['insert',*k,c]);assert bytes_<=cap
                    for k in requests:get(k)
                    now=sum(calc[:2])-sum(base[:2]);unsigned+=now
                    if expected_now is not None:assert now==expected_now and set(store)==expected_filled;counts['predicted_work_matches']+=1
                    assert calc==e['counts'] and log==e['events'] and dict(stat)==e['cache'] and bytes_==e['bytes'] and peakb==e['peak_bytes'];counts['arithmetic_ledgers']+=1;counts['cache_transitions']+=len(log);counts['epoch_ledgers']+=1;invalidations+=stat['invalidate'];fills+=stat['insert'];retargets+=stat['retarget']
                rows.append(dict(n=n,family=w['family'],seed=w['seed'],noise=p,pop=pop,locality=loc,policy=policy,kib=b['kib'],work=arith,unsigned_work=unsigned,seconds=b['seconds'],plan_seconds=plan_time,peak_bytes=peakb,invalidations=invalidations,fills=fills,retargets=retargets));counts['trajectories']+=1
        print('AUDIT',n,w['seed'],p,flush=True)
    assert len(seen)==18;out=dict(status='PASS',counts=dict(counts),source_sha256=sha(__file__),receipt_sha256=sha(path),rows=rows);(H/f'audit{n}.json').write_text(json.dumps(out,indent=2)+'\n');print(out['counts'],flush=True)
if __name__=='__main__':audit(int(sys.argv[1]))
