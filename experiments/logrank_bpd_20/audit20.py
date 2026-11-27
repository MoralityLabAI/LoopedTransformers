"""Independent reference, update-invalidation and logical byte ledger replay."""
import json, hashlib, math
from pathlib import Path
from collections import Counter
from reference19 import graph, reference
H=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def side(adj,j,k):
    seen={j};todo=[j]
    for a in todo:
        for b in adj[a]:
            if (a,b)==(j,k) or b in seen:continue
            seen.add(b);todo.append(b)
    return seen

def audit(n):
    r=json.loads((H/f'receipt{n}.json').read_text());assert r['source_sha256']==sha(H/'pilot20.py') and r['base_sha256']==sha(H/'base19.py') and r['protocol_sha256']==sha(H/'PROTOCOL.md');ws={w['seed']:w for w in r['worlds']};assert len(ws)==9;c=Counter();summary=[];seen=set()
    for a in r['results']:
        w=ws[a['seed']];p=a['noise'];assert (w['seed'],p) not in seen;seen.add((w['seed'],p));assert len(w['rows'])==5*n and all(w['rows'].count(1<<i)==2 for i in range(n));assert len(w['updates'])==3 and all(0<=i<len(w['rows']) and 0<=u<1 for i,u in w['updates']);qs=w['batches'];assert len(qs)==4 and all(len(set(map(tuple,b)))==len(b)==16 and all(0<=i<j<n for i,j in b) for b in qs);assert all(len(set(map(tuple,x))&set(map(tuple,y)))==8 for x,y in zip(qs,qs[1:]));bags,adj,parent=graph(w);peak=max(map(len,bags.values()));assert a['max_scope']==peak;c['graph_replays']+=1
        if a['status']=='refused':assert peak>6;c['refused']+=1;continue
        assert peak<=6;c['solved']+=1;pos={j:i for i,j in enumerate(bags)};owner=[min((j for j in range(n) if q>>j&1),key=pos.get) for q in w['rows']];sides={(j,k):side(adj,j,k) for j in adj for k in adj[j]};assert len(sides)==2*n and all(len(sides[j,k])+len(sides[k,j])==n+1 for j,k in sides)
        obs=[((q&w['hidden']).bit_count()&1)^int(u<p) for q,u in zip(w['rows'],w['uniforms'])];rows=w['rows'].copy();refs=[];updates=[]
        for epoch,b in enumerate(qs):
            if epoch:
                ix,u=w['updates'][epoch-1];row=w['rows'][ix];y=((row&w['hidden']).bit_count()&1)^int(u<p);obs.append(y);rows.append(row);updates.append((ix,y,owner[ix]))
            z,events=reference(dict(w,rows=rows,queries=b),obs,p);refs.append((str(z),[str(z-2*x) for x in events]));c['exact_reference_queries']+=len(events);c['exact_partitions']+=1
            if n==12:c['exhaustive_states']+=1<<n
        assert len(a['arms'])==10
        assert {(b['policy'],b['kib'],b['flush']) for b in a['arms']}=={('direct',0,False),('lru',16,False),('lru',64,False),('lru',256,False),('value',16,False),('value',64,False),('value',256,False),('lru',1024,False),('lru',64,True),('lru',1024,True)}
        for b in a['arms']:
            store={};charge=peakb=0;policy=b['policy'];cap=1024*b['kib'];counts=Counter();arith=0;assert len(b['epochs'])==4
            for epoch,e in enumerate(b['epochs']):
                assert e['epoch']==epoch and (e['partition'],e['signed'])==refs[epoch];c['arm_query_checks']+=16;arith+=sum(e['counts'][:2]);u=e['update'];assert (u is None)==(epoch==0)
                if epoch:
                    ix,y,j=updates[epoch-1];assert (u['index'],u['obs'],u['owner'])==(ix,y,j)
                if policy=='direct':continue
                ev=e['events'];invalid=[(j,k) for reason,j,k,v in ev if reason=='invalidate'];assert len(set(invalid))==len(invalid)
                if epoch:
                    expected={key for key in store if b['flush'] or u['owner'] in sides[key]};assert set(invalid)==expected and u['before']==len(store) and u['after']==len(store)-len(expected);assert u['affected_directions']==n and u['flush']==b['flush'];assert all(x[0]=='invalidate' for x in ev[:len(invalid)]);c['invalidation_replays']+=1;c['retained_directions_after_update']+=u['after'];c['invalidated_directions']+=len(expected)
                else:assert not invalid
                ledger=Counter()
                for reason,j,k,v in ev:
                    key=(j,k);assert key in sides and type(v)==int and v>0
                    if reason=='insert':assert key not in store;store[key]=v;charge+=v
                    else:assert reason in ('evict','invalidate') and store.pop(key)==v;charge-=v
                    ledger[reason]+=1;assert 0<=charge<=cap;peakb=max(peakb,charge);c['cache_transitions']+=1
                assert charge==e['bytes']==sum(store.values()) and peakb==e['peak_bytes'] and len(store)==e['entries'];m=e['cache'];assert all(m.get(x,0)==ledger[x] for x in ('insert','evict','invalidate'));assert m['requests']==m.get('hits',0)+m.get('misses',0) and m.get('misses',0)==m.get('insert',0)+m.get('rejected',0);counts.update(m);c['epoch_ledgers']+=1
            summary.append(dict(n=n,family=w['family'],seed=w['seed'],noise=p,policy=policy,kib=b['kib'],flush=b['flush'],arithmetic=arith,seconds=b['seconds'],peak_bytes=peakb,cache=dict(counts)))
            c['arm_trajectories']+=1
    assert len(seen)==18
    return dict(n=n,counts=dict(c),rows=summary,receipt_sha256=sha(H/f'receipt{n}.json'))

if __name__=='__main__':
    import sys
    n=int(sys.argv[1]);r=audit(n);r['status']='PASS';r['audit_sha256']=sha(__file__);r['reference_sha256']=sha(H/'reference19.py');(H/f'audit{n}.json').write_text(json.dumps(r,indent=2)+'\n');print(n,r['counts'])
