"""Independent replay: physical basis, integer cell bounds, partitions, intervals."""
import json,hashlib,heapq,sys
from pathlib import Path
from fractions import Fraction as F
from collections import Counter,defaultdict
H=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dot(a,b):return (a&b).bit_count()%2
class Basis:
    def __init__(self,rows):
        self.rows=[];self.piv={}
        for row in rows:
            a=row;c=1<<len(self.rows)
            while a:
                i=(a&-a).bit_length()-1
                if i not in self.piv:self.piv[i]=(a,c);self.rows.append(row);break
                b,t=self.piv[i];a^=b;c^=t
    def coord(self,a):
        c=0
        while a:
            i=(a&-a).bit_length()-1
            if i not in self.piv:return None
            b,t=self.piv[i];a^=b;c^=t
        return c
class AuditBounds:
    def __init__(self,rows,obs,k,p):
        self.k=k;self.rows=rows;self.obs=obs;rate=F(str(p));self.u=rate.numerator;self.v=rate.denominator-self.u;self.m=len(rows);self.level=[]
        self.weights=[self.u**e*self.v**(self.m-e) for e in range(self.m+1)]
        for d in range(k+1):
            groups=defaultdict(list)
            for a,y in zip(rows,obs):groups[a>>d].append((a&((1<<d)-1),y))
            fixed=groups.pop(0,[]);keys=sorted(groups,key=lambda a:(-len(groups[a]),a));basis=Basis(keys);selected=set(basis.rows)
            assert len(selected)<=k-d
            self.level.append((fixed,[(a,Counter(groups[a]),len(groups[a]),a in selected) for a in keys],len(selected),sum(map(len,groups.values()))))
    def bound(self,d,h,mode):
        fixed,groups,rank,nfree=self.level[d];base=self.u**sum(dot(a,h)!=y for a,y in fixed)*self.v**sum(dot(a,h)==y for a,y in fixed);size=1<<(self.k-d)
        if mode=='naive':return base*size*self.u**nfree,base*size*self.v**nfree
        lower=upper=base
        if mode=='weighted':
            items=[]
            for a,terms,n,chosen in groups:
                err=sum(count for (prefix,y),count in terms.items() if dot(prefix,h)!=y);pair=(self.u**err*self.v**(n-err),self.u**(n-err)*self.v**err)
                items.append((abs(n-2*err),a,pair))
            selected=set(Basis([a for _,a,_ in sorted(items,key=lambda x:(-x[0],x[1]))]).rows)
            for _,a,pair in items:
                if a in selected:lower*=sum(pair);upper*=sum(pair);size//=2
                else:lower*=min(pair);upper*=max(pair)
            lo,hi=lower*size,upper*size;old_l,old_u=self.bound(d,h,'independent');assert old_l<=lo<=hi<=old_u
            return lo,hi
        for a,terms,n,chosen in groups:
            err=sum(count for (prefix,y),count in terms.items() if dot(prefix,h)!=y);pair=(self.u**err*self.v**(n-err),self.u**(n-err)*self.v**err)
            if mode=='independent' and chosen:lower*=sum(pair);upper*=sum(pair);size//=2
            else:lower*=min(pair);upper*=max(pair)
        return lower*size,upper*size
    def weight(self,h):return self.weights[sum(dot(h,a)!=b for a,b in zip(self.rows,self.obs))]

def audit(paths):
    counts=Counter();groups=[];ids=set();nonmonotone=Counter()
    for path in paths:
        rec=json.loads(path.read_text());assert rec['protocol_sha256']==sha(H/'PROTOCOL.md');assert (rec['source_sha256']==sha(H/'weighted15.py') and rec['pilot_sha256']==sha(H/'pilot15.py')) if path.name.startswith('weighted') else rec['source_sha256']==sha(H/'pilot15.py')
        for w in rec['worlds']:
            k=w['rank'];p=w['noise'];seed=w['seed'];assert (seed,p) not in ids;ids.add((seed,p));raw=w['world'];tr=w['trace'];rr=raw['initial'].copy();obs=tr['initial_obs'].copy();spent=0;seen=set()
            assert obs==[dot(a,raw['hidden'])^int(u<p) for a,u in zip(rr,raw['uniforms'])]
            for j,t in enumerate(tr['trace']):
                i=t['j'];assert i not in seen and t['query']==raw['pool'][i] and t['cost']==raw['costs'][i];assert t['obs']==dot(t['query'],raw['hidden'])^int(raw['uniforms'][len(raw['initial'])+j]<p);assert t['expands']==(Basis(rr).coord(t['query']) is None);seen.add(i);spent+=t['cost'];rr.append(t['query']);obs.append(t['obs']);counts['trace_steps']+=1
            assert spent==tr['spent']<=24 and not any(spent+raw['costs'][i]<=24 for i in range(48) if i not in seen)
            f=Basis(rr);assert len(f.rows)==k;order=w['order'];assert sorted(order)==list(range(k));co=[f.coord(q) for q in rr];order2=sorted(range(k),key=lambda j:(-sum(q>>j&1 for q in co),j));assert order2==order
            def remap(q):return None if q is None else sum((q>>i&1)<<j for j,i in enumerate(order))
            rows=[remap(q) for q in co];targets=[remap(f.coord(q)) for q in raw['targets']];assert rows==w['rows'] and targets==w['targets'] and obs==w['obs'];assert [dot(q,raw['hidden']) for q in raw['targets']]==w['truth'];b=AuditBounds(rows,obs,k,p)
            exact_weights=[b.weight(h) for h in range(1<<k)] if k<=11 else None
            if exact_weights:
                z=sum(exact_weights);exact=[F(1,2) if q is None else F(sum(a for h,a in enumerate(exact_weights) if dot(h,q)),z) for q in targets];assert list(map(str,exact))==w['exact']
            for arm in w['arms']:
                mode=arm['mode'];heap=[];front={};leaves={};N=0;last=None;split_index=0
                def add(d,h):
                    nonlocal N
                    lo,hi=b.bound(d,h,mode);N+=1;counts['node_bounds']+=1
                    assert 0<lo<=hi
                    if d==k:assert lo==hi==b.weight(h) and h not in leaves;leaves[h]=lo;counts['leaf_weights']+=1
                    else:assert (d,h) not in front;front[d,h]=(lo,hi);heapq.heappush(heap,(-hi,-d,h))
                add(0,0);root=front[0,0]
                for snap in arm['snapshots']:
                    expected=snap['nodes'];assert expected<=snap['budget']
                    while N<expected:
                        d,h=arm['splits'][split_index];assert (d,h) in front
                        if split_index<k:assert (d,h)==(split_index,w['modal']&((1<<d)-1))
                        else:
                            while (-heap[0][1],heap[0][2]) not in front:heapq.heappop(heap)
                            assert (-heap[0][1],heap[0][2])==(d,h)
                        del front[d,h];add(d+1,h);add(d+1,h|(1<<d));split_index+=1;counts['splits']+=1
                    assert N==expected and (N+2>snap['budget'] or not front)
                    # Splits replace precisely one prefix cell by both children, preserving disjoint cover.
                    assert sum(1<<(k-d) for d,h in front)+len(leaves)==1<<k
                    W=sum(leaves.values());L=sum(l for l,u in front.values());U=sum(u for l,u in front.values());assert [str(W),str(L),str(U)]==[snap['weight'],snap['omitted_lower'],snap['omitted_upper']];assert [str(v) for v in root]==[snap['root_lower'],snap['root_upper']]
                    assert len(leaves)==snap['leaves'] and len(front)==snap['frontier'];assert abs(float(F(W,W+U))-snap['retained_lower'])<1e-15 and abs(float(F(W,W+L))-snap['retained_upper'])<1e-15
                    if exact_weights:assert W+L<=z<=W+U;counts['exact_mass_bounds']+=1
                    cov=correct=brier=width=0.
                    for i,(q,saved) in enumerate(zip(targets,snap['queries'])):
                        if q is None:lo=hi=pr=F(1,2)
                        else:
                            W1=sum(v for h,v in leaves.items() if dot(q,h));W0=W-W1;l=[W0,W1];u=[W0,W1]
                            for (d,h),(low,upper) in front.items():
                                if q>>d:u[0]+=upper;u[1]+=upper
                                else:
                                    j=dot(q,h);l[j]+=low;u[j]+=upper
                            lo=max(F(l[1],l[1]+u[0]),F(W1,W+U));hi=min(F(u[1],u[1]+l[0]),F(W1+U,W+U));pr=F(W1,W)
                        assert (lo,hi,pr)==tuple(F(saved[t]) for t in ('lo','hi','visited_probability'));assert 0<=lo<=hi<=1;cert=1 if lo>=F(9,10) else 0 if hi<=F(1,10) else None;assert cert==saved['cert'];wt=w['weights'][i];y=w['truth'][i];cov+=wt*(cert is not None);correct+=wt*(cert==y);brier+=wt*(float(pr)-y)**2;width+=wt*float(hi-lo)
                        if exact_weights:assert lo<=exact[i]<=hi;counts['exact_event_intervals']+=1
                        counts['event_intervals']+=1
                    for a,v in [('coverage',cov),('covered_correct',correct),('brier',brier),('mean_width',width)]:assert abs(snap[a]-v)<1e-12
                    if last is not None and snap['retained_lower']+1e-15<last:nonmonotone[mode]+=1
                    last=snap['retained_lower'];counts['snapshots']+=1;groups.append(dict(d=w['d'],seed=seed,noise=p,mode=mode,**{a:snap[a] for a in ('budget','nodes','leaves','frontier','retained_lower','retained_upper','coverage','covered_correct','brier','mean_width','seconds')}))
                assert split_index==len(arm['splits']) and set(leaves)==set(arm['leaf_ids']);counts['searches']+=1
            print('AUDIT',seed,p,flush=True)
    assert len(ids)==24
    return dict(status='PASS',**counts,retained_lower_decreases=nonmonotone,inputs={p.name:sha(p) for p in paths},source_sha256=sha(Path(__file__)),rows=groups,scope='Independent LSB elimination, integer cell bounds, exact prefix partition, priority replay, leaf weights, event intervals, exact small posterior and trace costs; no independent BP/acquisition policy replay.')
if __name__=='__main__':
    prefix='weighted' if len(sys.argv)>1 and sys.argv[1]=='weighted' else 'receipt';r=audit([H/f'{prefix}{k}.json' for k in (8,21,45)]);(H/('audit_weighted.json' if prefix=='weighted' else 'audit.json')).write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='rows'})
