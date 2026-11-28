"""Exact integer disjoint-subtree posterior repair."""
import hashlib,heapq,json,time,sys
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from base13 import Frame,world,run,dot
from bp import bp
H=Path(__file__).resolve().parent
MODES=('naive','grouped','independent');BUDGETS=(256,1024,4096)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def parity(a):return int(a).bit_count()&1
def remap(q,order):return sum(((q>>i)&1)<<j for j,i in enumerate(order))
class Bounds:
    def __init__(self,rows,obs,r,p,mode):
        self.r=r;self.mode=mode;self.counts=Counter(zip(rows,obs));self.m=len(rows);p=F(str(p));u=p.numerator;v=p.denominator-u
        self.pu=[u**e for e in range(self.m+1)];self.pv=[v**e for e in range(self.m+1)];self.exact=[self.pu[e]*self.pv[self.m-e] for e in range(self.m+1)];self.level=[]
        for d in range(r+1):
            mask=(1<<d)-1;fixed=[];groups=defaultdict(list);nf=0
            for (a,y),n in self.counts.items():
                rest=a&~mask;term=(a&mask,y,n)
                if rest:groups[rest].append(term);nf+=n
                else:fixed.append(term)
            piv={};selected=set()
            for a in sorted(groups,key=lambda a:(-sum(t[2] for t in groups[a]),a)):
                v=a
                while v and v.bit_length()-1 in piv:v^=piv[v.bit_length()-1]
                if v:piv[v.bit_length()-1]=v;selected.add(a)
            self.level.append((fixed,[(ts,sum(t[2] for t in ts),a in selected) for a,ts in groups.items()],nf,len(selected)))
    def bound(self,d,h):
        fixed,groups,nfree,rank=self.level[d];e=sum(n for a,y,n in fixed if parity(a&h)!=y);f=self.pu[e]*self.pv[self.m-nfree-e]
        if self.mode=='naive':return f*self.pu[nfree]<<(self.r-d),f*self.pv[nfree]<<(self.r-d)
        lo=hi=f
        for terms,n,selected in groups:
            e=sum(c for pre,y,c in terms if parity(pre&h)!=y);v0=self.pu[e]*self.pv[n-e];v1=self.pu[n-e]*self.pv[e]
            if self.mode=='independent' and selected:lo*=v0+v1;hi*=v0+v1
            else:lo*=min(v0,v1);hi*=max(v0,v1)
        free=self.r-d-(rank if self.mode=='independent' else 0)
        return lo<<free,hi<<free
    def weight(self,h):return self.exact[sum(n for (a,y),n in self.counts.items() if parity(a&h)!=y)]
def search(rows,obs,r,p,mode,seed,targets,truth,wt):
    start=time.perf_counter();bd=Bounds(rows,obs,r,p,mode);heap=[];leaves={};splits=[];nodes=L=U=W=0;Ws=[0]*len(targets);snaps=[]
    def add(d,h):
        nonlocal nodes,L,U,W
        lo,hi=bd.bound(d,h);nodes+=1;assert lo<=hi
        if d==r:
            assert lo==hi==bd.weight(h) and h not in leaves;leaves[h]=lo;W+=lo
            for j,q in enumerate(targets):
                if q is not None and parity(h&q):Ws[j]+=lo
        else:heapq.heappush(heap,(-hi,-d,h,lo));L+=lo;U+=hi
    def split(v):
        nonlocal L,U
        neg,nd,h,lo=v;d=-nd;L-=lo;U+=neg;splits.append((d,h));add(d+1,h);add(d+1,h|(1<<d))
    add(0,0);root=bd.bound(0,0)
    for d in range(r):
        i=next(i for i,v in enumerate(heap) if -v[1]==d and v[2]==seed&((1<<d)-1));v=heap[i];heap[i]=heap[-1];heap.pop();heapq.heapify(heap);split(v)
    def snapshot(budget):
        queries=[];coverage=correct=brier=width=0.
        for j,q in enumerate(targets):
            if q is None:lo=hi=pr=F(1,2)
            else:
                l1=Ws[j];l0=W-l1;u1=l1;u0=l0
                for neg,nd,h,low in heap:
                    d=-nd;upper=-neg
                    if q>>d:u1+=upper;u0+=upper
                    elif parity(q&h):l1+=low;u1+=upper
                    else:l0+=low;u0+=upper
                lo=F(l1,l1+u0);hi=F(u1,u1+l0);lo=max(lo,F(Ws[j],W+U));hi=min(hi,F(Ws[j]+U,W+U));pr=F(Ws[j],W)
            assert 0<=lo<=hi<=1
            cert=1 if lo>=F(9,10) else 0 if hi<=F(1,10) else None
            if cert is not None:coverage+=wt[j];correct+=wt[j]*(cert==truth[j])
            brier+=wt[j]*(float(pr)-truth[j])**2;width+=wt[j]*float(hi-lo)
            queries.append(dict(lo=str(lo),hi=str(hi),visited_probability=str(pr),cert=cert))
        snaps.append(dict(budget=budget,nodes=nodes,leaves=len(leaves),frontier=len(heap),retained_lower=float(F(W,W+U)),retained_upper=float(F(W,W+L)),weight=str(W),omitted_upper=str(U),omitted_lower=str(L),coverage=coverage,covered_correct=correct,brier=brier,mean_width=width,queries=queries,seconds=time.perf_counter()-start,root_lower=str(root[0]),root_upper=str(root[1])))
    for b in BUDGETS:
        while heap and nodes+2<=b:split(heapq.heappop(heap))
        snapshot(b)
    return dict(mode=mode,snapshots=snaps,splits=splits,leaf_ids=list(leaves))
def fixture():
    rng=np.random.default_rng(150015);checks=0
    for r in (2,3,4,5,6):
        for _ in range(8):
            rows=list(map(int,rng.integers(1,1<<r,4*r)));obs=list(map(int,rng.integers(0,2,len(rows))))
            for p in (.15,.35):
                bds=[Bounds(rows,obs,r,p,m) for m in MODES];exact=[bds[0].weight(h) for h in range(1<<r)]
                for d in range(r+1):
                    for h in range(1<<d):
                        z=sum(exact[x] for x in range(h,1<<r,1<<d));bounds=[b.bound(d,h) for b in bds];assert all(l<=z<=u for l,u in bounds);assert bounds[0][0]<=bounds[1][0]<=bounds[2][0] and bounds[0][1]>=bounds[1][1]>=bounds[2][1];checks+=1
    return checks
def main():
    checks=fixture();print('fixture cells',checks,flush=True);out=[];r0=int(sys.argv[1])
    for i in range(410,414):
        w=world(r0,i)
        for p in (.15,.35):
            trace,_=run(w,p,'induced_fixed');rr=w['initial']+[t['query'] for t in trace['trace']];obs=trace['initial_obs']+[t['obs'] for t in trace['trace']];f=Frame(rr);r=len(f.rows);rows=[f.coord(a) for a in rr];qs=[f.coord(a) for a in w['targets']];cnt=Counter(rows);order=sorted(range(r),key=lambda j:(-sum(n for q,n in cnt.items() if q>>j&1),j));mu,_=bp(rows,obs,p,r);seed=remap(sum(int(mu[j]<0)<<j for j in range(r)),order);rows=[remap(q,order) for q in rows];qs=[None if q is None else remap(q,order) for q in qs];truth=[dot(q,w['hidden']) for q in w['targets']];wt=w['weights'];arms=[search(rows,obs,r,p,m,seed,qs,truth,wt) for m in MODES];exact=None
            if r<=11:
                b=Bounds(rows,obs,r,p,'naive');weights=[b.weight(h) for h in range(1<<r)];z=sum(weights);exact=[str(F(1,2) if q is None else F(sum(v for h,v in enumerate(weights) if parity(h&q)),z)) for q in qs]
                for a in arms:
                    for s in a['snapshots']:
                        for x,y in zip(s['queries'],exact):assert F(x['lo'])<=F(y)<=F(x['hi'])
            out.append(dict(seed=w['seed'],d=w['d'],rank=r,noise=p,rows=rows,obs=obs,targets=qs,truth=truth,weights=wt,order=order,modal=seed,trace=trace,world=w,exact=exact,arms=arms));print(w['d'],w['seed'],p,[(a['mode'],round(a['snapshots'][-1]['retained_lower'],5),round(a['snapshots'][-1]['coverage'],3),a['snapshots'][-1]['leaves']) for a in arms],flush=True)
            (H/f'receipt{r0}.json').write_text(json.dumps(dict(evidence='development',protocol_sha256=sha(H/'PROTOCOL.md'),source_sha256=sha(__file__),fixture_cells=checks,worlds=out),separators=(',',':'))+'\n')
if __name__=='__main__':main()
