"""LBF-20H exact dependency agenda, learned scheduling, independent finite audit."""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np
from lbf20f_csp import Case, Solver, D, FULL, bits, generate

WINDOW=16
FEATURES=16
WIDTH=16
SEEDS=(107,109,113)
STEPS=320
BATCH=64
MULTIPLIERS=(1,2,4,8)
BASELINES=("ac_lcv","functional_fallback","singleton_ac","dsatur_fallback")
SCHEDULES=("dense_fifo","sparse_fifo","uniform","heuristic","lookahead")


def matrix(c,dom):
    rel=c.relations()
    return [tuple(((1<<a) if i==j else rel.get((i,j),(FULL,)*D)[a])&dom[j]
                  if dom[i]&(1<<a) else 0 for a in range(D))
            for i in range(c.n) for j in range(c.n)]


def meta(r):
    return (sum(x.bit_count() for x in r),
            sum(1<<i for i,row in enumerate(r) if row),
            r[0]|r[1]|r[2]|r[3])


def compose(left,right,old):
    out=[]
    for a,row in enumerate(left):
        z=0
        for b in bits(row):
            z|=right[b]
        out.append(old[a]&z)
    return tuple(out)


class Agenda:
    """Directed pair relations; a changed input re-enqueues every consumer.

    Every nontrivial initial composition is pending. A revision whose inputs
    have not changed stays valid even if its target shrinks. No-op screening
    uses only sufficient product-domain conditions, never predicted gains.
    """
    def __init__(self,c,dom,sparse=True):
        self.c=c;self.n=c.n;self.dom=dom;self.sparse=sparse
        n=self.n
        self.r=matrix(c,dom);self.m=list(map(meta,self.r))
        self.universal=[tuple(dom[j] if dom[i]&(1<<a) else 0 for a in range(D))
                        for i in range(n) for j in range(n)]
        self.edge=set(c.relations())
        self.queue=collections.deque();self.pending=set()
        self.stats=collections.Counter(revisions=0,updates=0,removed_pairs=0,
                                       diagonal_removed=0,screen_checks=0,screen_skips=0,
                                       candidate_scores=0,lookahead_probes=0,max_pending=0)
        self.contradiction=not all(dom) or any(not any(r) for r in self.r)
        if not self.contradiction:
            for i,k,j in itertools.product(range(n),repeat=3):
                self.push((i,k,j))

    def push(self,t):
        if t in self.pending:return
        if self.sparse:
            i,k,j=t;n=self.n;l=i*n+k;r=k*n+j
            self.stats["screen_checks"]+=1
            # Composition is the entire original endpoint product if a
            # universal operand meets complete endpoint coverage in the other.
            if ((self.r[l]==self.universal[l] and self.m[r][2]==self.dom[j]) or
                (self.r[r]==self.universal[r] and self.m[l][1]==self.dom[i])):
                self.stats["screen_skips"]+=1;return
        self.pending.add(t);self.queue.append(t)
        self.stats["max_pending"]=max(self.stats["max_pending"],len(self.pending))

    def preview(self,t):
        i,k,j=t;n=self.n
        self.stats["revisions"]+=1
        return compose(self.r[i*n+k],self.r[k*n+j],self.r[i*n+j])

    def gain(self,t,new):
        i,_,j=t
        removed=self.m[i*self.n+j][0]-sum(r.bit_count() for r in new)
        return removed*(4 if i==j else 1)+(16 if not any(new) else 0)

    def apply(self,t,new):
        i,k,j=t;n=self.n;p=i*n+j
        old=self.r[p]
        if new==old:return
        removed=self.m[p][0]-sum(r.bit_count() for r in new)
        assert removed>0 and all(a&b==a for a,b in zip(new,old))
        self.stats["updates"]+=1;self.stats["removed_pairs"]+=removed
        if i==j:self.stats["diagonal_removed"]+=removed
        self.r[p]=new;self.m[p]=meta(new)
        if not any(new):
            self.contradiction=True;return
        for v in range(n):
            self.push((i,j,v))
            self.push((v,i,j))

    def feature(self,t):
        i,k,j=t;n=self.n
        l,r,z=(i*n+k,k*n+j,i*n+j)
        dl,dr,dt=(self.m[l],self.m[r],self.m[z])
        di,dk,dj=(self.dom[v].bit_count() for v in (i,k,j))
        p=dl[0]/max(1,di*dk);q=dr[0]/max(1,dk*dj)
        estimate=dt[0]*(1-p*q)**dk
        return [dt[0]/16,p,q,dl[1].bit_count()/max(1,di),dr[2].bit_count()/max(1,dj),
                dl[2].bit_count()/max(1,dk),dr[1].bit_count()/max(1,dk),
                float(i==j),float((i,j) in self.edge),float((i,k) in self.edge),
                float((k,j) in self.edge),float(k==i or k==j),
                di/4,dj/4,dk/4,estimate/16]

    def take(self,k):
        return [self.queue.popleft() for _ in range(min(k,len(self.queue)))]

    def put_back(self,ts):
        self.queue.extendleft(reversed(ts))

    def step(self,policy,rng,net=None,remaining=1):
        if self.contradiction or not self.queue:return
        ts=self.take(1 if policy.endswith("fifo") else min(WINDOW,remaining) if policy=="lookahead" else WINDOW)
        if policy=="lookahead":
            news=[self.preview(t) for t in ts]
            self.stats["lookahead_probes"]+=len(ts)
            gains=[self.gain(t,v) for t,v in zip(ts,news)]
            j=max(range(len(ts)),key=lambda i:(gains[i],-i))
            # Exact no-op previews discharge pending work until an input
            # changes. Every preview is charged, including unselected ones.
            keep=[]
            for i,t in enumerate(ts):
                if i==j or gains[i]==0:self.pending.remove(t)
                else:keep.append(t)
            self.put_back(keep);self.apply(ts[j],news[j]);return
        if policy.endswith("fifo"):j=0
        elif policy=="uniform":j=rng.randrange(len(ts))
        else:
            fs=[self.feature(t) for t in ts]
            self.stats["candidate_scores"]+=len(ts)
            if policy=="heuristic":
                # Density approximation plus missing endpoint support.
                scores=[(f[-1]+f[0]*max(1-f[3],1-f[4]))*(4 if f[7] else 1) for f in fs]
            else:
                assert net is not None
                scores=net.predict(np.asarray(fs,dtype=np.float32))
            j=max(range(len(ts)),key=lambda i:(float(scores[i]),-i))
        t=ts[j];self.pending.remove(t)
        self.put_back(ts[:j]+ts[j+1:])
        self.apply(t,self.preview(t))

    def advance(self,policy,budget,rng,net=None):
        while self.stats["revisions"]<budget and self.queue and not self.contradiction:
            self.step(policy,rng,net,budget-self.stats["revisions"])
        assert self.stats["revisions"]<=budget

    def diagonal(self):
        if self.contradiction:return (0,)*self.n
        return tuple(sum(1<<a for a in range(D) if self.r[i*self.n+i][a]&(1<<a))
                     for i in range(self.n))

    def complete(self):
        return self.contradiction or not self.queue


def reference(c,dom):
    """Independent synchronous fixed point, explicit shared-value enumeration."""
    n=c.n;r=matrix(c,dom)
    while True:
        if any(not any(v) for v in r):return None
        new=[]
        for i,j in itertools.product(range(n),repeat=2):
            new.append(tuple(sum(1<<b for b in range(D) if r[i*n+j][a]&(1<<b) and
                                  all(any(r[i*n+k][a]&(1<<u) and r[k*n+j][u]&(1<<b)
                                          for u in range(D)) for k in range(n)))
                             for a in range(D)))
        if new==r:return new
        r=new


def color_cases(split,n,count,seed):
    rng=random.Random(seed);out=[]
    neq=tuple(7^(1<<a) if a<3 else 0 for a in range(D))
    for i in range(count):
        p=[rng.randrange(3) for _ in range(n)]
        edges=[]
        for a,b in itertools.combinations(range(n),2):
            if i%2==0 and p[a]==p[b]:continue
            if rng.random()<(.70 if i%2==0 else .48):
                edges.append((a,b,neq))
        out.append(Case(f"{split}_color_{i:03d}","color",n,(7,)*n,tuple(edges)))
    return out


def functional(c,dom):
    # Exact anchor propagation on total bijections; None outside fragment.
    s=Solver(c)
    if any(any(row.bit_count()!=1 for row in r) or len(set(r))!=D for r in s.rel.values()):
        return None
    seen=set();out=[0]*c.n
    for i in range(c.n):
        if i in seen:continue
        component={i};q=[i]
        for u in q:
            for v in s.nb[u]:
                if v not in component:component.add(v);q.append(v)
        seen|=component
        for a in bits(dom[i]):
            vals={i:a};ok=True
            for u in q:
                if not dom[u]&(1<<vals[u]):ok=False;break
                for j in s.nb[u]:
                    z=next(bits(s.rel[u,j][vals[u]]))
                    if j in vals and vals[j]!=z:ok=False;break
                    vals[j]=z
                if not ok:break
            if ok:
                for j,v in vals.items():out[j]|=1<<v
        if not out[i]:return (0,)*c.n
    return tuple(out)


def singleton(s,dom,stats):
    out=[]
    for i,m in enumerate(dom):
        keep=0
        for a in bits(m):
            q=list(dom);q[i]=1<<a
            stats["singleton_probes"]+=1
            if s.ac(tuple(q),stats) is not None:keep|=1<<a
        out.append(keep)
    return tuple(out)


def dsatur(c,dom,budget=64):
    """Exact finite DFS specialized to unequal-color edges, with cap."""
    n=c.n;nb=[set() for _ in range(n)]
    for i,j,r in c.edges:
        assert all(bool(r[a]&(1<<b))==(a!=b) for a in bits(dom[i]) for b in bits(dom[j]))
        nb[i].add(j);nb[j].add(i)
    vals=[-1]*n;nodes=0;capped=False
    def visit():
        nonlocal nodes,capped
        if nodes>=budget:capped=True;return None
        nodes+=1
        if all(v>=0 for v in vals):return tuple(vals)
        def forbidden(i):return {vals[j] for j in nb[i] if vals[j]>=0}
        i=max((i for i,v in enumerate(vals) if v<0),
              key=lambda i:(len(forbidden(i)),len(nb[i]),-i))
        f=forbidden(i)
        for a in bits(dom[i]):
            if a in f:continue
            vals[i]=a;v=visit();vals[i]=-1
            if v is not None:return v
            if capped:return None
        return None
    v=visit()
    if v is not None:assert c.valid(v)
    return {"status":"solved" if v is not None else "budget" if capped else "refuted",
            "solution":v,"nodes":nodes,"arc_revisions":0}


def audit():
    counts=collections.Counter()
    for c in generate("audit_h",4,96,2026112550,allow_unsat=True):
        worlds=list(itertools.product(range(D),repeat=c.n))
        good=[v for v in worlds if c.valid(v)]
        want=reference(c,c.domains)
        counts["systems"]+=1;counts["worlds"]+=len(worlds);counts["feasible_worlds"]+=len(good)
        for arm in SCHEDULES:
            a=Agenda(c,c.domains,arm!="dense_fifo");rng=random.Random(19)
            # Check soundness after each executed action on tiny cases.
            while not a.complete():
                a.step(arm,rng,remaining=WINDOW)
                assert a.stats["revisions"]<100000
                assert not good or not a.contradiction
                for v in good:
                    assert all(a.r[i*c.n+j][v[i]]&(1<<v[j]) for i in range(c.n) for j in range(c.n))
                counts["solution_preservation_checks"]+=len(good)
            assert (want is None)==a.contradiction,(c.root,arm)
            if want is not None:assert want==a.r,(c.root,arm)
            counts["fixed_point_comparisons"]+=1
    higher=[]
    for d in (3,4):
        n=d+1;mask=(1<<d)-1
        neq=tuple(mask^(1<<a) if a<d else 0 for a in range(D))
        c=Case(f"K{n}","clique",n,(mask,)*n,tuple((i,j,neq) for i,j in itertools.combinations(range(n),2)))
        assert reference(c,c.domains)==matrix(c,c.domains)
        worlds=list(itertools.product(range(d),repeat=n))
        assert not any(c.valid(v) for v in worlds)
        z=dsatur(c,c.domains,4096);assert z["status"]=="refuted"
        higher.append({"variables":n,"colors":d,"worlds":len(worlds),"pc_unchanged":True,
                       "dsatur_nodes":z["nodes"]})
    return {**dict(counts),"mismatches":0,"higher_order":higher}


class Net:
    def __init__(self,arm,seed):
        self.arm=arm;rng=np.random.default_rng(seed)
        self.w=[rng.normal(0,.08,(FEATURES,1)).astype(np.float32),np.zeros(1,dtype=np.float32)] if arm=="linear" else [
            rng.normal(0,.18,(FEATURES,WIDTH)).astype(np.float32),np.zeros(WIDTH,dtype=np.float32),
            rng.normal(0,.18,(WIDTH,1)).astype(np.float32),np.zeros(1,dtype=np.float32)]

    def predict(self,x):
        if self.arm=="linear":return (x@self.w[0]+self.w[1]).reshape(-1)
        return (np.tanh(x@self.w[0]+self.w[1])@self.w[2]+self.w[3]).reshape(-1)

    def fit(self,x,y,seed):
        rng=np.random.default_rng(seed+9001)
        m=[np.zeros_like(z) for z in self.w];v=[np.zeros_like(z) for z in self.w]
        for t in range(1,STEPS+1):
            ids=rng.integers(len(x),size=BATCH);a=x[ids];b=y[ids,None]
            if self.arm=="linear":
                p=a@self.w[0]+self.w[1];g=2*(p-b)/BATCH
                gs=[a.T@g,g.sum(0)]
            else:
                h=np.tanh(a@self.w[0]+self.w[1]);p=h@self.w[2]+self.w[3]
                g=2*(p-b)/BATCH;gh=(g@self.w[2].T)*(1-h*h)
                gs=[a.T@gh,gh.sum(0),h.T@g,g.sum(0)]
            for i,(z,g) in enumerate(zip(self.w,gs)):
                g=g+1e-4*z;m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g
                z-=.003*(m[i]/(1-.9**t))/(np.sqrt(v[i]/(1-.999**t))+1e-8)
        return float(np.mean((self.predict(x)-y)**2))

    def save(self,path):
        data=(json.dumps({"architecture":self.arm,"weights":[z.tolist() for z in self.w]},
                         sort_keys=True,separators=(",",":"))+"\n").encode()
        path.write_bytes(data);return hashlib.sha256(data).hexdigest()


def collect(train):
    x=[];y=[];roots=[];rng=random.Random(2026112551)
    for c in train:
        dom=Solver(c).ac(c.domains)
        if dom is None:continue
        a=Agenda(c,dom)
        for step in range(24):
            if a.complete():break
            ts=a.take(WINDOW)
            fs=[a.feature(t) for t in ts]
            news=[a.preview(t) for t in ts]
            gains=[a.gain(t,v) for t,v in zip(ts,news)]
            x.extend(fs);y.extend(math.log1p(g) for g in gains);roots.extend([c.root]*len(ts))
            j=max(range(len(ts)),key=lambda i:(gains[i],-i)) if step%2==0 else rng.randrange(len(ts))
            # Keep the other pending candidates; no evaluation roots seen.
            a.pending.remove(ts[j]);a.put_back(ts[:j]+ts[j+1:])
            a.apply(ts[j],news[j])
    return np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.float32),roots


def row(c,arm,mult,initial,filtered,exact,pre,search,extra):
    assert all(a&b==a for a,b in zip(exact,filtered)),(c.root,arm,mult)
    sat=any(exact)
    assert search["status"]!="refuted" or not sat,(c.root,arm)
    assert search["status"]!="solved" or (sat and c.valid(search["solution"]))
    total=sum(x.bit_count() for x in initial)-sum(x.bit_count() for x in exact)
    remaining=sum(x.bit_count() for x in filtered)-sum(x.bit_count() for x in exact)
    assert 0<=remaining<=total
    return {"root":c.root,"family":c.family,"n":c.n,"arm":arm,"multiplier":mult,
            "sat":sat,"status":search["status"],"nodes":search["nodes"],
            "solution":search["solution"],"initial_excess":total,"remaining_excess":remaining,
            "removed_excess":total-remaining,"filtered":filtered,"exact":exact,
            "pre_seconds":pre,"search_seconds":search["seconds"],
            "total_seconds":pre+search["seconds"],"search_arc_revisions":search["arc_revisions"],**extra}


def measure_search(s,dom):
    t=time.perf_counter()
    z=s.solve(dom,budget=64)
    return {**z,"seconds":time.perf_counter()-t}


def evaluate(c,exact,arm,net=None):
    t=time.perf_counter();s=Solver(c);root_stats=collections.Counter()
    dom=s.ac(c.domains,root_stats);initial=dom if dom is not None else (0,)*c.n
    if arm in BASELINES:
        q=initial
        if dom is not None and arm=="functional_fallback":
            f=functional(c,dom);q=dom if f is None else f
        elif dom is not None and arm=="singleton_ac":q=singleton(s,dom,root_stats)
        pre=time.perf_counter()-t;ts=time.perf_counter()
        if dom is not None and arm=="dsatur_fallback" and c.family=="color":
            z=dsatur(c,dom)
            z["seconds"]=time.perf_counter()-ts
        else:z=measure_search(s,q)
        return [row(c,arm,0,initial,q,exact,pre,z,{"revisions":0,"completed":None,
                    "root_arc_revisions":root_stats["arc_revisions"],"singleton_probes":root_stats["singleton_probes"]})]
    a=Agenda(c,initial,arm!="dense_fifo")
    pre=time.perf_counter()-t
    rng=random.Random(int.from_bytes(hashlib.sha256(c.root.encode()).digest()[:8],"big"))
    rows=[]
    for mult in MULTIPLIERS:
        t=time.perf_counter()
        a.advance(arm,mult*c.n*c.n,rng,net)
        q=a.diagonal();pre+=time.perf_counter()-t
        z=measure_search(s,q)
        rows.append(row(c,arm,mult,initial,q,exact,pre,z,
                        {**dict(a.stats),"completed":a.complete(),"root_arc_revisions":root_stats["arc_revisions"]}))
    return rows


def aggregate(rows):
    den=sum(r["initial_excess"] for r in rows);num=sum(r["removed_excess"] for r in rows)
    info=[r for r in rows if r["initial_excess"]]
    return {"roots":len(rows),"informative_roots":len(info),"initial_excess":den,
            "removed_excess":num,"pooled_excess_removed":num/den if den else None,
            "macro_excess_removed":statistics.mean(r["removed_excess"]/r["initial_excess"] for r in info) if info else None,
            "resolved":sum(r["status"] in ("solved","refuted") for r in rows),
            "mean_nodes":statistics.mean(r["nodes"] for r in rows),
            "mean_revisions":statistics.mean(r["revisions"] for r in rows),
            "mean_pre_ms":1000*statistics.mean(r["pre_seconds"] for r in rows),
            "mean_total_ms":1000*statistics.mean(r["total_seconds"] for r in rows),
            "mean_candidate_scores":statistics.mean(r.get("candidate_scores",0) for r in rows),
            "mean_screen_checks":statistics.mean(r.get("screen_checks",0) for r in rows),
            "mean_pair_deletions":statistics.mean(r.get("removed_pairs",0) for r in rows),
            "completed":sum(r["completed"] is True for r in rows)}


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    au=audit();print("AUDIT",json.dumps(au),flush=True)
    train=generate("train_h",6,96,2026112552,allow_unsat=True)+color_cases("train_h",6,24,2026112553)
    cohorts={"dev8":generate("dev_h",8,48,2026112554,allow_unsat=True),
             "stress12":generate("stress_h",12,36,2026112555,allow_unsat=True),
             "color12":color_cases("stress_h",12,24,2026112556)}
    all_cases={"train":train,**cohorts}
    ids=[(c.n,c.domains,c.edges) for cs in all_cases.values() for c in cs]
    # Exact labeled table identities can repeat in generated tiny color cases.
    # Require disjoint train/eval identities, not unique IDs within a cohort.
    train_ids={(c.n,c.domains,c.edges) for c in train}
    assert not train_ids&{(c.n,c.domains,c.edges) for cs in cohorts.values() for c in cs}
    task=(json.dumps({k:[c.__dict__ for c in cs] for k,cs in all_cases.items()},
                     sort_keys=True,separators=(",",":"))+"\n").encode()
    (out/"tasks.json").write_bytes(task)
    x,y,roots=collect(train)
    assert set(roots)<={c.root for c in train} and len(x)>0
    np.savez_compressed(out/"train_receipts.npz",x=x,y=y,root=np.asarray(roots))
    fits=[];nets={}
    for seed in SEEDS:
        for arch in ("linear","mlp"):
            net=Net(arch,seed);t=time.perf_counter();loss=net.fit(x,y,seed);sec=time.perf_counter()-t
            arm=f"{arch}_{seed}";nets[arm]=net
            sha=net.save(out/f"{arm}.json")
            fits.append({"architecture":arch,"seed":seed,"parameters":sum(z.size for z in net.w),
                         "steps":STEPS,"batch_size":BATCH,"checkpoint_sha256":sha,
                         "fit_seconds":sec,"train_mse":loss})
    rows=[];summary={}
    for name,cs in cohorts.items():
        truth={c.root:Solver(c).supports(c.domains) for c in cs}
        arms=[*BASELINES,*SCHEDULES,*nets]
        local=[]
        for i,c in enumerate(cs):
            off=i%len(arms)
            for arm in arms[off:]+arms[:off]:
                local.extend(evaluate(c,truth[c.root],arm,nets.get(arm)))
        rows.extend({**r,"cohort":name} for r in local)
        summary[name]={}
        for arm in arms:
            summary[name][arm]={str(m):aggregate([r for r in local if r["arm"]==arm and r["multiplier"]==m])
                                for m in ((0,) if arm in BASELINES else MULTIPLIERS)}
        # Keep seed-level data; summarize fixed primary B=2n² without choosing winners.
        print("COHORT",name,json.dumps({a:summary[name][a]["0" if a in BASELINES else "2"] for a in arms}),flush=True)
    for f in fits:
        arm=f'{f["architecture"]}_{f["seed"]}'
        f["cohorts"]={k:v[arm] for k,v in summary.items()}
    full={"schema":"lbf20h.revision-agenda.v1","scope":"development_only","audit":au,
          "task_sha256":hashlib.sha256(task).hexdigest(),
          "train_receipts_sha256":hashlib.sha256((out/"train_receipts.npz").read_bytes()).hexdigest(),
          "train_roots":len(train),"train_frame_roots":len(set(roots)),"train_candidates":len(x),
          "positive_gain_rate":float(np.mean(y>0)),"window":WINDOW,"multipliers":MULTIPLIERS,
          "rows":len(rows),"fits":fits,"cohorts":summary}
    (out/"summary.json").write_text(json.dumps(full,indent=2)+"\n")
    (out/"rows.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
    print("FIT_RECEIPTS",json.dumps([{k:v for k,v in f.items() if k!="cohorts"} for f in fits]),flush=True)
    print("COUNTS",json.dumps({k:v for k,v in full.items() if k not in ("fits","cohorts")}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);main(p.parse_args().out)
