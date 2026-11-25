"""LBF-20K: exact cutset conditioning with label-preserving residual quotienting.

Standard-library CPU. No oracle access in planning or inference. Branch caps
are checked before enumeration; a cap never means unsatisfiable.
"""
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

from lbf20f_csp import Case, Solver, D, FULL, bits, generate
from lbf20i_scope import minfill, make_cases, lift, project
from lbf20j_separator import Tree, dfs_support, exact_from_worlds, conditions

BAG_CAP=6
BRANCH_CAP=256
ARMS=("dfs_cover","direct_tree","degree_grouped","minfill_recompute",
      "minfill_cached","minfill_grouped")


def zero(c):
    return (0,)*c.n


def residual(c,dom,cut):
    keep=tuple(i for i in range(c.n) if i not in cut)
    ids={v:i for i,v in enumerate(keep)}
    rc=Case(c.root,c.family,len(keep),tuple(dom[i] for i in keep),
            tuple((ids[i],ids[j],r) for i,j,r in c.edges if i in ids and j in ids))
    return rc,keep


def width(c):
    return max((len(b)-1 for _,b in minfill(c)),default=-1)


def plan(c,dom,method,bag_cap=BAG_CAP,branch_cap=BRANCH_CAP,forced=()):
    cut=set(forced)
    stats=collections.Counter()
    def cost(cut):
        rc,_=residual(c,dom,cut)
        order=minfill(rc)
        stats["width_plans"]+=1
        return max((len(b) for _,b in order),default=0),sum(D**len(b) for _,b in order)
    w,slots=cost(cut)
    while w>bag_cap:
        candidates=[i for i in range(c.n) if i not in cut]
        if not candidates:raise AssertionError("empty residual cannot be too wide")
        if method=="degree":
            deg=collections.Counter()
            for i,j,_ in c.edges:
                if i not in cut and j not in cut:deg[i]+=1;deg[j]+=1
            v=min(candidates,key=lambda i:(-deg[i],dom[i].bit_count(),i))
            nxt=cut|{v};w,slots=cost(nxt)
        else:
            choices=[]
            for i in candidates:
                nw,ns=cost(cut|{i})
                choices.append((nw,dom[i].bit_count(),ns,i))
            w,_,slots,v=min(choices);nxt=cut|{v}
        if math.prod(dom[i].bit_count() for i in nxt)>branch_cap:
            return None,stats
        cut=nxt
    if math.prod(dom[i].bit_count() for i in cut)>branch_cap:return None,stats
    return tuple(sorted(cut)),stats


def restrict(c,dom,cut,values,keep,rel):
    assigned=dict(zip(cut,values))
    for i,j,r in c.edges:
        if i in assigned and j in assigned and not r[assigned[i]]&(1<<assigned[j]):
            return None
    d=[]
    for i in keep:
        m=dom[i]
        for j,a in assigned.items():
            if (j,i) in rel:m&=rel[j,i][a]
        if not m:return None
        d.append(m)
    return tuple(d)


class Cutset:
    def __init__(self,c,dom,method,mode,bag_cap=BAG_CAP,branch_cap=BRANCH_CAP,forced=()):
        self.c=c;self.dom=dom;self.mode=mode
        self.stats=collections.Counter()
        self.cut,stats=plan(c,dom,method,bag_cap,branch_cap,forced)
        self.stats.update(stats)
        self.available=self.cut is not None
        self.groups=[];self.tree=None;self.keep=()
        if not self.available:return
        rc,self.keep=residual(c,dom,set(self.cut))
        self.ids={v:i for i,v in enumerate(self.keep)}
        self.tree=Tree(rc,rc.domains) if self.keep else None
        assert self.tree is None or (self.tree.available and self.tree.width+1<=bag_cap)
        rel=c.relations()
        groups={}
        for vals in itertools.product(*(tuple(bits(dom[i])) for i in self.cut)):
            self.stats["enumerated_branches"]+=1
            d=restrict(c,dom,self.cut,vals,self.keep,rel)
            if d is None:
                self.stats["local_rejected"]+=1;continue
            self.stats["compatible_branches"]+=1
            if mode=="grouped":
                groups.setdefault(d,[]).append(vals)
            else:
                self.groups.append((d,[vals]))
        if mode=="grouped":self.groups=list(groups.items())
        self.stats["residual_groups"]=len(self.groups)
        self.cache=[]
        if mode!="recompute":
            for d,labels in self.groups:self.cache.append(self.calibrate(d,{}))
            self.stats["feasible_groups"]=sum(x is not None for x in self.cache)

    def calibrate(self,d,extra):
        self.stats["branch_calibrations"]+=1
        if self.tree is None:return []
        evidence={i:m for i,m in enumerate(d) if m!=self.tree.dom[i]}
        for i,m in extra.items():evidence[i]=evidence.get(i,self.tree.dom[i])&m
        return self.tree.calibrate(evidence,self.stats)

    def answer(self,cond,*,with_witness=False):
        assert self.available
        out=[0]*self.c.n;witness=None
        ext={self.ids[i]:m for i,m in cond.items() if i in self.ids}
        for j,(d,labels) in enumerate(self.groups):
            compatible=[v for v in labels if all(cond.get(i,FULL)&(1<<a) for i,a in zip(self.cut,v))]
            if not compatible:
                self.stats["groups_skipped_label"]+=1;continue
            if self.mode=="recompute":
                loc=self.calibrate(d,ext)
            else:
                loc=self.cache[j]
                for i,m in ext.items():
                    assert m.bit_count()==1
                    if loc is None:break
                    self.stats["conditioned_branches"]+=1
                    loc=self.tree.conditioned(loc,i,next(bits(m)),self.stats)
            if loc is None:continue
            for vals in compatible:
                for i,a in zip(self.cut,vals):out[i]|=1<<a
            if self.tree is not None:
                sup=self.tree.supports(loc)
                for i,m in zip(self.keep,sup):out[i]|=m
            if with_witness and witness is None:
                w=[-1]*self.c.n
                for i,a in zip(self.cut,compatible[0]):w[i]=a
                if self.tree is not None:
                    v=self.tree.witness(loc)
                    for i,a in zip(self.keep,v):w[i]=a
                witness=tuple(w)
                assert self.c.valid(witness)
                assert all(m&(1<<witness[i]) for i,m in cond.items())
        ans=tuple(out)
        if with_witness:
            assert bool(any(ans))==(witness is not None)
            return ans,witness
        return ans

    def logical_bits(self):
        # Nominal full bag-table allocation in the cached branch family.
        return 0 if self.mode=="recompute" or self.tree is None else self.tree.storage()*len(self.groups)


def dfs_cover(s,dom,cond,stats):
    """Exact support from complete DFS; each solution covers all its values."""
    d=tuple(m&cond.get(i,FULL) for i,m in enumerate(dom))
    ac=s.ac(d,stats)
    if ac is None:return zero(s.c)
    ans=[0]*s.c.n
    for i,m in enumerate(ac):
        for a in bits(m):
            if ans[i]&(1<<a):continue
            q=list(ac);q[i]=1<<a
            stats["support_probes"]+=1
            r=s.solve(tuple(q))
            stats["dfs_nodes"]+=r["nodes"]
            stats["arc_revisions"]+=r["arc_revisions"]
            assert r["status"] in ("solved","refuted")
            if r["solution"] is not None:
                for j,v in enumerate(r["solution"]):ans[j]|=1<<v
    return tuple(ans)


def audit():
    st=collections.Counter()
    for n,count,seed in ((4,32,2026112580),(5,24,2026112581)):
        for c in generate("audit_k",n,count,seed,allow_unsat=True):
            worlds=list(itertools.product(range(D),repeat=n))
            good=[v for v in worlds if c.valid(v)]
            # Force actual cutset branches even on a small-width graph.
            forced=(0,1) if n==5 else (0,)
            engines=[Cutset(c,c.domains,"minfill",mode,bag_cap=3,forced=forced)
                     for mode in ("recompute","cached","grouped")]
            assert all(x.available for x in engines)
            qs=[{}]+[{i:1<<a} for i in range(n) for a in range(D)]
            qs += [{0:1<<a,n-1:1<<b} for a in range(D) for b in range(D)]
            for cond in qs:
                ws=[v for v in good if all(m&(1<<v[i]) for i,m in cond.items())]
                expect=exact_from_worlds(c,ws)
                for e in engines:
                    ans,w=e.answer(cond,with_witness=True)
                    assert ans==expect,(c.root,e.mode,cond,ans,expect)
                    assert (w is not None)==bool(ws)
                    st["branch_union_checks"]+=1
                assert dfs_cover(Solver(c),c.domains,cond,collections.Counter())==expect
                st["condition_queries"]+=1
            st["systems"]+=1;st["worlds"]+=len(worlds)
    # Equal residual signatures with different cutset labels must not merge
    # incompatible label evidence. Also exercise disconnected residuals.
    c=Case("alias_audit","audit",4,(15,15,15,15),((0,1,(3,3,12,12)),(2,3,(15,15,15,15))))
    e=Cutset(c,c.domains,"minfill","grouped",forced=(0,))
    assert e.stats["residual_groups"]==2 and e.stats["compatible_branches"]==4
    assert e.answer({0:2})==(2,3,15,15)
    assert e.answer({1:4})==(12,4,15,15)
    assert e.answer({0:2,1:4})==zero(c)
    # Cap is explicit, never an empty union masquerading as UNSAT.
    cap=Cutset(c,c.domains,"minfill","grouped",forced=(0,1),branch_cap=1)
    assert not cap.available
    st["alias_checks"]=4
    return {**dict(st),"mismatches":0}


def fresh_cases():
    rng=random.Random(2026112582);out=[]
    for i in range(18):
        a=6+i//6;b=8+i//6;n=a+b
        perm=rng.sample(range(n),n)
        left=perm[:a];right=perm[a:]
        plant=[rng.randrange(D) for _ in range(n)]
        edges=[]
        family="aliased_bipartite" if i%2==0 else "general_bipartite"
        for u in left:
            for v in right:
                r=[]
                for x in range(D if family=="general_bipartite" else 2):
                    m=sum(1<<z for z in range(D) if rng.random()<.75)
                    r.append(m or (1<<rng.randrange(D)))
                if family=="aliased_bipartite":
                    r[plant[u]//2]|=1<<plant[v]
                    r=[r[x//2] for x in range(D)]
                else:r[plant[u]]|=1<<plant[v]
                if u<v:edges.append((u,v,tuple(r)))
                else:edges.append((v,u,tuple(sum(1<<x for x in range(D) if r[x]&(1<<z)) for z in range(D))))
        c=Case(f"fresh_k_{i:02d}",family,n,(FULL,)*n,tuple(sorted(edges)))
        assert c.valid(plant) and width(c)>BAG_CAP-1
        out.append(c)
    return out


def query_list(c,dom):
    qs=[{}]+[{v:1<<a} for v,a in conditions(c,dom)]
    # One joint condition on two public variables when available.
    vs=sorted((i for i,m in enumerate(dom) if m.bit_count()>1),
              key=lambda i:(dom[i].bit_count(),-len(Solver(c).nb[i]),i))
    if len(vs)>1:
        i,j=vs[:2];qs.append({i:1<<next(bits(dom[i])),j:1<<next(bits(dom[j]))})
    return qs


def run(c,arm,dom,qs,truth):
    lift.cache_clear();project.cache_clear()
    t0=time.perf_counter();s=Solver(c);ac=s.ac(c.domains)
    assert dom==(ac if ac is not None else zero(c))
    stats=collections.Counter();engine=None;tree=None
    if arm=="direct_tree":
        tree=Tree(c,dom);available=tree.available
    elif arm=="dfs_cover":available=True
    else:
        method="degree" if arm=="degree_grouped" else "minfill"
        mode=arm.split("_")[-1]
        engine=Cutset(c,dom,method,mode)
        available=engine.available
        stats=engine.stats
    setup=time.perf_counter()-t0
    if not available:
        return {"root":c.root,"family":c.family,"arm":arm,"n":c.n,"status":"cap",
                "setup_seconds":setup,"root_seconds":0.,"conditional_seconds":0.,
                "total_seconds":setup,"queries":0,"cutset":None,"stats":dict(stats),
                "logical_bits":0,"answers":[]}
    base=None;answers=[]
    t1=time.perf_counter()
    if arm=="dfs_cover":ans=dfs_cover(s,dom,{},stats)
    elif tree is not None:base=tree.calibrate({},stats);ans=tree.supports(base)
    else:ans=engine.answer({})
    answers.append(ans);root_seconds=time.perf_counter()-t1
    t1=time.perf_counter()
    for cond in qs[1:]:
        if arm=="dfs_cover":ans=dfs_cover(s,dom,cond,stats)
        elif tree is not None:
            loc=base
            for i,m in cond.items():loc=tree.conditioned(loc,i,next(bits(m)),stats)
            ans=tree.supports(loc)
        else:ans=engine.answer(cond)
        answers.append(ans)
    conditional_seconds=time.perf_counter()-t1
    assert answers==truth,(c.root,arm,answers,truth)
    snapshot=dict(stats)
    if engine is not None:
        # Separate witness audit, not part of measured support workload.
        for cond,expected in zip(qs,truth):
            ans,w=engine.answer(cond,with_witness=True)
            assert ans==expected and bool(any(ans))==(w is not None)
    return {"root":c.root,"family":c.family,"n":c.n,"arm":arm,"status":"exact",
            "setup_seconds":setup,"root_seconds":root_seconds,"conditional_seconds":conditional_seconds,
            "total_seconds":setup+root_seconds+conditional_seconds,"queries":len(qs),
            "cutset":list(engine.cut) if engine is not None else [],
            "residual_width":engine.tree.width if engine is not None and engine.tree is not None else None,
            "logical_bits":engine.logical_bits() if engine is not None else tree.storage() if tree is not None else 0,
            "stats":snapshot,"answers":answers}


def aggregate(rows):
    a=[r for r in rows if r["status"]=="exact"]
    mean=lambda k:statistics.mean(r[k] for r in a) if a else None
    stat=lambda k:statistics.mean(r["stats"].get(k,0) for r in a) if a else None
    return {"roots":len(rows),"exact":len(a),"caps":len(rows)-len(a),
            "queries":sum(r["queries"] for r in a),
            "mean_total_ms":1000*mean("total_seconds") if a else None,
            "mean_setup_ms":1000*mean("setup_seconds") if a else None,
            "mean_conditional_ms":1000*mean("conditional_seconds") if a else None,
            "mean_cutset_size":statistics.mean(len(r["cutset"]) for r in a) if a else None,
            "mean_logical_bits":mean("logical_bits"),
            **{k:stat(k) for k in ("enumerated_branches","compatible_branches","residual_groups",
                                   "branch_calibrations","messages","dfs_nodes","support_probes","width_plans")}}


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    au=audit();print("AUDIT",json.dumps(au),flush=True)
    retro=make_cases("stress_i",12,36,2026112562)
    high=[c for c in retro if width(c)>=BAG_CAP]
    low=[c for c in retro if width(c)<BAG_CAP]
    assert len(high)==8 and len(low)==28
    cohorts={"retrospective_capped":high,"retrospective_eligible":low,"fresh_bipartite":fresh_cases()}
    task=(json.dumps({k:[c.__dict__ for c in v] for k,v in cohorts.items()},
                     sort_keys=True,separators=(",",":"))+"\n").encode()
    (out/"tasks.json").write_bytes(task)
    rows=[];oracle=[];summary={}
    for co,cs in cohorts.items():
        local=[]
        for i,c in enumerate(cs):
            s=Solver(c);ac=s.ac(c.domains);dom=ac if ac is not None else zero(c)
            qs=query_list(c,dom)
            truth=[dfs_support(s,dom,q,collections.Counter()) for q in qs]
            oracle.append({"root":c.root,"cohort":co,"queries":qs,"answers":truth,"original_width":width(c)})
            off=i%len(ARMS)
            for arm in ARMS[off:]+ARMS[:off]:local.append(run(c,arm,dom,qs,truth))
        rows.extend({**r,"cohort":co} for r in local)
        common={r["root"] for r in local if r["arm"]=="minfill_grouped" and r["status"]=="exact"}
        summary[co]={"all":{a:aggregate([r for r in local if r["arm"]==a]) for a in ARMS},
                     "minfill_eligible":{a:aggregate([r for r in local if r["arm"]==a and r["root"] in common]) for a in ARMS}}
        print("COHORT",co,json.dumps(summary[co]),flush=True)
    result={"schema":"lbf20k.cutset-cache.v1","scope":"development_only","audit":au,
            "task_sha256":hashlib.sha256(task).hexdigest(),"rows":len(rows),
            "branch_cap":BRANCH_CAP,"bag_cap":BAG_CAP,"new_training_fits":0,"cohorts":summary}
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    for name,data in (("rows",rows),("oracle",oracle)):
        (out/f"{name}.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in data))
    print("COUNTS",json.dumps({k:v for k,v in result.items() if k!="cohorts"}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);main(p.parse_args().out)
