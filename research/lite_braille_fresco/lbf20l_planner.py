"""LBF-20L: budgeted cutset planning and incremental labeled branch grouping.

Every complete plan is exact under the existing calibrated-tree semantics.
Cost estimates choose a plan; no support is ever discarded by a score.
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
from lbf20j_separator import Tree, dfs_support, exact_from_worlds
from lbf20k_cutset import Cutset, residual, restrict, width, fresh_cases as old_fresh, query_list, dfs_cover

BAG_CAP=6
BRANCH_CAP=256
PLAN_CAP=160
BEAM=3
ARMS=("dfs_cover","direct_tree","degree_raw","degree_prefix","minfill_prefix",
      "profile_greedy","profile_beam","beam_hybrid")


def zero(c):return (0,)*c.n


def graph(c):
    adj=[0]*c.n
    for i,j,_ in c.edges:adj[i]|=1<<j;adj[j]|=1<<i
    return adj


def fast_order(adj,cut):
    """Same min-fill tie break as LBF-20I on original vertex IDs."""
    a=adj[:];remaining=((1<<len(a))-1)&~sum(1<<i for i in cut)
    out=[]
    while remaining:
        def key(i):
            nb=a[i]&remaining
            missing=sum((nb&~a[j]&~(1<<j)).bit_count() for j in bits(nb))//2
            return missing,nb.bit_count(),i
        i=min(bits(remaining),key=key);nb=a[i]&remaining
        out.append((i,tuple(bits(nb|(1<<i)))))
        for j in bits(nb):a[j]|=nb&~(1<<j)
        remaining&=~(1<<i)
    return out


class Planner:
    def __init__(self,c,dom,*,bag_cap=BAG_CAP,branch_cap=BRANCH_CAP,plan_cap=PLAN_CAP):
        self.c=c;self.dom=dom;self.adj=graph(c);self.rel=c.relations()
        self.bag_cap=bag_cap;self.branch_cap=branch_cap;self.plan_cap=plan_cap
        self.stats=collections.Counter();self.memo={}
    def profile_bound(self,cut):
        self.stats["profile_evaluations"]+=1
        rest=[i for i in range(self.c.n) if i not in cut]
        count=1
        for v in cut:
            profiles={tuple(self.rel[v,i][a]&self.dom[i] for i in rest if (v,i) in self.rel)
                      for a in bits(self.dom[v])}
            count*=len(profiles)
        return count
    def evaluate(self,cut,profiles=False):
        cut=tuple(sorted(cut))
        if cut not in self.memo:
            raw=math.prod(self.dom[i].bit_count() for i in cut)
            if raw>self.branch_cap:
                self.stats["branch_cap_rejections"]+=1
                return None
            if self.stats["width_plans"]>=self.plan_cap:
                self.stats["plan_cap_rejections"]+=1
                return None
            order=fast_order(self.adj,cut)
            b=max((len(v) for _,v in order),default=0)
            slots=sum(D**len(v) for _,v in order)
            self.stats["width_plans"]+=1
            self.memo[cut]={"cut":cut,"bag":b,"slots":slots,"raw":raw}
        rec=self.memo[cut]
        if profiles and "bound" not in rec:
            g=self.profile_bound(cut)
            assert g<=rec["raw"]
            rec["bound"]=g
            # Nominal bag slots plus a small linear cost of processing labels.
            # Proxy only, not an upper bound on measured runtime.
            rec["proxy"]=g*rec["slots"]+rec["raw"]*(self.c.n+len(cut))
        return rec
    def candidates(self,cut):
        rest=((1<<self.c.n)-1)&~sum(1<<i for i in cut)
        return sorted(bits(rest),key=lambda i:(-(self.adj[i]&rest).bit_count(),
                                               self.dom[i].bit_count(),i))
    def rank(self,r):
        return max(0,r["bag"]-self.bag_cap),r["proxy"],r["raw"],len(r["cut"]),r["cut"]
    def choose(self,method):
        profile=method.startswith("profile")
        root=self.evaluate((),profile)
        if root["bag"]<=self.bag_cap:return root
        if method!="profile_beam":
            cur=root
            while cur["bag"]>self.bag_cap:
                vs=self.candidates(cur["cut"])
                if method=="degree":
                    rec=self.evaluate((*cur["cut"],vs[0]))
                    if rec is None:return None
                    cur=rec
                else:
                    children=[(i,self.evaluate((*cur["cut"],i),profile)) for i in vs]
                    children=[(i,r) for i,r in children if r is not None]
                    if not children:return None
                    if method=="minfill":
                        _,cur=min(children,key=lambda x:(x[1]["bag"],self.dom[x[0]].bit_count(),
                                                         x[1]["slots"],x[0]))
                    else:_,cur=min(children,key=lambda x:self.rank(x[1]))
            return cur
        front=[root];best=None;first_depth=None
        # Explore at most one extra layer after the first feasible plan.
        for depth in range(1,self.c.n+1):
            nxt={}
            for r in front:
                for i in self.candidates(r["cut"]):
                    rec=self.evaluate((*r["cut"],i),True)
                    if rec is not None:
                        nxt[rec["cut"]]=rec
                        if rec["bag"]<=self.bag_cap:
                            if first_depth is None:first_depth=depth
                            if best is None or self.rank(rec)<self.rank(best):best=rec
                    if self.stats["width_plans"]>=self.plan_cap:break
                if self.stats["width_plans"]>=self.plan_cap:break
            if not nxt or self.stats["width_plans"]>=self.plan_cap:break
            if first_depth is not None and depth>=first_depth+1:break
            front=sorted(nxt.values(),key=self.rank)[:BEAM]
        return best


def raw_groups(c,dom,cut,stats):
    _,keep=residual(c,dom,set(cut));rel=c.relations();g=collections.defaultdict(list)
    for v in itertools.product(*(tuple(bits(dom[i])) for i in cut)):
        stats["raw_label_tests"]+=1
        d=restrict(c,dom,cut,v,keep,rel)
        if d is not None:g[d].append(v)
    return {d:tuple(sorted(vs)) for d,vs in g.items()}


def prefix_groups(c,dom,cut,stats):
    """Merge equal unassigned-domain states at every prefix.

    The original label prefixes travel with each state. Restricting a future
    variable depends only on that state's masks and the next chosen value.
    """
    if not all(dom):return {}
    rel=c.relations();states={tuple(dom):[()]}
    remaining=set(range(c.n))
    stats["peak_prefix_states"]=1
    for v in cut:
        remaining.remove(v)
        out=collections.defaultdict(list)
        for d,labels in sorted(states.items()):
            for a in bits(d[v]):
                stats["prefix_transitions"]+=1
                ds=list(d);ds[v]=0;ok=True
                for j in sorted(remaining):
                    if (v,j) in rel:
                        stats["prefix_intersections"]+=1
                        ds[j]&=rel[v,j][a]
                        if not ds[j]:ok=False;break
                if ok:
                    out[tuple(ds)].extend(x+(a,) for x in labels)
                    stats["label_extensions"]+=len(labels)
        states=out
        stats["peak_prefix_states"]=max(stats["peak_prefix_states"],len(states))
    keep=sorted(remaining)
    return {tuple(d[i] for i in keep):tuple(sorted(labels)) for d,labels in states.items()}


class Compiled(Cutset):
    """Use 20K's exact answer/calibration methods, with a preselected plan."""
    def __init__(self,c,dom,rec,mode,stats):
        self.c=c;self.dom=dom;self.mode="grouped";self.stats=stats
        self.cut=rec["cut"];self.available=True
        rc,self.keep=residual(c,dom,set(self.cut))
        self.ids={v:i for i,v in enumerate(self.keep)}
        self.tree=Tree(rc,rc.domains) if self.keep else None
        assert self.tree is None or self.tree.available
        g=(raw_groups if mode=="raw" else prefix_groups)(c,dom,self.cut,stats)
        self.groups=list(sorted(g.items()))
        stats["compatible_labels"]=sum(len(vs) for vs in g.values())
        stats["residual_groups"]=len(g)
        if "bound" in rec:assert len(g)<=rec["bound"]
        assert stats["compatible_labels"]<=rec["raw"]
        self.cache=[self.calibrate(d,{}) for d,_ in self.groups]
        stats["feasible_groups"]=sum(x is not None for x in self.cache)


def audit():
    st=collections.Counter()
    for n,count,seed in ((4,32,2026112590),(5,24,2026112591)):
        for c in generate("audit_l",n,count,seed,allow_unsat=True):
            dom=c.domains;adj=graph(c);p=Planner(c,dom,bag_cap=3,plan_cap=1000)
            worlds=list(itertools.product(range(D),repeat=n))
            good=[w for w in worlds if c.valid(w)]
            for k in range(min(3,n)+1):
                for cut in itertools.combinations(range(n),k):
                    rc,keep=residual(c,dom,set(cut))
                    fast=fast_order(adj,cut)
                    slow=[(keep[i],tuple(keep[j] for j in b)) for i,b in minfill(rc)]
                    assert fast==slow,(c.root,cut,fast,slow)
                    rec=p.evaluate(cut,True);assert rec is not None
                    raw=raw_groups(c,dom,cut,collections.Counter())
                    pref=prefix_groups(c,dom,cut,collections.Counter())
                    assert raw==pref,(c.root,cut,raw,pref)
                    assert len(raw)<=rec["bound"]<=rec["raw"]
                    st["profile_checks"]+=1;st["group_map_checks"]+=1;st["order_checks"]+=1
            for method in ("degree","minfill","profile_greedy","profile_beam"):
                p=Planner(c,dom,bag_cap=3)
                rec=p.choose(method);assert rec is not None
                e=Compiled(c,dom,rec,"prefix",collections.Counter())
                qs=[{}]+[{i:1<<a} for i in range(n) for a in range(D)]
                qs += [{0:1<<a,n-1:1<<b} for a in range(D) for b in range(D)]
                for q in qs:
                    expect=exact_from_worlds(c,[v for v in good if all(m&(1<<v[i]) for i,m in q.items())])
                    got,w=e.answer(q,with_witness=True)
                    assert got==expect and (w is not None)==bool(any(expect))
                    st["support_checks"]+=1
            st["systems"]+=1;st["worlds"]+=len(worlds)
    # Explicit plan-cap and branch-cap behavior; neither implies a refutation.
    c=Case("cap","audit",7,(FULL,)*7,tuple((i,j,(15,15,15,15)) for i,j in itertools.combinations(range(7),2)))
    assert Planner(c,c.domains,bag_cap=6,plan_cap=1).choose("profile_beam") is None
    assert Planner(c,c.domains,bag_cap=6,branch_cap=1).choose("profile_beam") is None
    st["cap_checks"]=2
    return {**dict(st),"mismatches":0}


def fresh():
    rng=random.Random(2026112592);out=[]
    for i in range(24):
        kind=("degree_decoy","alias_core","overlap_core")[i%3]
        if kind=="degree_decoy":
            # Core K9 plus two high-degree hubs outside that core. Removing
            # hubs does not lower the core width; the branch cap matters.
            core=9;n=21;edges=set(itertools.combinations(range(core),2))
            edges.update((h,j) for h in (9,10) for j in range(11,n))
            edges.update(((0,9),(1,10)))
            alias=set(rng.sample(range(core),3))
        elif kind=="alias_core":
            core=8+(i//3)%2;n=core+4
            edges=set(itertools.combinations(range(core),2))
            for j in range(core,n):edges.add((rng.randrange(core),j))
            alias=set(rng.sample(range(core),3))
        else:
            n=14;edges=set(itertools.combinations(range(7),2))
            edges.update(itertools.combinations(range(4,11),2))
            edges.update(((2,11),(8,12),(11,13),(12,13)))
            alias=set(rng.sample(range(11),4))
        p=rng.sample(range(n),n);plant=[rng.randrange(D) for _ in range(n)]
        # Relation class rows are equal for aliased vertices on every edge;
        # all rows and columns have support, preventing trivial AC shrinkage.
        cls={v:2 if v in alias else 4 for v in range(n)}
        relations=[]
        for u,v in sorted(edges):
            a,b=cls[u],cls[v]
            allow={(x,y) for x in range(a) for y in range(b) if rng.random()<.78}
            def group(x,k):return x//(D//k)
            pair=(group(plant[u],a),group(plant[v],b));allow.add(pair)
            for x in range(a):
                if not any(xx==x for xx,yy in allow):allow.add((x,rng.randrange(b)))
            for y in range(b):
                if not any(yy==y for xx,yy in allow):allow.add((rng.randrange(a),y))
            if len(allow)==a*b:
                allow.remove(next((x,y) for x in range(a) for y in range(b) if (x,y)!=pair))
            r=tuple(sum(1<<y for y in range(D) if (group(x,a),group(y,b)) in allow)
                    for x in range(D))
            x,y=p[u],p[v]
            if x>y:
                x,y=y,x
                r=tuple(sum(1<<a0 for a0 in range(D) if r[a0]&(1<<b0)) for b0 in range(D))
            relations.append((x,y,r))
        c=Case(f"fresh_l_{i:02d}",kind,n,(FULL,)*n,tuple(sorted(relations)))
        witness=[0]*n
        for old,new in enumerate(p):witness[new]=plant[old]
        assert c.valid(witness) and width(c)>=BAG_CAP
        assert Solver(c).ac(c.domains)==c.domains
        out.append(c)
    return out


def run(c,arm,dom,qs,truth):
    lift.cache_clear();project.cache_clear()
    t0=time.perf_counter();s=Solver(c)
    ac=s.ac(c.domains);assert dom==(ac if ac is not None else zero(c))
    stats=collections.Counter();rec=None;engine=None;tree=None;reason=None;fallback=False
    method={"degree_raw":"degree","degree_prefix":"degree","minfill_prefix":"minfill",
            "profile_greedy":"profile_greedy","profile_beam":"profile_beam","beam_hybrid":"profile_beam"}.get(arm)
    tp=time.perf_counter()
    if not all(dom):
        mode="contradiction"
    elif arm=="dfs_cover":mode="dfs"
    elif arm=="direct_tree":
        tree=Tree(c,dom)
        mode="tree" if tree.available else "cap";reason=None if tree.available else "width_cap"
    else:
        p=Planner(c,dom);rec=p.choose(method);stats.update(p.stats)
        if rec is None:
            reason="plan_or_branch_cap"
            if arm=="beam_hybrid":mode="dfs";fallback=True
            else:mode="cap"
        else:mode="cutset"
    planning=time.perf_counter()-tp
    if mode=="cutset":
        engine=Compiled(c,dom,rec,"raw" if arm=="degree_raw" else "prefix",stats)
    setup=time.perf_counter()-t0
    if mode=="cap":
        return {"root":c.root,"family":c.family,"arm":arm,"status":"cap","reason":reason,
                "planning_seconds":planning,"setup_seconds":setup,"total_seconds":setup,
                "query_seconds":0.,"queries":0,"fallback":False,"plan":None,"stats":dict(stats),
                "logical_bits":0,"answers":[]}
    t1=time.perf_counter();answers=[]
    base=tree.calibrate({},stats) if mode=="tree" else None
    for q in qs:
        if mode=="contradiction":ans=zero(c)
        elif mode=="dfs":ans=dfs_cover(s,dom,q,stats)
        elif mode=="tree":
            loc=base
            for i,m in q.items():loc=tree.conditioned(loc,i,next(bits(m)),stats)
            ans=tree.supports(loc)
        else:ans=engine.answer(q)
        answers.append(ans)
    query_seconds=time.perf_counter()-t1
    assert answers==truth,(c.root,arm,answers,truth)
    snap=dict(stats)
    if engine is not None:
        # Audit selected plan's full label map against raw enumeration.
        assert dict(engine.groups)==raw_groups(c,dom,engine.cut,collections.Counter())
        for q,expected in zip(qs,truth):
            ans,w=engine.answer(q,with_witness=True)
            assert ans==expected and (w is not None)==bool(any(ans))
    return {"root":c.root,"family":c.family,"arm":arm,"status":"exact","reason":reason,
            "planning_seconds":planning,"setup_seconds":setup,"total_seconds":setup+query_seconds,
            "query_seconds":query_seconds,"queries":len(qs),"fallback":fallback,"plan":rec,"stats":snap,
            "logical_bits":engine.logical_bits() if engine is not None else tree.storage() if mode=="tree" else 0,
            "answers":answers}


def aggregate(rows):
    a=[r for r in rows if r["status"]=="exact"]
    plans=[r for r in a if r["plan"] is not None]
    mean=lambda k:statistics.mean(r[k] for r in a) if a else None
    stat=lambda k:statistics.mean(r["stats"].get(k,0) for r in plans) if plans else None
    return {"roots":len(rows),"exact":len(a),"caps":len(rows)-len(a),
            "fallbacks":sum(r["fallback"] for r in a),"queries":sum(r["queries"] for r in a),
            "mean_total_ms":1000*mean("total_seconds") if a else None,
            "mean_planning_ms":1000*mean("planning_seconds") if a else None,
            "mean_query_ms":1000*mean("query_seconds") if a else None,
            "mean_logical_bits":mean("logical_bits"),
            "planned_roots":len(plans),
            "mean_cutset_size":statistics.mean(len(r["plan"]["cut"]) for r in plans) if plans else None,
            **{k:stat(k) for k in ("width_plans","compatible_labels","residual_groups","branch_calibrations",
                                  "messages","raw_label_tests","prefix_transitions","prefix_intersections")},
            "mean_profile_bound":statistics.mean(r["plan"]["bound"] for r in plans if "bound" in r["plan"])
                                  if any("bound" in r["plan"] for r in plans) else None,
            "cap_mean_ms":1000*statistics.mean(r["total_seconds"] for r in rows if r["status"]=="cap")
                          if len(a)<len(rows) else None}


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    au=audit();print("AUDIT",json.dumps(au),flush=True)
    retro=make_cases("stress_i",12,36,2026112562)
    cohorts={"retro_capped":[c for c in retro if width(c)>=BAG_CAP],
             "retro_eligible":[c for c in retro if width(c)<BAG_CAP],
             "retro_bipartite":old_fresh(),"fresh_mixed":fresh()}
    assert [len(v) for v in cohorts.values()]==[8,28,18,24]
    tasks=(json.dumps({k:[c.__dict__ for c in v] for k,v in cohorts.items()},
                      sort_keys=True,separators=(",",":"))+"\n").encode()
    (out/"tasks.json").write_bytes(tasks)
    rows=[];oracle=[];summary={}
    for co,cs in cohorts.items():
        local=[]
        for i,c in enumerate(cs):
            s=Solver(c);ac=s.ac(c.domains);dom=ac if ac is not None else zero(c)
            qs=query_list(c,dom)
            truth=[dfs_support(s,dom,q,collections.Counter()) for q in qs]
            oracle.append({"cohort":co,"root":c.root,"queries":qs,"answers":truth})
            off=i%len(ARMS)
            for arm in ARMS[off:]+ARMS[:off]:
                local.append(run(c,arm,dom,qs,truth))
        rows.extend({**r,"cohort":co} for r in local)
        summary[co]={a:aggregate([r for r in local if r["arm"]==a]) for a in ARMS}
        summary[co]["families"]={f:{a:aggregate([r for r in local if r["arm"]==a and r["family"]==f])
                                   for a in ARMS} for f in sorted({r["family"] for r in local})}
        common=set.intersection(*[{r["root"] for r in local if r["arm"]==a and r["status"]=="exact"}
                                  for a in ("degree_prefix","minfill_prefix","profile_greedy","profile_beam")])
        summary[co]["common_planner_eligible"]={a:aggregate([r for r in local if r["arm"]==a and r["root"] in common])
                                                  for a in ARMS}
        print("COHORT",co,json.dumps({a:summary[co][a] for a in ARMS}),flush=True)
        print("FAMILIES",co,json.dumps({f:{a:{k:x[k] for k in ("exact","caps","mean_total_ms","residual_groups","width_plans")}
                                     for a,x in v.items()} for f,v in summary[co]["families"].items()}),flush=True)
    result={"schema":"lbf20l.profile-cutset.v1","scope":"development_only","audit":au,
            "task_sha256":hashlib.sha256(tasks).hexdigest(),"rows":len(rows),"plan_cap":PLAN_CAP,
            "branch_cap":BRANCH_CAP,"bag_cap":BAG_CAP,"new_training_fits":0,"cohorts":summary}
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    for name,data in (("rows",rows),("oracle",oracle)):
        (out/f"{name}.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in data))
    print("COUNTS",json.dumps({k:v for k,v in result.items() if k!="cohorts"}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);main(p.parse_args().out)
