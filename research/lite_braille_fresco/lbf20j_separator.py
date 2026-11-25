"""LBF-20J: exact separator-conditioned support with calibrated bag trees.

All inference is Boolean and width-capped. Shared local evidence is never
silently treated as independent. Exact support labels are audit-only.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import random
import statistics
import time
from pathlib import Path

from lbf20f_csp import Case, Solver, D, FULL, bits, generate
from lbf20i_scope import (full, code, digit, lift, project, positions, initial,
                          minfill, make_cases, color_case, Engine, choose)

CAP=6
ARMS=("exact_dfs", "repeat_up", "recalibrate", "cached_outward")
NCOND=3


def zero(c):
    return (0,)*c.n


class Tree:
    """Elimination bags, forest parent links, local factors and messages."""
    def __init__(self,c,dom):
        self.c=c
        self.dom=tuple(dom)
        order=minfill(c)
        self.bags=tuple(b for _,b in order)
        self.owner={v:i for i,(v,b) in enumerate(order)}
        self.width=max(len(b)-1 for b in self.bags)
        self.available=self.width+1<=CAP
        self.parent=[]
        self.children=[[] for _ in self.bags]
        self.sep=[]
        self.local=[]
        self.roots=[]
        self.base=None
        if not self.available:return
        for i,(v,b) in enumerate(order):
            s=tuple(x for x in b if x!=v)
            p=min((self.owner[x] for x in s),default=None)
            self.parent.append(p)
            self.sep.append(s)
            if p is None:self.roots.append(i)
            else:
                assert p>i and set(s)<=set(self.bags[p])
                self.children[p].append(i)
        self.neighbors=[tuple(([p] if p is not None else [])+self.children[i])
                        for i,p in enumerate(self.parent)]
        # Check running intersection directly, and factor coverage.
        for v in range(c.n):
            containing={i for i,b in enumerate(self.bags) if v in b}
            seen={self.owner[v]};todo=list(seen)
            for i in todo:
                for j in self.neighbors[i]:
                    if j in containing and j not in seen:
                        seen.add(j);todo.append(j)
            assert seen==containing,(v,seen,containing)
        self.local=[full(len(b)) for b in self.bags]
        tables=initial(c,dom)
        for f in [*((v,) for v in range(c.n)),*((i,j) for i,j,_ in c.edges)]:
            i=min(self.owner[v] for v in f)
            b=self.bags[i]
            assert set(f)<=set(b)
            self.local[i]&=lift(len(b),positions(b,f),tables[f])

    def storage(self):
        return sum(D**len(b) for b in self.bags) if self.available else 0

    def evidence(self,cond):
        loc=self.local[:]
        for v,m in cond.items():
            i=self.owner[v];b=self.bags[i]
            loc[i]&=lift(len(b),positions(b,(v,)),m)
        return loc

    def send(self,src,dst,mask,stats):
        b=self.bags[src];q=self.bags[dst]
        s=tuple(sorted(set(b)&set(q)))
        stats["messages"]+=1
        stats["separator_slots"]+=D**len(s)
        msg=project(len(b),positions(b,s),mask)
        return lift(len(q),positions(q,s),msg)

    def inward(self,cond,stats):
        loc=self.evidence(cond)
        stats["inward_runs"]+=1
        for i,b in enumerate(self.bags):
            stats["bag_visits"]+=1
            stats["nominal_bag_slots"]+=D**len(b)
            if not loc[i]:return None
            p=self.parent[i]
            if p is not None:
                loc[p]&=self.send(i,p,loc[i],stats)
        return loc

    def calibrate(self,cond,stats):
        loc=self.inward(cond,stats)
        if loc is None:return None
        # Projecting a calibrated parent that already includes this child's
        # upward message is valid: that message is redundant on the child.
        for i in reversed(range(len(self.bags))):
            p=self.parent[i]
            if p is not None:
                loc[i]&=self.send(p,i,loc[p],stats)
                stats["bag_visits"]+=1
                stats["nominal_bag_slots"]+=D**len(self.bags[i])
                assert loc[i]
        return loc

    def conditioned(self,base,v,a,stats):
        """Single-bag evidence; update away from that bag only."""
        stats["cached_queries"]+=1
        if base is None:return None
        loc=base[:]
        i=self.owner[v];b=self.bags[i]
        loc[i]&=lift(len(b),positions(b,(v,)),1<<a)
        stats["bag_visits"]+=1
        stats["nominal_bag_slots"]+=D**len(b)
        if not loc[i]:return None
        todo=[(i,None)]
        for u,p in todo:
            for w in self.neighbors[u]:
                if w==p:continue
                new=loc[w]&self.send(u,w,loc[u],stats)
                stats["bag_visits"]+=1
                stats["nominal_bag_slots"]+=D**len(self.bags[w])
                assert new
                # If an entire neighbor marginal is unchanged, all of the
                # feasible separator assignments on that side still extend
                # to the evidence. No further outward propagation required.
                if new!=loc[w]:
                    loc[w]=new
                    todo.append((w,u))
                else:
                    stats["unchanged_subtree_stops"]+=1
        return loc

    def supports(self,loc):
        if loc is None:return zero(self.c)
        return tuple(project(len(self.bags[i]),positions(self.bags[i],(v,)),loc[i])
                     for v,i in sorted(self.owner.items()))

    def witness(self,loc,cond=None):
        if loc is None:return None
        assigned={}
        for i in reversed(range(len(self.bags))):
            b=self.bags[i];s=self.sep[i]
            m=loc[i]
            if s:
                m&=lift(len(b),positions(b,s),1<<code(assigned[v] for v in s))
            assert m
            a=next(bits(m))
            for k,v in enumerate(b):
                val=digit(a,k)
                assert v not in assigned or assigned[v]==val
                assigned[v]=val
        v=tuple(assigned[i] for i in range(self.c.n))
        assert self.c.valid(v)
        if cond:
            assert all(m&(1<<v[i]) for i,m in cond.items())
        return v


def repeat_support(t,cond,stats):
    # Same compiled decomposition, reuse index/mask caches, no deliberate
    # penalty from reconstructing the graph or choosing a new elimination order.
    domains=[m&cond.get(i,FULL) for i,m in enumerate(t.dom)]
    ans=[]
    for i,m in enumerate(domains):
        keep=0
        for a in bits(m):
            q={**cond,i:1<<a}
            stats["support_probes"]+=1
            if t.inward(q,stats) is not None:keep|=1<<a
        ans.append(keep)
    return tuple(ans)


def dfs_support(s,dom,cond,stats):
    d=tuple(m&cond.get(i,FULL) for i,m in enumerate(dom))
    ac=s.ac(d,stats)
    if ac is None:return zero(s.c)
    ans=[]
    for i,m in enumerate(ac):
        keep=0
        for a in bits(m):
            q=list(ac);q[i]=1<<a
            stats["support_probes"]+=1
            r=s.solve(tuple(q))
            stats["dfs_nodes"]+=r["nodes"]
            stats["arc_revisions"]+=r["arc_revisions"]
            assert r["status"] in ("solved","refuted")
            if r["solution"] is not None:keep|=1<<a
        ans.append(keep)
    return tuple(ans)


def exact_from_worlds(c,worlds):
    return tuple(sum(1<<a for a in range(D) if any(v[i]==a for v in worlds))
                 for i in range(c.n))


def bag_masks(t,worlds):
    return [sum(1<<i for i in {code(v[j] for j in b) for v in worlds}) for b in t.bags]


def audit():
    stats=collections.Counter()
    for n,count,seed in ((4,48,2026112570),(5,32,2026112571)):
        for c in generate("audit_j",n,count,seed,allow_unsat=True):
            worlds=list(itertools.product(range(D),repeat=n))
            ws=[v for v in worlds if c.valid(v)]
            t=Tree(c,c.domains);assert t.available
            a=collections.Counter();base=t.calibrate({},a)
            assert (base is None)==(not ws)
            assert t.supports(base)==exact_from_worlds(c,ws)
            if ws:
                assert base==bag_masks(t,ws);t.witness(base)
                stats["bag_marginal_checks"]+=len(t.bags)
            assert repeat_support(t,{},a)==exact_from_worlds(c,ws)
            for v in range(n):
                for value in range(D):
                    sub=[w for w in ws if w[v]==value]
                    expected=exact_from_worlds(c,sub)
                    r=t.conditioned(base,v,value,a)
                    assert t.supports(r)==expected
                    assert (r is None)==(not sub)
                    assert t.supports(t.calibrate({v:1<<value},a))==expected
                    if sub:
                        assert r==bag_masks(t,sub)
                        t.witness(r,{v:1<<value})
                        stats["bag_marginal_checks"]+=len(t.bags)
                    stats["conditional_queries"]+=1
            stats["systems"]+=1;stats["worlds"]+=len(worlds)
    # Explicit disconnected components, isolated vertex and unsatisfiable
    # evidence on one component (which invalidates every global marginal).
    r=tuple(FULL^(1<<a) for a in range(D))
    c=Case("disconnected","audit",5,(3,3,7,7,1),((0,1,r),(2,3,r)))
    t=Tree(c,c.domains);a=collections.Counter();base=t.calibrate({},a)
    assert len(t.roots)==3 and t.supports(base)==c.domains
    assert t.supports(t.conditioned(base,0,0,a))==(1,2,7,7,1)
    assert t.conditioned(base,4,1,a) is None
    stats["disconnected_checks"]=3
    return {**dict(stats),"mismatches":0}


def fresh_cases():
    rng=random.Random(2026112572);out=[]
    for i in range(24):
        if i%2==0:
            # Overlapping four-variable bags in a chain with pair separators.
            n=(12,18,24)[(i//2)%3]
            v=rng.sample(range(n),n);plant=[rng.randrange(D) for _ in range(n)]
            pairs=set()
            for start in range(0,n-2,2):
                pairs.update(tuple(sorted(x)) for x in itertools.combinations(v[start:start+4],2))
            edges=[]
            for a,b in sorted(pairs):
                p=rng.uniform(.38,.70)
                rows=[sum(1<<j for j in range(D) if rng.random()<p) for _ in range(D)]
                rows[plant[a]]|=1<<plant[b]
                edges.append((a,b,tuple(rows)))
            out.append(Case(f"fresh_j_{i:02d}","separator_chain",n,(FULL,)*n,tuple(edges)))
        else:
            # Multiple odd wheels share one selector but have independent
            # color vertices. The selector is deliberately hidden by relabeling.
            blocks=2+(i//2)%3;n=1+6*blocks
            v=rng.sample(range(n),n);sel=v[0];ds=[FULL]*n;ds[sel]=3
            neq=tuple(FULL^(1<<a) for a in range(D))
            edges=[]
            for j in range(blocks):
                hub=v[1+6*j];rim=v[2+6*j:7+6*j]
                ee=[(hub,z) for z in rim]+[(rim[k],rim[(k+1)%5]) for k in range(5)]
                edges.extend((min(a,b),max(a,b),neq) for a,b in ee)
                rows=(7,15,0,0)
                for u in [hub,*rim]:
                    if sel<u:edges.append((sel,u,rows))
                    else:edges.append((u,sel,tuple(sum(1<<a for a in range(D) if rows[a]&(1<<b)) for b in range(D))))
            out.append(Case(f"fresh_j_{i:02d}","linked_wheel",n,tuple(ds),tuple(sorted(edges))))
    return out


def conditions(c,dom):
    # Fixed deterministic public policy. Pick minimum nontrivial domain,
    # then high degree, and probe its first two values. Third query, if
    # available, on another variable. Includes both conditional-wheel modes.
    s=Solver(c)
    order=sorted((i for i,m in enumerate(dom) if m.bit_count()>1),
                 key=lambda i:(dom[i].bit_count(),-len(s.nb[i]),i))
    if not order:return []
    i=order[0];qs=[(i,a) for a in list(bits(dom[i]))[:2]]
    if len(order)>1:qs.append((order[1],next(bits(dom[order[1]]))))
    return qs[:NCOND]


def check_supports(c,answers,expected):
    assert answers==expected,(c.root,answers,expected)


def run(c,arm,dom,queries,truth):
    lift.cache_clear();project.cache_clear()
    stats=collections.Counter()
    t0=time.perf_counter()
    s=Solver(c)
    # All arms receive the same initial AC domains. Charge constructing AC.
    actual=s.ac(c.domains)
    assert dom==(actual if actual is not None else zero(c))
    t=None
    if arm!="exact_dfs":
        t=Tree(c,dom)
    setup=time.perf_counter()-t0
    if t is not None and not t.available:
        return {"root":c.root,"family":c.family,"arm":arm,"status":"width_cap",
                "n":c.n,"width":t.width,"setup_seconds":setup,"query_count":0,
                "root_seconds":0.,"conditional_seconds":0.,"total_seconds":setup,
                "logical_bag_bits":0,"stats":{}}
    t1=time.perf_counter()
    if arm=="exact_dfs":
        supports=dfs_support(s,dom,{},stats)
    elif arm=="repeat_up":
        supports=repeat_support(t,{},stats)
    else:
        base=t.calibrate({},stats);supports=t.supports(base)
    root_seconds=time.perf_counter()-t1
    answers=[supports]
    t1=time.perf_counter()
    for v,a in queries:
        cond={v:1<<a}
        if arm=="exact_dfs":q=dfs_support(s,dom,cond,stats)
        elif arm=="repeat_up":q=repeat_support(t,cond,stats)
        elif arm=="recalibrate":q=t.supports(t.calibrate(cond,stats))
        else:q=t.supports(t.conditioned(base,v,a,stats))
        answers.append(q)
    conditional_seconds=time.perf_counter()-t1
    check_supports(c,answers,truth)
    # Witness recovery is audited separately from the timed support workload.
    if t is not None and arm in ("recalibrate","cached_outward"):
        t.witness(base)
        for v,a in queries:
            scratch=collections.Counter()
            q=t.calibrate({v:1<<a},scratch) if arm=="recalibrate" else t.conditioned(base,v,a,scratch)
            t.witness(q,{v:1<<a})
    return {"root":c.root,"family":c.family,"arm":arm,"status":"exact","n":c.n,
            "width":t.width if t is not None else None,"setup_seconds":setup,
            "root_seconds":root_seconds,"conditional_seconds":conditional_seconds,
            "total_seconds":setup+root_seconds+conditional_seconds,
            "query_count":len(answers),"logical_bag_bits":t.storage() if t is not None else 0,
            "stats":dict(stats),"answers":answers}


def selector_audit(c,dom):
    # This structural family has exactly one two-value selector domain.
    v=next(i for i,m in enumerate(c.domains) if m==3)
    out=[]
    s=Solver(c)
    for arm in ("singleton_ac","singleton_four","cached_tree","repeat_inward","exact_dfs"):
        lift.cache_clear();project.cache_clear();stats=collections.Counter()
        t0=time.perf_counter();mask=0
        if arm in ("cached_tree","repeat_inward"):
            t=Tree(c,dom);assert t.available
            base=t.calibrate({},stats) if arm=="cached_tree" else None
        for a in bits(dom[v]):
            d=list(dom);d[v]=1<<a
            if arm=="singleton_ac":ok=s.ac(tuple(d),stats) is not None
            elif arm=="singleton_four":
                ac=s.ac(tuple(d),stats)
                if ac is None:ok=False
                else:
                    e=Engine(c,ac)
                    pc=tuple(itertools.combinations(range(c.n),3))
                    e.propagate(pc)
                    if not e.contradiction:
                        bags=choose(c,e,"tight_pair",2*c.n)
                        e.propagate((*pc,*bags))
                    stats.update(e.stats);ok=not e.contradiction
            elif arm=="cached_tree":ok=t.conditioned(base,v,a,stats) is not None
            elif arm=="repeat_inward":ok=t.inward({v:1<<a},stats) is not None
            else:
                z=s.solve(tuple(d));ok=z["status"]=="solved"
                stats["dfs_nodes"]+=z["nodes"];stats["arc_revisions"]+=z["arc_revisions"]
            if ok:mask|=1<<a
        seconds=time.perf_counter()-t0
        assert mask&2, (c.root,arm)
        if arm in ("cached_tree","repeat_inward","exact_dfs"):
            assert mask==2,(c.root,arm,mask)
        out.append({"root":c.root,"family":c.family,"n":c.n,"arm":arm,
                    "selector":v,"support":mask,"seconds":seconds,"stats":dict(stats)})
    return out


def agg(rows):
    exact=[r for r in rows if r["status"]=="exact"]
    def mean(k):return statistics.mean(r[k] for r in exact) if exact else None
    def stat(k):return statistics.mean(r["stats"].get(k,0) for r in exact) if exact else None
    return {"roots":len(rows),"exact_roots":len(exact),"width_caps":len(rows)-len(exact),
            "queries":sum(r["query_count"] for r in exact),
            **{k:mean(k) for k in ("setup_seconds","root_seconds","conditional_seconds","total_seconds","logical_bag_bits")},
            **{k:stat(k) for k in ("messages","bag_visits","nominal_bag_slots","support_probes","dfs_nodes","unchanged_subtree_stops")}}


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    au=audit();print("AUDIT",json.dumps(au),flush=True)
    cohorts={"retrospective8":make_cases("dev_i",8,48,2026112561),
             "retrospective12":make_cases("stress_i",12,36,2026112562),
             "fresh_separator":fresh_cases()}
    task=(json.dumps({k:[c.__dict__ for c in v] for k,v in cohorts.items()},
                     sort_keys=True,separators=(",",":"))+"\n").encode()
    (out/"tasks.json").write_bytes(task)
    rows=[];sel=[];answers=[];summary={}
    for co,cs in cohorts.items():
        local=[]
        for i,c in enumerate(cs):
            s=Solver(c);ac=s.ac(c.domains);dom=ac if ac is not None else zero(c)
            queries=conditions(c,dom)
            truth=[dfs_support(s,dom,{},collections.Counter())]
            truth.extend(dfs_support(s,dom,{v:1<<a},collections.Counter()) for v,a in queries)
            answers.append({"root":c.root,"cohort":co,"conditions":queries,"supports":truth})
            off=i%len(ARMS)
            for arm in ARMS[off:]+ARMS[:off]:
                local.append(run(c,arm,dom,queries,truth))
            if c.family in ("conditional_wheel","linked_wheel"):
                sel.extend({**r,"cohort":co} for r in selector_audit(c,dom))
        rows.extend({**r,"cohort":co} for r in local)
        eligible={r["root"] for r in local if r["arm"]=="cached_outward" and r["status"]=="exact"}
        summary[co]={"all":{a:agg([r for r in local if r["arm"]==a]) for a in ARMS},
                     "common_width_eligible":{a:agg([r for r in local if r["arm"]==a and r["root"] in eligible]) for a in ARMS}}
        print("COHORT",co,json.dumps(summary[co]),flush=True)
    sa={}
    for arm in ("singleton_ac","singleton_four","cached_tree","repeat_inward","exact_dfs"):
        r=[x for x in sel if x["arm"]==arm]
        sa[arm]={"roots":len(r),"exact_selector":sum(x["support"]==2 for x in r),
                 "mean_seconds":statistics.mean(x["seconds"] for x in r)}
    result={"schema":"lbf20j.separator-cache.v1","scope":"development_only","audit":au,
            "task_sha256":hashlib.sha256(task).hexdigest(),"rows":len(rows),
            "selector_rows":len(sel),"selectors":sa,"cohorts":summary,"new_training_fits":0}
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    for name,data in (("rows",rows),("selector_rows",sel),("oracle_answers",answers)):
        (out/f"{name}.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in data))
    print("COUNTS",json.dumps({k:v for k,v in result.items() if k!="cohorts"}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);main(p.parse_args().out)
