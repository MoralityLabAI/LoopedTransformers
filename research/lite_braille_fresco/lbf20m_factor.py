"""LBF-20M: factorized cutset labels and component-scoped separator caches.

A layered deterministic DAG represents complete compatible cutset labels.
A terminal is usable only if every residual component is feasible in that
same terminal under the query. Scores and oracle labels do not enter inference.
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
from lbf20i_scope import make_cases, lift, project
from lbf20j_separator import Tree, dfs_support, exact_from_worlds
from lbf20k_cutset import residual, width, fresh_cases as bipartite, query_list, dfs_cover
from lbf20l_planner import Planner, Compiled, raw_groups, fresh as mixed

STATE_CAP=4096
EDGE_CAP=16384
RAW_CAP=256
BAG_CAP=6
ARMS=("dfs_cover","direct_tree","flat_forest","mdd_forest","mdd_components","mdd_query_cache")


def zero(c):return (0,)*c.n


class Capped(Exception):
    pass


class Diagram:
    def __init__(self,c,dom,cut,stats,state_cap=STATE_CAP,edge_cap=EDGE_CAP):
        assert len(set(cut))==len(cut) and all(dom)
        self.cut=tuple(cut);self.stats=stats
        self.keep=tuple(i for i in range(c.n) if i not in cut)
        self.layers=[(tuple(dom),)];self.edges=[]
        self.paths=[(1,)]
        rel=c.relations();remaining=set(range(c.n))
        stats["diagram_states"]=1
        for v in self.cut:
            remaining.remove(v)
            ids={};edges=[];counts=[]
            for src,d in enumerate(self.layers[-1]):
                for a in bits(d[v]):
                    stats["prefix_transitions"]+=1
                    nxt=list(d);nxt[v]=0
                    for j in sorted(remaining):
                        if (v,j) in rel:
                            stats["prefix_intersections"]+=1
                            nxt[j]&=rel[v,j][a]
                            if not nxt[j]:break
                    else:
                        key=tuple(nxt)
                        if key not in ids:
                            if stats["diagram_states"]>=state_cap:
                                raise Capped("diagram_state_cap")
                            ids[key]=len(ids);counts.append(0)
                            stats["diagram_states"]+=1
                        dst=ids[key]
                        if stats["diagram_edges"]>=edge_cap:
                            raise Capped("diagram_edge_cap")
                        edges.append((src,a,dst));stats["diagram_edges"]+=1
                        counts[dst]+=self.paths[-1][src]
            self.layers.append(tuple(ids));self.edges.append(tuple(edges));self.paths.append(tuple(counts))
        stats["terminals"]=len(self.layers[-1])
        stats["compatible_labels"]=sum(self.paths[-1])
        stats["peak_layer_states"]=max(map(len,self.layers))
        assert stats["compatible_labels"]<=math.prod(dom[i].bit_count() for i in cut)

    def forward(self,cond):
        reach=[{0}]
        for v,edges in zip(self.cut,self.edges):
            allowed=cond.get(v,FULL)
            self.stats["forward_edge_scans"]+=len(edges)
            reach.append({d for s,a,d in edges if s in reach[-1] and allowed&(1<<a)})
        return reach

    def backward(self,reach,good,cond,out):
        viable=[set() for _ in self.layers];viable[-1]=set(good)
        for k in reversed(range(len(self.cut))):
            v=self.cut[k];allowed=cond.get(v,FULL)
            self.stats["backward_edge_scans"]+=len(self.edges[k])
            for s,a,d in self.edges[k]:
                if s in reach[k] and d in viable[k+1] and allowed&(1<<a):
                    viable[k].add(s);out[v]|=1<<a
        assert bool(viable[0])==bool(good)
        return viable

    def one_label(self,viable,cond):
        s=0;label=[]
        for k,v in enumerate(self.cut):
            allowed=cond.get(v,FULL)
            a,d=next((a,d) for src,a,d in self.edges[k]
                     if src==s and d in viable[k+1] and allowed&(1<<a))
            label.append(a);s=d
        return tuple(label),s

    def label_map(self):
        # Exhaustive debug oracle ONLY for tiny finite audits.
        assert sum(self.paths[-1])<=4096
        current={0:[()]}
        for edges in self.edges:
            nxt=collections.defaultdict(list)
            for s,a,d in edges:nxt[d].extend(p+(a,) for p in current.get(s,[]))
            current=nxt
        return {tuple(self.layers[-1][i][v] for v in self.keep):tuple(sorted(labels))
                for i,labels in current.items()}


def components(c,keep):
    left=set(keep);adj={i:[] for i in keep}
    for i,j,_ in c.edges:
        if i in left and j in left:adj[i].append(j);adj[j].append(i)
    result=[]
    while left:
        root=min(left);q=[root];left.remove(root)
        for i in q:
            for j in adj[i]:
                if j in left:left.remove(j);q.append(j)
        result.append(tuple(sorted(q)))
    return result


class Unit:
    def __init__(self,c,dom,vs,stats):
        self.vs=vs;self.ids={v:i for i,v in enumerate(vs)};self.stats=stats
        ids=self.ids
        rc=Case(c.root,c.family,len(vs),tuple(dom[v] for v in vs),
                tuple((ids[i],ids[j],r) for i,j,r in c.edges if i in ids and j in ids))
        self.tree=Tree(rc,rc.domains)
        if not self.tree.available:raise Capped("residual_width_cap")
        self.base={};self.memo={}
        self.stats["compiled_units"]+=1

    def build(self,d):
        if d in self.base:
            self.stats["base_hits"]+=1;return
        evidence={i:m for i,m in enumerate(d) if m!=self.tree.dom[i]}
        self.stats["base_calibrations"]+=1
        loc=self.tree.calibrate(evidence,self.stats)
        self.base[d]=(loc,self.tree.supports(loc))

    def query(self,d,cond,memo):
        key=tuple(sorted((self.ids[v],m) for v,m in cond.items() if v in self.ids))
        loc,sup=self.base[d]
        if not key or loc is None:return loc,sup
        ck=(d,key)
        if memo and ck in self.memo:
            self.stats["query_hits"]+=1;return self.memo[ck]
        self.stats["query_misses"]+=1
        for i,m in key:
            assert m.bit_count()==1
            if loc is None:break
            self.stats["conditioned_calls"]+=1
            loc=self.tree.conditioned(loc,i,next(bits(m)),self.stats)
        ans=(loc,self.tree.supports(loc))
        if memo:self.memo[ck]=ans
        return ans

    def storage(self):
        # Nominal capacity of successful stored bag tables and unary masks.
        # No claim about allocator/RSS or Python DAG/label overhead.
        base=sum(loc is not None for loc,_ in self.base)
        memo=sum(loc is not None for loc,_ in self.memo.values())
        return (base*self.tree.storage()+len(self.base)*D*len(self.vs),
                memo*self.tree.storage()+len(self.memo)*D*len(self.vs))


class Factorized:
    def __init__(self,c,dom,cut,mode,stats,state_cap=STATE_CAP,edge_cap=EDGE_CAP):
        self.c=c;self.cut=tuple(cut);self.stats=stats;self.mode=mode
        self.dag=Diagram(c,dom,cut,stats,state_cap,edge_cap)
        keep=self.dag.keep
        parts=[keep] if mode=="mdd_forest" and keep else components(c,keep)
        self.units=[Unit(c,dom,vs,stats) for vs in parts]
        self.signatures=[]
        for d in self.dag.layers[-1]:
            sig=tuple(tuple(d[v] for v in u.vs) for u in self.units)
            self.signatures.append(sig)
            for u,k in zip(self.units,sig):u.build(k)
        self.stats["whole_forest_entries"]=len(self.signatures)
        self.stats["component_entries"]=sum(len(u.base) for u in self.units)
        self.stats["feasible_component_entries"]=sum(loc is not None for u in self.units for loc,_ in u.base.values())

    def answer(self,cond,*,with_witness=False):
        reach=self.dag.forward(cond);out=[0]*self.c.n
        good=set()
        for t in sorted(reach[-1]):
            results=[]
            for unit,d in zip(self.units,self.signatures[t]):
                self.stats["component_checks"]+=1
                loc,sup=unit.query(d,cond,self.mode=="mdd_query_cache")
                if loc is None:break
                results.append((unit,sup))
            else:
                good.add(t)
                for unit,sup in results:
                    for i,m in zip(unit.vs,sup):out[i]|=m
        viable=self.dag.backward(reach,good,cond,out)
        ans=tuple(out)
        if not with_witness:return ans
        if not good:
            assert not any(ans)
            return ans,None
        label,t=self.dag.one_label(viable,cond)
        witness=[-1]*self.c.n
        for i,a in zip(self.cut,label):witness[i]=a
        for u,d in zip(self.units,self.signatures[t]):
            loc,_=u.query(d,cond,self.mode=="mdd_query_cache")
            w=u.tree.witness(loc);assert w is not None
            for i,a in zip(u.vs,w):witness[i]=a
        w=tuple(witness)
        assert self.c.valid(w)
        assert all(m&(1<<w[i]) for i,m in cond.items())
        return ans,w

    def storage(self):
        p=[u.storage() for u in self.units]
        return sum(x for x,_ in p),sum(y for _,y in p)


def audit():
    st=collections.Counter()
    for n,count,seed in ((4,32,2026112600),(5,24,2026112601)):
        for c in generate("audit_m",n,count,seed,allow_unsat=True):
            worlds=list(itertools.product(range(D),repeat=n))
            good=[v for v in worlds if c.valid(v)]
            cut=(0,) if n==4 else (0,1)
            fs=[Factorized(c,c.domains,cut,mode,collections.Counter()) for mode in ARMS[3:]]
            ref=raw_groups(c,c.domains,cut,collections.Counter())
            for e in fs:
                assert e.dag.label_map()==ref
                assert e.stats["compatible_labels"]==sum(map(len,ref.values()))
                st["label_map_checks"]+=1
            qs=[{}]+[{i:1<<a} for i in range(n) for a in range(D)]
            qs += [{0:1<<a,n-1:1<<b} for a in range(D) for b in range(D)]
            for q in qs:
                want=exact_from_worlds(c,[w for w in good if all(m&(1<<w[i]) for i,m in q.items())])
                for e in fs:
                    got,w=e.answer(q,with_witness=True)
                    assert got==want and (w is not None)==bool(any(want)),(c.root,e.mode,q,got,want)
                    st["support_checks"]+=1
                st["queries"]+=1
            st["systems"]+=1;st["worlds"]+=len(worlds)
    # Opposing conditions on separate residual components must share one
    # cutset terminal. Mixing marginal feasibility from different terminals
    # would incorrectly certify this query.
    c=Case("cross_terminal","audit",3,(3,3,3),((0,1,(1,2,0,0)),(0,2,(2,1,0,0))))
    for mode in ARMS[3:]:
        e=Factorized(c,c.domains,(0,),mode,collections.Counter())
        assert e.answer({1:1,2:1})==zero(c)
        assert e.answer({1:1})==(1,1,2)
        st["cross_terminal_checks"]+=2
    # Empty and full cutsets and independent disconnected components.
    c=Case("boundary","audit",3,(3,3,3),((0,1,(3,3,0,0)),))
    for cut in ((),(0,1,2)):
        for mode in ARMS[3:]:
            e=Factorized(c,c.domains,cut,mode,collections.Counter())
            assert e.answer({})==c.domains
            assert e.answer({0:1,2:2},with_witness=True)[0]==(1,3,2)
            st["boundary_checks"]+=2
    for sc,ec in ((1,EDGE_CAP),(STATE_CAP,1)):
        try:Diagram(c,c.domains,(0,1),collections.Counter(),sc,ec)
        except Capped:st["cap_checks"]+=1
        else:raise AssertionError("cap did not stop diagram")
    return {**dict(st),"mismatches":0}


def fresh():
    rng=random.Random(2026112602);out=[]
    for i in range(12):
        kind=("coherent_alias","independent_modes","coupled_modes")[i%3];j=i//3
        k=(6+j) if kind=="coherent_alias" else (5+j//2)
        blocks=k-1 if kind=="coupled_modes" else k
        n=k+6*blocks
        es=[]
        for v in range(k-1):
            r=(3,3,12,12) if kind=="coherent_alias" else (15,15,15,7)
            es.append((v,v+1,r))
        for b in range(blocks):
            vs=range(k+6*b,k+6*(b+1))
            es.extend((u,v,(15,15,15,7)) for u,v in itertools.combinations(vs,2))
            es.extend((b,v,(7,7,14,14)) for v in vs)
            if kind=="coupled_modes":es.extend((b+1,v,(11,11,13,13)) for v in vs)
        # Random local value permutations and vertex relabeling preserve the
        # structural opportunities but hide literal value and vertex IDs.
        p=rng.sample(range(n),n);val=[rng.sample(range(D),D) for _ in range(n)]
        edges=[]
        for u,v,r in es:
            rows=[0]*D
            for a in range(D):
                for b in bits(r[a]):rows[val[u][a]]|=1<<val[v][b]
            x,y=p[u],p[v]
            if x>y:
                x,y=y,x
                rows=[sum(1<<a for a in range(D) if rows[a]&(1<<b)) for b in range(D)]
            edges.append((x,y,tuple(rows)))
        c=Case(f"fresh_m_{i:02d}",kind,n,(FULL,)*n,tuple(sorted(edges)))
        # Preserve the supplied structural cut order across all arms.
        cut=tuple(p[v] for v in range(k))
        assert Solver(c).ac(c.domains)==c.domains
        assert width(c)>=BAG_CAP and width(residual(c,c.domains,set(cut))[0])<BAG_CAP
        out.append((c,cut))
    return out


def fresh_queries(c,dom,cut):
    # Public forced-cutset workload, with both label and residual evidence.
    i,j=cut[0],cut[-1];r=next(v for v in range(c.n) if v not in cut)
    a,b=list(bits(dom[i]))[:2];d=list(bits(dom[j]))[-1]
    x,y=list(bits(dom[r]))[:2]
    return [{},{i:1<<a},{i:1<<b},{j:1<<d},{r:1<<x},{r:1<<y},
            {i:1<<a,j:1<<d},{i:1<<a,r:1<<x}]


def run(c,dom,cut,qs,truth,arm):
    lift.cache_clear();project.cache_clear()
    st=collections.Counter();t0=time.perf_counter()
    s=Solver(c);ac=s.ac(c.domains)
    assert dom==(ac if ac is not None else zero(c))
    raw=math.prod(dom[i].bit_count() for i in cut)
    engine=None;tree=None;base=None;reason=None
    try:
        if not all(dom):mode="contradiction"
        elif arm=="dfs_cover":mode="dfs"
        elif arm=="direct_tree":
            mode="tree";tree=Tree(c,dom)
            if not tree.available:raise Capped("width_cap")
        elif arm=="flat_forest":
            mode="flat"
            if raw>RAW_CAP:raise Capped("raw_label_cap")
            engine=Compiled(c,dom,{"cut":cut,"raw":raw},"prefix",st)
        else:
            mode="factor"
            engine=Factorized(c,dom,cut,arm,st)
    except Capped as exc:
        reason=str(exc)
        sec=time.perf_counter()-t0
        return {"root":c.root,"family":c.family,"arm":arm,"status":"cap","reason":reason,
                "n":c.n,"raw_product":raw,"cutset_size":len(cut),"setup_seconds":sec,
                "query_seconds":0.,"total_seconds":sec,"queries":0,"answers":[],
                "base_bits":0,"query_bits":0,"stats":dict(st)}
    setup=time.perf_counter()-t0
    t1=time.perf_counter();answers=[]
    if mode=="tree":base=tree.calibrate({},st)
    for q in qs:
        if mode=="contradiction":ans=zero(c)
        elif mode=="dfs":ans=dfs_cover(s,dom,q,st)
        elif mode=="tree":
            loc=base
            for i,m in q.items():loc=tree.conditioned(loc,i,next(bits(m)),st)
            ans=tree.supports(loc)
        else:ans=engine.answer(q)
        answers.append(ans)
    sec=time.perf_counter()-t1
    assert answers==truth,(c.root,arm,answers,truth)
    if mode=="factor":bb,qb=engine.storage()
    elif mode=="flat":
        # Same successful base-table and unary-mask capacity convention.
        bb=(sum(loc is not None for loc in engine.cache)*engine.tree.storage()
            +len(engine.cache)*D*len(engine.keep)) if engine.tree is not None else 0
        qb=0
    elif mode=="tree":bb=tree.storage() if base is not None else 0;qb=0
    else:bb=qb=0
    snap=dict(st)
    if engine is not None:
        for q,want in zip(qs,truth):
            got,w=engine.answer(q,with_witness=True)
            assert got==want and (w is not None)==bool(any(want))
    return {"root":c.root,"family":c.family,"arm":arm,"status":"exact","reason":reason,
            "n":c.n,"raw_product":raw,"cutset_size":len(cut),"setup_seconds":setup,
            "query_seconds":sec,"total_seconds":setup+sec,"queries":len(qs),"answers":answers,
            "base_bits":bb,"query_bits":qb,"stats":snap}


def aggregate(rows):
    a=[r for r in rows if r["status"]=="exact"]
    mean=lambda k:statistics.mean(r[k] for r in a) if a else None
    stat=lambda k:sum(r["stats"].get(k,0) for r in a)
    return {"roots":len(rows),"exact":len(a),"caps":len(rows)-len(a),
            "cap_reasons":dict(collections.Counter(r["reason"] for r in rows if r["status"]=="cap")),
            "queries":sum(r["queries"] for r in a),"mean_ms":1000*mean("total_seconds") if a else None,
            "mean_setup_ms":1000*mean("setup_seconds") if a else None,
            "mean_query_ms":1000*mean("query_seconds") if a else None,
            "base_bits":sum(r["base_bits"] for r in a),"query_bits":sum(r["query_bits"] for r in a),
            "raw_product":sum(r["raw_product"] for r in a),
            **{k:stat(k) for k in ("diagram_states","diagram_edges","compatible_labels","terminals",
                                   "compiled_units","base_calibrations","base_hits","query_hits","query_misses",
                                   "conditioned_calls","messages","forward_edge_scans","backward_edge_scans",
                                   "support_probes","dfs_nodes","branch_calibrations","residual_groups")}}


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    au=audit();print("AUDIT",json.dumps(au),flush=True)
    retro=[c for c in make_cases("stress_i",12,36,2026112562) if width(c)>=BAG_CAP]
    cohorts={"retro_capped":retro,"retro_bipartite":bipartite(),"retro_mixed":mixed()}
    assert [len(v) for v in cohorts.values()]==[8,18,24]
    planned={}
    for co,cs in cohorts.items():
        planned[co]=[]
        for c in cs:
            ac=Solver(c).ac(c.domains);dom=ac if ac is not None else zero(c)
            start=time.perf_counter()
            rec=Planner(c,dom).choose("profile_greedy") if all(dom) else {"cut":()}
            elapsed=time.perf_counter()-start
            assert rec is not None,(c.root,"retrospective planner cap")
            planned[co].append((c,rec["cut"],elapsed))
    planned["fresh_explosion"]=[(c,cut,0.) for c,cut in fresh()]
    tasks=(json.dumps({k:[{"case":c.__dict__,"cut":cut} for c,cut,_ in cs] for k,cs in planned.items()},
                      sort_keys=True,separators=(",",":"))+"\n").encode()
    (out/"tasks.json").write_bytes(tasks)
    rows=[];oracle=[];summary={}
    for co,cs in planned.items():
        local=[]
        for i,(c,cut,planning) in enumerate(cs):
            s=Solver(c);ac=s.ac(c.domains);dom=ac if ac is not None else zero(c)
            qs=fresh_queries(c,dom,cut) if co=="fresh_explosion" else query_list(c,dom)
            truth=[dfs_support(s,dom,q,collections.Counter()) for q in qs]
            oracle.append({"root":c.root,"cohort":co,"cut":cut,"queries":qs,"answers":truth,
                           "shared_plan_seconds":planning})
            off=i%len(ARMS)
            for arm in ARMS[off:]+ARMS[:off]:
                row=run(c,dom,cut,qs,truth,arm)
                local.append({**row,"shared_plan_seconds":planning})
            print("ROOT",co,c.root,c.n,len(cut),flush=True)
        rows.extend({**r,"cohort":co} for r in local)
        summary[co]={a:aggregate([r for r in local if r["arm"]==a]) for a in ARMS}
        summary[co]["families"]={f:{a:aggregate([r for r in local if r["family"]==f and r["arm"]==a])
                                   for a in ARMS} for f in sorted({r["family"] for r in local})}
        print("COHORT",co,json.dumps({a:summary[co][a] for a in ARMS}),flush=True)
        print("FAMILIES",co,json.dumps({f:{a:{k:x[k] for k in ("exact","mean_ms","base_bits","query_bits","base_calibrations","messages","compatible_labels","diagram_states","diagram_edges","query_hits")}
                                     for a,x in v.items()} for f,v in summary[co]["families"].items()}),flush=True)
    result={"schema":"lbf20m.factorized-cache.v1","scope":"development_only","audit":au,
            "task_sha256":hashlib.sha256(tasks).hexdigest(),"rows":len(rows),
            "state_cap":STATE_CAP,"edge_cap":EDGE_CAP,"raw_cap":RAW_CAP,"new_training_fits":0,
            "cohorts":summary}
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    for name,data in (("rows",rows),("oracle",oracle)):
        (out/f"{name}.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in data))
    print("COUNTS",json.dumps({k:v for k,v in result.items() if k!="cohorts"}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);main(p.parse_args().out)
