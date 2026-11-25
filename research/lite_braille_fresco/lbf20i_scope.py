"""LBF-20I: exact local bag joins, selective ternary persistence and elimination.

Standard-library CPU implementation. Sound local pruning never reads exact
evaluation labels. Work units and actual wall time are reported separately.
"""
from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import itertools
import json
import random
import statistics
import time
from pathlib import Path
from lbf20f_csp import Case, Solver, D, FULL, bits, generate

ARMS = ("ac_lcv", "singleton", "dsatur_fallback", "ve6",
        "pair", "tight_pair", "tight_ternary", "uniform_ternary",
        "minfill_ternary", "full_ternary")
CAP = 128


def full(k):
    return (1 << D**k) - 1


def digit(t, p):
    return (t // D**p) % D


def code(values):
    return sum(a * D**p for p, a in enumerate(values))


def subindex(t, pos):
    return code(digit(t, p) for p in pos)


@functools.lru_cache(None)
def partitions(k, pos):
    out = [0] * D**len(pos)
    for t in range(D**k):
        out[subindex(t, pos)] |= 1 << t
    return tuple(out)


@functools.lru_cache(32768)
def lift(k, pos, mask):
    if mask == full(len(pos)):
        return full(k)
    p = partitions(k, pos)
    result = 0
    for i in bits(mask):
        result |= p[i]
    return result


@functools.lru_cache(32768)
def project(k, pos, mask):
    return sum(1 << i for i, block in enumerate(partitions(k, pos)) if block & mask)


def positions(bag, face):
    return tuple(bag.index(i) for i in face)


def scopes(bag, k):
    return itertools.combinations(bag, k)


def initial(c, dom):
    rel = c.relations()
    tables = {(i,): dom[i] for i in range(c.n)}
    for i, j in itertools.combinations(range(c.n), 2):
        rows = rel.get((i,j),(FULL,)*D)
        tables[i,j] = sum(1 << code((a,b)) for a in bits(dom[i]) for b in bits(dom[j]&rows[a]))
    return tables


class Engine:
    def __init__(self, c, dom):
        self.c = c
        self.tables = initial(c, dom)
        self.stats = collections.Counter(bag_revisions=0, face_lifts=0, face_projections=0,
                                         nominal_assignment_slots=0, removed_entries=0, updates=0)
        self.contradiction = any(not v for v in self.tables.values())

    def promote(self, bags, rank):
        # Persist every ternary face of promoted bags, or only existing pairs.
        if rank >= 3:
            for bag in bags:
                for face in scopes(bag,3):
                    self.tables.setdefault(face, full(3))
        if rank >= 4:
            for bag in bags:
                for face in scopes(bag,4):
                    self.tables.setdefault(face, full(4))

    def propagate(self, bags):
        bags = tuple(sorted(set(bags)))
        faces = {}
        consumers = collections.defaultdict(list)
        for bag in bags:
            faces[bag] = [(face, positions(bag,face)) for k in range(1,len(bag)+1)
                          for face in scopes(bag,k) if face in self.tables]
            for face, pos in faces[bag]:
                consumers[face].append(bag)
        pending = set(bags)
        q = collections.deque(bags)
        while q and not self.contradiction:
            bag = q.popleft()
            pending.remove(bag)
            k = len(bag)
            self.stats["bag_revisions"] += 1
            self.stats["nominal_assignment_slots"] += D**k
            if self.stats["bag_revisions"] > 200000:
                raise RuntimeError("closure revision cap exceeded")
            joint = full(k)
            for face,pos in faces[bag]:
                joint &= lift(k,pos,self.tables[face])
                self.stats["face_lifts"] += 1
                if not joint:
                    self.contradiction = True
                    return
            # All projections come from the same join. Changing a face queues
            # every bag depending on it, including the current bag.
            for face,pos in faces[bag]:
                old = self.tables[face]
                new = project(k,pos,joint)
                self.stats["face_projections"] += 1
                assert new & old == new
                if new != old:
                    self.tables[face] = new
                    self.stats["updates"] += 1
                    self.stats["removed_entries"] += old.bit_count()-new.bit_count()
                    for b in consumers[face]:
                        if b not in pending:
                            pending.add(b); q.append(b)

    def domain(self):
        return (0,)*self.c.n if self.contradiction else tuple(self.tables[i,] for i in range(self.c.n))

    def storage(self):
        # Logical packed bits, not Python object/RSS memory.
        return sum(D**len(face) for face in self.tables)


def reference(c,dom,bags,rank):
    """Independent synchronous tuple enumeration; used only for tiny audit."""
    tab = initial(c,dom)
    for bag in bags:
        for k in range(3,rank+1):
            for f in scopes(bag,k):
                tab.setdefault(f,full(k))
    if any(not v for v in tab.values()):
        return None
    while True:
        nxt=dict(tab)
        for bag in bags:
            faces=[f for k in range(1,len(bag)+1) for f in scopes(bag,k) if f in tab]
            vals=[]
            for v in itertools.product(range(D),repeat=len(bag)):
                a=dict(zip(bag,v))
                if all(tab[f] & (1<<code(a[i] for i in f)) for f in faces):
                    vals.append(a)
            if not vals:return None
            for f in faces:
                allowed={code(v[i] for i in f) for v in vals}
                nxt[f]&=sum(1<<a for a in allowed)
        if any(not v for v in nxt.values()):return None
        if nxt==tab:return tab
        tab=nxt


def minfill(c):
    adj=[set() for _ in range(c.n)]
    for i,j,_ in c.edges:adj[i].add(j);adj[j].add(i)
    remaining=set(range(c.n));order=[]
    while remaining:
        def key(i):
            nb=sorted(adj[i]&remaining)
            return (sum(b not in adj[a] for a,b in itertools.combinations(nb,2)),len(nb),i)
        i=min(remaining,key=key)
        nb=adj[i]&remaining
        order.append((i,tuple(sorted(nb|{i}))))
        for a,b in itertools.combinations(nb,2):adj[a].add(b);adj[b].add(a)
        remaining.remove(i)
    return order


def ve(c,dom,cap=6):
    """Exact Boolean bucket elimination with witness reconstruction, or width cap."""
    order=minfill(c)
    width=max(len(b)-1 for _,b in order)
    if width+1>cap:return {"status":"width_cap","solution":None,"width":width,"slots":0}
    factors={(i,):d for i,d in enumerate(dom)}
    it=initial(c,dom)
    for i,j,_ in c.edges:factors[i,j]=it[i,j]
    hist=[];slots=0
    for i,expected in order:
        touched=[f for f in factors if i in f]
        bag=tuple(sorted(set().union(*(set(f) for f in touched)))) if touched else (i,)
        assert set(bag)<=set(expected)
        joint=full(len(bag));slots+=D**len(bag)
        for f in touched:
            joint&=lift(len(bag),positions(bag,f),factors.pop(f))
        if not joint:return {"status":"refuted","solution":None,"width":width,"slots":slots}
        hist.append((i,bag,joint))
        rest=tuple(j for j in bag if j!=i)
        if rest:
            p=project(len(bag),positions(bag,rest),joint)
            factors[rest]=factors.get(rest,full(len(rest)))&p
    assigned={}
    for i,bag,joint in reversed(hist):
        for a in bits(dom[i]):
            vals=[a if j==i else assigned[j] for j in bag]
            if joint&(1<<code(vals)):
                assigned[i]=a;break
        assert i in assigned
    v=tuple(assigned[i] for i in range(c.n))
    assert c.valid(v)
    return {"status":"solved","solution":v,"width":width,"slots":slots}


def choose(c,engine,policy,count):
    allbags=list(itertools.combinations(range(c.n),4))
    if policy=="full_ternary":return allbags
    if policy=="uniform_ternary":
        rng=random.Random(int.from_bytes(hashlib.sha256(c.root.encode()).digest()[:8],"big"))
        rng.shuffle(allbags);return allbags[:count]
    t=engine.tables
    def key(b):
        return (sum(1-t[f].bit_count()/max(1,t[f[0],].bit_count()*t[f[1],].bit_count())
                    for f in scopes(b,2)),b)
    ordered=sorted(allbags,key=key,reverse=True)
    if policy=="minfill_ternary":
        priority=[]
        # Favor bags from the actual elimination plan; pad small bags to a
        # four-scope with the tightest available containing superbag.
        for _,bag in minfill(c):
            if len(bag)>4:continue
            bs=set(bag)
            hit=next((b for b in ordered if bs<=set(b)),None)
            if hit is not None and hit not in priority:priority.append(hit)
        ordered=priority+[b for b in ordered if b not in priority]
    return ordered[:count]


def color_case(root,n,edges,d=3,domains=None,family="color"):
    mask=(1<<d)-1
    neq=tuple(mask^(1<<a) if a<d else 0 for a in range(D))
    return Case(root,family,n,tuple(domains or [mask]*n),
                tuple((i,j,neq) for i,j in sorted({tuple(sorted(e)) for e in edges})))


def make_cases(split,n,count,seed):
    rng=random.Random(seed)
    out=[]
    mixed=generate(split+"_mixed",n,count//4,seed+100,allow_unsat=True)
    # Remove functional parity rings from the hard nonfunctional cohort.
    mixed=generate(split+"_mixed",n,3*(count//4),seed+101,allow_unsat=True)
    mixed=[c for c in mixed if c.family!="parity_ring"][:count//4]
    assert len(mixed)==count//4
    for i in range(count):
        family=i%4
        if family==0:
            out.append(mixed[i//4]);continue
        perm=rng.sample(range(n),n)
        if family==1:
            # Random colorable planted or unrestricted graph.
            col=[rng.randrange(3) for _ in range(n)]
            e=[(u,v) for u,v in itertools.combinations(range(n),2)
               if (i//4%2 or col[u]!=col[v]) and rng.random()<(.48 if i//4%2 else .65)]
            out.append(color_case(f"{split}_{i:03d}",n,e,family="random_color"))
        elif family==2:
            # Wheel with alternating even/odd rim, random relabeling and
            # attached tree leaves. Odd wheel is a K4-free obstruction.
            rim=5 if (i//4)%2 else 4
            e=[(0,j) for j in range(1,rim+1)]
            e += [(j,1+(j%rim)) for j in range(1,rim+1)]
            for j in range(rim+1,n):e.append((j,rng.randrange(j)))
            out.append(color_case(f"{split}_{i:03d}",n,[(perm[u],perm[v]) for u,v in e],family="wheel"))
        else:
            # Conditional wheel: selector value 0 restricts odd-wheel vertices
            # to three colors, value 1 allows four. All local facts public.
            # The selector's impossible value may require a larger bag.
            sel,hub=perm[0],perm[1];rim=perm[2:7]
            e=[(hub,j) for j in rim]+[(rim[j],rim[(j+1)%5]) for j in range(5)]
            for j in perm[7:]:e.append((j,rng.choice(rim)))
            base=color_case(f"{split}_{i:03d}",n,e,4,family="conditional_wheel")
            ds=[FULL]*n;ds[sel]=3
            rows=tuple([7,15,0,0])
            # Orient selector restriction correctly under sorted endpoints.
            extra=[]
            for j in [hub,*rim]:
                if sel<j:extra.append((sel,j,rows))
                else:extra.append((j,sel,tuple(sum(1<<a for a in range(D) if rows[a]&(1<<b)) for b in range(D))))
            out.append(Case(base.root,base.family,n,tuple(ds),tuple(sorted((*base.edges,*extra)))))
    assert len({c.root for c in out})==count
    return out


def audit():
    count=collections.Counter()
    for c in generate("audit_i",4,48,2026112560,allow_unsat=True):
        worlds=list(itertools.product(range(D),repeat=c.n))
        feasible=[v for v in worlds if c.valid(v)]
        pc=tuple(itertools.combinations(range(c.n),3));quads=tuple(itertools.combinations(range(c.n),4))
        for rank in (2,3):
            e=Engine(c,c.domains);e.promote(quads,rank);e.propagate((*pc,*quads))
            r=reference(c,c.domains,(*pc,*quads),rank)
            assert (r is None)==e.contradiction,(c.root,rank)
            if r is not None:assert r==e.tables
            for v in feasible:
                assert not e.contradiction
                for f,m in e.tables.items():
                    assert m&(1<<code(v[i] for i in f))
                    count["support_checks"]+=1
            count["fixed_table_checks"]+=1
        z=ve(c,c.domains);assert (z["status"]=="solved")==bool(feasible)
        count["elimination_checks"]+=1
        count["systems"]+=1;count["worlds"]+=len(worlds)
    ob=[]
    for d in (3,4):
        n=d+1
        c=color_case(f"K{n}",n,itertools.combinations(range(n),2),d)
        pc=tuple(itertools.combinations(range(n),3))
        e=Engine(c,c.domains);e.propagate(pc)
        assert not e.contradiction and e.domain()==c.domains
        quad=tuple(itertools.combinations(range(n),4))
        e.promote(quad,3);e.propagate((*pc,*quad))
        four=e.contradiction
        assert four==(d==3)
        if not four:
            five=tuple(itertools.combinations(range(n),5))
            e.promote(five,4);e.propagate((*pc,*quad,*five))
            assert e.contradiction
        z=ve(c,c.domains);assert z["status"]=="refuted"
        assert not any(c.valid(v) for v in itertools.product(range(d),repeat=n))
        ob.append({"case":c.root,"pair_contradiction":False,"four_scope_contradiction":four,
                   "five_scope_contradiction":True if d==4 else None,"ve_width":z["width"]})
    return {**dict(count),"obstructions":ob,"mismatches":0}


def dsatur(c,dom,budget=CAP):
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
        def f(i):return {vals[j] for j in nb[i] if vals[j]>=0}
        i=max((i for i,v in enumerate(vals) if v<0),key=lambda i:(len(f(i)),len(nb[i]),-i))
        for a in bits(dom[i]):
            if a in f(i):continue
            vals[i]=a;v=visit();vals[i]=-1
            if v is not None:return v
            if capped:return None
        return None
    v=visit()
    return {"status":"solved" if v is not None else "budget" if capped else "refuted","solution":v,"nodes":nodes}


def run(c,truth,arm):
    lift.cache_clear();project.cache_clear()
    t=time.perf_counter();s=Solver(c);ac=s.ac(c.domains)
    initial_dom=ac if ac is not None else (0,)*c.n
    q=initial_dom;e=None;bags=[];used_ve=None;ss=0
    before=sum(m.bit_count() for m in initial_dom)-sum(m.bit_count() for m in truth)
    pair_excess=None;pair_empty=None
    if ac is not None:
        if arm=="singleton":
            masks=[]
            for i,m in enumerate(ac):
                keep=0
                for a in bits(m):
                    d=list(ac);d[i]=1<<a;ss+=1
                    if s.ac(tuple(d)) is not None:keep|=1<<a
                masks.append(keep)
            q=tuple(masks)
        elif arm=="ve6":
            used_ve=ve(c,ac)
        elif arm in ("pair","tight_pair","tight_ternary","uniform_ternary","minfill_ternary","full_ternary"):
            pc=list(itertools.combinations(range(c.n),3))
            e=Engine(c,ac);e.propagate(pc);q=e.domain()
            pair_excess=sum(m.bit_count() for m in q)-sum(m.bit_count() for m in truth)
            pair_empty=e.contradiction
            if arm!="pair" and not e.contradiction and any(m.bit_count()>1 for m in q):
                bags=choose(c,e,arm,2*c.n)
                rank=2 if arm=="tight_pair" else 3
                e.promote(bags,rank)
                e.propagate((*pc,*bags));q=e.domain()
    pre=time.perf_counter()-t
    t=time.perf_counter()
    if used_ve is not None and used_ve["status"]!="width_cap":
        z={**used_ve,"nodes":0}
    elif arm=="dsatur_fallback" and ac is not None and c.family in ("wheel","random_color"):
        z=dsatur(c,q)
    else:z=s.solve(q,budget=CAP)
    sec=time.perf_counter()-t
    assert z["status"]!="refuted" or not any(truth),(c.root,arm)
    assert z["status"]!="solved" or (any(truth) and c.valid(z["solution"]))
    assert all(a&b==a for a,b in zip(truth,q)),(c.root,arm)
    excess=sum(m.bit_count() for m in q)-sum(m.bit_count() for m in truth)
    assert 0<=excess<=before
    return {"root":c.root,"family":c.family,"n":c.n,"arm":arm,"sat":bool(any(truth)),
            "status":z["status"],"nodes":z["nodes"],"solution":z["solution"],
            "initial_excess":before,"pair_excess":pair_excess,"remaining_excess":excess,
            "pc_contradiction":pair_empty,"filter_contradiction":not all(q),
            "quad_bags":len(bags),"ternary_scopes":sum(len(f)==3 for f in e.tables) if e else 0,
            "logical_bits":e.storage() if e else 0,"pre_seconds":pre,"search_seconds":sec,
            "total_seconds":pre+sec,"ve":used_ve,"singleton_probes":ss,
            "stats":dict(e.stats) if e else {}}


def aggregate(rows):
    mean=lambda key:statistics.mean(r[key] for r in rows)
    rem=sum(r["remaining_excess"] for r in rows);base=sum(r["initial_excess"] for r in rows)
    return {"roots":len(rows),"sat":sum(r["sat"] for r in rows),
            "resolved":sum(r["status"]!="budget" for r in rows),
            "refuted_in_filter":sum(r["filter_contradiction"] for r in rows),
            "initial_excess":base,"remaining_excess":rem,"fraction_removed":1-rem/base if base else None,
            "mean_nodes":mean("nodes"),"mean_pre_ms":1000*mean("pre_seconds"),
            "mean_total_ms":1000*mean("total_seconds"),"mean_quads":mean("quad_bags"),
            "mean_ternary_scopes":mean("ternary_scopes"),"mean_logical_bits":mean("logical_bits"),
            "mean_bag_revisions":statistics.mean(r["stats"].get("bag_revisions",0) for r in rows),
            "mean_nominal_slots":statistics.mean(r["stats"].get("nominal_assignment_slots",0) for r in rows),
            "ve_exact":sum(r["ve"] is not None and r["ve"]["status"]!="width_cap" for r in rows),
            "ve_width_cap":sum(r["ve"] is not None and r["ve"]["status"]=="width_cap" for r in rows)}


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    au=audit();print("AUDIT",json.dumps(au),flush=True)
    cohorts={"dev8":make_cases("dev_i",8,48,2026112561),
             "stress12":make_cases("stress_i",12,36,2026112562)}
    task=(json.dumps({k:[c.__dict__ for c in v] for k,v in cohorts.items()},
                     sort_keys=True,separators=(",",":"))+"\n").encode()
    (out/"tasks.json").write_bytes(task)
    rows=[];summary={}
    for name,cs in cohorts.items():
        local=[]
        for i,c in enumerate(cs):
            truth=Solver(c).supports(c.domains)
            offset=i%len(ARMS)
            for arm in ARMS[offset:]+ARMS[:offset]:
                local.append(run(c,truth,arm))
        rows.extend({**r,"cohort":name} for r in local)
        summary[name]={a:aggregate([r for r in local if r["arm"]==a]) for a in ARMS}
        family={f:{a:aggregate([r for r in local if r["arm"]==a and r["family"]==f])
                    for a in ARMS} for f in sorted({r["family"] for r in local})}
        summary[name]["families"]=family
        # Paired gains over same-scope no-ternary control and full pair closure.
        print("COHORT",name,json.dumps({a:summary[name][a] for a in ARMS}),flush=True)
        print("FAMILIES",name,json.dumps({f:{a:{"excess":x["remaining_excess"],"refuted":x["refuted_in_filter"]}
             for a,x in v.items()} for f,v in family.items()}),flush=True)
    full_summary={"schema":"lbf20i.adaptive-scope.v1","scope":"development_only","audit":au,
                  "task_sha256":hashlib.sha256(task).hexdigest(),"rows":len(rows),
                  "new_training_fits":0,"cohorts":summary}
    (out/"summary.json").write_text(json.dumps(full_summary,indent=2)+"\n")
    (out/"rows.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
    print("COUNTS",json.dumps({k:v for k,v in full_summary.items() if k!="cohorts"}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);main(p.parse_args().out)
