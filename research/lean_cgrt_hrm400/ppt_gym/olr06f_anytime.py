"""OLR-06F: anytime cost-partitioned abstractions under a total work cap.

Public graph exploration is a proposal mechanism. Partial Dijkstra tables
supply lower bounds; only the separate native replay/compiler certifies
the successful proof paths.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
import heapq
import math
import random
import time

from .data import digest, sha256_file
from .equality_suite import ALPHABET, RULE_COUNT, BANDS, choose_shell, neighbors, system_lineage
from .olr06d_search import partitions, hamming, MAX_REWRITES
from .olr06e_multigoal import (
    Root, COUNTS, Meter, bounded, exact, certify, load as load_old,
    read, write,
)

SEED = 20261228
OLD_SHA = "dd50ca18548b334c8b943b0122ece1305c3d223f7ff4f5f96b79c1f8d74e83e8"
CAPS = (128, 256, 512)
ARMS = (
    "cached_independent", "demand", "hamming",
    "eager2", "eager3", "cyclic_quarter", "demand_quarter", "demand_half",
)
REG = "olr06f.anytime_registry.v1"
SCORE = "olr06f.anytime_native_score.v1"


def freeze(old, out):
    if out.exists() or sha256_file(old/"manifest.json") != OLD_SHA:
        raise ValueError("pinned 06E source bank and create-only registry required")
    previous = load_old(old)
    excluded = set(read(old/"excluded.json"))
    excluded.update(r.lineage for r in previous)
    if len(excluded) != 276:
        raise ValueError("prior alpha lineage census changed")
    used = set(excluded)
    out.mkdir(parents=True)
    write(out/"excluded.json", sorted(excluded))
    pairs = ["".join(p) for p in product(ALPHABET, repeat=2)]
    available = list(combinations(pairs, 2))
    files = {}
    for split, (count, k) in COUNTS.items():
        rng = random.Random(f"olr06f/{SEED}/{split}")
        rows = []
        for attempt in range(count*1000):
            if len(rows) == count:
                break
            rules = tuple((a,b) if rng.randrange(2) else (b,a)
                          for a,b in rng.sample(available,RULE_COUNT))
            lineage = system_lineage(rules)
            if lineage in used:
                continue
            n = 10 if split == "transfer" else 7+len(rows)%3
            source = "".join(rng.choice(ALPHABET) for _ in range(n))
            targets = []
            for j in range(k):
                t = choose_shell(source,rules,BANDS[j%3],rng)
                if t is None or t in targets:
                    break
                targets.append(t)
            if len(targets) != k:
                continue
            rng.shuffle(targets)
            r = Root(split,rules,source,tuple(targets)).validate()
            used.add(lineage)
            rows.append(asdict(r))
        if len(rows) != count:
            raise ValueError("new root generator exhausted")
        path = out/(split+".json")
        write(path, rows)
        files[path.name] = sha256_file(path)
    files["excluded.json"] = sha256_file(out/"excluded.json")
    m = {"schema":REG,"seed":SEED,"counts":{s:list(v) for s,v in COUNTS.items()},
         "old_manifest_sha256":OLD_SHA,"excluded_lineages":276,"files":files}
    write(out/"manifest.json",m)
    return m


def load(bank):
    m = read(bank/"manifest.json")
    if (m["schema"]!=REG or m["seed"]!=SEED or m["old_manifest_sha256"]!=OLD_SHA
            or m["counts"]!={s:list(v) for s,v in COUNTS.items()}):
        raise ValueError("unregistered anytime cohort")
    for name,h in m["files"].items():
        if sha256_file(bank/name)!=h:
            raise ValueError("frozen root bytes changed")
    seen = set(read(bank/"excluded.json"))
    if len(seen)!=276:
        raise ValueError("prior lineage count differs")
    roots=[]
    for split,(count,_) in COUNTS.items():
        rows=read(bank/(split+".json"))
        if len(rows)!=count:
            raise ValueError("registered root count differs")
        for row in rows:
            if set(row)!=set(Root.__dataclass_fields__):
                raise ValueError("unexpected public metadata")
            r=Root(row["split"],tuple(tuple(e) for e in row["equations"]),
                   row["source"],tuple(row["targets"])).validate()
            if r.split!=split or r.lineage in seen:
                raise ValueError("alpha lineage or split overlap")
            seen.add(r.lineage)
            roots.append(r)
    return roots


class PartialTable:
    """Incremental exact reverse Dijkstra in half-rewrite cost units."""
    def __init__(self,rules,target,left,right):
        self.rules,self.target,self.left,self.right=rules,target,left,right
        self.directed=tuple(e for a,b in rules for e in ((a,b),(b,a)))
        self.dist={target:0}
        self.heap=[(0,target)]
        self.settled={}
        self.expansions=self.edges=0

    def frontier(self):
        while self.heap:
            d,w=self.heap[0]
            if w not in self.settled and self.dist[w]==d:
                return d
            heapq.heappop(self.heap)
        return math.inf

    def complete(self):
        return not math.isfinite(self.frontier())

    def bound(self,pattern):
        # An unsettled state cannot have a shorter distance than the current
        # Dijkstra frontier minimum. Once exhausted, unreachable is infinity.
        if pattern in self.settled:
            return self.settled[pattern]
        return self.frontier()

    def step(self):
        if self.complete():
            raise ValueError("cannot charge an exhausted abstract table")
        d,w=heapq.heappop(self.heap)
        self.settled[w]=d
        self.expansions+=1
        moves=[(m.next_word,2) for m in neighbors(w,self.rules)]
        for a,b in self.directed:
            if self.left and w[0]==a[1] and a[1]!=b[1]:
                moves.append((b[1]+w[1:],1))
            if self.right and w[-1]==a[0] and a[0]!=b[0]:
                moves.append((w[:-1]+b[0],1))
        self.edges+=len(moves)
        for v,c in moves:
            nd=d+c
            if nd<self.dist.get(v,math.inf):
                if v in self.settled:
                    raise ValueError("abstract Dijkstra settled distance decreased")
                self.dist[v]=nd
                heapq.heappush(self.heap,(nd,v))


class AnytimePDB:
    """Shared partial pattern tables; all available partitions remain admissible."""
    def __init__(self,rules,goal,width=3):
        self.goal=goal
        self.parts=tuple(p for p in partitions(len(goal))
                         if all(hi-lo<=width for lo,hi in p))
        self.tables={}
        self.windows={}
        for part in self.parts:
            for lo,hi in part:
                key=(goal[lo:hi],lo>0,hi<len(goal))
                if key not in self.tables:
                    self.tables[key]=PartialTable(rules,*key)
                self.windows[lo,hi]=key
        self.order=tuple(sorted(self.tables,key=lambda k:(len(k[0]),k)))
        self.cursor=0
        self.expansions=self.edges=0

    def bound(self,w):
        if len(w)!=len(self.goal):
            raise ValueError("word outside pattern scope")
        best=hamming(w,self.goal)
        for part in self.parts:
            d=sum(self.tables[self.windows[lo,hi]].bound(w[lo:hi]) for lo,hi in part)
            if not math.isfinite(d):
                return math.inf
            best=max(best,(d+1)//2)
        return best

    def choose(self,mode,words=()):
        live=[k for k in self.order if not self.tables[k].complete()]
        if not live:
            return None
        if mode=="eager":
            return live[0]
        if mode=="cyclic":
            for _ in self.order:
                key=self.order[self.cursor%len(self.order)]
                self.cursor+=1
                if key in live:
                    return key
            raise ValueError("inconsistent live abstraction roster")
        if mode!="demand":
            raise ValueError("unknown table scheduler")
        counts=defaultdict(int)
        for w in words:
            for (lo,hi),key in self.windows.items():
                table=self.tables[key]
                if key in live and w[lo:hi] not in table.settled:
                    counts[key]+=1
        # No unobserved future utility or reference path. Refine tables used
        # by current queue heads; charge every abstract enumeration.
        if not counts:
            return None
        return min(counts,key=lambda k:(-Fraction(counts[k],4**len(k[0])),len(k[0]),k))

    def step(self,key):
        table=self.tables[key]
        before=table.edges
        table.step()
        self.expansions+=1
        self.edges+=table.edges-before


class Queue:
    """One target's search, with lazy rekeying of monotone increasing bounds."""
    def __init__(self,target,source,bound):
        self.source,self.bound=source,bound
        h=bound(target)
        self.heap=[(1.5*h,h,0,target,(target,))]
        self.best={target:0}
        self.closed={}
        self.done=False
        self.path=None
        self.rekeys=0

    def peek(self):
        if self.done:
            return None
        while self.heap:
            entry=self.heap[0]
            _,old,g,w,path=entry
            if g!=self.best[w] or g>=self.closed.get(w,math.inf) or g>=MAX_REWRITES:
                heapq.heappop(self.heap)
                continue
            h=self.bound(w)
            if h<old:
                raise ValueError("partial pattern bound decreased")
            if not math.isfinite(h):
                heapq.heappop(self.heap)
                continue
            if h>old:
                heapq.heapreplace(self.heap,(g+1.5*h,h,g,w,path))
                self.rekeys+=1
                continue
            return entry
        self.done=True
        return None

    def step(self,meter):
        if self.peek() is None:
            return
        _,h,g,w,path=heapq.heappop(self.heap)
        self.closed[w]=g
        for move in meter.visit(w):
            v=move.next_word
            if v==self.source:
                self.path=list(path+(v,))
                self.done=True
                return
            if g+1<self.best.get(v,math.inf):
                self.best[v]=g+1
                hv=self.bound(v)
                if math.isfinite(hv):
                    heapq.heappush(self.heap,(g+1+1.5*hv,hv,g+1,v,path+(v,)))


def path_audit(root,row,cap):
    if row["total_work"]>cap or row["logical"]!=row["expansions"]+row["hits"]:
        raise ValueError("total work or cache accounting violated")
    if row["total_work"]!=row["expansions"]+row["abstract_expansions"]:
        raise ValueError("unmetered abstract/concrete enumeration")
    if set(row["found"])-set(root.targets):
        raise ValueError("unknown target")
    for target,q in row["found"].items():
        path=q["words"]
        total=q["fresh_at"]+q["abstract_at"]
        if (path[0]!=target or path[-1]!=root.source
                or not 1<=len(path)-1<=MAX_REWRITES
                or q["found_at"] not in (total,total+1)
                or not 1<=q["found_at"]<=cap
                or total>row["total_work"]):
            raise ValueError("path/discovery budget mismatch")
        for a,b in zip(path,path[1:]):
            if not any(m.next_word==b for m in neighbors(a,root.equations)):
                raise ValueError("nonpublic rewrite edge")


def anytime(root,arm,cap):
    if arm not in ARMS or type(cap) is not int or cap<1:
        raise ValueError("invalid bounded policy")
    if arm in ("cached_independent","demand"):
        row=bounded(root,arm,cap)
        row.update(total_work=row["expansions"],rekeys=0)
        for q in row["found"].values():
            q["abstract_at"]=0
        path_audit(root,row,cap)
        return row
    start=time.perf_counter()
    meter=Meter(root.equations,cap,True)
    pdb=None if arm=="hamming" else AnytimePDB(root.equations,root.source,2 if arm=="eager2" else 3)
    bound=(lambda w:hamming(w,root.source)) if pdb is None else pdb.bound
    found={}
    setup_seconds=0.
    def abstract_work():
        return pdb.expansions if pdb is not None else 0
    def total():
        return meter.expansions+abstract_work()
    def step_table(key):
        nonlocal setup_seconds
        t=time.perf_counter()
        pdb.step(key)
        setup_seconds+=time.perf_counter()-t
    if arm.startswith("eager"):
        while total()<cap:
            key=pdb.choose("eager")
            if key is None:
                break
            step_table(key)
    fronts=[Queue(t,root.source,bound) for t in root.targets]
    cursor=0
    period=2 if arm=="demand_half" else 4
    while total()<cap and not all(q.done for q in fronts):
        # The cycle is independent of the resource cap, so a larger cap
        # extends exactly the same prefix, including after a cache hit.
        if pdb is not None and not arm.startswith("eager") and (meter.logical+abstract_work())%period==0:
            if arm=="cyclic_quarter":
                key=pdb.choose("cyclic")
            else:
                heads=[q.peek() for q in fronts]
                key=pdb.choose("demand",[e[3] for e in heads if e is not None])
            if key is not None:
                step_table(key)
                continue
        for _ in fronts:
            i=cursor%len(fronts)
            cursor+=1
            if fronts[i].done:
                continue
            q=fronts[i]
            q.step(meter)
            if q.path is not None:
                found[root.targets[i]]={"words":q.path,"fresh_at":meter.expansions,
                        "abstract_at":abstract_work(),
                        "found_at":meter.required_cap+abstract_work()}
            break
    row={"found":found,**meter.stats(),"total_work":total(),
         "abstract_expansions":abstract_work(),"abstract_edges":pdb.edges if pdb else 0,
         "rekeys":sum(q.rekeys for q in fronts),"pdb_seconds":setup_seconds,
         "seconds":time.perf_counter()-start}
    path_audit(root,row,cap)
    return row


def at(row,cap):
    return sum(q["found_at"]<=cap for q in row["found"].values())


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r["split"],r["arm"]].append(r)
    result={}
    for (split,arm),group in sorted(groups.items()):
        k=COUNTS[split][1];n=len(group)
        result[f"{split}:{arm}"]={
            "roots":n,"leaves":n*k,
            **{f"full{c}":sum(bool(r["joint_lemma"]) and at(r,c)==k for r in group) for c in CAPS},
            **{f"leaves{c}":sum(at(r,c) for r in group) for c in CAPS},
            "full_final":sum(bool(r["joint_lemma"]) for r in group),
            "leaf_final":sum(len(r["found"]) for r in group),
            **{f"mean_{key}":sum(r[key] for r in group)/n for key in
               ("total_work","expansions","edges","logical","logical_edges","hits",
                "abstract_expansions","abstract_edges","seconds","pdb_seconds","rekeys")},
            "native_leaf_tactics":sum(q["native_tactics"] for r in group for q in r["found"].values()),
        }
    return result


def contrast(rows,split,left,right,cap):
    index={(r["root"],r["arm"]):r for r in rows if r["split"]==split}
    ids=sorted({i for i,a in index});k=COUNTS[split][1]
    diffs=[int(at(index[i,left],cap)==k)-int(at(index[i,right],cap)==k) for i in ids]
    rng=random.Random(f"olr06f/{split}/{left}/{right}/{cap}")
    mean=lambda x:sum(x)/len(x)
    draws=sorted(mean([rng.choice(diffs) for _ in ids]) for _ in range(5000))
    return {"difference":mean(diffs),"ci95":[draws[125],draws[4875]],"roots":len(ids)}


def score(bank,out):
    if out.exists():
        raise ValueError("create-only evaluation required")
    roots=load(bank);out.mkdir(parents=True)
    start=time.perf_counter();rows=[]
    for root in roots:
        for arm in ARMS:
            r=anytime(root,arm,max(CAPS))
            rows.append({"root":root.root,"split":root.split,"arm":arm,**r})
    write(out/"frozen_bounded.json",rows)
    for root in roots:
        r=exact(root)
        r.update(total_work=r["expansions"],rekeys=0)
        for q in r["found"].values():
            q["abstract_at"]=0
        rows.append({"root":root.root,"split":root.split,"arm":"exact",**r})
    cert=certify(roots,rows,out)
    write(out/"root_scores.json",rows)
    summary=summarize(rows)
    best={s:max(summary[f"{s}:{a}"]["full512"] for a in ARMS) for s in COUNTS}
    k_by_id={r.root:len(r.targets) for r in roots}
    solved={r["root"] for r in rows if r["arm"] in ARMS and r["joint_lemma"]
            and at(r,512)==k_by_id[r["root"]]}
    total_best=sum(best.values())
    portfolio_gap=(len(solved)-total_best)/len(roots)
    for split,(count,_) in COUNTS.items():
        if summary[f"{split}:exact"]["full_final"]!=count:
            raise ValueError("unrestricted native ceiling failed")
    files=("frozen_bounded.json","root_scores.json","native_episodes.json","conjunctions.lean",
           "conjunction_compiler.log","leaves/standalone.lean","leaves/compiler.log","leaves/compiler_receipt.json")
    m={"schema":SCORE,"registry_sha256":sha256_file(bank/"manifest.json"),
       "roots":len(roots),"target_obligations":sum(len(r.targets) for r in roots),
       "policy_trials":len(rows),"bounded_frozen_trials":len(roots)*len(ARMS),
       "certificate":cert,"scorecard":summary,"best_full512_by_split":best,
       "hindsight_portfolio_full512":len(solved),"hindsight_gap":portfolio_gap,
       "headroom":total_best<=.8*len(roots) and portfolio_gap>=.1,
       "contrasts":{f"{s}:demand_quarter-{a}:{cap}":contrast(rows,s,"demand_quarter",a,cap)
                    for s in COUNTS for a in ARMS if a!="demand_quarter" for cap in CAPS},
       "model_training_updates":0,"architecture_promotion_eligible":False,
       "elapsed_seconds":time.perf_counter()-start,
       "files":{f:sha256_file(out/f) for f in files}}
    write(out/"manifest.json",m)
    return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest="op",required=True)
    a=sp.add_parser("freeze");a.add_argument("--old-bank",type=Path,required=True);a.add_argument("--out",type=Path,required=True)
    b=sp.add_parser("score");b.add_argument("--bank",type=Path,required=True);b.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    from .data import canonical
    print(canonical(freeze(args.old_bank,args.out) if args.op=="freeze" else score(args.bank,args.out)),flush=True)


if __name__=="__main__":main()
