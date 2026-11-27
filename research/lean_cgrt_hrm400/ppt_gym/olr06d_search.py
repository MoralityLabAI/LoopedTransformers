"""OLR-06D: public exact abstractions, source-only rewrite distances, bounded search.

Reuse the upstream native equality grammar. These planners propose paths.
A capped miss is censored, never a logical negative receipt.
"""
from __future__ import annotations
from collections import deque
from dataclasses import asdict
from itertools import combinations, product
from pathlib import Path
import heapq
import json
import math
import random
import time

from .data import canonical,digest,sha256_file
from .equality_suite import (
    EqCase,ALPHABET,RULE_COUNT,BANDS,choose_shell,neighbors,system_lineage,
)

SEED=20261220
COUNTS={"train":48,"calibration":12,"dev":16,"transfer":16}
OLD_SHA="d928d8b5ffbbd9760e02fc9af903704bdac0cea7adc0b6e5be9103e7f2153874"
FRAMES=12
CAPS=(64,128)
MAX_REWRITES=15
SCHEMA="olr06d.opaque_rewrite_registry.v1"


def make_case(rules,source,target,split):
    lineage=system_lineage(rules)
    # All non-training cohorts are new unsealed calibration-type cases.
    # The external registry records the distinct calibration/dev/transfer roles.
    native_split="train" if split=="train" else "calibration"
    root=digest([lineage,native_split,rules,source,target])[:24]
    return EqCase(root,lineage,native_split,"indexed_unary_equations",rules,source,target).validate()


def source_ball(case):
    """Complete preceding BFS layers give exact distances to every reached word.

    Stop after target discovery and at least FRAMES nontrivial states.
    Only TRAIN and source calibration roots may request this teacher.
    """
    parent={case.source:None};dist={case.source:0}
    todo=deque([case.source]);expansions=edges=0
    while todo and (case.target not in parent or len(parent)<FRAMES+1):
        word=todo.popleft()
        if dist[word]>=11:continue
        if expansions>=16000:break
        moves=neighbors(word,case.equations)
        expansions+=1;edges+=len(moves)
        for move in moves:
            v=move.next_word
            if v not in parent:
                parent[v]=word;dist[v]=dist[word]+1;todo.append(v)
    if case.target not in parent or len(parent)<FRAMES+1:
        raise ValueError("bounded teacher cannot supply registered reference/frames")
    return parent,dist,expansions,edges


def path_to_goal(parent,word):
    path=[word]
    while parent[path[-1]] is not None:path.append(parent[path[-1]])
    return path


def source_frames(case):
    parent,dist,exp,edges=source_ball(case)
    rng=random.Random(case.root+"/olr06d_frames")
    buckets={}
    for w,d in sorted(dist.items()):
        if w not in (case.target,case.source):buckets.setdefault(d,[]).append(w)
    for values in buckets.values():rng.shuffle(values)
    selected=[case.target]
    while len(selected)<FRAMES:
        for d in sorted(buckets):
            if buckets[d] and len(selected)<FRAMES:selected.append(buckets[d].pop())
    return [(word,dist[word],path_to_goal(parent,word)) for word in selected],exp,edges


def write(path,obj):path.write_text(canonical(obj)+"\n",encoding="utf-8")


def freeze(out:Path,old:Path):
    if out.exists() or sha256_file(old)!=OLD_SHA:
        raise ValueError("new registry and pinned prior lineage manifest required")
    prev=json.loads(old.read_text(encoding="utf-8"))
    excluded={v for shard in prev["shards"].values() for v in shard["lineages"]}
    if len(excluded)!=160:raise ValueError("prior LSA lineage census changed")
    used=set(excluded);out.mkdir(parents=True)
    write(out/"excluded_lineages.json",sorted(excluded))
    pairs=["".join(p) for p in product(ALPHABET,repeat=2)]
    available=list(combinations(pairs,2));files={}
    for split,count in COUNTS.items():
        rng=random.Random(f"OLR06D/{SEED}/{split}");rows=[];attempts=0
        while len(rows)<count:
            attempts+=1
            if attempts>count*1000:raise ValueError("root generator exhausted")
            rules=tuple((b,a) if rng.randrange(2) else (a,b)
                        for a,b in rng.sample(available,RULE_COUNT))
            lineage=system_lineage(rules)
            if lineage in used:continue
            band=BANDS[len(rows)%3]
            length=10 if split=="transfer" else 7+(len(rows)//3)%3
            source="".join(rng.choice(ALPHABET) for _ in range(length))
            target=choose_shell(source,rules,band,rng)
            if target is None:continue
            case=make_case(rules,source,target,split)
            if split in ("train","calibration"):
                try:source_ball(case)
                except ValueError:continue
            used.add(lineage);rows.append(asdict(case))
        file=out/(split+".json");write(file,rows)
        files[file.name]=sha256_file(file)
    files["excluded_lineages.json"]=sha256_file(out/"excluded_lineages.json")
    m={"schema":SCHEMA,"seed":SEED,"counts":COUNTS,"old_manifest_sha256":OLD_SHA,
       "excluded_old_lineages":160,"files":files,"old_sealed_rows_parsed":False,
       "source_frame_gate":"at least 12 nontrivial BFS states",
       "frames_per_source_root":FRAMES}
    write(out/"manifest.json",m);return m


def load_bank(out):
    m=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    if (m["schema"]!=SCHEMA or m["seed"]!=SEED or m["counts"]!=COUNTS
            or m["old_manifest_sha256"]!=OLD_SHA):
        raise ValueError("unregistered 06D bank")
    for n,h in m["files"].items():
        if sha256_file(out/n)!=h:raise ValueError("frozen registry changed")
    seen=set(json.loads((out/"excluded_lineages.json").read_text()))
    if len(seen)!=160:raise ValueError("prior lineage census changed")
    groups={}
    for split,n in COUNTS.items():
        rows=json.loads((out/(split+".json")).read_text())
        if len(rows)!=n:raise ValueError("registered split count changed")
        cases=[]
        for row in rows:
            row["equations"]=tuple(tuple(e) for e in row["equations"])
            c=EqCase(**row).validate()
            if c.lineage in seen or c.split!=("train" if split=="train" else "calibration"):
                raise ValueError("equation-system overlap or role drift")
            seen.add(c.lineage);cases.append(c)
        groups[split]=cases
    return m,groups


def hamming(word,goal):return (sum(a!=b for a,b in zip(word,goal,strict=True))+1)//2


def partitions(n):
    """Disjoint intervals with every offset at widths one, two and three."""
    result=set()
    for width in (1,2,3):
        for offset in range(width):
            cuts=sorted({0,n,*range(offset,n,width)})
            result.add(tuple(zip(cuts,cuts[1:])))
    return tuple(sorted(result))


class PatternDB:
    """Admissible consistent bound from cost-partitioned public abstractions.

    All costs are in half-units. The boundary abstracts the outside symbol,
    so it only adds possibilities. Each concrete rewrite charges <=2 total.
    """
    def __init__(self,rules,goal):
        self.rules=rules;self.goal=goal;self.parts=partitions(len(goal))
        self.tables={};self.windows={};self.expansions=0;self.edges=0
        start=time.perf_counter()
        for part in self.parts:
            for lo,hi in part:
                key=(goal[lo:hi],lo>0,hi<len(goal))
                if key not in self.tables:self.tables[key]=self._table(*key)
                self.windows[lo,hi]=self.tables[key]
        self.setup_seconds=time.perf_counter()-start

    def _table(self,target,left,right):
        # Reversible projected graph; compute distances by Dijkstra from target.
        dist={target:0};queue=[(0,target)]
        directed=[e for a,b in self.rules for e in ((a,b),(b,a))]
        while queue:
            cost,w=heapq.heappop(queue)
            if cost!=dist[w]:continue
            self.expansions+=1
            options=[(m.next_word,2) for m in neighbors(w,self.rules)]
            for a,b in directed:
                if left and w[0]==a[1] and a[1]!=b[1]:
                    options.append((b[1]+w[1:],1))
                if right and w[-1]==a[0] and a[0]!=b[0]:
                    options.append((w[:-1]+b[0],1))
            self.edges+=len(options)
            for v,c in options:
                nc=cost+c
                if nc<dist.get(v,math.inf):
                    dist[v]=nc;heapq.heappush(queue,(nc,v))
        return dist

    def bound(self,word):
        if len(word)!=len(self.goal):raise ValueError("word length differs")
        best=hamming(word,self.goal)
        for part in self.parts:
            d=sum(self.windows[lo,hi].get(word[lo:hi],math.inf) for lo,hi in part)
            if not math.isfinite(d):return math.inf
            best=max(best,(d+1)//2)
        return best


def search(case,heuristic,*,cap=128,batch_rank=None):
    """Same weighted best-first loop for classical and learned readouts.

    Count one expansion per public neighbor enumeration. A generated goal
    is terminal; successful paths have at most 15 rewrites.
    """
    start=time.perf_counter();expanded=edges=model_batches=model_rows=0
    gbest={case.target:0};closed={};cache={}
    def score(words):
        nonlocal model_batches,model_rows
        unseen=list(dict.fromkeys(w for w in words if w not in cache))
        if unseen:
            base=[heuristic(w) for w in unseen]
            if batch_rank is not None:
                delta=batch_rank(unseen,base)
                if len(delta)!=len(unseen) or any(not math.isfinite(x) or x<0 for x in delta):
                    raise ValueError("invalid learned residual ranking")
                model_batches+=1;model_rows+=len(unseen)
            else:delta=[0.]*len(unseen)
            cache.update(zip(unseen,[h+d for h,d in zip(base,delta,strict=True)]))
        return [cache[w] for w in words]
    h=score([case.target])[0]
    queue=[(1.5*h,h,0,case.target,(case.target,))]
    result=None
    while queue and expanded<cap:
        _,h,g,w,path=heapq.heappop(queue)
        if g!=gbest[w] or g>=closed.get(w,math.inf) or g>=MAX_REWRITES:
            continue
        closed[w]=g;expanded+=1
        moves=neighbors(w,case.equations);edges+=len(moves)
        candidates=[]
        for m in moves:
            v=m.next_word
            if v==case.source:
                result=path+(v,);break
            if g+1<gbest.get(v,math.inf):
                gbest[v]=g+1;candidates.append(v)
        if result is not None:break
        for v,hv in zip(candidates,score(candidates),strict=True):
            if math.isfinite(hv):heapq.heappush(queue,(g+1+1.5*hv,hv,g+1,v,path+(v,)))
    return {"found":result is not None,"words":list(result) if result else [],
            "expansions":expanded,"candidate_edges_enumerated":edges,
            "rewrite_distance":len(result)-1 if result else None,
            "model_batches":model_batches,"model_rows":model_rows,
            "planning_seconds":time.perf_counter()-start,"expansion_cap":cap,
            "status":"proposal_found" if result else "censored_budget_or_frontier"}
