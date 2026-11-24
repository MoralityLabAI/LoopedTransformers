"""OLR-06E: shared public rewrite frontiers, exact proposal ledger and Lean certificates.

Discovered edges and paths are proposals until independent native verification.
No learning, no private teacher paths in bounded search, no negative proofs
from exhausted search. The unit of budget is a fresh concrete neighbor enumeration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict,deque
from dataclasses import asdict,dataclass
from fractions import Fraction
from itertools import combinations,product
from pathlib import Path
import heapq
import json
import math
import random
import re
import subprocess
import time

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from .data import canonical,digest,sha256_file
from .equality_suite import ALPHABET,RULE_COUNT,BANDS,choose_shell,neighbors,system_lineage,at_arg
from .equality_search import plan,run_live,replay,BUDGET,EXACT_CAP
from .equality_preflight import compile_scripts
from .olr06d_search import load_bank,make_case,PatternDB,MAX_REWRITES

SEED=20261224
OLD_SHA="ee03f60eaf6f5b0cfc7930ac79d3fcc2215615853740816a4c0cba7fd7969179"
COUNTS={"dev":(12,3),"transfer":(12,5)}
CAPS=(128,256,512)
ARMS=("independent","cached_independent","reverse","balanced","demand","pdb")
REG="olr06e.multigoal.registry.v1"
SCORE="olr06e.multigoal.score.v1"


def write(p,obj):p.write_text(canonical(obj)+"\n",encoding="utf-8")
def read(p):return json.loads(p.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Root:
    split:str
    equations:tuple[tuple[str,str],...]
    source:str
    targets:tuple[str,...]

    @property
    def lineage(self):return system_lineage(self.equations)

    @property
    def root(self):return digest([REG,self.split,self.lineage,self.equations,self.source,self.targets])[:24]

    def cases(self):
        return tuple(make_case(self.equations,self.source,t,self.split) for t in self.targets)

    def validate(self):
        if self.split not in COUNTS or len(self.targets)!=COUNTS[self.split][1]:
            raise ValueError("wrong registered target count")
        if len(set(self.targets))!=len(self.targets):
            raise ValueError("duplicate obligations")
        if (self.split=="transfer" and len(self.source)!=10 or
                self.split=="dev" and not 7<=len(self.source)<=9):
            raise ValueError("incorrect registered length")
        self.cases()
        return self


def freeze(old,out):
    if out.exists() or sha256_file(old/"manifest.json")!=OLD_SHA:
        raise ValueError("pinned 06D bank and fresh output required")
    _,groups=load_bank(old)
    excluded=set(read(old/"excluded_lineages.json"))
    excluded.update(c.lineage for g in groups.values() for c in g)
    if len(excluded)!=252:raise ValueError("prior alpha-lineage census changed")
    used=set(excluded);out.mkdir(parents=True)
    write(out/"excluded.json",sorted(excluded))
    pairs=["".join(p) for p in product(ALPHABET,repeat=2)]
    available=list(combinations(pairs,2));files={}
    for split,(count,k) in COUNTS.items():
        rng=random.Random(f"olr06e/{SEED}/{split}")
        rows=[]
        for attempt in range(count*1000):
            if len(rows)==count:break
            rules=tuple((a,b) if rng.randrange(2) else (b,a)
                        for a,b in rng.sample(available,RULE_COUNT))
            lineage=system_lineage(rules)
            if lineage in used:continue
            n=10 if split=="transfer" else 7+len(rows)%3
            source="".join(rng.choice(ALPHABET) for _ in range(n))
            targets=[]
            for j in range(k):
                t=choose_shell(source,rules,BANDS[j%3],rng)
                if t is None or t in targets:break
                targets.append(t)
            if len(targets)!=k:continue
            rng.shuffle(targets)
            root=Root(split,rules,source,tuple(targets)).validate()
            used.add(lineage);rows.append(asdict(root))
        if len(rows)!=count:raise ValueError("root generator exhausted")
        path=out/(split+".json");write(path,rows);files[path.name]=sha256_file(path)
    files["excluded.json"]=sha256_file(out/"excluded.json")
    m={"schema":REG,"seed":SEED,"counts":{s:list(v) for s,v in COUNTS.items()},
       "old_manifest_sha256":OLD_SHA,"excluded_lineages":252,"files":files}
    write(out/"manifest.json",m);return m


def load(out):
    m=read(out/"manifest.json")
    if (m["schema"]!=REG or m["seed"]!=SEED or m["old_manifest_sha256"]!=OLD_SHA
            or m["counts"]!={s:list(v) for s,v in COUNTS.items()}):
        raise ValueError("unregistered cohort")
    for name,sha in m["files"].items():
        if sha256_file(out/name)!=sha:raise ValueError("frozen root bytes changed")
    seen=set(read(out/"excluded.json"))
    if len(seen)!=252:raise ValueError("lineage exclusion altered")
    roots=[]
    for split,(count,k) in COUNTS.items():
        rows=read(out/(split+".json"))
        if len(rows)!=count:raise ValueError("root count changed")
        for r in rows:
            if set(r)!=set(Root.__dataclass_fields__):raise ValueError("hidden labels in public root")
            root=Root(r["split"],tuple(tuple(e) for e in r["equations"]),
                      r["source"],tuple(r["targets"])).validate()
            if root.split!=split or root.lineage in seen:raise ValueError("lineage leakage")
            seen.add(root.lineage);roots.append(root)
    return roots


class Meter:
    def __init__(self,rules,cap,cached):
        if type(cap) is not int or cap<1:raise ValueError("positive cap required")
        self.rules,self.cap,self.cached=rules,cap,cached
        self.expansions=self.edges=self.logical=self.logical_edges=self.hits=0
        self.cache={}
        self.required_cap=0

    def visit(self,w):
        if self.expansions>=self.cap:raise ValueError("fresh search cap exceeded")
        self.logical+=1
        if self.cached and w in self.cache:
            moves=self.cache[w];self.hits+=1
            self.required_cap=self.expansions+1
        else:
            moves=neighbors(w,self.rules);self.expansions+=1;self.edges+=len(moves)
            self.required_cap=self.expansions
            if self.cached:self.cache[w]=moves
        self.logical_edges+=len(moves)
        return moves

    def stats(self):
        return {k:getattr(self,k) for k in ("expansions","edges","logical","logical_edges","hits")}


def path_up(parent,w):
    out=[w]
    while parent[out[-1]] is not None:out.append(parent[out[-1]])
    return out


class Bidir:
    """One target's ordinary bidirectional BFS, stepped one expansion at a time."""
    def __init__(self,target,source):
        self.parents=[{target:None},{source:None}]
        self.front=[[target],[source]];self.side=None
        self.todo=deque();self.new=[];self.done=False;self.path=None

    def step(self,meter):
        if self.done:return
        if self.side is None:
            if not self.front[0] or not self.front[1]:
                self.done=True;return
            self.side=int(len(self.front[1])<len(self.front[0]))
            self.todo=deque(self.front[self.side]);self.new=[]
        side=self.side;w=self.todo.popleft()
        for move in meter.visit(w):
            n=move.next_word
            if n in self.parents[side]:continue
            self.parents[side][n]=w;self.new.append(n)
            if n in self.parents[1-side]:
                path=list(reversed(path_up(self.parents[0],n)))+path_up(self.parents[1],n)[1:]
                if len(path)-1<=MAX_REWRITES:self.path=path
                self.done=True;return
        if not self.todo:
            self.front[side]=self.new;self.side=None


class PDBFront:
    """Round-robin weighted best-first search with a shared immutable PDB."""
    def __init__(self,target,source,pdb):
        self.source=source;self.pdb=pdb;h=pdb.bound(target)
        self.queue=[(1.5*h,h,0,target,(target,))]
        self.best={target:0};self.closed={};self.done=False;self.path=None

    def step(self,meter):
        if self.done:return
        while self.queue:
            _,h,g,w,path=heapq.heappop(self.queue)
            if g==self.best[w] and g<self.closed.get(w,math.inf) and g<MAX_REWRITES:break
        else:self.done=True;return
        self.closed[w]=g
        for move in meter.visit(w):
            v=move.next_word
            if v==self.source:
                self.path=list(path+(v,));self.done=True;return
            if g+1<self.best.get(v,math.inf):
                self.best[v]=g+1;hv=self.pdb.bound(v)
                if math.isfinite(hv):
                    heapq.heappush(self.queue,(g+1+1.5*hv,hv,g+1,v,path+(v,)))


def result_row(meter,found,pdb,start):
    return {"found":found,**meter.stats(),
            "abstract_expansions":pdb.expansions if pdb else 0,
            "abstract_edges":pdb.edges if pdb else 0,
            "pdb_seconds":pdb.setup_seconds if pdb else 0.,
            "seconds":time.perf_counter()-start}


def round_robin(root,arm,cap):
    start=time.perf_counter()
    meter=Meter(root.equations,cap,arm!="independent")
    pdb=PatternDB(root.equations,root.source) if arm=="pdb" else None
    fronts=[PDBFront(t,root.source,pdb) if pdb else Bidir(t,root.source)
            for t in root.targets]
    found={}
    while meter.expansions<cap and not all(f.done for f in fronts):
        for i,f in enumerate(fronts):
            if meter.expansions>=cap:break
            if f.done:continue
            f.step(meter)
            if f.path is not None:
                found[root.targets[i]]={"words":f.path,"found_at":meter.required_cap,"fresh_at":meter.expansions}
    return result_row(meter,found,pdb,start)


def reverse(root,cap):
    start=time.perf_counter();meter=Meter(root.equations,cap,False)
    parent={root.source:None};depth={root.source:0};queue=deque([root.source])
    pending=set(root.targets);found={}
    while queue and pending and meter.expansions<cap:
        w=queue.popleft()
        if depth[w]>=MAX_REWRITES:continue
        for m in meter.visit(w):
            v=m.next_word
            if v in parent:continue
            parent[v]=w;depth[v]=depth[w]+1;queue.append(v)
            if v in pending:
                pending.remove(v);found[v]={"words":path_up(parent,v),"found_at":meter.required_cap,"fresh_at":meter.expansions}
    return result_row(meter,found,None,start)


class ProposalLedger:
    """Exact discovered rewrite graph with mergeable component frontiers."""
    def __init__(self,root):
        self.root=root;self.adj={};self.parent={};self.front={};self.size={};self.level={}
        for w in (root.source,*root.targets):self.add(w,0)

    def add(self,w,level):
        if w not in self.parent:
            self.parent[w]=w;self.front[w]={w};self.size[w]=1
            self.level[w]=level;self.adj[w]=set()

    def find(self,w):
        while self.parent[w]!=w:
            self.parent[w]=self.parent[self.parent[w]];w=self.parent[w]
        return w

    def join(self,w,v):
        a,b=self.find(w),self.find(v)
        if a==b:return
        if (self.size[a],a)<(self.size[b],b):a,b=b,a
        self.parent[b]=a;self.size[a]+=self.size.pop(b)
        self.front[a].update(self.front.pop(b))

    def expand(self,w,meter):
        self.front[self.find(w)].remove(w)
        for move in meter.visit(w):
            v=move.next_word;self.add(v,self.level[w]+1)
            self.adj[w].add(v);self.adj[v].add(w);self.join(w,v)

    def select(self,pending,mode):
        source=self.find(self.root.source)
        count=defaultdict(int)
        for t in pending:count[self.find(t)]+=1
        count[source]=len(pending)
        options=[]
        for c,demand in count.items():
            if not self.front[c]:continue
            w=min(self.front[c],key=lambda t:(self.level[t],t))
            divisor=demand if mode=="demand" else 1
            options.append((Fraction(len(self.front[c]),divisor),self.level[w],w))
        return min(options)[2] if options else None

    def connected_paths(self,pending):
        source=self.root.source
        wanted={t for t in pending if self.find(t)==self.find(source)}
        if not wanted:return {}
        parent={source:None};depth={source:0};queue=deque([source]);paths={}
        while queue and wanted:
            w=queue.popleft()
            if depth[w]>=MAX_REWRITES:continue
            for v in sorted(self.adj[w]):
                if v in parent:continue
                parent[v]=w;depth[v]=depth[w]+1;queue.append(v)
                if v in wanted:
                    wanted.remove(v);paths[v]=path_up(parent,v)
        return paths


def shared(root,arm,cap):
    start=time.perf_counter();meter=Meter(root.equations,cap,False)
    graph=ProposalLedger(root);pending=set(root.targets);found={}
    while pending and meter.expansions<cap:
        w=graph.select(pending,arm)
        if w is None:break
        graph.expand(w,meter)
        for t,path in graph.connected_paths(pending).items():
            found[t]={"words":path,"found_at":meter.required_cap,"fresh_at":meter.expansions};pending.remove(t)
    return result_row(meter,found,None,start)


def bounded(root,arm,cap=512):
    if arm in ("independent","cached_independent","pdb"):
        r=round_robin(root,arm,cap)
    elif arm=="reverse":r=reverse(root,cap)
    elif arm in ("balanced","demand"):r=shared(root,arm,cap)
    else:raise ValueError("unregistered bounded arm")
    if r["expansions"]>cap or r["logical"]!=r["expansions"]+r["hits"]:
        raise ValueError("expansion accounting drift")
    if any(q["found_at"]>cap for q in r["found"].values()):
        raise ValueError("discovery prefix exceeds cap")
    validate_paths(root,r)
    return r


def exact(root):
    t=time.perf_counter();exp=edges=0;found={}
    for case in root.cases():
        p=plan(case,cap=EXACT_CAP)
        if not p["found"] or p["rewrite_distance"]>MAX_REWRITES:
            raise ValueError("registered exact ceiling failed")
        exp+=p["expansions"];edges+=p["candidate_edges_enumerated"]
        found[case.target]={"words":p["words"],"found_at":exp,"fresh_at":exp}
    return {"found":found,"expansions":exp,"edges":edges,"logical":exp,
            "logical_edges":edges,"hits":0,"abstract_expansions":0,"abstract_edges":0,
            "pdb_seconds":0.,"seconds":time.perf_counter()-t}


def validate_paths(root,r):
    if set(r["found"])-set(root.targets):raise ValueError("unregistered target")
    for t,q in r["found"].items():
        words=q["words"]
        if (words[0]!=t or words[-1]!=root.source or not 1<=len(words)-1<=MAX_REWRITES
                or not 1<=q["fresh_at"]<=r["expansions"]
                or q["found_at"] not in (q["fresh_at"],q["fresh_at"]+1)):
            raise ValueError("path or cap contract violated")
        for a,b in zip(words,words[1:]):
            if not any(m.next_word==b for m in neighbors(a,root.equations)):
                raise ValueError("path not in public rewrite graph")


def certify(roots,rows,out):
    """Deduplicated native leaves; independently compile the full original conjunction."""
    byroot={r.root:r for r in roots};cache={};items=[];episodes=[]
    start=time.perf_counter()
    with LeanRepl(imports=("Lean",),timeout=60,exe=config.REPL_EXE,
                  cwd=config.LEAN_DIR,env=config.lean_env()) as lean:
        if str(lean.version)!="4.19.0":raise ValueError("pinned native Lean required")
        for row in rows:
            r=byroot[row["root"]];cases={c.target:c for c in r.cases()}
            for t,q in row["found"].items():
                case=cases[t];key=(case.root,tuple(q["words"]))
                if key not in cache:
                    ep=run_live(lean,case,"bidirectional_bfs",words=q["words"])
                    if not ep["live_kernel"] or not replay(lean,case,ep):
                        raise ValueError("native proposal path failed replay")
                    label=f"p{len(items)}"
                    items.append((case,label,ep["script"]));episodes.append(ep)
                    cache[key]=(f"EqTransportCert.t_{case.root}_{label}",ep["effort"],len(episodes)-1)
                q["lemma"],q["native_tactics"],q["native_episode"]=cache[key]
    leaf_seconds=time.perf_counter()-start
    leaf_dir=out/"leaves";leaf_dir.mkdir()
    leaf=compile_scripts(items,leaf_dir,timeout=300)
    source=(leaf_dir/"standalone.lean").read_text(encoding="utf-8")
    lines=[source,"namespace OLR06EConjunction"]
    joined={};names=[]
    for row in rows:
        root=byroot[row["root"]];row["joint_lemma"]=None
        if len(row["found"])!=len(root.targets):continue
        lemmas=tuple(row["found"][t]["lemma"] for t in root.targets)
        key=(root.root,lemmas)
        if key not in joined:
            name=f"j{len(joined)}";joined[key]=name;names.append(name)
            binder=root.cases()[0].binders()
            goal=" ∧ ".join(f"P {at_arg(t)}" for t in root.targets)
            args="a b c d x P "+" ".join(f"e{i}" for i in range(RULE_COUNT))+" w"
            terms=[f"({lemma} {args})" for lemma in lemmas]
            term=terms[-1]
            for t in reversed(terms[:-1]):term=f"(And.intro {t} {term})"
            lines.append(f"theorem {name} {binder} : {goal} := by\n  exact {term}")
        row["joint_lemma"]="OLR06EConjunction."+joined[key]
    lines.append("end OLR06EConjunction")
    lines.extend(f"#print axioms OLR06EConjunction.{n}" for n in names)
    path=out/"conjunctions.lean";path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    run=subprocess.run([str(config.LEAN_EXE),str(path.resolve())],env=config.lean_env(),
                       capture_output=True,text=True,encoding="utf-8",timeout=300)
    logs=run.stdout+run.stderr;(out/"conjunction_compiler.log").write_text(logs,encoding="utf-8")
    clean=set(re.findall(r"'OLR06EConjunction\.([a-zA-Z0-9_]+)' does not depend on any axioms",logs))
    if run.returncode or clean!=set(names):
        raise ValueError("full conjunction compiler rejected:\n"+logs[-3500:])
    write(out/"native_episodes.json",episodes)
    return {"native_leaf_replays":len(episodes),"leaf_declarations":leaf["passed"],
            "joint_declarations":len(names),"native_leaf_seconds":leaf_seconds,
            "axiom_free":leaf["passed"]+len(names)}


def count_at(row,cap,joint=False):
    overhead=row["abstract_expansions"] if joint else 0
    return sum(q["found_at"]+overhead<=cap for q in row["found"].values())


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:groups[row["split"],row["arm"]].append(row)
    result={}
    for (split,arm),group in groups.items():
        k=COUNTS[split][1];n=len(group)
        result[f"{split}:{arm}"]={
            "roots":n,"leaves":n*k,"full_final":sum(bool(r["joint_lemma"]) for r in group),
            "leaf_final":sum(len(r["found"]) for r in group),
            **{f"full{c}":sum(bool(r["joint_lemma"]) and count_at(r,c)==k for r in group) for c in CAPS},
            **{f"leaves{c}":sum(count_at(r,c) for r in group) for c in CAPS},
            **{f"combined_full{c}":sum(bool(r["joint_lemma"]) and count_at(r,c,True)==k for r in group) for c in CAPS},
            **{f"mean_{key}":sum(r[key] for r in group)/n for key in
               ("expansions","edges","logical","logical_edges","hits","abstract_expansions","abstract_edges","seconds","pdb_seconds")},
            "total_native_leaf_tactics":sum(q["native_tactics"] for r in group for q in r["found"].values())}
    return result


def contrast(rows,split,left,right,cap):
    idx={(r["root"],r["arm"]):r for r in rows if r["split"]==split}
    ids=sorted({i for i,a in idx});k=COUNTS[split][1]
    diffs=[int(count_at(idx[i,left],cap)==k)-int(count_at(idx[i,right],cap)==k) for i in ids]
    rng=random.Random(f"olr06e/{split}/{left}/{right}/{cap}")
    mean=lambda a:sum(a)/len(a)
    vals=sorted(mean([rng.choice(diffs) for _ in ids]) for _ in range(5000))
    return {"difference":mean(diffs),"ci95":[vals[125],vals[4875]],"roots":len(ids)}


def score(bank,out):
    if out.exists():raise ValueError("create-only evaluation")
    roots=load(bank);out.mkdir(parents=True);start=time.perf_counter();rows=[]
    for root in roots:
        for arm in ARMS:
            rows.append({"root":root.root,"split":root.split,"arm":arm,**bounded(root,arm,max(CAPS))})
    write(out/"frozen_bounded.json",rows)
    for root in roots:rows.append({"root":root.root,"split":root.split,"arm":"exact",**exact(root)})
    cert=certify(roots,rows,out)
    write(out/"root_scores.json",rows)
    summary=summarize(rows)
    for s,(count,k) in COUNTS.items():
        if summary[f"{s}:exact"]["full_final"]!=count:raise ValueError("native conjunction ceiling failed")
    best={s:max(summary[f"{s}:{a}"]["full256"] for a in ARMS) for s in COUNTS}
    files=("frozen_bounded.json","root_scores.json","native_episodes.json","conjunctions.lean",
           "conjunction_compiler.log","leaves/standalone.lean","leaves/compiler.log","leaves/compiler_receipt.json")
    report={"schema":SCORE,"bank_sha256":sha256_file(bank/"manifest.json"),
            "roots":len(roots),"target_obligations":sum(len(r.targets) for r in roots),
            "policy_trials":len(rows),"bounded_frozen_trials":len(roots)*len(ARMS),
            "certificate":cert,"scorecard":summary,
            "contrasts":{f"{s}:demand-{a}:{c}":contrast(rows,s,"demand",a,c)
                         for s in COUNTS for a in ARMS if a!="demand" for c in CAPS},
            "best_full256":best,"headroom_at_256":sum(best.values())<=.8*len(roots),
            "model_training_updates":0,"architecture_promotion_eligible":False,
            "elapsed_seconds":time.perf_counter()-start,
            "files":{f:sha256_file(out/f) for f in files}}
    write(out/"manifest.json",report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest="op",required=True)
    a=sp.add_parser("freeze");a.add_argument("--old-bank",type=Path,required=True);a.add_argument("--out",type=Path,required=True)
    b=sp.add_parser("score");b.add_argument("--bank",type=Path,required=True);b.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    print(canonical(freeze(args.old_bank,args.out) if args.op=="freeze" else score(args.bank,args.out)),flush=True)


if __name__=="__main__":main()
