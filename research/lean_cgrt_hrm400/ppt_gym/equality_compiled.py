"""LSA-02N: prospective hard-cohort compiled projection potentials."""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
from itertools import combinations, combinations_with_replacement, product
import heapq
import json
import math
from pathlib import Path
import random
import time

import torch

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from .data import canonical, digest, sha256_file
from .equality_suite import ALPHABET, RULE_COUNT, EqCase, choose_shell, neighbors, system_lineage
from .equality_search import BUDGET, EXACT_CAP, run_live, replay, path_moves
from .equality_preflight import compile_scripts, _native_backend, _rows, _write
from .equality_ordering import SUITE_SHA, EDGE_CAPS, DEV_N, _jsonl, _public_plan, hamming
from . import equality_residual as prior

DEV_SEED = 2026112404
M_SHA = "be142147c008a06e55bb01967a90cf27b27b128984d5f4d59cae147b64bb69b3"
PAIRS_SHA = "24ade5f3b9830c2539f60397003a2ec3bc30dd0b39ee1c4d0d7ff59d18f4b4a5"
CAPS = (16, 32, 64, 256, 1024)
PUBLIC = ("hamming", "bidirectional", "pair_quarter", "pair_full", "triple_quarter", "triple_full")
LEARNED = ("quadratic", "linear", "sham107", "sham109")
Q = 256
MONOMIALS = tuple(combinations_with_replacement(range(prior.FEATURES), 2))


def excluded(suite, old_l, old_m):
    _, seen = prior.excluded(suite, old_l)
    if sha256_file(old_m / "manifest.json") != M_SHA:
        raise ValueError("LSA-02M development manifest mismatch")
    m = json.loads((old_m / "manifest.json").read_text())
    lins = set(m["lineages"])
    if len(lins) != DEV_N or seen.intersection(lins):
        raise ValueError("LSA-02M development lineage violation")
    return seen | lins


def freeze(suite, old_l, old_m, out):
    seen = excluded(suite, old_l, old_m)
    rng = random.Random(DEV_SEED)
    available = list(combinations(prior.PAIRS, 2))
    rows = []
    attempts = 0
    while len(rows) < DEV_N:
        attempts += 1
        if attempts > DEV_N*500:
            raise RuntimeError("hard development generation failed")
        rules = tuple((b,a) if rng.randrange(2) else (a,b)
                      for a,b in rng.sample(available,RULE_COUNT))
        lin = system_lineage(rules)
        if lin in seen:
            continue
        source = "".join(rng.choice(ALPHABET) for _ in range(8+len(rows)%3))
        target = choose_shell(source,rules,(10,11),rng)
        if target is None:
            continue
        root = digest([lin,"calibration",rules,source,target])[:24]
        case = EqCase(root,lin,"calibration","indexed_unary_equations",rules,
                      source,target).validate()
        rows.append(dict(vars(case)))
        seen.add(lin)
    out.mkdir(parents=True,exist_ok=False)
    _rows(out/"roots.jsonl",rows)
    m = {"schema":"lsa02n.hard_dev.v1","seed":DEV_SEED,"roots":DEV_N,
         "suite_sha256":SUITE_SHA,"prior_l_sha256":prior.OLD_DEV_SHA,"prior_m_sha256":M_SHA,
         "lineages":[r["lineage"] for r in rows],
         "rows_sha256":sha256_file(out/"roots.jsonl"),
         "generator_sha256":sha256_file(Path(__file__))}
    _write(out/"manifest.json",m)
    return m


def load_dev(suite,old_l,old_m,out):
    seen = excluded(suite,old_l,old_m)
    m = json.loads((out/"manifest.json").read_text())
    if (m["schema"]!="lsa02n.hard_dev.v1" or m["seed"]!=DEV_SEED or m["roots"]!=DEV_N
        or m["suite_sha256"]!=SUITE_SHA or m["prior_l_sha256"]!=prior.OLD_DEV_SHA
        or m["prior_m_sha256"]!=M_SHA or m["rows_sha256"]!=sha256_file(out/"roots.jsonl")
        or m["generator_sha256"]!=sha256_file(Path(__file__))):
        raise ValueError("new development custody mismatch")
    cases = []
    for row in _jsonl(out/"roots.jsonl"):
        if set(row)!=set(EqCase.__dataclass_fields__):
            raise ValueError("private development metadata")
        row["equations"] = tuple(tuple(p) for p in row["equations"])
        c = EqCase(**row).validate()
        if c.split!="calibration" or c.lineage in seen or len(c.source)!=8+len(cases)%3:
            raise ValueError("development lineage/length violation")
        cases.append(c)
        seen.add(c.lineage)
    if len(cases)!=DEV_N or [c.lineage for c in cases]!=m["lineages"]:
        raise ValueError("development census mismatch")
    return cases


def bound(values,k):
    if not all(math.isfinite(v) for v in values):
        return math.inf
    return max(max(values),math.ceil(sum(values)/(k+1)))


class Projection:
    """Exact BFS in a finite relaxed window graph; outside letters unconstrained."""
    def __init__(self,case,k):
        self.k = k
        self.tables = []
        self.states = self.checks = self.expansions = self.edges = 0
        n = len(case.source)
        words = ["".join(x) for x in product(ALPHABET,repeat=k)]
        oriented = [(a,b) for a,b in case.equations]+[(b,a) for a,b in case.equations]
        for i in range(n-k+1):
            adj = {}
            for w in words:
                self.states += 1
                opts = set()
                for j in range(max(0,i-1),min(n-2,i+k-1)+1):
                    lo,hi = max(i,j),min(i+k,j+2)
                    for a,b in oriented:
                        self.checks += 1
                        if all(w[p-i]==a[p-j] for p in range(lo,hi)):
                            v = list(w)
                            for p in range(lo,hi):
                                v[p-i] = b[p-j]
                            opts.add("".join(v))
                adj[w] = opts-{w}
            target = case.source[i:i+k]
            d,q = {target:0},deque([target])
            while q:
                w = q.popleft()
                self.expansions += 1
                self.edges += len(adj[w])
                for v in sorted(adj[w]):
                    if v not in d:
                        d[v] = d[w]+1
                        q.append(v)
            self.tables.append(d)

    def values(self,w):
        return [d.get(w[i:i+self.k],math.inf) for i,d in enumerate(self.tables)]

    def __call__(self,w):
        return bound(self.values(w),self.k)


def basis(x,quadratic):
    return list(x)+([x[i]*x[j] for i,j in MONOMIALS] if quadratic else [])


def avg(xs,quadratic):
    vectors = [basis(x,quadratic) for x in xs]
    return [sum(col)/len(vectors) for col in zip(*vectors)]


def fit(samples,arm):
    quadratic = arm!="linear"
    differences,targets,strata = [],[],defaultdict(list)
    for i,s in enumerate(samples):
        hard = [x for x,m in zip(s["hard_features"],s["mask"]) if m]
        easy = [x for x,m in zip(s["easy_features"],s["mask"]) if m]
        dh = bound([round(16*x[6]) for x in hard],2)
        de = bound([round(16*x[6]) for x in easy],2)
        a,b = avg(hard,quadratic),avg(easy,quadratic)
        differences.append([x-y for x,y in zip(a,b)])
        targets.append(s["gap"]-(s["hard_hamming"]-s["easy_hamming"]+0.25*(dh-de)))
        strata[s["kind"],len(hard)].append(i)
    if arm.startswith("sham"):
        rng = random.Random(int(arm[4:]))
        for ix in strata.values():
            y = [targets[i] for i in ix]
            rng.shuffle(y)
            for i,v in zip(ix,y):
                targets[i] = v
    x = torch.tensor(differences,dtype=torch.float64)
    y = torch.tensor(targets,dtype=torch.float64)
    n,p = x.shape
    beta = torch.linalg.solve(x.T@x/n+0.01*torch.eye(p,dtype=torch.float64),x.T@y/n)
    if not torch.isfinite(beta).all():
        raise ValueError("invalid fitted coefficients")
    return beta.tolist(),{"parameters":p,"pairs":n,"fit_mse":float((x@beta-y).square().mean()),
                          "zero_mse":float(y.square().mean()),"coefficient_sha256":digest(beta.tolist())}


class Scorer:
    def __init__(self,case,arm,coeff=None):
        self.case,self.arm = case,arm
        self.pair = prior.Features(case) if arm!="hamming" else None
        self.triple = Projection(case,3) if arm.startswith("triple") else None
        self.products = 0
        self.tables = None
        if coeff is not None:
            if len(coeff) not in (prior.FEATURES,prior.FEATURES+len(MONOMIALS)):
                raise ValueError("invalid coefficient size")
            quadratic = len(coeff)>prior.FEATURES
            self.tables = []
            for table in self.pair.tables:
                row = {}
                for w,(x,d,_) in table.items():
                    z = basis(x,quadratic)
                    self.products += len(z)
                    row[w] = (d,round(Q*sum(a*b for a,b in zip(coeff,z))))
                self.tables.append(row)
            self.goal_sum = sum(t[case.source[i:i+2]][1] for i,t in enumerate(self.tables))
        if (coeff is not None)!=(arm in LEARNED):
            raise ValueError("model/scorer arm mismatch")

    def __call__(self,w):
        h = Q*hamming(w,self.case.source)
        if self.pair is None:
            return h
        if self.tables is None:
            p = bound([e[1] for e in self.pair.entries(w)],2)
            if self.triple is not None:
                p = max(p,self.triple(w))
            return h+(Q if self.arm.endswith("full") else Q//4)*p
        entries = [t[w[i:i+2]] for i,t in enumerate(self.tables)]
        p = bound([e[0] for e in entries],2)
        residual = round((sum(e[1] for e in entries)-self.goal_sum)/len(entries))
        return h+(Q//4)*p+max(-2*Q,min(2*Q,residual))

    def costs(self):
        p,t = self.pair,self.triple
        return {
            "setup_states":(p.setup_states if p else 0)+(t.states if t else 0),
            "setup_rule_checks":(p.rule_checks if p else 0)+(t.checks if t else 0),
            "setup_abstract_expansions":(p.abstract_expansions if p else 0)+(t.expansions if t else 0),
            "setup_abstract_edges":(p.abstract_edges if p else 0)+(t.edges if t else 0),
            "setup_coefficient_products":self.products,
        }


def search(case,cap,arm,coeff=None):
    if arm=="bidirectional" or arm=="exact_ceiling":
        return dict(_public_plan(case,cap),scored_states=0,setup_states=0,
                    setup_rule_checks=0,setup_coefficient_products=0,setup_seconds=0)
    t0 = time.perf_counter()
    scorer = Scorer(case,arm,coeff)
    setup = time.perf_counter()-t0
    start = case.target
    cache = {case.source:0,start:scorer(start)}
    heap = [(cache[start],hamming(start,case.source),0,start,(start,))]
    best = {start:0}
    expansions=edges=0
    path=()
    while heap:
        _,_,g,w,trace=heapq.heappop(heap)
        if best[w]!=g:
            continue
        if w==case.source:
            path=trace
            break
        if g>=BUDGET-1:
            continue
        if expansions>=cap:
            break
        moves=neighbors(w,case.equations)
        expansions+=1
        edges+=len(moves)
        for v in sorted({m.next_word for m in moves}):
            if g+1>=best.get(v,math.inf):
                continue
            best[v]=g+1
            if v not in cache:
                cache[v]=scorer(v)
            heapq.heappush(heap,(cache[v],hamming(v,case.source),g+1,v,(*trace,v)))
    return dict(scorer.costs(),found=bool(path),words=list(path),expansion_cap=cap,
                expansions=expansions,candidate_edges_enumerated=edges,
                planning_seconds=time.perf_counter()-t0,setup_seconds=setup,
                scored_states=len(cache)-1,model_batches=0,model_states_scored=0)


def certify(cases,records,out):
    lookup={c.root:c for c in cases}
    items,episodes,certs=[],[],{}
    certdir=out/"certificates"
    certdir.mkdir()
    with LeanRepl(imports=("Lean",),timeout=60,exe=config.REPL_EXE,
                  cwd=config.LEAN_DIR,env=config.lean_env()) as repl:
        backend=_native_backend(repl)
        for row in records:
            row.update(certified=False,native_tactics=0,certificate_id=None)
            if not row["found"]:
                continue
            case=lookup[row["root"]]
            key=(case.root,tuple(row["words"]))
            if key not in certs:
                path_moves(case,row["words"])
                label=f"p{len(certs)}"
                t=time.perf_counter()
                ep=run_live(repl,case,"bidirectional_bfs",words=row["words"])
                if not ep["live_kernel"] or not replay(repl,case,ep):
                    raise ValueError("native certificate/replay failed")
                certs[key]=label,ep["effort"]
                items.append((case,label,ep["script"]))
                episodes.append({"certificate_id":label,"episode":ep,
                                 "native_and_replay_seconds":time.perf_counter()-t})
            row["certificate_id"],row["native_tactics"]=certs[key]
    n=0
    for i in range(0,len(items),128):
        chunk=certdir/str(i//128)
        chunk.mkdir()
        n+=compile_scripts(items[i:i+128],chunk,timeout=600)["passed"]
    if n!=len(certs):
        raise ValueError("native compile census mismatch")
    for ep in episodes:
        ep["episode"]["standalone_verified"]=True
    for row in records:
        row["certified"]=row["certificate_id"] is not None
    depths={r["root"]:len(r["words"])-1 for r in records
            if r["arm"]=="exact_ceiling" and r["certified"]}
    if len(depths)!=DEV_N or not all(d in (10,11) for d in depths.values()):
        raise ValueError("hard cohort exact-distance gate failed")
    for row in records:
        row["reference_rewrite_count"]=depths[row["root"]]
    _rows(out/"native_episodes.jsonl",episodes)
    _rows(out/"root_results.jsonl",records)
    return backend,n


def summarize(records):
    summary={}
    for arm in (*PUBLIC,*LEARNED,"exact_ceiling"):
        summary[arm]={}
        for cap in (CAPS if arm!="exact_ceiling" else (EXACT_CAP,)):
            rows=[r for r in records if r["arm"]==arm and r["expansion_cap"]==cap]
            if len(rows)!=DEV_N:
                raise ValueError("arm/cap census mismatch")
            s={"episodes":len(rows),"certified":sum(r["certified"] for r in rows),
               "edge_threshold_certified":{str(e):sum(r["certified"] and r["candidate_edges_enumerated"]<=e
                                                     for r in rows) for e in EDGE_CAPS},
               "joint_cap_certified":sum(r["certified"] and r["expansions"]+r["native_tactics"]<=cap for r in rows),
               "by_distance":{}}
            for key in ("expansions","candidate_edges_enumerated","planning_seconds","setup_seconds",
                        "setup_states","setup_rule_checks","setup_abstract_expansions","setup_abstract_edges",
                        "setup_coefficient_products","scored_states","native_tactics"):
                s["mean_"+key]=sum(r.get(key,0) for r in rows)/len(rows)
            for d in (10,11):
                sub=[r for r in rows if r["reference_rewrite_count"]==d]
                s["by_distance"][str(d)]={"episodes":len(sub),"certified":sum(r["certified"] for r in sub)}
            summary[arm][str(cap)]=s
    return summary


def gate(records):
    z={(r["root"],r["arm"]):int(r["certified"]) for r in records if r["expansion_cap"]==64}
    roots=sorted({r for r,_ in z})
    if len(roots)!=DEV_N or len(z)!=DEV_N*(len(PUBLIC)+len(LEARNED)):
        raise ValueError("primary census mismatch")
    def mean(rs,a):
        return sum(z[r,a] for r in rs)/len(rs)
    def delta(rs):
        return mean(rs,"quadratic")-max(mean(rs,a) for a in PUBLIC)
    rng=random.Random(DEV_SEED)
    b=sorted(delta([rng.choice(roots) for _ in roots]) for _ in range(2000))
    shams={s:mean(roots,"quadratic")-mean(roots,s) for s in ("sham107","sham109")}
    sham=sum(shams.values())/2
    return {"quadratic_minus_strongest_public":delta(roots),
            "strongest_public":max(PUBLIC,key=lambda a:mean(roots,a)),
            "bootstrap95":[b[50],b[1949]],"quadratic_minus_mean_sham":sham,
            "sham_seed_differences":shams,"quadratic_minus_linear":mean(roots,"quadratic")-mean(roots,"linear"),
            "developmental_gate":delta(roots)>=3/32 and b[50]>0 and sham>=2/32 and min(shams.values())>0}


def run(suite,old_l,old_m,teacher,dev,out):
    cases=load_dev(suite,old_l,old_m,dev)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    samples,source=prior.build_pairs(suite,teacher,out/"teacher")
    if (len(samples)!=994 or source["offpath_native_compiled"]!=760
        or sha256_file(out/"teacher/pairs.jsonl")!=PAIRS_SHA):
        raise ValueError("registered LSA-02M certified pair source mismatch")
    coeffs,fits={},{}
    for arm in LEARNED:
        a,receipt=fit(samples,arm)
        b,duplicate=fit(samples,arm)
        if receipt["coefficient_sha256"]!=duplicate["coefficient_sha256"] or a!=b:
            raise ValueError("duplicate coefficient fit mismatch")
        coeffs[arm]=a
        fits[arm]=dict(receipt,coefficients=a,duplicate_sha256=duplicate["coefficient_sha256"])
    _write(out/"fits.json",fits)
    records=[]
    for i,c in enumerate(cases):
        arms=(*PUBLIC,*LEARNED)
        arms=arms[i%len(arms):]+arms[:i%len(arms)]
        for arm in arms:
            for cap in CAPS:
                records.append(dict(search(c,cap,arm,coeffs.get(arm)),arm=arm,root=c.root,lineage=c.lineage))
        records.append(dict(search(c,EXACT_CAP,"exact_ceiling"),arm="exact_ceiling",root=c.root,lineage=c.lineage))
    _rows(out/"proposed_paths.jsonl",records)
    backend,n=certify(cases,records,out)
    m={"schema":"lsa02n.compiled_projection_development.v1","roots":len(cases),"backend":backend,
       "teacher":source,"fits":{k:{x:y for x,y in v.items() if x!="coefficients"} for k,v in fits.items()},
       "native_eval_proofs":n,"summary":summarize(records),"gate":gate(records),
       "development_sha256":sha256_file(dev/"manifest.json"),"sealed_contents_parsed":False,
       "scientific_eligible":False,"architecture_promotion_eligible":False,
       "root_results_sha256":sha256_file(out/"root_results.jsonl")}
    _write(out/"manifest.json",m)
    return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("freeze","run"))
    for arg in ("suite","old-l","old-m","out"):
        p.add_argument("--"+arg,type=Path,required=True)
    p.add_argument("--teacher",type=Path)
    p.add_argument("--dev",type=Path)
    a=p.parse_args()
    if a.command=="freeze":
        m=freeze(a.suite,a.old_l,a.old_m,a.out)
    else:
        if a.teacher is None or a.dev is None:
            p.error("run requires --teacher and --dev")
        m=run(a.suite,a.old_l,a.old_m,a.teacher,a.dev,a.out)
    print(canonical(m))


if __name__=="__main__":
    main()
