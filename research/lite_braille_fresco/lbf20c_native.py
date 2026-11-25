"""Registered LBF-20C: frozen guard scorer; compare equal-cache planners."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import lbf20b_native_guarded as b
from lbf20_ground_witness import WeightedWitness, expression, replace
from lbf20c_pareto import ParetoWitness, replay, at, audit

HERE = Path(__file__).resolve().parent
SHA = "0471fe790ba1045ba991af10929be9a110a8bce9ccb4864a0f2f39f40de0c22e"
ARMS = ("lazy_steps", "pareto_steps", "pareto_guards", "pareto_risk", "probe_all")
H = 20


def fresh():
    rng = random.Random(2026112505)
    cases = []
    positives = ["P","n + 0 = n","n + 2 ≤ m + 2","n < m + 3",
                 "List.reverse (List.reverse xs) = xs","xs ++ [] = xs",
                 "List.length xs ≤ List.length xs + 2"]
    negatives = ["Q","False","n + 1 ≤ n","m + 2 ≤ n"]
    for i in range(18):
        a,c,d,e,f,g = [(x,) for x in rng.sample(list("abcdef"),6)]
        kind = ("sharing","misleading","mixed")[i%3]
        shared, x, y, alt = rng.sample(positives,4)
        bad = rng.choice(negatives)
        if kind == "misleading":
            shared = bad
        long = [b.Rule(a,c,shared), b.Rule(c,d,shared),
                b.Rule(e,f,shared if kind!="mixed" else alt), b.Rule(f,g,shared if kind!="mixed" else alt)]
        direct = [b.Rule(a,d,x), b.Rule(e,g,y)]
        if kind == "mixed":
            direct[0] = b.Rule(a,d,bad)
        rules = long+direct+[b.Rule(("p",a,e),("p",d,g),rng.choice(negatives)),
                            b.Rule(("p",e,a),("p",g,d),rng.choice(negatives))]
        rng.shuffle(rules)
        nodes = [("p",a,e) if rng.random()<.5 else ("p",e,a) for _ in range(rng.randint(2,4))]
        source = b.join_pairs(rng,nodes)
        def target(t):
            if t==a: return d
            if t==e: return g
            return (t[0],*(target(u) for u in t[1:]))
        cases.append(b.Case(f"fresh_{i:02d}","dev",kind,source,target(source),tuple(rules)))
    return cases


def run_policy(ep,w):
    case = ep.case
    # Exact guard text equivalence within one fixed context.
    guards = list(dict.fromkeys(r.guard for r in case.rules))
    group = [guards.index(r.guard) for r in case.rules]
    reps = [group.index(j) for j in range(len(guards))]
    rules = [(r.lhs,r.rhs,g) for r,g in zip(case.rules,group)]
    known = failed = 0
    sid = ep.root["sid"]
    script = []
    prob = [b.probability(w,s) for s in guards]
    planning_seconds = 0.
    selections = []
    oracle = frontier = None
    if ep.arm.startswith("pareto"):
        ts = time.perf_counter()
        oracle = ParetoWitness(case.source,rules,H)
        frontier = oracle.frontier(case.source,case.target)
        planning_seconds += time.perf_counter()-ts
    def probe(g):
        nonlocal sid,known,failed
        assert not (known|failed)&(1<<g)
        j = reps[g]
        tac = b.guard_tactic(case.rules[j],j)
        reply = ep.attempt(sid,tac,{"kind":"guard","guard_group":g,"guard":guards[g]})
        if reply is None: return False
        if reply["ok"]:
            b.check_state(reply["state"],case.source,case.target)
            sid = reply["state"]["sid"]
            script.append(tac)
            known |= 1<<g
        else:
            failed |= 1<<g
        return True
    if ep.arm=="probe_all":
        for g in range(len(guards)):
            if not probe(g): break
    while ep.alive():
        ts = time.perf_counter()
        if frontier is not None:
            choices = [(m,p) for m,p in frontier.items() if not m&failed]
            if not choices: break
            def key(pair):
                m,p=pair
                new=m&~known
                count=new.bit_count()
                risk=sum(1/max(.05,prob[g]) for g in range(len(guards)) if new&(1<<g))
                if ep.arm=="pareto_steps": return len(p),count,m,p
                if ep.arm=="pareto_guards": return count,len(p),m,p
                return risk,len(p),m,p
            mask,path=min(choices,key=key)
        else:
            active=[i for i,g in enumerate(group) if not (1<<g)&failed]
            wa=WeightedWitness([case.source],[(case.rules[i].lhs,case.rules[i].rhs) for i in active],H)
            terms=wa.witness(case.source,case.target)
            if terms is None: break
            edges=b.path_edges(case,terms,set(active))
            path=tuple((i,pos) for _,i,pos,_ in edges)
            _,mask=replay(case.source,path,rules)
        planning_seconds+=time.perf_counter()-ts
        assert replay(case.source,path,rules)==(case.target,mask)
        assert not mask&failed
        unknown=list(dict.fromkeys(group[i] for i,_ in path if not (1<<group[i])&known))
        selections.append({"mask":mask,"steps":len(path),"unknown":unknown,
                           "known":known,"failed":failed})
        if unknown:
            if not probe(unknown[0]): break
            continue
        proofs=[]
        word=case.source
        for i,pos in path:
            r=case.rules[i]
            assert at(word,pos)==r.lhs
            ctx=expression(replace(word,pos,("HOLE",)),"z")
            proofs.append(f"congrArg (fun z : T => {ctx}) (h{i} guard_{reps[group[i]]})")
            word=replace(word,pos,r.rhs)
        assert word==case.target
        if proofs:
            term=proofs[-1]
            for pr in reversed(proofs[:-1]):
                term=f"Eq.trans ({pr}) ({term})"
            tac="exact "+term
        else: tac="rfl"
        reply=ep.attempt(sid,tac,{"kind":"deferred_proof","rewrite_steps":len(path),
                                 "guard_mask":mask,"proved_groups":known})
        if reply is None: break
        assert reply["ok"],reply
        ep.certify(reply["state"],(*script,tac))
        break
    return {"groups":len(guards),"rules":len(case.rules),"known_groups":known.bit_count(),
            "failed_groups":failed.bit_count(),"planning_seconds":planning_seconds,
            "frontier_size":None if frontier is None else len(frontier),
            "automaton_states":None if oracle is None else oracle.n,
            "selections":selections}


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    raw=(HERE/"results/lbf20c/frozen_lbf20b_guard_ranker.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SHA
    w=json.loads(raw)["weights"]
    audit_result=audit()
    (out/"pareto_audit.json").write_text(json.dumps(audit_result,indent=2)+"\n")
    print("PARETO_AUDIT",json.dumps(audit_result),flush=True)
    rows=[]
    proofs=["import Lean",""]
    old=b.generate("dev",18)
    new=fresh()
    assert len({c.statement() for c in [*old,*new]})==36
    with b.LeanRepl(imports=("Lean",),timeout=20) as repl:
        assert str(repl.version).startswith("4.19.0")
        for cohort,cases in (("retrospective",old),("planted_tradeoff",new)):
            for case in cases:
                # Rotate arm order independently of outcome.
                offset=int(case.root.split("_")[-1])%len(ARMS)
                for arm in ARMS[offset:]+ARMS[:offset]:
                    ep=b.Episode(repl,case,arm)
                    extra=run_policy(ep,w)
                    row={**ep.finish(),**extra,"cohort":cohort}
                    rows.append(row)
                    if row["kernel"] is not None:
                        name=f"lbf20c_{case.root}_{arm}"
                        proofs += [f"theorem {name} : {case.statement()} := by",
                                   "  "+case.intros(),*("  "+t for t in row["script"]),
                                   f"#print axioms {name}",""]
                    print(cohort,case.root,arm,row["status"],row["calls"],
                          round(row["wall_seconds"],4),flush=True)
    (out/"Witnesses.lean").write_text("\n".join(proofs),encoding="utf-8")
    (out/"episodes.jsonl").write_text("".join(b.canonical(r)+"\n" for r in rows),encoding="utf-8")
    ag={}
    for cohort in ("retrospective","planted_tradeoff"):
        ag[cohort]={}
        for arm in ARMS:
            sub=[r for r in rows if r["cohort"]==cohort and r["arm"]==arm]
            ref={r["root"]:r for r in rows if r["cohort"]==cohort and r["arm"]=="lazy_steps"}
            wins=[r["calls"]-ref[r["root"]]["calls"] for r in sub if r["closed"] and ref[r["root"]]["closed"]]
            def mean(k): return statistics.mean(r[k] for r in sub)
            ag[cohort][arm]={"roots":len(sub),"closed":sum(r["closed"] for r in sub),
                "mean_calls":mean("calls"),"mean_rejects":mean("rejects"),
                "mean_wall_seconds":mean("wall_seconds"),"mean_planning_seconds":mean("planning_seconds"),
                "mean_groups":mean("groups"),"mean_rules":mean("rules"),
                "closure_at_calls":{str(k):sum(r["closed"] and r["calls"]<=k for r in sub) for k in (4,8,16,24)},
                "paired_closed_vs_lazy_steps_n":len(wins),
                "calls_delta_on_paired_closures":statistics.mean(wins) if wins else None,
                "calls_wins_ties_losses": [sum(v<0 for v in wins),sum(v==0 for v in wins),sum(v>0 for v in wins)],
                "status_counts":dict(collections.Counter(r["status"] for r in sub)),
                "family":{f:{"n":sum(r["family"]==f for r in sub),
                             "closed":sum(r["family"]==f and r["closed"] for r in sub),
                             "mean_calls":statistics.mean(r["calls"] for r in sub if r["family"]==f)}
                          for f in sorted({r["family"] for r in sub})}}
    summary={"schema":"lbf20c.premise-pareto.v1","scope":"development_only",
             "frozen_checkpoint_sha256":SHA,"new_training_fits":0,"audit":audit_result,
             "episodes":len(rows),"native_tactic_calls":sum(r["calls"] for r in rows),
             "native_kernel_certificates":sum(r["kernel"] is not None for r in rows),
             "cohorts":ag}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("RESULTS",json.dumps(summary),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    main(p.parse_args().out)
