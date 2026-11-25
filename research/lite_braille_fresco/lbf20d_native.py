"""Retrospective native LBF-20C replay plus paired fresh public-context variants.

The exact DP is exact only for its supplied static probability model. All
native controllers maintain the actual verified local guard witnesses. A
separate post-run audit probes guards in the original context.
"""
from __future__ import annotations
import argparse
import collections
import dataclasses
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
import lbf20b_native_guarded as b
import lbf20c_native as c
from lbf20c_pareto import ParetoWitness, replay, at
from lbf20_ground_witness import expression,replace
from lbf20d_dp import Problem,bits

HERE=Path(__file__).resolve().parent
ARMS=("risk_path","dp_independent","dp_mixture","risk_context")
H=20

@dataclasses.dataclass(frozen=True)
class ContextCase(b.Case):
    premise: str
    def statement(self):
        return super().statement().replace("(hp : P)",f"(hp : {self.premise})")

def pairs():
    rng=random.Random(2026112511)
    out=[]
    for i in range(6):
        a,x,t,d,y,u=[(z,) for z in rng.sample(list("abcdef"),6)]
        rules=[b.Rule(a,x,"P"),b.Rule(x,t,"P"),b.Rule(d,y,"P"),b.Rule(y,u,"P"),
               b.Rule(a,t,"Q"),b.Rule(d,u,"Q"),
               b.Rule(("p",a,d),("p",t,u),"False"),
               b.Rule(("p",d,a),("p",u,t),"n + 1 ≤ n")]
        rng.shuffle(rules)
        source=b.join_pairs(rng,[("p",a,d) if rng.random()<.5 else ("p",d,a)
                                  for _ in range(rng.randint(2,4))])
        def target(v):
            if v==a:return t
            if v==d:return u
            return (v[0],*(target(z) for z in v[1:]))
        for q in ("P","Q"):
            out.append(ContextCase(f"ctx_{i:02d}_{q}","dev",f"paired_context_{i:02d}",source,target(source),tuple(rules),q))
    return out

def run_policy(ep,w):
    case=ep.case
    start=time.perf_counter()
    guards=list(dict.fromkeys(r.guard for r in case.rules))
    group=[guards.index(r.guard) for r in case.rules]
    reps=[group.index(g) for g in range(len(guards))]
    rules=[(r.lhs,r.rhs,g) for r,g in zip(case.rules,group)]
    wa=ParetoWitness(case.source,rules,H)
    frontier=wa.frontier(case.source,case.target)
    p=[min(.99,max(.01,b.probability(w,g))) for g in guards]
    base_p=list(p)
    if ep.arm=="risk_context":
        g=ep.root["goals"][0]
        hyps=ep.repl.contexts.get(g["ctx"],g.get("hyps",[]))
        for i,guard in enumerate(guards):
            # Actual native public hypotheses, not generator labels.
            if any(h.get("type")=={"a":guard} for h in hyps):
                p[i]=.99
    if ep.arm=="dp_mixture":
        # Preserves marginals; fixed uncalibrated common-mode model.
        delta=[.8*min(v,1-v) for v in p]
        model=Problem(frontier,[[v-d for v,d in zip(p,delta)],[v+d for v,d in zip(p,delta)]],[.5,.5],all_queries=False)
    else:
        model=Problem(frontier,[p],[1.],all_queries=False)
    planning=time.perf_counter()-start
    known=failed=0
    sid=ep.root["sid"]
    script=[]
    while ep.alive():
        ts=time.perf_counter()
        live=[(m,path) for m,path in frontier.items() if not m&failed]
        if not live: break
        ready=[(m,path) for m,path in live if not m&~known]
        if ready:
            mask,path=min(ready,key=lambda a:(len(a[1]),a[0],a[1]))
            assert replay(case.source,path,rules)==(case.target,mask)
            proofs=[];word=case.source
            for i,pos in path:
                r=case.rules[i];assert at(word,pos)==r.lhs
                ctx=expression(replace(word,pos,("HOLE",)),"z")
                proofs.append(f"congrArg (fun z : T => {ctx}) (h{i} guard_{reps[group[i]]})")
                word=replace(word,pos,r.rhs)
            assert word==case.target
            if proofs:
                term=proofs[-1]
                for pr in reversed(proofs[:-1]): term=f"Eq.trans ({pr}) ({term})"
                tac="exact "+term
            else:tac="rfl"
            planning+=time.perf_counter()-ts
            reply=ep.attempt(sid,tac,{"kind":"deferred_proof","steps":len(path),"mask":mask})
            if reply is None:break
            assert reply["ok"],reply
            ep.certify(reply["state"],(*script,tac))
            break
        if ep.arm.startswith("dp_"):
            _,j=model.solve(known,failed)
            assert j is not None
        else:
            m,path=min(live,key=lambda a:(sum(1/max(.05,p[g]) for g in bits(a[0]&~known)),len(a[1]),a[0],a[1]))
            j=next(group[i] for i,_ in path if not (known|failed)&(1<<group[i]))
        planning+=time.perf_counter()-ts
        assert not (known|failed)&(1<<j)
        k=reps[j]
        tac=b.guard_tactic(case.rules[k],k)
        reply=ep.attempt(sid,tac,{"kind":"guard","group":j,"guard":guards[j],"marginal":base_p[j]})
        if reply is None:break
        if reply["ok"]:
            b.check_state(reply["state"],case.source,case.target)
            sid=reply["state"]["sid"];script.append(tac);known|=1<<j
        else:failed|=1<<j
    return {"planning_seconds":planning,"frontier_size":len(frontier),"groups":len(guards),
            "dp_states":len(model.q),"known":known,"failed":failed}

def static_audit(repl,case,rows):
    repl.reset()
    root=repl.init(case.statement())
    seen={}
    for i,r in enumerate(case.rules):
        if r.guard in seen: continue
        o=repl.run(root["sid"],b.guard_tactic(r,i))
        assert type(o.get("ok")) is bool
        if o["ok"]: b.check_state(o["state"],case.source,case.target)
        seen[r.guard]=o["ok"]
    changes=[]
    for row in rows:
        for e in row["events"]:
            if e["kind"]=="guard" and e["accepted"]!=seen[e["guard"]]:
                changes.append({"arm":row["arm"],"guard":e["guard"],"static":seen[e["guard"]],"dynamic":e["accepted"]})
    return {"root":case.root,"static":seen,"queries":len(seen),"differences":changes}

def main(out):
    out.mkdir(parents=True,exist_ok=False)
    raw=(HERE/"results/lbf20c/frozen_lbf20b_guard_ranker.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest()==c.SHA
    w=json.loads(raw)["weights"]
    cohorts={"retrospective":b.generate("dev",18)+c.fresh(),"paired_context":pairs()}
    rows=[];audits=[];proofs=["import Lean",""]
    with b.LeanRepl(imports=("Lean",),timeout=20) as repl:
        assert str(repl.version).startswith("4.19.0")
        for co,cases in cohorts.items():
            for idx,case in enumerate(cases):
                sub=[]
                offset=idx%len(ARMS)
                for arm in ARMS[offset:]+ARMS[:offset]:
                    ep=b.Episode(repl,case,arm)
                    extra=run_policy(ep,w)
                    row={**ep.finish(),**extra,"cohort":co}
                    rows.append(row);sub.append(row)
                    if row["kernel"] is not None:
                        name=f"lbf20d_{case.root}_{arm}"
                        proofs.extend([f"theorem {name} : {case.statement()} := by","  "+case.intros(),
                                       *("  "+t for t in row["script"]),f"#print axioms {name}",""])
                    print(co,case.root,arm,row["status"],row["calls"],round(row["wall_seconds"],4),flush=True)
                # Audit only after every arm has completed this root; no update.
                audits.append(static_audit(repl,case,sub))
    (out/"Witnesses.lean").write_text("\n".join(proofs))
    (out/"episodes.jsonl").write_text("".join(b.canonical(r)+"\n" for r in rows))
    (out/"static_audit.json").write_text(json.dumps(audits,indent=2)+"\n")
    agg={}
    for co in cohorts:
        base={r["root"]:r for r in rows if r["cohort"]==co and r["arm"]=="risk_path"}
        agg[co]={}
        for arm in ARMS:
            sub=[r for r in rows if r["cohort"]==co and r["arm"]==arm]
            paired=[r["calls"]-base[r["root"]]["calls"] for r in sub if r["closed"] and base[r["root"]]["closed"]]
            agg[co][arm]={"roots":len(sub),"closed":sum(r["closed"] for r in sub),
                 "mean_calls":statistics.mean(r["calls"] for r in sub),
                 "mean_wall_seconds":statistics.mean(r["wall_seconds"] for r in sub),
                 "mean_planning_seconds":statistics.mean(r["planning_seconds"] for r in sub),
                 "mean_dp_states":statistics.mean(r["dp_states"] for r in sub),
                 "paired_closed_n":len(paired),"paired_delta":statistics.mean(paired) if paired else None,
                 "wins_ties_losses":[sum(x<0 for x in paired),sum(x==0 for x in paired),sum(x>0 for x in paired)],
                 "statuses":dict(collections.Counter(r["status"] for r in sub))}
    summary={"schema":"lbf20d.native-static-guard.v1","scope":"development_only","episodes":len(rows),
             "native_tactic_calls":sum(r["calls"] for r in rows),
             "native_kernel_certificates":sum(r["kernel"] is not None for r in rows),
             "audit_queries":sum(r["queries"] for r in audits),
             "static_dynamic_disagreements":sum(len(r["differences"]) for r in audits),
             "cohorts":agg}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("NATIVE_RESULTS",json.dumps(summary),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);main(p.parse_args().out)
