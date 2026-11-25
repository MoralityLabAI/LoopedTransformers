"""LBF-20F native certificates for shared existential assignments.
All allowed pairs are public local assumptions. Exact finite planning is legal;
there is no artificial requirement to verify discarded symbolic assignments.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
import torch
from lbf20f_csp import generate,Solver,bits
from lbf20f_learning import Net,score,ARMS,BUDGET

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"lean_cgrt_hrm400"))
from lcgrt.lean_repl import LeanRepl

CALL_CAP=96
WALL_CAP=4.
POLICIES=("deferred_lcv","deferred_bp","deferred_pc","deferred_direct1","deferred_recur4","deferred_untied4","live_lcv")


def layout(c):
    names=[];binders=[];atoms=[]
    for i in range(c.n):
        names.append(f"u{i}")
    binders.append("("+" ".join(names)+" : Nat → Prop)")
    relnames=[f"r{e}" for e in range(len(c.edges))]
    binders.append("("+" ".join(relnames)+" : Nat → Nat → Prop)")
    names+=relnames
    for i,d in enumerate(c.domains):
        atoms.append(f"u{i} x{i}")
        for a in bits(d):
            name=f"hu{i}_{a}";names.append(name);binders.append(f"({name} : u{i} {a})")
    for k,(i,j,rows) in enumerate(c.edges):
        atoms.append(f"r{k} x{i} x{j}")
        for a,r in enumerate(rows):
            for b in bits(r):
                name=f"hr{k}_{a}_{b}";names.append(name);binders.append(f"({name} : r{k} {a} {b})")
    goal=" ∧ ".join(atoms)
    for i in reversed(range(c.n)):
        goal=f"∃ x{i} : Nat, "+goal
    return "∀ "+" ".join(binders)+", "+goal,"intro "+" ".join(names)


def fact_names(c,v):
    return [*(f"hu{i}_{a}" for i,a in enumerate(v)),
            *(f"hr{k}_{v[i]}_{v[j]}" for k,(i,j,_) in enumerate(c.edges))]


def proof_term(c,v):
    f=fact_names(c,v)
    p=f[-1]
    for name in reversed(f[:-1]):p=f"And.intro {name} ({p})"
    for a in reversed(v):p=f"Exists.intro {a} ({p})"
    return p


def episode(repl,c,arm,models):
    started=time.perf_counter();stmt,intros=layout(c)
    t0=time.perf_counter();repl.reset();state=repl.init(stmt);init=time.perf_counter()-t0
    assert len(state["goals"])==1
    p=arm.removeprefix("deferred_").removeprefix("live_")
    model=models.get(p)
    s=Solver(c)
    result=s.solve(policy=p if model is None else "lcv",budget=BUDGET,
                   scorer=(lambda s,d,i:score(model,s,d,i)) if model is not None else None)
    v=result["solution"]
    script=[];events=[];kernel=None;status=result["status"]
    if v is not None:
        assert c.valid(v)
        if arm=="live_lcv":
            tactics=[*(f"refine ⟨{a}, ?_⟩" for a in v)]
            fs=fact_names(c,v)
            for i,f in enumerate(fs):
                if i<len(fs)-1:tactics.append("constructor")
                tactics.append("exact "+f)
        else:tactics=["exact "+proof_term(c,v)]
        for tac in tactics:
            if len(events)>=CALL_CAP or time.perf_counter()-started>=WALL_CAP:
                status="native_cap";break
            t0=time.perf_counter();o=repl.run(state["sid"],tac);dt=time.perf_counter()-t0
            events.append({"tactic":tac,"accepted":o["ok"],"seconds":dt,
                           "before":[g["target"] for g in state["goals"]],
                           "after":[g["target"] for g in o["state"]["goals"]] if o["ok"] else []})
            assert o["ok"],(c.root,arm,tac,o)
            state=o["state"];script.append(tac)
        if state["closed"]:
            t0=time.perf_counter();cert=repl.kernel(state["sid"]);kt=time.perf_counter()-t0
            assert cert["ok"] and cert["axioms"]==[],cert
            kernel={"axioms":[],"term_sha256":hashlib.sha256(cert["term"].encode()).hexdigest(),"seconds":kt}
            status="closed" if time.perf_counter()-started<=WALL_CAP else "late_closed"
    return {"root":c.root,"family":c.family,"arm":arm,"status":status,"closed":status=="closed",
            "symbolic":result,"native_calls":len(events),"init_seconds":init,
            "wall_seconds":time.perf_counter()-started,"kernel":kernel,
            "script":script,"events":events},stmt,intros


def metavariable_audit(repl):
    # Same x,y across both relations. One attractive local fact disagrees
    # on y; the explicit tuple must not silently select independent witnesses.
    stmt="∀ (P Q : Nat → Nat → Prop) (hp : P 1 2) (bad : Q 1 0) (good : Q 1 2), ∃ x y : Nat, P x y ∧ Q x y"
    repl.reset();s=repl.init(stmt)
    wrong=repl.run(s["sid"],"exact ⟨1, 2, hp, bad⟩")
    assert not wrong["ok"],wrong
    partial=repl.run(s["sid"],"refine ⟨?_, ?_, ?_, ?_⟩")
    trace={"wrong_tuple_rejected":True,"partial_accepted":partial["ok"],
           "partial_goals":[g["target"] for g in partial["state"]["goals"]] if partial["ok"] else [],
           "partial_error":str(partial.get("error",""))}
    if partial["ok"]:
        goals=partial["state"]["goals"]
        for i,g in enumerate(goals):
            if g["target"].get("raw","").startswith("P "):
                f=repl.focus(partial["state"]["sid"],i)
                trace["focus_accepted"]=f["ok"]
                if f["ok"]:
                    o=repl.run(f["state"]["sid"],"exact hp")
                    trace["partial_fact_accepted"]=o["ok"]
                    trace["partial_fact_goals"]=[g["target"] for g in o["state"]["goals"]] if o["ok"] else []
                    trace["partial_fact_error"]=str(o.get("error",""))
                break
    # Independent terminal proof from original state, not the speculative state.
    ok=repl.run(s["sid"],"exact ⟨1, 2, hp, good⟩")
    assert ok["ok"] and ok["state"]["closed"],ok
    cert=repl.kernel(ok["state"]["sid"])
    assert cert["ok"] and cert["axioms"]==[],cert
    trace["correct_tuple_kernel_closed"]=True
    return trace


def main(out,fit_dir):
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((fit_dir/"summary.json").read_text())
    models={}
    for arm in ARMS:
        f=next(f for f in manifest["fits"] if f["architecture"]==arm and f["seed"]==107)
        p=fit_dir/f"{arm}_107.pt"
        assert hashlib.sha256(p.read_bytes()).hexdigest()==f["checkpoint_sha256"]
        model=Net(arm);model.load_state_dict(torch.load(p,map_location="cpu",weights_only=True));model.eval();models[arm]=model
    cases=generate("native",6,12,2026112518)
    assert all(Solver(c).solve()["status"]=="solved" for c in cases)
    rows=[];proofs=["import Lean",""]
    with LeanRepl(imports=("Lean",),timeout=20) as repl:
        assert str(repl.version).startswith("4.19.0")
        for i,c in enumerate(cases):
            off=i%len(POLICIES)
            for arm in POLICIES[off:]+POLICIES[:off]:
                r,stmt,intros=episode(repl,c,arm,models);rows.append(r)
                if r["kernel"]:
                    name=f"lbf20f_{c.root}_{arm}"
                    proofs.extend([f"theorem {name} : {stmt} := by","  "+intros,
                                   *("  "+x for x in r["script"]),f"#print axioms {name}",""])
                print(c.root,arm,r["status"],r["symbolic"]["nodes"],r["native_calls"],flush=True)
        audit=metavariable_audit(repl)
    (out/"Witnesses.lean").write_text("\n".join(proofs),encoding="utf-8")
    (out/"episodes.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows),encoding="utf-8")
    ag={}
    for arm in POLICIES:
        rs=[r for r in rows if r["arm"]==arm]
        ag[arm]={"roots":len(rs),"closed":sum(r["closed"] for r in rs),
                 "mean_nodes":statistics.mean(r["symbolic"]["nodes"] for r in rs),
                 "mean_native_calls":statistics.mean(r["native_calls"] for r in rs),
                 "mean_symbolic_seconds":statistics.mean(r["symbolic"]["wall_seconds"] for r in rs),
                 "mean_wall_seconds":statistics.mean(r["wall_seconds"] for r in rs),
                 "status_counts":dict(collections.Counter(r["status"] for r in rs))}
    summary={"schema":"lbf20f.native.v1","scope":"development_only","roots":len(cases),"episodes":len(rows),
             "native_tactic_calls":sum(r["native_calls"] for r in rows),
             "native_kernel_certificates":sum(r["kernel"] is not None for r in rows),
             "metavariable_audit":audit,"arms":ag}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("NATIVE_RESULTS",json.dumps(summary),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    p.add_argument("--fit-dir",type=Path,required=True);a=p.parse_args();main(a.out,a.fit_dir)
