"""OLR-METTA-04A: native Lean dependent-witness proof preflight.

This module never infers proof success from arithmetic. Its generator supplies
frozen source statements; Lean checks every tactic and final proof. The
independent standalone compiler replays each complete original theorem.

The source-only witness reference is used for assertions AFTER native verdicts.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from .data import canonical, digest, sha256_file
from .live_lean import candidate_menu, public_state

SCHEMA = "ppt.olr-dependent-native.v1"
ROOT_MENU = ("apply r0", "apply r1")
PROOF_TAILS = {
    "manual": ("apply h0", "apply h1", "apply h0"),
    "fast": ("apply h0", "exact h1 _ h0"),
}
ALLOW = set(ROOT_MENU) | {"rfl", "apply h0", "apply h1", "exact h1 _ h0"}
STMT = re.compile(
    r"^∀ \(P Q : Nat → Prop\) \(R : Prop\) "
    r"\(h0 : P \((\d+) \+ (\d+)\)\) "
    r"\(h1 : ∀ n : Nat, P n → Q n\) "
    r"\(r0 : \((\d+) \+ (\d+) = (\d+)\) → P (\d+) → Q (\d+) → R\) "
    r"\(r1 : \((\d+) \+ (\d+) = (\d+)\) → P (\d+) → Q (\d+) → R\), R$"
)


@dataclass(frozen=True)
class DependentWorld:
    root_id: str
    pair_id: str
    split: str
    world: int
    a: int
    b: int
    delta: int

    @property
    def target(self) -> int:
        return self.a+self.b

    def candidates(self) -> tuple[int,int]:
        n=self.target
        return ((n,n+self.delta) if self.world==0
                else (n+self.delta,n))

    def statement(self) -> str:
        c0,c1=self.candidates()
        a,b=self.a,self.b
        return (
            "∀ (P Q : Nat → Prop) (R : Prop) "
            f"(h0 : P ({a} + {b})) "
            "(h1 : ∀ n : Nat, P n → Q n) "
            f"(r0 : ({a} + {b} = {c0}) → P {c0} → Q {c0} → R) "
            f"(r1 : ({a} + {b} = {c1}) → P {c1} → Q {c1} → R), R"
        )

    def reference_branch(self) -> int:
        """PRIVATE generator-derived audit label, not model input."""
        return self.world


def generate_pairs(split: str, *, seed: int, count: int, low: int, high: int,
                   excluded: frozenset[tuple[int,int,int]]=frozenset()):
    if split not in ("train", "dev", "ood") or not 1<=count<=96:
        raise ValueError("invalid dependent-root cohort")
    if not 17<=low<=high<=44:
        raise ValueError("outside bounded numerical source grammar")
    rng=random.Random(f"olr-dependent-v1/{split}/{seed}/{low}/{high}")
    seen=set(excluded)
    roots=[]
    for i in range(count):
        for retry in range(600):
            a,b=rng.randint(low,high),rng.randint(low,high)
            delta=rng.choice((1,2,3))
            if (a,b,delta) not in seen:
                break
        else:
            raise ValueError("source generator exhausted distinct lineages")
        seen.add((a,b,delta))
        pair_id=digest([SCHEMA,split,seed,i,a,b,delta])
        for world in (0,1):
            root_id=digest([SCHEMA,pair_id,world])
            roots.append(DependentWorld(root_id,pair_id,split,world,a,b,delta))
    return tuple(roots)


def validate_root(lean:LeanRepl, world:DependentWorld) -> dict:
    """Native root/shape check, optionally used before model actions."""
    lean.reset()
    root=lean.init(world.statement(),pp=True)
    pub=public_state(lean,root)
    menu=candidate_menu(lean,root)
    if menu!=ROOT_MENU or len(pub["goals"])!=1:
        raise ValueError(f"dependent root grammar changed: {menu}")
    # The public typed context is available, but this preflight emits only
    # hashes and shape; no witness label is exported to model observations.
    return {"state":root,"public":pub,"source_sha256":digest(pub),
            "menu":list(menu),"theorem_sha256":digest(world.statement())}


def _tactic(lean:LeanRepl,state:dict,action:str) -> tuple[dict,dict]:
    if action not in ALLOW:
        raise ValueError("tactic outside dependent proof whitelist")
    before=digest(public_state(lean,state))
    result=lean.run(state["sid"],action)
    if type(result.get("ok")) is not bool:
        raise ValueError("malformed native Lean tactic response")
    new=result["state"] if result["ok"] else state
    return new,{"kind":"tactic","tactic":action,"ok":result["ok"],
                "before":before,"after":digest(public_state(lean,new))}


def native_proof(lean:LeanRepl, world:DependentWorld,
                 *, first:int,tail:str) -> dict:
    """Execute and independently replay a two/three-pending-goal proof."""
    if first not in (0,1) or tail not in PROOF_TAILS:
        raise ValueError("unregistered branch or continuation")
    root=validate_root(lean,world)
    state=root["state"]
    events=[]
    script=[]
    state,e=_tactic(lean,state,f"apply r{first}")
    if not e["ok"]:
        raise ValueError("candidate root was rejected before its guard")
    events.append(e);script.append(e["tactic"])
    goals=public_state(lean,state)["goals"]
    if len(goals)!=3:
        raise ValueError("dependent candidate did not open three goals")
    first_goal=goals[0]["target"]
    pending=[g["target"] for g in goals[1:]]
    state,e=_tactic(lean,state,"rfl")
    events.append(e)
    if not e["ok"]:
        if first==world.reference_branch():
            raise ValueError("native guard contradicted the generator witness")
        return {"root_id":world.root_id,"statement_sha256":root["theorem_sha256"],
                "first":first,"tail":tail,"status":"guard_rejected",
                "kernel_closed":False,"calls":2,"events":events,
                "pending_after_apply":pending,"guard":first_goal,"script":[]}
    script.append("rfl")
    if first!=world.reference_branch():
        raise ValueError("wrong-world native guard unexpectedly passed")
    for action in PROOF_TAILS[tail]:
        state,e=_tactic(lean,state,action)
        if not e["ok"]:
            raise ValueError(f"native dependent proof failed at {action}")
        events.append(e);script.append(action)
    if public_state(lean,state)["goals"]:
        raise ValueError("native dependent proof left open goals")
    receipt=lean.kernel(state["sid"])
    if receipt.get("ok") is not True or receipt.get("axioms")!=[]:
        raise ValueError("kernel or axiom audit rejected dependent proof")
    events.append({"kind":"kernel","before":digest(public_state(lean,state)),
                   "after":digest(public_state(lean,state)),"ok":True})
    return {"root_id":world.root_id,"statement_sha256":root["theorem_sha256"],
            "first":first,"tail":tail,"status":"kernel_closed",
            "kernel_closed":True,"calls":2+len(PROOF_TAILS[tail]),"events":events,
            "pending_after_apply":pending,"guard":first_goal,"script":script}


def replay_proof(lean:LeanRepl,world:DependentWorld,ep:dict):
    root=validate_root(lean,world)
    state=root["state"]
    calls=0;closed=False
    script=[]
    for item in ep["events"]:
        before=digest(public_state(lean,state))
        if before!=item["before"]:
            raise ValueError("dependent replay source state mismatch")
        if item["kind"]=="tactic":
            state,actual=_tactic(lean,state,item["tactic"])
            if (actual!=item):
                raise ValueError("dependent native tactic outcome drifted")
            calls+=1
            if actual["ok"]:script.append(item["tactic"])
        elif item["kind"]=="kernel":
            receipt=lean.kernel(state["sid"])
            closed=receipt.get("ok") is True and receipt.get("axioms")==[]
            if not closed or digest(public_state(lean,state))!=item["after"]:
                raise ValueError("kernel replay of full theorem failed")
        else:
            raise ValueError("unknown dependent replay event")
    if (calls!=ep["calls"] or closed!=ep["kernel_closed"]
            or (closed and script!=ep["script"])
            or (not closed and ep["script"])):
        raise ValueError("dependent replay/accounting discrepancy")
    return True


def compile_certified(successes:list[tuple[DependentWorld,dict]],out:Path):
    """Independent compilation for the original quantified dependent theorem."""
    lines=["namespace OLRDependent"]
    unique={}
    for world,ep in successes:
        if not ep["kernel_closed"] or ep["root_id"]!=world.root_id:
            raise ValueError("uncertified dependent source")
        unique[world.statement(),tuple(ep["script"])]=ep
    for i,((statement,script),_) in enumerate(sorted(unique.items())):
        if STMT.fullmatch(statement) is None:
            raise ValueError("unsafe dependent theorem source")
        if (len(script)<4 or script[0] not in ROOT_MENU or script[1]!="rfl"
                or tuple(script[2:]) not in PROOF_TAILS.values()):
            raise ValueError("unsafe standalone dependent tactic script")
        lines.append(f"theorem dep_{i} {statement[2:-3]} : R := by")
        lines.extend("  "+a for a in script)
    lines.append("end OLRDependent")
    lines.extend(f"#print axioms OLRDependent.dep_{i}" for i in range(len(unique)))
    text="\n".join(lines)+"\n"
    out.write_text(text,encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        file=Path(tmp)/"Dependent.lean"
        file.write_text(text,encoding="utf-8")
        process=subprocess.run([str(config.LEAN_EXE),str(file)],
            env=config.lean_env(),capture_output=True,text=True,encoding="utf-8",
            timeout=120,check=False)
    detail=process.stdout+process.stderr
    clean=sum(f"'OLRDependent.dep_{i}' does not depend on any axioms" in detail
              for i in range(len(unique)))
    if process.returncode!=0 or clean!=len(unique):
        raise ValueError(f"standalone dependent Lean failed: rc={process.returncode}, "
                         f"clean={clean}/{len(unique)}, {detail[-2000:]}")
    return {"compiled":len(unique),"axiom_free":clean,
            "source_sha256":sha256_file(out)}


def preflight(out:Path, *, seed:int=20261130, pairs:int=3):
    if out.exists() or not 1<=pairs<=8:
        raise ValueError("new artifact directory and 1..8 pairs required")
    roots=generate_pairs("train",seed=seed,count=pairs,low=17,high=30)
    receipts=[];winners=[]
    with LeanRepl(timeout=60) as lean:
        if not str(lean.version).startswith("4.19.0"):
            raise ValueError("pinned Lean 4.19.0 required")
        for world in roots:
            for branch in (0,1):
                for tail in PROOF_TAILS:
                    ep=native_proof(lean,world,first=branch,tail=tail)
                    if not replay_proof(lean,world,ep):
                        raise ValueError("dependent source replay refused")
                    receipts.append({"root":world.root_id,"pair":world.pair_id,
                                     "world":world.world,**ep})
                    if ep["kernel_closed"]:
                        winners.append((world,ep))
    out.mkdir(parents=True)
    independent=compile_certified(winners,out/"standalone.lean")
    (out/"episodes.jsonl").write_text(
        "".join(canonical(ep)+"\n" for ep in receipts),encoding="utf-8")
    report={"schema":SCHEMA,"source_seed":seed,"pairs":pairs,"worlds":len(roots),
            "native_executions":len(receipts),"native_replays":len(receipts),
            "native_proofs":len(winners),"standalone":independent,
            "manual_tactic_calls":5,"fast_repair_tactic_calls":4,
            "evidence":"native_Lean_dependent_type_preflight",
            "architecture_promotion_eligible":False,
            "lean_binary_sha256":sha256_file(config.LEAN_EXE),
            "repl_binary_sha256":sha256_file(config.REPL_EXE)}
    report["artifacts"]={n:sha256_file(out/n)
                         for n in ("episodes.jsonl","standalone.lean")}
    (out/"manifest.json").write_text(canonical(report)+"\n",encoding="utf-8")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261130)
    p.add_argument("--pairs",type=int,default=3)
    args=p.parse_args(argv)
    print(canonical(preflight(args.out,seed=args.seed,pairs=args.pairs)))


if __name__=="__main__":
    main()
