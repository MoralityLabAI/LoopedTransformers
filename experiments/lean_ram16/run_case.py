"""Score one previously untouched RAM-16 original MathCan theorem in Lean4.35.

Exact original 14-policy primary/secondary/ablation matrix: 8+4+2.
All source-protocol and model SHA gates run before any original TEST goal
is opened. Results are single-use, with every attempted compiler action
and accepted entire theorem source preserved for third independent replay.
"""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import harness
from native_replay import sha

PHASES=("primary","secondary","ablation")

def run(index:int,phases=PHASES):
    config=harness.CONFIG
    if subprocess.check_output(["git","-C",str(harness.SRC),"rev-parse","HEAD"],text=True).strip()!=config["test_source_commit"]:
        raise ValueError("original TEST source commit changed")
    if (harness.SRC/"lean-toolchain").read_text().strip()!=config["lean_toolchain"]:
        raise ValueError("different Lean compiler version")
    if not harness.LEAN.is_file():raise FileNotFoundError("pinned Lean4.35 missing")
    if harness.LOCKED_SOURCE["case_count"]!=10:raise AssertionError("locked original theorem count")
    ranker=harness.Rankers()
    # Original TEST source modules are opened only after independently
    # recorded four-checkpoint selected DEV hashes are verified by Rankers.
    sources=harness.module_sources()
    cases=[(module,root) for module,(_,roots) in harness.LOCKED.items() for root in roots]
    if len(cases)!=10:raise AssertionError("original source target count changed")
    if not 0<=index<10:raise ValueError("case index outside original registrar")
    module,theorem=cases[index]
    for phase in phases:
        if phase=="primary":
            budget=harness.PRIMARY;policies=harness.POLICIES;ablation=False
        elif phase=="secondary":
            budget=harness.SECONDARY;policies=harness.SECOND;ablation=False
        elif phase=="ablation":
            budget=harness.PRIMARY;policies=harness.ABLATED;ablation=True
        else:raise ValueError("unregistered RAM16 scoring phase")
        order=policies[index%len(policies):]+policies[:index%len(policies)]
        for policy in order:
            folder=harness.OUT/phase/f"{index:02d}"/policy
            if folder.exists():
                raise FileExistsError("original first-scored output already exists: "+str(folder))
            result=harness.search(sources[module],theorem,ranker,policy,budget,folder,
                ablation=ablation)
            if result["logical_evaluations"]>budget["evaluations"]:
                raise AssertionError("score over original Lean tactic budget")
            if result["accepted"] and (not result["certificate"] or not result["replay"]):
                raise AssertionError("accepted source without paired original compiler proofs")
            print("RAM16_PROSPECTIVE_SCORE",json.dumps(dict(
                    phase=phase,index=index,module=module,theorem=theorem,policy=policy,
                    accepted=result["accepted"],reason=result["reason"],
                    original_calls=result["logical_evaluations"],
                    elapsed=round(result["elapsed_seconds"],5)),
                  sort_keys=True),flush=True)
    final=harness.OUT/"case_completed";final.mkdir(parents=True,exist_ok=True)
    item=dict(index=index,module=module,theorem=theorem,source_commit=config["test_source_commit"],
        original_source_sha256=sha(sources[module].encode()),phases=list(phases),
        source_test_read_only=True,model_selection_frozen_before_test=True,
        original_source_tests_replaced_by_generated_tactics=True,
        score_class="first_prospective_theorem_family")
    (final/f"{index:02d}.json").write_text(json.dumps(item,indent=2)+"\n")
    print("RAM16_CASE_FINISHED",index,module,theorem,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",type=int,required=True)
    p.add_argument("--phase",choices=(*PHASES,"all"),default="all")
    a=p.parse_args()
    run(a.case,PHASES if a.phase=="all" else (a.phase,))
