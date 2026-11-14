"""FC-13 preregistered fresh original MathCan transfer with a HARD Lean deadline.

An independent FC-13 experiment; FC-12 training weights and action generator
are immutable. Both model and native share a fixed per-theorem 20 s wall
budget, 12 *actual* Lean subprocess calls and a separate full proof gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from fc12_policy import (FC12Loop, graph, hash_file, read_jsonl,
                         SOURCE_MATHCAN, EARLY_TRAIN_GOAL_SHA,
                         EARLY_TRAIN_PROBE_SHA)

FROZEN_MODEL_RUN=43547762837
FC13_REGISTRATION="309fed65caf67da46db8cb981757a9326bc59fbc"
SOURCE_SHA=SOURCE_MATHCAN
CHECKPOINTS={
    17:"55bc16117e045279ba0e723c171e45b4fe05b4c6eb8a1b9709dd1b59cc9987a2",
    29:"0e11d74cf1be7889caf824671099d052884dc3722211e773dd126d233671a805",
}
DEADLINE_MS=20000.
MAX_LEAN_CALLS=12
MAX_TRANSITIONS=3
PER_CALL_TIMEOUT_MS=8000.
TOOLCHAIN="leanprover/lean4:v4.35.0"


def frozen_weights(directory:Path)->dict[int,FC12Loop]:
    manifest=json.loads((directory/"train_manifest.json").read_text())
    if manifest["train_actual_goal_sha"]!=EARLY_TRAIN_GOAL_SHA:
        raise AssertionError("FC13 train-state digest changed")
    if manifest["train_probe_data_sha"]!=EARLY_TRAIN_PROBE_SHA:
        raise AssertionError("FC13 train-action effects changed")
    if manifest["train_heldout_selector_not_available"] is not True:
        raise AssertionError("frozen model had heldout source access")
    if {x["seed"] for x in manifest["models"]}!={17,29}:
        raise AssertionError("changed seed set")
    result={}
    for row in manifest["models"]:
        seed=row["seed"]
        file=directory/row["filename"]
        if (row["checkpoint_sha256"]!=CHECKPOINTS[seed]
                or hash_file(file)!=CHECKPOINTS[seed]):
            raise AssertionError("original FC12 frozen model changed")
        model=FC12Loop()
        model.load_state_dict(torch.load(file,map_location="cpu",weights_only=True))
        if sum(p.numel() for p in model.parameters())!=62401:
            raise AssertionError("model architecture changed")
        model.eval()
        result[seed]=model
    return result


def trace_original_test(source_root:Path,mathcan:Path,out:Path):
    """Compile original source reference proofs, never use proof text at inference."""
    from research.refutation_bench.fc13 import test_data
    if (mathcan/"lean-toolchain").read_text().strip()!=TOOLCHAIN:
        raise AssertionError("original compiler toolchain differs")
    receipt=test_data.emit(mathcan,out)
    for path in test_data.TEST_MODULES:
        stem=Path(path).stem
        proc=subprocess.run(["lean",str(out/(stem+".lean"))],
            capture_output=True,text=True,timeout=120)
        (out/(stem+".log")).write_text(proc.stdout+proc.stderr,encoding="utf8")
        if proc.returncode!=0 or "error:" in proc.stdout+proc.stderr or "sorryAx" in proc.stdout+proc.stderr:
            raise AssertionError("original oracle theorem compilation failed: "+path)
    extraction=test_data.extract(out)
    rows=read_jsonl(out/"actual_goal_states.jsonl")
    initial={(row["source_file"],row["theorem"]):row["actual_lean_goal_state"]
             for row in rows if row["stage_index"]==0}
    expected={(path,name) for path,names in test_data.TEST_MODULES.items() for name in names}
    if set(initial)!=expected or len(expected)!=16:
        raise AssertionError("FC13 original heldout theorem selection changed")
    return initial,receipt,extraction


def run_controller(
    original:str,theorem:str,initial_goal:str,
    model:FC12Loop|None,*,deadline_ms:float=DEADLINE_MS,
    compile_fn=None,clock=None,record_candidates:bool=True
)->dict:
    """Finite hard deadline with process-level timeouts, not just loop checks."""
    from research.refutation_bench.fc12 import core,scratch
    if compile_fn is None:
        compile_fn=scratch.compile_candidate
    if clock is None:
        clock=time.perf_counter
    now=clock
    began=now()
    stop_at=began+deadline_ms/1000.
    chosen=[]
    state=initial_goal
    attempted=[]
    calls=0
    transitions=0
    certified=False
    reason="goal_transition_budget"
    def left()->float:
        return (stop_at-now())*1000.

    for step in range(MAX_TRANSITIONS):
        if state.count("⊢")!=1:
            reason="multigoal_unsupported"
            break
        if left()<=0.:
            reason="hard_deadline"
            break
        if model is None:
            ranked=list(core.FAMILIES)
        else:
            with torch.no_grad():
                logits=model(graph([state]))[0]
                ranked=[core.FAMILIES[i] for i in logits.argsort(descending=True).tolist()]
        progressed=False
        pool=core.generate(state)
        for family in ranked:
            for script in pool[family]:
                if calls>=MAX_LEAN_CALLS:
                    reason="lean_subprocess_budget"
                    break
                if left()<=0.:
                    reason="hard_deadline"
                    break
                if not scratch.valid_script(script):
                    raise AssertionError("candidate script outside frozen original grammar")
                before=left()
                timeout_s=min(PER_CALL_TIMEOUT_MS,before)/1000.
                if timeout_s<=0.:
                    reason="hard_deadline"
                    break
                prefix=[*chosen,script]
                started=now()
                probe=compile_fn(original,theorem,prefix,probe=True,
                                 timeout_seconds=timeout_s)
                ended=now()
                calls+=1
                ran_past=ended>stop_at
                row={
                    "step":step,"family":family,"script":script,
                    "candidate_source_sha256":probe.get("source_sha256"),
                    "scratch_accepted":bool(probe["accepted"]),
                    "scratch_has_sorry":True,
                    "scratch_goal_count":int(probe.get("real_remaining_goal_count",0)),
                    "compiler_returncode":probe.get("compiler_returncode"),
                    "compiler_timed_out":bool(probe.get("timed_out",False)),
                    "compiler_ms":probe.get("lean_wall_ms"),
                    "call_timeout_seconds":round(timeout_s,6),
                    "call_wall_ms":round((ended-started)*1000,5),
                    "expired_at_return":ran_past,
                    "error_excerpt":probe.get("compiler_error_excerpt","")[:250],
                }
                attempted.append(row)
                if ran_past or left()<=0.:
                    reason="hard_deadline"
                    break
                if not probe["accepted"]:
                    continue
                goals=probe.get("real_remaining_goal_states",[])
                count=sum(g.count("⊢") for g in goals)
                if count!=row["scratch_goal_count"]:
                    raise AssertionError("mismatched compiler goal accounting")
                if count>1:
                    reason="multigoal_unsupported"
                    break
                if count==0:
                    if calls>=MAX_LEAN_CALLS:
                        reason="lean_subprocess_budget"
                        break
                    if left()<=0.:
                        reason="hard_deadline"
                        break
                    timeout_s=min(PER_CALL_TIMEOUT_MS,left())/1000.
                    fin_start=now()
                    final=compile_fn(original,theorem,prefix,probe=False,
                                     timeout_seconds=timeout_s)
                    fin_end=now()
                    calls+=1
                    ok=bool(final["accepted"] and not final.get("timed_out") and
                            fin_end<=stop_at and left()>=0.)
                    attempted.append({
                        "step":step,"family":family,"script":script,
                        "final_full_original_theorem":True,
                        "final_without_sorry":True,
                        "candidate_source_sha256":final.get("source_sha256"),
                        "compiler_returncode":final.get("compiler_returncode"),
                        "compiler_timed_out":bool(final.get("timed_out",False)),
                        "compiler_ms":final.get("lean_wall_ms"),
                        "call_timeout_seconds":round(timeout_s,6),
                        "call_wall_ms":round((fin_end-fin_start)*1000,5),
                        "expired_at_return":fin_end>stop_at,
                        "certified":ok,
                    })
                    if ok:
                        chosen=prefix
                        certified=True
                        reason="kernel_certified"
                    else:
                        reason="hard_deadline" if fin_end>=stop_at else "final_lean_rejected"
                    break
                nxt="\n".join(goals).strip()
                if nxt and nxt!=state.strip():
                    chosen=prefix
                    state=nxt
                    progressed=True
                    transitions+=1
                    reason="goal_advanced"
                    break
            if certified or reason in ("hard_deadline","multigoal_unsupported",
                          "lean_subprocess_budget","final_lean_rejected"):
                break
            if progressed:
                break
        if certified or reason in ("hard_deadline","multigoal_unsupported",
                         "lean_subprocess_budget","final_lean_rejected"):
            break
        if not progressed:
            reason="no_valid_progress"
            break
    elapsed=(now()-began)*1000
    # This final invariant prevents a successful compile that ended after the
    # deadline from ever being certified even if it returned zero.
    if elapsed>deadline_ms and certified:
        certified=False
        reason="hard_deadline"
    return {
        "accepted":certified,"reason":reason,
        "controller":"native_order" if model is None else "typedloop",
        "chosen_actions":chosen,
        "transition_count":transitions,
        "lean_subprocess_count":calls,
        "tactic_trials":attempted,
        "elapsed_ms":round(elapsed,6),
        "deadline_ms":deadline_ms,
        "deadline_overrun_ms":round(max(0,elapsed-deadline_ms),6),
        "certificate_completed_within_hard_deadline":certified,
        "original_reference_used_as_fallback":False,
    }


def run_eval(source:Path,mathcan:Path,weights:Path,out:Path):
    from research.refutation_bench.fc13 import test_data
    torch.set_num_threads(2)
    students=frozen_weights(weights)
    out.mkdir(parents=True,exist_ok=True)
    initial,source_receipt,extraction=trace_original_test(source,mathcan,out/"original_oracle")
    originals={p:(mathcan/p).read_text(encoding="utf8") for p in test_data.TEST_MODULES}
    cases=[(p,name) for p,names in test_data.TEST_MODULES.items() for name in names]
    results=[]
    for seed in (17,29):
        paired=[]
        by_controller={"model":[],"native":[]}
        for file,theorem in cases:
            goal=initial[file,theorem]
            m=run_controller(originals[file],theorem,goal,students[seed])
            n=run_controller(originals[file],theorem,goal,None)
            m.update(seed=seed,source_file=file,theorem=theorem)
            n.update(seed=seed,source_file=file,theorem=theorem)
            by_controller["model"].append(m)
            by_controller["native"].append(n)
            print("FC13_PAIR",json.dumps({
                "seed":seed,"source_file":file,"theorem":theorem,
                "student_certified":m["accepted"],"native_certified":n["accepted"],
                "student_reason":m["reason"],"native_reason":n["reason"],
                "student_elapsed_ms":m["elapsed_ms"],"native_elapsed_ms":n["elapsed_ms"],
                "student_calls":m["lean_subprocess_count"],
                "native_calls":n["lean_subprocess_count"]},sort_keys=True),flush=True)
        for controller,rows in by_controller.items():
            dest=out/f"fc13_seed{seed}_{controller}.jsonl"
            raw="".join(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n" for row in rows)
            dest.write_text(raw,encoding="utf8")
        count_m=sum(x["accepted"] for x in by_controller["model"])
        count_n=sum(x["accepted"] for x in by_controller["native"])
        ms_m=sum(x["elapsed_ms"] for x in by_controller["model"])
        ms_n=sum(x["elapsed_ms"] for x in by_controller["native"])
        results.append({
            "seed":seed,"student_certified":count_m,"native_certified":count_n,
            "student_total_wall_ms":round(ms_m,5),"native_total_wall_ms":round(ms_n,5),
            "student_ms_per_certificate":round(ms_m/count_m,5) if count_m else None,
            "native_ms_per_certificate":round(ms_n/count_n,5) if count_n else None,
            "model_only":sum(m["accepted"] and not n["accepted"]
                      for m,n in zip(by_controller["model"],by_controller["native"])),
            "native_only":sum(n["accepted"] and not m["accepted"]
                      for m,n in zip(by_controller["model"],by_controller["native"])),
            "student_subprocesses":sum(x["lean_subprocess_count"] for x in by_controller["model"]),
            "native_subprocesses":sum(x["lean_subprocess_count"] for x in by_controller["native"]),
            "student_output_sha256":hash_file(out/f"fc13_seed{seed}_model.jsonl"),
            "native_output_sha256":hash_file(out/f"fc13_seed{seed}_native.jsonl"),
        })
    learning=all(x["student_certified"]>x["native_certified"] for x in results)
    compute=learning and all(x["student_ms_per_certificate"] is not None
                and x["native_ms_per_certificate"] is not None
                and x["student_ms_per_certificate"]<x["native_ms_per_certificate"]
                for x in results)
    report={
        "experiment":"FC13_new_modules_strict_original_Lean435_hard_deadline",
        "preregistration_commit":FC13_REGISTRATION,
        "original_mathcan_commit":SOURCE_SHA,
        "frozen_model_training_run":FROZEN_MODEL_RUN,
        "frozen_train_state_sha256":EARLY_TRAIN_GOAL_SHA,
        "frozen_train_effect_sha256":EARLY_TRAIN_PROBE_SHA,
        "checkpoint_sha256":CHECKPOINTS,
        "heldout_16_original_oracle_compiled":True,
        "reference_theorems":len(cases),
        "registered_wall_deadline_ms":DEADLINE_MS,
        "max_actual_lean_subprocesses":MAX_LEAN_CALLS,
        "max_goal_transitions":MAX_TRANSITIONS,
        "results":results,
        "registered_learning_criterion":learning,
        "registered_compute_criterion":compute,
    }
    (out/"FC13_report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("FC13_CRITERIA",json.dumps(report,sort_keys=True),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--mathcan",type=Path,required=True)
    p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run_eval(a.source,a.mathcan,a.weights,a.out)


if __name__=="__main__":
    main()
