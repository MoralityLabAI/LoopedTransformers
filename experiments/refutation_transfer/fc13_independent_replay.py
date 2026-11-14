"""FC-13 independent post-score replay of frozen hard-deadline proof certificates.

Rebuild original theorem script prefixes independently of fc13_policy.py.
All 64 recorded chosen scripts are separately compiled under Lean4.35.
Only original score-run certificates count toward proof coverage: a proof
compiled in this untimed replay cannot retroactively cross the hard deadline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from fc12_independent_replay import reconstruct_original,compile_lean
from fc13_policy import (CHECKPOINTS,DEADLINE_MS,MAX_LEAN_CALLS,
                         FROZEN_MODEL_RUN,SOURCE_SHA)

EVAL_RUN=43554219466
TEST_SOURCE="e8a780254b5cf41d479095f4f22bcfc4aa76ed72"
CASES={
    "MathCan/HookMultiTick.lean":(
        "distribute_sound","three_account_conservation",
        "before_swap_exact_input","specified_credit_restores_debit",
    ),
    "MathCan/Settlement.lean":(
        "marked_total_conserved","exact_five_account_partition",
        "terminal_pool_floor_budget","certificate_gap_identity",
    ),
    "MathCan/TOEJordan.lean":(
        "metric_vanishes_at_double_root","mixed_vanishes_at_double_root",
        "metric_derivative_vanishes_at_root","rank_one_root_kernel",
    ),
    "MathCan/TOENonAbelian.lean":(
        "flavor_exchange","signed_color_norm",
        "signed_color_dot","mixed_under_flavor_exchange",
    ),
}


def file_sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(artifact:Path,mathcan:Path,out:Path,lean:str="lean")->dict:
    if (mathcan/"lean-toolchain").read_text().strip()!="leanprover/lean4:v4.35.0":
        raise AssertionError("original Lean toolchain mismatch")
    original_commit=subprocess.check_output(
        ["git","-C",str(mathcan),"rev-parse","HEAD"],text=True).strip()
    if original_commit!=SOURCE_SHA:
        raise AssertionError("original MathCan source was changed")
    report=json.loads((artifact/"FC13_report.json").read_text())
    assert report["original_mathcan_commit"]==SOURCE_SHA
    assert report["frozen_model_training_run"]==FROZEN_MODEL_RUN
    assert {int(k):v for k,v in report["checkpoint_sha256"].items()}==CHECKPOINTS
    assert report["reference_theorems"]==16
    assert report["registered_wall_deadline_ms"]==DEADLINE_MS
    assert report["max_actual_lean_subprocesses"]==MAX_LEAN_CALLS
    assert {x["seed"] for x in report["results"]}=={17,29}

    expected={(f,n) for f,names in CASES.items() for n in names}
    files={f:(mathcan/f).read_text(encoding="utf8") for f in CASES}
    rows=[]
    for summary in report["results"]:
        seed=summary["seed"]
        for kind in ("model","native"):
            result_path=artifact/f"fc13_seed{seed}_{kind}.jsonl"
            if file_sha(result_path)!=summary[f"{'student' if kind=='model' else 'native'}_output_sha256"]:
                raise AssertionError("frozen scored candidate output digest changed")
            cases=[json.loads(x) for x in result_path.read_text().splitlines()]
            keys={(r["source_file"],r["theorem"]) for r in cases}
            if keys!=expected or len(cases)!=16:
                raise AssertionError("source selection or theorem case order changed")
            for row in cases:
                file=row["source_file"]
                theorem=row["theorem"]
                if row["lean_subprocess_count"]!=len(row["tactic_trials"]):
                    raise AssertionError("did not account for every actual Lean call")
                if not 0<=row["lean_subprocess_count"]<=MAX_LEAN_CALLS:
                    raise AssertionError("compiler subprocess budget breached")
                if len(row["chosen_actions"])>3:
                    raise AssertionError("script transition budget breached")
                if row["deadline_ms"]!=DEADLINE_MS:
                    raise AssertionError("deadline changed")
                if row["accepted"] and row["elapsed_ms"]>DEADLINE_MS:
                    raise AssertionError("late proof received score credit")
                if row["accepted"] and row["reason"]!="kernel_certified":
                    raise AssertionError("untrusted scratch or incomplete proof accepted")
                if row["accepted"] and not any(
                    x.get("final_full_original_theorem") and x.get("certified")
                    for x in row["tactic_trials"]
                ):
                    raise AssertionError("accepted theorem lacks separate final Lean proof")

                generated=reconstruct_original(files[file],theorem,row["chosen_actions"])
                evidence=compile_lean(generated,lean,90)
                if row["accepted"] and not evidence["compiled_sorry_free"]:
                    raise AssertionError("original certificate failed independent compiler")
                if row["accepted"]:
                    final=[x for x in row["tactic_trials"] if x.get("final_full_original_theorem")]
                    if len(final)!=1 or final[0]["candidate_source_sha256"]!=evidence["source_sha256"]:
                        raise AssertionError("independent certificate source hash differs")
                rows.append({
                    "seed":seed,"controller":kind,
                    "source_file":file,"theorem":theorem,
                    "frozen_certified":row["accepted"],
                    "independent_full_compiled":evidence["compiled_sorry_free"],
                    "independent_source_sha256":evidence["source_sha256"],
                    "compiler_returncode":evidence["returncode"],
                    "compiler_error_excerpt":evidence["error_excerpt"],
                    "chosen_actions":row["chosen_actions"],
                    "frozen_reason":row["reason"],
                    "frozen_elapsed_ms":row["elapsed_ms"],
                    "frozen_compiler_calls":row["lean_subprocess_count"],
                    "frozen_deadline_overrun_ms":row["deadline_overrun_ms"],
                    "unscored_semantically_valid":bool(
                         evidence["compiled_sorry_free"] and not row["accepted"]),
                })
    if len(rows)!=64:
        raise AssertionError("must independently recompile all 64 original cases")
    out.mkdir(parents=True,exist_ok=True)
    corpus="".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in rows)
    (out/"fc13_independent_compiler_replay.jsonl").write_text(corpus,encoding="utf8")
    receipt={
        "experiment":"FC13_original_frozen_hard_deadline_independent_replay",
        "original_scored_run":EVAL_RUN,
        "original_mathcan_commit":SOURCE_SHA,
        "heldout_source_revision":TEST_SOURCE,
        "recompiled_cases":len(rows),
        "false_positive_certificates":sum(x["frozen_certified"] and
            not x["independent_full_compiled"] for x in rows),
        "unscored_semantically_valid_count":sum(x["unscored_semantically_valid"] for x in rows),
        "replay_jsonl_sha256":hashlib.sha256(corpus.encode("utf8")).hexdigest(),
        "original_report_sha256":file_sha(artifact/"FC13_report.json"),
        "per_seed":[{
            "seed":seed,
            "model_certified":sum(x["frozen_certified"] for x in rows if
                              x["seed"]==seed and x["controller"]=="model"),
            "native_certified":sum(x["frozen_certified"] for x in rows if
                              x["seed"]==seed and x["controller"]=="native"),
        } for seed in (17,29)],
    }
    (out/"FC13_independent_replay_receipt.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print("FC13_INDEPENDENT_REPLAY",json.dumps(receipt,sort_keys=True),flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact",type=Path,required=True)
    p.add_argument("--mathcan",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",default="lean")
    args=p.parse_args()
    audit(args.artifact,args.mathcan,args.out,args.lean)


if __name__=="__main__":
    main()
