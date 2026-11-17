"""LS-SAE07 source-only fixed-roster admission audit; no model access.

A failure of a frozen exact-hole source seed is recorded as a source failure
without replacement, a smaller study or changed denominator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from mechinterp.sae_causal_transfer.native_hard_task_v1 import families as FAM
from mechinterp.sae_causal_transfer.native_trm_v1 import families as S2

SCHEMA="LS-SAE07/model-blind-hard-puzzle-source-admission-v1"
PROTOCOL="2641f05d5cc9a1d6c7359543c44693b151d3c3cf"


def evaluate():
    tested={}
    complete=set();inputs=set()
    any_fail=False
    for key in FAM.BASES:
        rows=[]
        refused=[]
        for i in range(FAM.SIZES[key]):
            seed=FAM.BASES[key]+i
            full=S2.full_solution(seed)
            if S2.rectangles(full)<=0:
                raise AssertionError("registered MRV source orbit invariant violated")
            try:
                puzzle=FAM.dig_hard(full,seed+1000000,FAM.HOLES[key])
            except ValueError as error:
                refused.append({"grid_seed":seed,"reason":str(error)})
                any_fail=True
                continue
            if tuple(full) in complete or tuple(puzzle) in inputs:
                raise AssertionError("registered seed duplicated source across families")
            complete.add(tuple(full));inputs.add(tuple(puzzle))
            rows.append({"grid_seed":seed,"unique_holes":FAM.HOLES[key],
                         "source_solution_sha256":hashlib.sha256("".join(map(str,full)).encode()).hexdigest(),
                         "visible_puzzle_sha256":hashlib.sha256("".join(map(str,puzzle)).encode()).hexdigest()})
        tested[key]={"requested_slots":FAM.SIZES[key],
                     "admitted_slots":len(rows),
                     "refused_slots":refused,
                     "admitted_source_receipts":rows,
                     "source_cohort_complete":not refused}
    return {"schema":SCHEMA,"preregistration_commit":PROTOCOL,
            "all_generated_from_registered_source_grids":True,
            "all_source_admission_before_pretrained_model_access":True,
            "full_scientific_cohort_executable":not any_fail,
            "per_family":tested}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=evaluate()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:{"admitted":v["admitted_slots"],"requested":v["requested_slots"],
                         "rejected":v["refused_slots"]} for k,v in result["per_family"].items()},indent=2))
    print("LS-SAE07_FULL_SOURCE_ADMISSION",result["full_scientific_cohort_executable"])


if __name__=="__main__":
    main()
