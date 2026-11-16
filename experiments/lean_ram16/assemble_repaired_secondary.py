"""RAM-16 original first-run evidence assembly for independent Lean replay.

Only secondary records with the original zero-probe request/max-goal
config omission are replaced by separately labeled registered-budget
secondary correction receipts. Original primary and original-grammar
ablation scores and every accepted source remain byte-identical.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path

SOURCE_RUN=43805791781
CORRECTED_SECONDARY_RUN=43806362281
PHASES={"primary":("native","frequency","shape_frequency","loop64_task_only",
     "loop64_structural","flat71_task_only","flat71_structural","frozen_loop64_proxy"),
 "secondary":("native","shape_frequency","loop64_structural","flat71_structural"),
 "ablation":("native","loop64_structural")}
COUNTS={"primary":80,"secondary":40,"ablation":20}
FIRST_SECONDARY_INVALID_REASON="unexpected_error"
FIRST_SECONDARY_INVALID_CALLS=0
SOURCE_COMMIT="fdbe1581e95427016699169fd19c9e915a40bcf9"

def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()

def load_receipt(root:Path,phase:str,index:int,policy:str):
    path=root/phase/f"{index:02d}"/policy/"case.json"
    if not path.is_file():raise FileNotFoundError(str(path))
    row=json.loads(path.read_text())
    if row["policy"]!=policy or not row["theorem"] or row["logical_evaluations"]<0:
        raise AssertionError("source-provenance case metadata mismatch")
    return path,row


def assemble(first:Path,repair:Path,output:Path):
    if output.exists():raise FileExistsError("cannot overwrite first original case matrix")
    for path in (first,repair):
        if not path.is_dir():raise FileNotFoundError(str(path))
    # Verify every old secondary attempt was invalid before replacing any
    # machine-readable *audit view*. Keep full untouched old artifacts.
    original_invalid=[]
    for i in range(10):
        for policy in PHASES["secondary"]:
            path,row=load_receipt(first,"secondary",i,policy)
            if row["accepted"] or row["logical_evaluations"]!=FIRST_SECONDARY_INVALID_CALLS:
                raise AssertionError("an original secondary actually executed compiler actions")
            if row["reason"]!=FIRST_SECONDARY_INVALID_REASON:
                raise AssertionError("different original failure not covered by amendment E")
            original_invalid.append(dict(case_index=i,policy=policy,
                original_case_json_sha256=sha(path.read_bytes()),
                original_failure_reason=row["reason"],
                original_compiler_tactic_calls=0))
    output.mkdir(parents=True)
    allnew=[]
    copied={}
    for phase,policies in PHASES.items():
        source=repair if phase=="secondary" else first
        saved=[]
        for i in range(10):
            for policy in policies:
                path,row=load_receipt(source,phase,i,policy)
                budget=row["budget"]
                if phase=="secondary":
                    if budget!={"seconds":60,"evaluations":96,
                                "transitions":16,"request_seconds":3,"max_goals":8}:
                        raise AssertionError("corrected secondary changed original registered search budget")
                else:
                    if budget!={"seconds":30,"evaluations":48,
                                "transitions":10,"request_seconds":3,"max_goals":8}:
                        raise AssertionError("original primary/ablation scientific budget differs")
                if row["logical_evaluations"]>budget["evaluations"]:
                    raise AssertionError("candidate budget exceeded")
                if row["accepted"] and (row["reason"]!="independently_certified" or
                      row["elapsed_seconds"]>budget["seconds"]):
                    raise AssertionError("unverified or late first source proof")
                source_case=path.parent
                dest=output/phase/f"{i:02d}"/policy
                if dest.exists():raise FileExistsError("duplicate first case")
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copytree(source_case,dest)
                outpath=dest/"case.json"
                if outpath.read_bytes()!=path.read_bytes():
                    raise AssertionError("original scored case bytes changed during assembly")
                item=dict(phase=phase,index=i,policy=policy,
                    source_run=CORRECTED_SECONDARY_RUN if phase=="secondary" else SOURCE_RUN,
                    original_case_json_sha256=sha(path.read_bytes()),
                    assembled_case_json_sha256=sha(outpath.read_bytes()),
                    accepted=bool(row["accepted"]),
                    reason=row["reason"],calls=row["logical_evaluations"])
                saved.append(item);allnew.append(item)
        if len(saved)!=COUNTS[phase]:raise AssertionError("incomplete policy matrix: "+phase)
        copied[phase]=dict(cases=len(saved),certified=sum(v["accepted"] for v in saved),
            source_run=saved[0]["source_run"])
    if len(allnew)!=140 or len(original_invalid)!=40:
        raise AssertionError("matrix or invalid earlier secondary count drifted")
    record=dict(schema="RAM16-amendment-E-original-primary-validated-secondary-composite",
        original_first_source_run=SOURCE_RUN,corrected_secondary_source_run=CORRECTED_SECONDARY_RUN,
        source_commit=SOURCE_COMMIT,
        original_first_secondary_error_count=len(original_invalid),
        original_first_secondary_had_actual_Lean_tactic_calls=False,
        original_first_secondary_still_preserved_in_separate_artifacts=True,
        unchanged_original_primary_and_ablation_case_count=100,
        corrected_secondary_case_count=40,
        all_composite_case_count=len(allnew),
        copied=copied,
        primary_and_ablation_first_scored_bytes_preserved=True,
        new_source_or_model_checkpoint_selection_performed=False,
        original_RAM08_PEF_cases_scored=0,original_RAM10_cases_scored=0,
        original_invalid_secondary_case_digest=sha(json.dumps(original_invalid,sort_keys=True).encode()),
        audited_composite_case_manifest_sha256=sha(json.dumps(allnew,sort_keys=True).encode()))
    (output/"RAM16_ASSEMBLY_RECEIPT.json").write_text(json.dumps(record,indent=2)+"\n")
    (output/"RAM16_ORIGINAL_SECONDARY_INVALID_CASES.json").write_text(
        json.dumps(original_invalid,indent=2)+"\n")
    print("RAM16_AUDIT_ASSEMBLY",json.dumps(record,sort_keys=True),flush=True)
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--corrected-secondary",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    assemble(a.original.resolve(),a.corrected_secondary.resolve(),a.out.resolve())
