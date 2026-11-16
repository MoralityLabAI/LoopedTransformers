"""RAM-16 postscore composite: preserve immutable first-run and two repairs.

Merge exact original primary sources from Actions 43805791781, secondary-only
operational correction from 43806362281, and *postscore* original-20 ablation
correction, each archived under a distinct run. Never overwrite or relabel the
original zero-call secondary or expanded-grammar nominal ablation receipts.
This script is source-only; it performs no model inference or Lean evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
OLD_RUN=43805791781
SECOND_RUN=43806362281
PHASES={
  "primary":["native","frequency","shape_frequency","loop64_task_only",
      "loop64_structural","flat71_task_only","flat71_structural","frozen_loop64_proxy"],
  "secondary":["native","shape_frequency","loop64_structural","flat71_structural"],
  "ablation":["native","loop64_structural"]
}
COUNTS={p:10*len(names) for p,names in PHASES.items()}


def digest(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_case(folder:Path,*,phase:str,index:int,policy:str,original_invalid:str|None=None):
    file=folder/"case.json"
    if not file.is_file():
        raise FileNotFoundError("missing immutable case "+str(file))
    data=json.loads(file.read_text())
    if data["policy"]!=policy or data["theorem"]=="":
        raise AssertionError("policy or theorem drift: "+str(file))
    if data["logical_evaluations"]<0 or data["elapsed_seconds"]<0:
        raise AssertionError("invalid native resource accounting")
    actual=data["grammar_receipt"]["ablation_original_20"]
    if original_invalid=="ablation":
        if actual is not False:
            raise AssertionError("first nominal ablation did not exhibit recorded reset")
    elif original_invalid=="secondary":
        if data["reason"]!="unexpected_error" or data["logical_evaluations"]!=0:
            raise AssertionError("first secondary invalidity changed")
    elif actual!=(phase=="ablation"):
        raise AssertionError("corrected ablation or expanded grammar flag mismatch")
    return dict(phase=phase,index=index,policy=policy,theorem=data["theorem"],
      accepted=bool(data["accepted"]),reason=data["reason"],
      logical_evaluations=data["logical_evaluations"],
      source_sha256=data["original_source_sha256"],
      original_receipt_sha256=digest(file),
      original_grammar_used=actual,original_status=original_invalid)


def copy_case(root:Path,output:Path,phase:str,index:int,policy:str,invalid:str|None=None):
    path=root/phase/f"{index:02d}"/policy
    report=read_case(path,phase=phase,index=index,policy=policy,original_invalid=invalid)
    new=(output/("invalid_original_"+phase if invalid else phase)/
         f"{index:02d}"/policy)
    new.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(path,new,symlinks=False)
    if digest(new/"case.json")!=report["original_receipt_sha256"]:
        raise AssertionError("original first-run bytes changed during copy")
    return report


def compose(original:Path,secondary:Path,ablation:Path,out:Path,
            *,ablation_run_id:int):
    if out.exists():
        raise FileExistsError("composite output already exists; preserve initial scoring")
    if ablation_run_id in (OLD_RUN,SECOND_RUN) or ablation_run_id<=0:
        raise ValueError("corrected ablation must have its own immutable Actions run")
    test=json.loads((HERE/"LOCKED_TEST_MANIFEST.json").read_text())
    if test["case_count"]!=10 or not test["selected_before_goal_state_access"]:
        raise AssertionError("original source-locked TEST manifest changed")
    if test["test_source_commit"]!="fdbe1581e95427016699169fd19c9e915a40bcf9":
        raise AssertionError("original source revision changed")
    if len(PHASES["primary"])!=8 or len(PHASES["secondary"])!=4 or len(PHASES["ablation"])!=2:
        raise AssertionError("registered policy list drifted")
    out.mkdir(parents=True)
    rows=[]
    for index in range(10):
        expected_module=test["cases"][index//2]["module"]
        expected_theorem=test["cases"][index//2]["theorems"][index%2]
        for phase in PHASES:
            if phase=="primary":
                source=original
            elif phase=="secondary":
                source=secondary
            else:
                source=ablation
            for policy in PHASES[phase]:
                item=copy_case(source,out,phase,index,policy)
                if item["theorem"]!=expected_theorem:
                    raise AssertionError("frozen TEST source theorem drift: "+str(item))
                item["module"]=expected_module
                rows.append(item)
        # Preserve invalid original secondary and nominal ablation without
        # allowing either to receive the corrected phase label.
        for phase in ("secondary","ablation"):
            for policy in PHASES[phase]:
                extra=copy_case(original,out,phase,index,policy,invalid=phase)
                if extra["theorem"]!=expected_theorem:
                    raise AssertionError("original invalid case theorem differs")
                extra["module"]=expected_module
                rows.append(extra)
    assert len(rows)==200, len(rows) # 140 corrected/valid + 60 preserved invalid
    primary=[r for r in rows if r["phase"]=="primary" and not r["original_status"]]
    corrected_secondary=[r for r in rows if r["phase"]=="secondary" and not r["original_status"]]
    corrected_ablation=[r for r in rows if r["phase"]=="ablation" and not r["original_status"]]
    if (len(primary),len(corrected_secondary),len(corrected_ablation))!=(80,40,20):
        raise AssertionError("original registered phase matrix incomplete")
    if sum(1 for r in rows if r["original_status"]=="secondary")!=40:
        raise AssertionError("missing original failed secondary receipts")
    if sum(1 for r in rows if r["original_status"]=="ablation")!=20:
        raise AssertionError("missing original expanded-grammar nominal ablations")
    for phase,items in (("primary",primary),("secondary",corrected_secondary),
                        ("ablation",corrected_ablation)):
        if len({(r["index"],r["policy"]) for r in items})!=COUNTS[phase]:
            raise AssertionError("duplicate original source case")
    result=dict(schema="RAM16-combined-postscore-audit-input-v1",
       original_source_commit=test["test_source_commit"],case_count=10,
       original_primary_actions_run=OLD_RUN,
       original_first_secondary_and_nominal_ablation_run=OLD_RUN,
       corrected_secondary_actions_run=SECOND_RUN,
       corrected_ablation_actions_run=ablation_run_id,
       primary_policy_cases=80,corrected_secondary_cases=40,
       corrected_original_grammar_ablation_cases=20,
       preserved_invalid_original_secondary_cases=40,
       preserved_invalid_original_ablation_cases=20,
       three_compiler_source_audit_pending=True,
       original_primary_modified=False,
       new_training_or_model_selection=False,
       original_RAM08_PEF_scored=0,
       original_RAM10_test_scored=0,
       scientific_qualification="Primary is original prospective; secondary and original-grammar ablation were operationally repaired after primary outcomes, with original invalid receipts preserved",
       original_policy_case_count=140,
       original_byte_hashes=[dict(phase=r["phase"],index=r["index"],policy=r["policy"],
             sha256=r["original_receipt_sha256"],original_status=r["original_status"])
             for r in rows])
    (out/"COMPOSITE_PROVENANCE.json").write_text(json.dumps(result,indent=2)+"\n")
    print("RAM16_COMPOSITE_READY",json.dumps({k:v for k,v in result.items()
          if k!="original_byte_hashes"},sort_keys=True),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--secondary",type=Path,required=True)
    p.add_argument("--ablation",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--ablation-run-id",type=int,required=True)
    a=p.parse_args()
    compose(a.original.resolve(),a.secondary.resolve(),a.ablation.resolve(),
            a.out.resolve(),ablation_run_id=a.ablation_run_id)
