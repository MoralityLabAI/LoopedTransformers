"""NEC-03W/F01 source-fit CALIBRATION behavioral factorial pilot, CPU ONLY.

This operates exclusively on the ORIGINAL dev5-000..007 source fit context
groups16..23. The physical pair/sham roster, candidate IDs, source coefficients
and exact source-instance catalog are frozen separately BEFORE model outcomes.

Ten native modes: baseline, A, B, AB, C, D, CD, delayed AB+target rescue,
delayed AB+sham rescue, same-visit identity addback. One original recurrent
source visit per arm is manipulated; all other source-selected visits retain
the frozen full-affine donor. All 12 passes run; weights are not updated.

This is developmental functional CALIBRATION, not source query evaluation or
population confirmation. A separately registered independent CPU replay must
verify every original input, source label, geometry and logit.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from .nec03v_geometry import read_pack,file_hash
from .nec03v_source_attest import attest,sha256_file,canonical
from .nec03w_shortlist_registry import registered_shortlist
from .nec03w_f01_registry import physical_method_lock

SCHEMA="NEC-03W/F01/original-source-functional-calibration-factorial-behavior/v1"
ARMS=("composed","additive")
CAL_GROUPS=tuple(range(16,24))
CASES=256
ROOTS=tuple(range(8))
ORIGINAL_ZERO=(3,5)
MODES=("baseline","remove_A","remove_B","remove_AB","sham_C","sham_D",
       "sham_CD","late_target_rescue","late_matched_sham_rescue",
       "same_visit_exact_addback")


def sha_json(value):
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def input_gate(root:int,source:Path,t:Path,reset:Path|None,
               pack:Path,codes:Path,code_audit:Path,
               shortlist:Path,shortlist_audit:Path,
               physical:Path,physical_audit:Path):
    if type(root) is not int or root not in ROOTS:
        raise PermissionError("source F01 behavioral calibration restricted to historical original roots 0..7")
    from .nec03w_f01_behavior_registry import registered_behavior
    lock=registered_behavior()
    physical_method_lock()
    registered_shortlist()
    original=attest(root,source,t,reset)
    source_manifest,shards,packsha=read_pack(pack)
    calibration=json.loads(codes.read_text("utf-8"))
    audited_codes=json.loads(code_audit.read_text("utf-8"))
    selected=json.loads(shortlist.read_text("utf-8"))
    selected_audit=json.loads(shortlist_audit.read_text("utf-8"))
    phenotype=json.loads(physical.read_text("utf-8"))
    phenotype_audit=json.loads(physical_audit.read_text("utf-8"))
    registered_source=lock["original_source_input_roots"][str(root)]
    if (packsha!=registered_source["source_pack_manifest_sha256"]
            or source_manifest["world_index"]!=root
            or source_manifest["source_checkpoint_receipts"]!=original["source_checkpoint_receipts"]
            or calibration["world_index"]!=root
            or calibration["original_source_pack_manifest_sha256"]!=packsha
            or file_hash(codes)!=registered_source["source_calibration_codes_sha256"]
            or audited_codes["independent_code_audit_passed"] is not True
            or audited_codes["original_source_calibration_codes_sha256"]!=file_hash(codes)
            or audited_codes["original_source_query_groups_32_to_39_read"] is not False
            or calibrated_violations(calibration)
            or selected["world_index"]!=root
            or selected["new_native_model_forwards"]!=0
            or selected["new_task_labels_read"]!=0
            or selected["selection_roster_sha256"]!=registered_source["shortlist_roster_sha256"]
            or selected_audit["source_shortlist_sha256"]!=file_hash(shortlist)
            or selected_audit["original_source_site_selection_independently_recomputed"] is not True
            or phenotype["world_index"]!=root
            or phenotype["world_id"]!=original["world_id"]
            or phenotype["source_pack_sha256"]!=packsha
            or phenotype["physical_roster_sha256"]!=registered_source["physical_roster_sha256"]
            or file_hash(physical)!=registered_source["physical_roster_file_sha256"]
            or phenotype["task_labels_or_logits_read"]!=0
            or phenotype["new_model_forwards"]!=0
            or phenotype["factorial_behavior_executed"] is not False
            or phenotype["source_query_groups32_to39_not_read"] is not True
            or phenotype["selected_shortlist_roster_sha256"]!=selected["selection_roster_sha256"]
            or phenotype_audit["independent_original_source_physical_audit_passed"] is not True
            or phenotype_audit["world_index"]!=root
            or phenotype_audit["physical_roster_file_sha256"]!=file_hash(physical)
            or phenotype_audit["physical_roster_sha256"]!=phenotype["physical_roster_sha256"]
            or phenotype_audit["original_source_case_arm_records"]!=512
            or phenotype_audit["task_labels_read"]!=0
            or phenotype_audit["new_model_forwards"]!=0
            or set(phenotype["arms"])!=set(ARMS)):
        raise ValueError("F01 original source calibration physical or independently audited method/source SHA changed")
    for arm in ARMS:
        if (len(phenotype["arms"][arm]["cases"])!=CASES
                or phenotype["arms"][arm]["denominator"]!=CASES
                or phenotype["arms"][arm]["admissible_case_count"]!=
                   phenotype_audit["admissible_original_cases_by_arm"][arm]):
            raise ValueError("F01 original full 256-case physical denominator changed")
        selected_site=phenotype["arms"][arm]["original_source_site"]
        if selected_site is None:
            if root not in ORIGINAL_ZERO or shards[arm]["visits"]:
                raise ValueError("F01 original eligible arm unexpectedly lacks source site")
        elif (str(selected_site["visit"]) not in shards[arm]["visits"]
              or str(selected_site["visit"]) not in calibration["arms"][arm]["positive_codes_by_visit"]):
            raise ValueError("F01 selected recurrent site no longer belongs to frozen original source")
        elif root not in ORIGINAL_ZERO and phenotype["arms"][arm]["admissible_case_count"]<64:
            raise ValueError("F01 original physically feasible case coverage below registered threshold")
    return lock,original,shards,calibration,selected,phenotype,phenotype_audit


def calibrated_violations(item):
    return (
        item.get("original_calibration_group_indices")!=list(CAL_GROUPS)
        or item.get("source_fit_group_indices_used_by_discovery")!=list(range(16))
        or item.get("previously_scored_source_groups_excluded")!=list(range(24,32))
        or item.get("source_only_query_groups_excluded")!=list(range(32,40))
        or item.get("source_query_groups_32_to_39_read") is not False
        or item.get("task_labels_read")!=0
        or item.get("model_or_SAE_optimizer_updates")!=0
        or item.get("behavioral_factorial_interventions_executed")!=0
        or item.get("recruited_or_confirmation_world_reads")!=0
    )


def execute(root:int,source:Path,t:Path,reset:Path|None,
            pack:Path,codes:Path,code_audit:Path,
            shortlist:Path,shortlist_audit:Path,
            physical:Path,physical_audit:Path,out:Path,
            *,allow_registered_source_behavior:bool=False):
    lock,old,shards,coeffs,chosen,roster,independent=input_gate(
        root,source,t,reset,pack,codes,code_audit,
        shortlist,shortlist_audit,physical,physical_audit)
    if not allow_registered_source_behavior:
        return {
            "schema":SCHEMA,"dry_run":True,"world_id":old["world_id"],
            "source_groups":list(CAL_GROUPS),"cases_per_arm":CASES,
            "physical_sha":roster["physical_roster_sha256"],
            "admitted_cases_by_arm":independent["admissible_original_cases_by_arm"],
            "new_native_forwards":0,"task_labels_read":0,
            "protected_world_reads":0,"scientific_gate":"REDESIGN_OR_PILOT_ONLY",
        }
    if out.exists():
        raise FileExistsError("original source F01 behavioral output must be fresh")
    import torch
    if torch.version.cuda is not None or getattr(torch.version,"hip",None) is not None:
        raise RuntimeError("source F01 behavioral native runner requires CPU-only PyTorch")
    from .nec03w_native import native_forward,source_donor_modes,MODES as NATIVE_MODES
    from .nec03w_factorial import factorial_metrics
    from .nec03l_five_input_worlds import labels
    from .nec03r_source_recoverability import load_source_arm
    from research.pef07.pef07 import as_tensor,model_digest

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if tuple(NATIVE_MODES)!=MODES:
        raise ValueError("F01 frozen native factorial mode count changed")
    saved=[]
    by_arm={}
    all_forward_counts={}
    for arm in ARMS:
        frozen=load_source_arm(root,arm,source,reset=reset)
        model=frozen["model"].eval()
        model_before=model_digest(model)
        if (model_before!=old["source_checkpoint_receipts"][arm]["native_sha256"]
                or frozen["sae_sha256"]!=old["source_checkpoint_receipts"][arm]["sae_sha256"]):
            raise ValueError("original source teacher or SAE is not the frozen register")
        panel=frozen["catalog"]["train"][16*32:24*32]
        if len(panel)!=CASES:
            raise ValueError("original source calibration group panel must be 256 cases")
        true_input_sha=sha_json([list(row) for row in panel])
        if true_input_sha!=coeffs["arms"][arm]["original_original_source_input_sha256"]:
            raise ValueError("original source calibration input bit history SHA changed")
        selected=roster["arms"][arm]["original_source_site"]
        z_by_visit={}
        for visit in sorted(frozen["source_sites"]):
            rows=coeffs["arms"][arm]["positive_codes_by_visit"][str(visit)]
            if len(rows)!=CASES:
                raise ValueError("original source SAE selected visit missing 256 coefficients")
            z=torch.zeros((CASES,128),dtype=model.block.mlp_down.weight.dtype)
            for i,items in enumerate(rows):
                for feature,value in items:
                    z[i,int(feature)]=float(value)
            z_by_visit[int(visit)]=z
        if set(z_by_visit)!=set(frozen["source_sites"]):
            raise ValueError("F01 original selected source-site roster changed")
        result_rows=[]
        n_calls=0
        correct={mode:0 for mode in MODES}
        for group_offset,group in enumerate(CAL_GROUPS):
            cases=panel[group_offset*32:(group_offset+1)*32]
            torch_bits=as_tensor(cases)
            truth=labels(torch_bits,frozen["world"],control=(arm=="additive")).tolist()
            original_32=roster["arms"][arm]["cases"][group_offset*32:(group_offset+1)*32]
            if (len(original_32)!=32 or
                    any(row["group"]!=group or row["mask"]!=i
                        for i,row in enumerate(original_32))):
                raise ValueError("original source selected physical case order changed")
            geometry=[row["geometry"] for row in original_32]
            if selected is not None:
                plans=source_donor_modes(
                    frozen,{site:z[group_offset*32:(group_offset+1)*32]
                            for site,z in z_by_visit.items()},
                    target_visit=selected["visit"],
                    matched_geometry=geometry)
            else:
                plans={name:{"donors":{},"gates":{},"rescue":None} for name in MODES}
            outputs={}
            errors={}
            with torch.no_grad():
                for mode in MODES:
                    logits=native_forward(model,torch_bits,**plans[mode])
                    n_calls+=1
                    outputs[mode]=logits.tolist()
                    errors[mode]=[int(x!=y) for x,y in zip(
                        logits.argmax(1).tolist(),truth)]
                    correct[mode]+=sum(1-value for value in errors[mode])
            baseline=outputs["baseline"]
            exact=outputs["same_visit_exact_addback"]
            for case_id,geo in enumerate(geometry):
                if not geo["admissible"]:
                    if any(
                        max(abs(x-y) for x,y in zip(baseline[case_id],outputs[name][case_id]))>1e-6
                        for name in MODES
                    ):
                        raise ValueError("source F01 ineligible original case had a nonzero causal change")
                if max(abs(x-y) for x,y in zip(baseline[case_id],exact[case_id]))>1e-4:
                    raise ValueError("F01 same-visit exact restored source donor failed numerical identity")
                if selected is not None and selected["visit"]==11:
                    for name in ("late_target_rescue","late_matched_sham_rescue"):
                        if max(abs(x-y) for x,y in zip(
                            outputs["remove_AB"][case_id],outputs[name][case_id]))>1e-6:
                            raise ValueError("last source recurrent visit permits no later memory rescue")
            case_rows=[]
            for mask in range(32):
                valid=bool(geometry[mask]["admissible"])
                eligible_late=valid and selected is not None and selected["visit"]<11
                case_rows.append({
                    "eligible":valid,"delayed_rescue_eligible":eligible_late,
                    "errors":{mode:errors[mode][mask] for mode in MODES},
                })
            result_rows.extend(case_rows)
            saved.append({
                "schema":SCHEMA,"world_index":root,"arm":arm,
                "source_calibration_group":group,"focal_mask_order":list(range(32)),
                "source_input_histories":[list(row) for row in cases],
                "teacher_truth":truth,
                "original_physical_case_sha256":[sha_json(row) for row in original_32],
                "eligible":[v["eligible"] for v in case_rows],
                "delayed_rescue_eligible":[v["delayed_rescue_eligible"] for v in case_rows],
                "mode_logits":outputs,
                "task_outcomes_used_for_source_quartet_selection":False,
            })
        by_arm[arm]={"factorial":factorial_metrics(result_rows),
                     "correct_by_mode":correct,
                     "physical_admitted":roster["arms"][arm]["admissible_case_count"]}
        all_forward_counts[arm]=n_calls
        if model_digest(model)!=model_before:
            raise ValueError("original source-trained native model weights mutated")
    interaction=(by_arm["composed"]["factorial"]["factorial_group_minus_sham_interaction"]-
                 by_arm["additive"]["factorial"]["factorial_group_minus_sham_interaction"])
    rescue=(by_arm["composed"]["factorial"]["delayed_target_minus_sham_rescue_benefit"]-
            by_arm["additive"]["factorial"]["delayed_target_minus_sham_rescue_benefit"])
    effect=0. if root in ORIGINAL_ZERO else interaction
    out.mkdir(parents=True,exist_ok=False)
    raw=out/"factorial_source_calibration_raw.jsonl"
    raw.write_text("".join(canonical(item)+"\n" for item in saved),encoding="utf-8")
    result={
        "schema":SCHEMA,"world_id":old["world_id"],"world_index":root,
        "original_calibration_groups":list(CAL_GROUPS),
        "source_cases_per_arm":CASES,
        "original_source_pack_manifest_sha256":file_hash(pack/"manifest.json"),
        "source_code_calibration_sha256":file_hash(codes),
        "physical_roster_file_sha256":file_hash(physical),
        "physical_roster_sha256":roster["physical_roster_sha256"],
        "physical_independent_audit_file_sha256":file_hash(physical_audit),
        "source_F01_behavior_method_lock_sha256":file_hash(
            Path(__file__).with_name("NEC03W_F01_BEHAVIOR_LOCK.json")),
        "selected_source_visit_by_arm":{
            arm:(roster["arms"][arm]["original_source_site"] or {}).get("visit")
            for arm in ARMS},
        "factorial_outcome_by_arm":by_arm,
        "raw_root_composed_minus_additive_source_interaction":interaction,
        "registered_root_ITT_source_interaction":effect,
        "raw_root_composed_minus_additive_delayed_rescue":rescue,
        "root_fixed_original_ITT_zero":root in ORIGINAL_ZERO,
        "original_source_behavior_raw_file":raw.name,
        "original_source_behavior_raw_sha256":file_hash(raw),
        "native_forward_calls_by_arm":all_forward_counts,
        "new_original_source_behavior_native_forward_calls":sum(all_forward_counts.values()),
        "native_tied_recurrent_visits":12,
        "new_optimizer_updates":0,"source_quartet_selected_by_task_labels":False,
        "source_query32_to39_outcomes_read":False,
        "old_NEC03V_source24_to31_outcomes_read":False,
        "recruited_reserved_embargoed_confirmation_reads":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
        "population_or_confirmation_claim":False,
    }
    (out/"factorial_source_calibration_report.json").write_text(canonical(result)+"\n",encoding="utf-8")
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--nec03t-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--original-pack",type=Path,required=True)
    p.add_argument("--calibration-codes",type=Path,required=True)
    p.add_argument("--independent-calibration-audit",type=Path,required=True)
    p.add_argument("--label-blind-shortlist",type=Path,required=True)
    p.add_argument("--independent-shortlist-audit",type=Path,required=True)
    p.add_argument("--physical",type=Path,required=True)
    p.add_argument("--independent-physical",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--allow-registered-source-calibration-behavior",action="store_true")
    a=p.parse_args(argv)
    value=execute(a.world_index,a.source_root,a.nec03t_root,a.reset_root,
                  a.original_pack,a.calibration_codes,a.independent_calibration_audit,
                  a.label_blind_shortlist,a.independent_shortlist_audit,
                  a.physical,a.independent_physical,a.out,
                  allow_registered_source_behavior=a.allow_registered_source_calibration_behavior)
    print(canonical(value))
    return value


if __name__=="__main__":
    main()
