"""NEC-03W/F02: label-blind paired-case physical support overlay.

For each original dev5-000..007 source-fit calibration16..23 case, form a
shared *composed AND additive* admissibility mask using the already frozen
and independently audited NEC03W/F01 full 256-case physical rosters.

This makes any future composed-minus-additive comparison use equal original
case support and retains the original 256-case denominator with a zero
intervention for BOTH arms when either arm is ineligible.

ZERO native/SAE forwards, ZERO task labels/logits, ZERO parameter updates.
This new mask and >=95% paired physical coverage criterion are *post-root0*
design diagnostics, disclosed as such. Do not retroactively rescore F01.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .nec03v_geometry import file_hash
from .nec03w_f02_registry import registered_paired
from .nec03w_factorial import encode

SCHEMA="NEC-03W/F02/eight-original-source-paired-physical-mask/v1"
ROOTS=tuple(range(8))
ELIGIBLE_ROOTS=(0,1,2,4,6,7)
ITT_ZEROS=(3,5)
GROUPS=tuple(range(16,24))
CASES=256
BALANCED_COVERAGE_GATE=.95


def derive_one(root:int,body:dict):
    if type(root) is not int or root not in ROOTS:
        raise PermissionError("F02 only evaluates original eight source-trained worlds")
    if (body.get("schema")!="NEC-03W/F01/unscored-original-calibration-factorial-physical-roster/v1"
            or body.get("world_index")!=root
            or body.get("world_id")!=f"dev5-{root:03d}"
            or body.get("source_groups")!=list(GROUPS)
            or body.get("source_calibration_cases_per_arm")!=CASES
            or body.get("new_model_forwards")!=0
            or body.get("task_labels_or_logits_read")!=0
            or body.get("factorial_behavior_executed") is not False
            or body.get("source_query_groups32_to39_not_read") is not True
            or body.get("recruited_reserved_embargoed_confirmation_reads")!=0
            or body.get("scientific_gate")!="REDESIGN_OR_PILOT_ONLY"
            or set(body.get("arms",{}))!={"composed","additive"}):
        raise AssertionError("F02 source-only original physical case ownership changed")
    eligible={}
    for arm in ("composed","additive"):
        source=body["arms"][arm]
        rows=source["cases"]
        if (not isinstance(rows,list) or len(rows)!=CASES
                or source["denominator"]!=CASES
                or source["original_fixed_ITT_root"]!=(root in ITT_ZEROS)):
            raise AssertionError("F02 original arm denominator or fixed root status changed")
        valid=[]
        for i,row in enumerate(rows):
            if (row.get("group")!=GROUPS[i//32]
                    or row.get("mask")!=i%32
                    or type(row["geometry"].get("admissible")) is not bool):
                raise AssertionError("source F02 case/mask or physical admissibility invalid")
            valid.append(row["geometry"]["admissible"])
        if sum(valid)!=source["admissible_case_count"]:
            raise AssertionError("F02 original independent physical admitted count changed")
        eligible[arm]=valid
    common=[bool(c and a) for c,a in zip(eligible["composed"],eligible["additive"])]
    effective=len([x for x in common if x])
    pre={"composed":sum(eligible["composed"]),"additive":sum(eligible["additive"])}
    if root in ELIGIBLE_ROOTS and (
        pre["composed"]<64 or pre["additive"]<64):
        raise AssertionError("pre-registered minimal original source control feasibility changed")
    return {
        "world_index":root,"world_id":body["world_id"],
        "source_groups":list(GROUPS),
        "paired_case_admissible":common,
        "original_composed_admissible":eligible["composed"],
        "original_additive_admissible":eligible["additive"],
        "original_case_count_per_arm":CASES,
        "previously_admitted_by_arm":pre,
        "new_paired_case_admitted":effective,
        "paired_case_fraction":effective/CASES,
        "composed_cases_zeroed_by_pairing":pre["composed"]-effective,
        "additive_cases_zeroed_by_pairing":pre["additive"]-effective,
        "meets_new_paired_95_percent_gate":effective/CASES>=BALANCED_COVERAGE_GATE,
        "original_source_eligible_root":root in ELIGIBLE_ROOTS,
        "registered_original_ITT_fixed_zero_root":root in ITT_ZEROS,
        "input_source_physical_roster_sha256":body["physical_roster_sha256"],
        "mechanical_effect":"later source and sham lesions must be zero on BOTH teacher arms for all unpaired cases",
    }


def generate(files:dict[int,Path]):
    registered_paired()
    if set(files)!=set(ROOTS):
        raise ValueError("F02 requires all eight exact original source physics files")
    here=Path(__file__).parent
    manifest=json.loads((here/"NEC03W_F01_EIGHT_ROOT_PHYSICAL_RECEIPTS.json").read_text("utf-8"))
    if (manifest.get("schema")!="NEC-03W/F01/independent-original-eight-source-physical-summary/v1"
            or manifest.get("eight_roots_independently_passed")!=8
            or manifest.get("original_six_source_eligible_worlds_matched_both_arm_case_count")!=1176
            or manifest.get("task_labels_or_logits_accessed_for_physical_sweep")!=0
            or manifest.get("recruited_reserved_embargoed_confirmation_reads")!=0):
        raise ValueError("prior eight-root original physical archive has changed")
    roots={}
    for root in ROOTS:
        identity=manifest["roots"][str(root)]
        file=files[root]
        if (identity["world_id"]!=f"dev5-{root:03d}"
                or identity["physical_roster_file_sha256"]!=file_hash(file)
                or identity["original_512_case_arm_denominator"]!=512):
            raise ValueError("prior original root F01 physical case archive file SHA changed")
        data=json.loads(file.read_text(encoding="utf-8"))
        if (data.get("physical_roster_sha256")!=identity["physical_roster_sha256"]
                or data.get("source_pack_sha256")!=identity["original_source_pack_manifest_sha256"]):
            raise ValueError("root physical roster or original source geometry digest changed")
        row=derive_one(root,data)
        if row["previously_admitted_by_arm"]!=identity["admitted_original_256_cases_per_arm"]:
            raise ValueError("original independent physical source arm counts changed")
        row["physical_source_file_sha256"]=file_hash(file)
        roots[str(root)]=row
    denominator=len(ELIGIBLE_ROOTS)*CASES
    paired=sum(roots[str(root)]["new_paired_case_admitted"] for root in ELIGIBLE_ROOTS)
    total=sum(row["new_paired_case_admitted"] for row in roots.values())
    return {
        "schema":SCHEMA,
        "source_fit_calibration_groups":list(GROUPS),
        "original_source_eligible_roots":list(ELIGIBLE_ROOTS),
        "original_ITT_fixed_zero_roots":list(ITT_ZEROS),
        "eight_root_case_arm_denominator":8*2*CASES,
        "original_eligible_root_paired_case_denominator":denominator,
        "original_eligible_root_paired_case_count":paired,
        "original_eligible_root_paired_case_fraction":paired/denominator,
        "eligible_roots_meeting_new_95_percent_paired_gate":[
            root for root in ELIGIBLE_ROOTS if roots[str(root)]["meets_new_paired_95_percent_gate"]],
        "eligible_roots_failing_new_95_percent_paired_gate":[
            root for root in ELIGIBLE_ROOTS if not roots[str(root)]["meets_new_paired_95_percent_gate"]],
        "all_eight_roots_paired_case_count_including_ITT_zero_roots":total,
        "root_count":8,
        "roots":roots,
        "case_mask_intent":"original input mask index preserved; unpaired case must zero interventions in both teacher arms, full denominator 256",
        "new_paired_95_percent_gate_declared_after_root0_behavior":True,
        "original_root0_behavior_result_was_already_seen":True,
        "any_existing_behavioral_case_reinterpreted":False,
        "new_model_forwards":0,"task_labels_or_logits_read":0,
        "query_group32_to39_behavior_read":False,
        "recruited_reserved_embargoed_confirmation_reads":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
        "population_or_confirmation_claim":False,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-physical",type=Path,action="append",required=True,
                   help="Exact file 'world index=path' for each of eight roots")
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("F02 original source paired physical overlay must be a new output")
    files={}
    for value in args.root_physical:
        head,sep,tail=str(value).partition("=")
        if not sep or not head.isdecimal() or int(head) not in ROOTS or int(head) in files:
            raise ValueError("F02 requires eight unique root=file paths, roots0..7 only")
        files[int(head)]=Path(tail)
    result=generate(files)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(encode(result)+"\n",encoding="utf-8")
    print(encode({
        "original_eligible_root_paired_case_count":result["original_eligible_root_paired_case_count"],
        "original_eligible_root_paired_case_fraction":result["original_eligible_root_paired_case_fraction"],
        "eligible_roots_failing_new_95_percent_paired_gate":result["eligible_roots_failing_new_95_percent_paired_gate"],
        "new_model_forwards":0,"scientific_gate":"REDESIGN_OR_PILOT_ONLY"}))
    return result


if __name__=="__main__":
    main()
