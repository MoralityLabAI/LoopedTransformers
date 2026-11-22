"""Independent NEC-03W/F02 original-source paired-case mask and custody audit.

Recomputes original two-arm feature pairing eligibility and full-denominator
256-case original source masks directly from the SHA-locked F01 independent
physical artifacts, independent of the F02 mask-generation implementation.

This is post-root0, purely source physical engineering. No teacher forward,
source label, logit, optimizer step, source-query or protected-world read.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .nec03v_geometry import file_hash
from .nec03w_factorial import encode
from .nec03w_f02_registry import registered_paired

SCHEMA="NEC-03W/F02/independent-eight-original-source-paired-physical-mask-audit/v1"


def verify(files:dict[int,Path],output:Path):
    registered_paired()
    snapshot=json.loads(output.read_text(encoding="utf-8"))
    source_manifest=json.loads(
        (Path(__file__).with_name("NEC03W_F01_EIGHT_ROOT_PHYSICAL_RECEIPTS.json")).read_text("utf-8"))
    if (snapshot.get("schema")!="NEC-03W/F02/eight-original-source-paired-physical-mask/v1"
            or snapshot.get("root_count")!=8 or set(files)!=set(range(8))
            or set(snapshot.get("roots",{}))!=set(str(i) for i in range(8))
            or snapshot.get("source_fit_calibration_groups")!=list(range(16,24))
            or snapshot.get("original_source_eligible_roots")!=[0,1,2,4,6,7]
            or snapshot.get("original_ITT_fixed_zero_roots")!=[3,5]
            or snapshot.get("eight_root_case_arm_denominator")!=4096
            or snapshot.get("original_eligible_root_paired_case_denominator")!=1536
            or snapshot.get("new_model_forwards")!=0
            or snapshot.get("task_labels_or_logits_read")!=0
            or snapshot.get("query_group32_to39_behavior_read") is not False
            or snapshot.get("recruited_reserved_embargoed_confirmation_reads")!=0
            or snapshot.get("new_paired_95_percent_gate_declared_after_root0_behavior") is not True
            or snapshot.get("original_root0_behavior_result_was_already_seen") is not True
            or snapshot.get("any_existing_behavioral_case_reinterpreted") is not False
            or snapshot.get("scientific_gate")!="REDESIGN_OR_PILOT_ONLY"
            or snapshot.get("population_or_confirmation_claim") is not False):
        raise AssertionError("F02 paired original case mask or posthoc scientific interpretation changed")
    total_eligible=0
    total_all=0
    paired_per_eligible={}
    passing=[]
    failing=[]
    checked=0
    for root in range(8):
        source=files[root]
        sha=file_hash(source)
        pinned=source_manifest["roots"][str(root)]
        if sha!=pinned["physical_roster_file_sha256"]:
            raise AssertionError("original source physical roster artifact is not the frozen SHA")
        model=json.loads(source.read_text(encoding="utf-8"))
        saved=snapshot["roots"][str(root)]
        if (model.get("physical_roster_sha256")!=pinned["physical_roster_sha256"]
                or model.get("source_pack_sha256")!=pinned["original_source_pack_manifest_sha256"]
                or model.get("world_index")!=root
                or model.get("source_groups")!=list(range(16,24))
                or model.get("new_model_forwards")!=0
                or model.get("task_labels_or_logits_read")!=0
                or set(model.get("arms",{}))!={"composed","additive"}
                or saved.get("world_index")!=root
                or saved.get("world_id")!=f"dev5-{root:03d}"
                or saved.get("physical_source_file_sha256")!=sha
                or saved.get("input_source_physical_roster_sha256")!=pinned["physical_roster_sha256"]
                or saved.get("registered_original_ITT_fixed_zero_root")!=(root in (3,5))
                or saved.get("original_source_eligible_root")!=(root in (0,1,2,4,6,7))):
            raise AssertionError("original F02 paired source world and physical SHA identity mismatch")
        arms={}
        for arm in ("composed","additive"):
            cases=model["arms"][arm]["cases"]
            if (len(cases)!=256 or
                    model["arms"][arm]["original_fixed_ITT_root"]!=(root in (3,5))):
                raise AssertionError("original 256-case source mask or ITT status changed")
            flags=[]
            for i,case in enumerate(cases):
                if (case.get("group")!=16+i//32 or
                        case.get("mask")!=i%32 or
                        type(case["geometry"].get("admissible")) is not bool):
                    raise AssertionError("original source 32-mask chronology changed")
                flags.append(case["geometry"]["admissible"])
            if (flags!=saved["original_"+arm+"_admissible"]
                    or sum(flags)!=pinned["admitted_original_256_cases_per_arm"][arm]):
                raise AssertionError("source paired F02 original arm eligibility differs from physical archive")
            arms[arm]=flags
        paired=[c and a for c,a in zip(arms["composed"],arms["additive"])]
        count=sum(paired)
        if (paired!=saved["paired_case_admissible"]
                or saved["new_paired_case_admitted"]!=count
                or saved["original_case_count_per_arm"]!=256
                or saved["previously_admitted_by_arm"]!={
                    arm:sum(flags) for arm,flags in arms.items()}
                or saved["composed_cases_zeroed_by_pairing"]!=sum(arms["composed"])-count
                or saved["additive_cases_zeroed_by_pairing"]!=sum(arms["additive"])-count
                or saved["paired_case_fraction"]!=count/256
                or saved["meets_new_paired_95_percent_gate"]!=(count/256>=.95)):
            raise AssertionError("independent 256-case original two-arm intersection disagrees")
        total_all+=count
        checked+=256*2
        if root in (0,1,2,4,6,7):
            total_eligible+=count
            paired_per_eligible[str(root)]=count
            (passing if count/256>=.95 else failing).append(root)
    if (snapshot["all_eight_roots_paired_case_count_including_ITT_zero_roots"]!=total_all
            or snapshot["original_eligible_root_paired_case_count"]!=total_eligible
            or snapshot["original_eligible_root_paired_case_fraction"]!=total_eligible/1536
            or snapshot["eligible_roots_meeting_new_95_percent_paired_gate"]!=passing
            or snapshot["eligible_roots_failing_new_95_percent_paired_gate"]!=failing):
        raise AssertionError("independent paired-root coverage denominator or new gate changed")
    return {
        "schema":SCHEMA,
        "original_roots_independently_rechecked":8,
        "original_physical_case_arm_records_rechecked":checked,
        "source_eligible_paired_case_count":total_eligible,
        "source_eligible_paired_case_denominator":1536,
        "source_eligible_paired_case_fraction":total_eligible/1536,
        "original_source_eligible_root_paired_counts":paired_per_eligible,
        "eligible_roots_failing_new_95_percent_paired_gate":failing,
        "eligible_roots_passing_new_95_percent_paired_gate":passing,
        "original_root0_behavior_previously_seen":True,
        "original_root0_behavior_reinterpreted":False,
        "independent_original_source_paired_mask_audit_passed":True,
        "source_f02_paired_roster_file_sha256":file_hash(output),
        "new_model_forwards":0,"task_labels_read":0,
        "source_query32_to39_read":False,
        "recruited_reserved_embargoed_confirmation_reads":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-physical",type=Path,action="append",required=True)
    p.add_argument("--paired",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("independent original F02 source paired audit output must be fresh")
    original={}
    for item in args.root_physical:
        root,suffix,value=str(item).partition("=")
        if not suffix or not root.isdecimal() or int(root) not in range(8) or int(root) in original:
            raise ValueError("F02 independent audit needs unique 0..7 original physical roots")
        original[int(root)]=Path(value)
    result=verify(original,args.paired)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(encode(result)+"\n",encoding="utf-8")
    print(encode(result))
    return result


if __name__=="__main__":
    main()
