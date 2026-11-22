"""NEC-03W/F01 independent original calibration physical roster audit.

Uses an independently implemented 32D physical coordinate summation and
original code provenance. Verifies the entire 256-case denominator for BOTH
original teacher arms, group/mask identity, selected quartet and pair split,
positive-code membership, source/sham disjointness, nonamplifying gains and
matched projected output. Does not import the F01 physical producer.

The selected sham identity is additionally checked against the registered
SHA proposal search; therefore this is independent PHYSICAL arithmetic,
but the selector's own source-label-free proposal procedure is shared.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from .nec03v_geometry import read_pack,file_hash
from .nec03w_factorial import matched_pair_geometry,encode
from .nec03w_independent import verify_pair_geometry
from .nec03w_shortlist_registry import registered_shortlist
from .nec03w_f01_registry import physical_method_lock

SCHEMA="NEC-03W/F01/independent-original-calibration-physical-roster-audit/v1"


def checksum(obj):
    return sha256(encode(obj).encode("utf-8")).hexdigest()


def verify(pack:Path,codes:Path,calibration_audit:Path,
           shortlist:Path,shortlist_audit:Path,physical:Path):
    physical_method_lock()
    registered_shortlist()
    manifest,shards,packsha=read_pack(pack)
    root=manifest["world_index"]
    calibration=json.loads(codes.read_text(encoding="utf-8"))
    prior_audit=json.loads(calibration_audit.read_text(encoding="utf-8"))
    chosen=json.loads(shortlist.read_text(encoding="utf-8"))
    chosen_audit=json.loads(shortlist_audit.read_text(encoding="utf-8"))
    result=json.loads(physical.read_text(encoding="utf-8"))
    source_digest=result.pop("physical_roster_sha256",None)
    if source_digest!=checksum(result):
        raise AssertionError("original source 256-case physical roster checksum changed")
    result["physical_roster_sha256"]=source_digest
    if (result.get("schema")!="NEC-03W/F01/unscored-original-calibration-factorial-physical-roster/v1"
            or result.get("world_id")!=manifest["world_id"]
            or result.get("world_index")!=root
            or result.get("source_groups")!=list(range(16,24))
            or result.get("source_calibration_cases_per_arm")!=256
            or result.get("source_pack_sha256")!=packsha
            or result.get("source_calibration_positive_codes_sha256")!=file_hash(codes)
            or result.get("independent_source_calibration_codes_audit_sha256")!=file_hash(calibration_audit)
            or result.get("label_blind_shortlist_sha256")!=file_hash(shortlist)
            or result.get("independent_shortlist_audit_sha256")!=file_hash(shortlist_audit)
            or result.get("selected_shortlist_roster_sha256")!=chosen["selection_roster_sha256"]
            or result.get("physical_sham_proposals_per_case")!=96
            or result.get("minimum_admissible_source_calibration_cases")!=64
            or result.get("original_fixed_ITT_zero_roots")!=[3,5]
            or result.get("original_source_group24_to31_behavior_not_read") is not True
            or result.get("source_query_groups32_to39_not_read") is not True
            or result.get("new_model_forwards")!=0 or result.get("task_labels_or_logits_read")!=0
            or result.get("factorial_behavior_executed") is not False
            or result.get("delayed_rescue_executed") is not False
            or result.get("new_optimizer_updates")!=0
            or result.get("recruited_reserved_embargoed_confirmation_reads")!=0
            or result.get("scientific_gate")!="REDESIGN_OR_PILOT_ONLY"
            or calibration.get("world_index")!=root
            or calibration.get("source_query_groups_32_to_39_read") is not False
            or calibration.get("task_labels_read")!=0
            or prior_audit.get("independent_code_audit_passed") is not True
            or prior_audit.get("original_source_calibration_codes_sha256")!=file_hash(codes)
            or chosen.get("new_task_labels_read")!=0
            or chosen.get("new_native_model_forwards")!=0
            or chosen_audit.get("original_source_site_selection_independently_recomputed") is not True
            or chosen_audit.get("source_shortlist_sha256")!=file_hash(shortlist)
            or set(result["arms"])!={"composed","additive"}):
        raise AssertionError("original source F01 physical roster lineage or custody changed")
    arithmetic_checks=0
    reason_histogram={}
    eligible={}
    for arm in ("composed","additive"):
        entries=result["arms"][arm]
        planned=chosen["selected_physical_source_candidate_per_arm"][arm]
        selected=entries["original_source_site"]
        if not isinstance(entries["cases"],list) or len(entries["cases"])!=256:
            raise AssertionError("original 256-case source-panel denominator incomplete")
        if planned is None:
            if selected is not None or root not in (3,5):
                raise AssertionError("only original ineligible source root may lack selected site")
            active=[]
            decoder=None
        else:
            if (not isinstance(selected,dict)
                    or selected.get("visit")!=planned["visit"]
                    or selected.get("quartet")!=planned["quartet"]
                    or selected.get("pair_A")!=planned["partition"][0]
                    or selected.get("pair_B")!=planned["partition"][1]
                    or selected.get("source_candidate_index")!=planned["candidate_index"]
                    or selected.get("source_partition_index")!=planned["partition_index"]
                    or selected.get("calibration_positive_support")!=planned["full_source_calibration_support_count"]):
                raise AssertionError("original source label-blind quartet/site/partition changed")
            visit=str(selected["visit"])
            decoder=shards[arm]["visits"][visit]["decoder32"]
            active=calibration["arms"][arm]["positive_codes_by_visit"][visit]
            if len(active)!=256:
                raise AssertionError("original source coefficient calibration count changed")
        valid=0
        for i,row in enumerate(entries["cases"]):
            grp,mask=16+i//32,i%32
            if (set(row)!={"group","mask","geometry"}
                    or row["group"]!=grp or row["mask"]!=mask):
                raise AssertionError("source F01 group/mask original chronological order changed")
            geo=row["geometry"]
            if selected is None:
                if (geo!={"schema":"NEC-03W/matched-factorial-physical-sham/v1",
                         "admissible":False,"reason":"original_source_ineligible_no_SAE_site"}):
                    raise AssertionError("original absent source site unexpectedly has intervention")
                reason_histogram["original_source_ineligible_no_SAE_site"]=(
                    reason_histogram.get("original_source_ineligible_no_SAE_site",0)+1)
                continue
            expected=matched_pair_geometry(
                active[i],decoder,tuple(selected["quartet"]),
                [selected["pair_A"],selected["pair_B"]],
                root=root,arm=arm,group=grp,mask=mask,
                visit=int(visit),trials=96)
            if geo!=expected:
                raise AssertionError("source original 2x2 physical sham or eligibility was tampered")
            if geo["admissible"]:
                independent=verify_pair_geometry(active[i],decoder,geo)
                if (independent["group_and_sham_pairs_disjoint"] is not True
                        or independent["physical_coordinates_recomputed"]!=192
                        or independent["new_model_forwards"]!=0):
                    raise AssertionError("independently reconstructed F01 source physical vectors diverged")
                arithmetic_checks+=192
                valid+=1
            else:
                reason=geo.get("reason","unspecified")
                reason_histogram[reason]=reason_histogram.get(reason,0)+1
        if (entries["denominator"]!=256 or
                entries["admissible_case_count"]!=valid or
                entries["invalid_paired_zero_case_count"]!=256-valid or
                entries["original_fixed_ITT_root"]!=(root in (3,5)) or
                (root not in (3,5) and valid<64)):
            raise AssertionError("original root physical F01 admissible-case denominator changed")
        eligible[arm]=valid
    return {
        "schema":SCHEMA,"world_index":root,"world_id":manifest["world_id"],
        "original_source_pack_manifest_sha256":packsha,
        "physical_roster_sha256":source_digest,
        "physical_roster_file_sha256":file_hash(physical),
        "physical_32D_coordinates_independently_recomputed":arithmetic_checks,
        "original_source_case_arm_records":512,
        "admissible_original_cases_by_arm":eligible,
        "paired_zero_reason_histogram":reason_histogram,
        "original_root_ITT_fixed_zero":root in (3,5),
        "independent_original_source_physical_audit_passed":True,
        "new_model_forwards":0,"task_labels_read":0,
        "source_query_group32_to39_behavior_opened":False,
        "recruited_embargoed_or_confirmation_reads":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-pack",type=Path,required=True)
    p.add_argument("--calibration-codes",type=Path,required=True)
    p.add_argument("--independent-calibration-audit",type=Path,required=True)
    p.add_argument("--label-blind-shortlist",type=Path,required=True)
    p.add_argument("--independent-shortlist-audit",type=Path,required=True)
    p.add_argument("--physical",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():
        raise FileExistsError("independent F01 source physical audit requires a fresh output")
    value=verify(a.original_pack,a.calibration_codes,a.independent_calibration_audit,
                 a.label_blind_shortlist,a.independent_shortlist_audit,a.physical)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(encode(value)+"\n",encoding="utf-8")
    print(encode(value))
    return value


if __name__=="__main__":
    main()
