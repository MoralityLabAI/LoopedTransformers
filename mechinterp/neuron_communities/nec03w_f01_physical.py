"""NEC-03W/F01: frozen original-source CALIBRATION factorial physical roster.

ZERO MODEL FORWARDS. Derive a full 256-case x two-arm source-only original
calibration16..23 intervention roster from already independently audited
frozen quartet shortlist and TopK16 code/decoder artifacts. NEVER read labels,
logits, query32..39 coefficients, recruits, reserved or confirmations.

Each case has one of: an admitted 4-code 2x2 group+sham physical geometry,
a registered original code-support/matching invalid no-op, or the original
source-ineligible arm no-source-site no-op. The count stays 256 per teacher.
This roster must be frozen by SHA in a separate pre-behavior commit.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .nec03v_geometry import read_pack,file_hash
from .nec03w_factorial import matched_pair_geometry,encode
from .nec03w_calibration_shortlist import load_calibration
from .nec03w_shortlist_registry import registered_shortlist
from .nec03w_independent import verify_pair_geometry
from .nec03w_f01_registry import physical_method_lock

SCHEMA="NEC-03W/F01/unscored-original-calibration-factorial-physical-roster/v1"
ARMS=("composed","additive")
GROUPS=tuple(range(16,24))
CASES=256
PHYSICAL_TRIALS=96
MIN_CASES=64


def checksum(obj):
    return sha256(encode(obj).encode("utf-8")).hexdigest()


def validate_original_shortlist(pack:Path,codes:Path,code_audit:Path,
                                shortlist:Path,independent_shortlist:Path):
    registered_shortlist()
    physical_method_lock()
    manifest,shards,source,_,_,packsha=load_calibration(pack,codes,code_audit)
    pick=json.loads(shortlist.read_text(encoding="utf-8"))
    reviewer=json.loads(independent_shortlist.read_text(encoding="utf-8"))
    root=manifest["world_index"]
    import copy
    saved_hash=pick.get("selection_roster_sha256")
    copy_of_pick=copy.deepcopy(pick)
    copy_of_pick.pop("selection_roster_sha256",None)
    if (pick.get("schema")!="NEC-03W/source-fit-calibration-label-blind-selection/v1"
            or saved_hash!=checksum(copy_of_pick)
            or pick.get("world_index")!=root
            or pick.get("world_id")!=manifest["world_id"]
            or pick.get("parent_source_fit_manifest_sha256")!=packsha
            or pick.get("original_source_calibration_codes_sha256")!=file_hash(codes)
            or pick.get("original_source_calibration_code_audit_sha256")!=file_hash(code_audit)
            or pick.get("original_source_query_group_ids_unscored")!=list(range(32,40))
            or pick.get("new_native_model_forwards")!=0
            or pick.get("new_task_labels_read")!=0
            or pick.get("new_factorial_or_rescue_behavior_scored") is not False
            or reviewer.get("schema")!="NEC-03W/independently-audited-original-source-shortlist/v1"
            or reviewer.get("world_index")!=root
            or reviewer.get("source_shortlist_sha256")!=file_hash(shortlist)
            or reviewer.get("original_source_pack_sha256")!=packsha
            or reviewer.get("original_source_calibration_coefficients_sha256")!=file_hash(codes)
            or reviewer.get("candidate_source_support_independently_recomputed") is not True
            or reviewer.get("original_source_site_selection_independently_recomputed") is not True
            or reviewer.get("model_forwards")!=0
            or reviewer.get("task_labels_read")!=0
            or reviewer.get("source_query_behavior_read") is not False
            or reviewer.get("recruited_or_confirmation_reads")!=0):
        raise ValueError("source-only F01 shortlist/independent audit provenance changed")
    return manifest,shards,source,pick,reviewer


def build(pack:Path,codes:Path,code_audit:Path,
          shortlist:Path,independent_shortlist:Path):
    origin,shards,positive,pick,_=validate_original_shortlist(
        pack,codes,code_audit,shortlist,independent_shortlist)
    root=origin["world_index"]
    arms={}
    for arm in ARMS:
        saved=pick["selected_physical_source_candidate_per_arm"][arm]
        cases=[]
        if saved is None:
            if (root not in (3,5) or
                    shards[arm]["visits"] or positive["arms"][arm]["positive_codes_by_visit"]):
                raise ValueError("only original source-ineligible arm may omit a selected candidate")
            for j in range(CASES):
                cases.append({"group":GROUPS[j//32],"mask":j%32,
                              "geometry":{"schema":"NEC-03W/matched-factorial-physical-sham/v1",
                                          "admissible":False,
                                          "reason":"original_source_ineligible_no_SAE_site"}})
            selected=None
        else:
            site=str(saved["visit"])
            cohort=positive["arms"][arm]["positive_codes_by_visit"][site]
            decoder=shards[arm]["visits"][site]["decoder32"]
            if (saved["candidate_index"] not in range(12)
                    or saved["partition_index"] not in range(3)
                    or saved["full_source_calibration_support_count"]<MIN_CASES
                    or not isinstance(cohort,list) or len(cohort)!=CASES
                    or saved["quartet"]!=sorted(saved["quartet"])
                    or saved["partition"] is None):
                raise ValueError("original source F01 selected candidate/partition preflight failed")
            for j in range(CASES):
                group,mask=GROUPS[j//32],j%32
                geo=matched_pair_geometry(
                    cohort[j],decoder,tuple(saved["quartet"]),saved["partition"],
                    root=root,arm=arm,group=group,mask=mask,visit=int(site),
                    trials=PHYSICAL_TRIALS)
                if geo["admissible"]:
                    independent=verify_pair_geometry(cohort[j],decoder,geo)
                    if independent["physical_coordinates_recomputed"]!=192:
                        raise AssertionError("independent original source F01 32D physical components incomplete")
                cases.append({"group":group,"mask":mask,"geometry":geo})
            selected={
                "visit":int(site),"quartet":saved["quartet"],
                "pair_A":saved["partition"][0],
                "pair_B":saved["partition"][1],
                "source_candidate_index":saved["candidate_index"],
                "source_partition_index":saved["partition_index"],
                "calibration_positive_support":saved["full_source_calibration_support_count"],
                "prebehavior_physical_sample_cases":saved["physical_sample_cases"],
                "prebehavior_physical_sample_admitted":saved["matched_physical_sample_count"],
            }
        valid=sum(x["geometry"]["admissible"] for x in cases)
        if valid<MIN_CASES and root not in (3,5):
            raise ValueError("registered 64-case per-arm functional calibration physical coverage gate failed")
        arms[arm]={
            "original_source_site":selected,
            "cases":cases,
            "admissible_case_count":valid,
            "invalid_paired_zero_case_count":CASES-valid,
            "denominator":CASES,
            "original_fixed_ITT_root":root in (3,5),
        }
    result={
        "schema":SCHEMA,
        "world_id":origin["world_id"],"world_index":root,
        "source_groups":list(GROUPS),
        "source_calibration_cases_per_arm":CASES,
        "source_pack_sha256":file_hash(pack/"manifest.json"),
        "source_calibration_positive_codes_sha256":file_hash(codes),
        "independent_source_calibration_codes_audit_sha256":file_hash(code_audit),
        "label_blind_shortlist_sha256":file_hash(shortlist),
        "independent_shortlist_audit_sha256":file_hash(independent_shortlist),
        "selected_shortlist_roster_sha256":pick["selection_roster_sha256"],
        "original_fixed_ITT_zero_roots":[3,5],
        "physical_sham_proposals_per_case":PHYSICAL_TRIALS,
        "minimum_admissible_source_calibration_cases":MIN_CASES,
        "arms":arms,
        "original_source_group24_to31_behavior_not_read":True,
        "source_query_groups32_to39_not_read":True,
        "new_model_forwards":0,"task_labels_or_logits_read":0,
        "factorial_behavior_executed":False,"delayed_rescue_executed":False,
        "new_optimizer_updates":0,
        "recruited_reserved_embargoed_confirmation_reads":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    result["physical_roster_sha256"]=checksum(result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-pack",type=Path,required=True)
    p.add_argument("--calibration-codes",type=Path,required=True)
    p.add_argument("--independent-calibration-audit",type=Path,required=True)
    p.add_argument("--label-blind-shortlist",type=Path,required=True)
    p.add_argument("--independent-shortlist-audit",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():
        raise FileExistsError("NEC03W original source physical roster output must be fresh")
    out=build(a.original_pack,a.calibration_codes,
              a.independent_calibration_audit,a.label_blind_shortlist,
              a.independent_shortlist_audit)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(encode(out)+"\n",encoding="utf-8")
    print(encode({"schema":out["schema"],"world_index":out["world_index"],
                  "physical_roster_sha256":out["physical_roster_sha256"],
                  "admissible_case_count_by_arm":{
                      arm:x["admissible_case_count"] for arm,x in out["arms"].items()},
                  "new_model_forwards":0,"scientific_gate":"REDESIGN_OR_PILOT_ONLY"}))
    return out


if __name__=="__main__":
    main()
