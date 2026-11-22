"""NEC-03W: label-blind original-source calibration shortlist, zero model inference.

Selects ONE original source visit / quartet / 2+2 partition per frozen
original arm using only positive code IDs and independently audited source
coefficient/decoder geometry from original source FIT groups 16..23.

The original quartet MENU was pre-ranked on source groups 0..15. This
shortlist uses group16..23 co-support and physical matching only, NO labels,
logits, model forwards, optimization, prior source behavior24..31 or source
query32..39 outcomes. All source contexts were seen by the original SAE fit,
therefore this selection is DEVELOPMENTAL, not a prospective fresh-data test.

The fixed mechanical sample takes masks (0,8,16,24) in EACH of the eight
source calibration groups, 32 cases. The full 256-case source calibration
support is recorded, and any later behavioral scoring must keep the entire
original denominator and zero unsuccessful matched controls.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .nec03v_geometry import read_pack,file_hash
from .nec03w_factorial import build_roster,matched_pair_geometry,encode,digest
from .nec03w_independent import verify_candidate_roster,verify_pair_geometry
from .nec03w_candidate_registry import registered
from .nec03w_shortlist_registry import registered_shortlist

SCHEMA="NEC-03W/source-fit-calibration-label-blind-selection/v1"
ARM_NAMES=("composed","additive")
CALIBRATION_GROUPS=tuple(range(16,24))
SAMPLE_MASKS=(0,8,16,24)
PHYSICAL_SAMPLE=tuple(i*32+mask for i in range(8) for mask in SAMPLE_MASKS)
MIN_CALIBRATION_CO_SUPPORT=64
SHAM_PROPOSALS=96
MAX_CALIBRATION_CANDIDATES_PER_SITE=1
SPLITS_PER_CANDIDATE=3


def _source_sha(data):
    return sha256(encode(data).encode("utf-8")).hexdigest()


def load_calibration(pack:Path, codes:Path, audit:Path):
    pin=registered()
    manifest,shards,parent_sha=read_pack(pack)
    root=manifest["world_index"]
    if pin["parent_source_manifest_sha256_by_root"][str(root)]!=parent_sha:
        raise ValueError("original source quartet universe SHA differs from pinned parent archive")
    data=json.loads(codes.read_text(encoding="utf-8"))
    checked=json.loads(audit.read_text(encoding="utf-8"))
    if (data.get("schema")!="NEC-03W/original-source-functional-calibration-positive-codes/v1"
            or data.get("world_index")!=root
            or data.get("world_id")!=manifest["world_id"]
            or data.get("original_source_pack_manifest_sha256")!=parent_sha
            or data.get("original_calibration_group_indices")!=list(CALIBRATION_GROUPS)
            or data.get("source_only_query_groups_excluded")!=list(range(32,40))
            or data.get("source_query_groups_32_to_39_read") is not False
            or data.get("task_labels_read")!=0
            or data.get("model_logits_stored") is not False
            or data.get("model_or_SAE_optimizer_updates")!=0
            or data.get("behavioral_factorial_interventions_executed")!=0
            or data.get("recruited_or_confirmation_world_reads")!=0
            or checked.get("schema")!="NEC-03W/independently-replayed-source-calibration-topk16/v1"
            or checked.get("world_index")!=root
            or checked.get("original_source_calibration_codes_sha256")!=file_hash(codes)
            or checked.get("independent_code_audit_passed") is not True
            or checked.get("original_source_query_groups_32_to_39_read") is not False
            or checked.get("new_source_only_native_forward_calls")!=data["native_model_forward_calls"]
            or checked.get("task_labels_read")!=0
            or checked.get("recruited_embargoed_or_confirmation_reads")!=0
            or checked.get("factorial_behavior_or_rescue_scored") is not False
            or checked.get("scientific_gate")!="REDESIGN_OR_PILOT_ONLY"):
        raise ValueError("original source calibration code audit or protected cohort custody failed")
    if set(data.get("arms",{}))!=set(ARM_NAMES):
        raise ValueError("both original source model arms required")
    for arm in ARM_NAMES:
        original_visits=shards[arm]["visits"]
        case_visits=data["arms"][arm]["positive_codes_by_visit"]
        if (set(original_visits)!=set(case_visits)
                or data["arms"][arm]["selected_original_source_visits"]!=
                   sorted(int(v) for v in original_visits)):
            raise ValueError("original source SAE visits differ from original source-fit archive")
        for visit,rows in case_visits.items():
            if not isinstance(rows,list) or len(rows)!=256:
                raise ValueError("complete 256 source calibration cases per original visit required")
            for i,items in enumerate(rows):
                if (not isinstance(items,list) or len(items)>16
                        or any(not isinstance(p,list) or len(p)!=2
                               or type(p[0]) is not int or type(p[1]) not in (int,float)
                               or p[1]<=1e-7 for p in items)
                        or [p[0] for p in items]!=
                           original_visits[visit]["source_fit_active"][512+i]):
                    raise ValueError("positive code support or coefficient differs from original source fit")
    return manifest,shards,data,checked,build_roster(pack),parent_sha


def shortlist(pack:Path,codes:Path,code_audit:Path):
    registered_shortlist()
    orig,shards,source,audit,roster,packsha=load_calibration(pack,codes,code_audit)
    world=orig["world_index"]
    all_sites={}
    chosen={}
    for arm in ARM_NAMES:
        entries={}
        for visit,source_spec in sorted(roster["arms"][arm]["visits"].items(),key=lambda t:int(t[0])):
            batch=source["arms"][arm]["positive_codes_by_visit"][visit]
            decoder=shards[arm]["visits"][visit]["decoder32"]
            counts=[]
            for position,item in enumerate(source_spec["candidates"]):
                quartet=item["quartet"]
                n=sum(len(c)==16 and set(quartet).issubset(p[0] for p in c)
                      for c in batch)
                counts.append({"candidate_index":position,
                               "quartet":quartet,
                               "calibration_support_count":n})
            candidates=sorted(
                [x for x in counts if x["calibration_support_count"]>=MIN_CALIBRATION_CO_SUPPORT],
                key=lambda x:(-x["calibration_support_count"],x["candidate_index"]))
            tested=[]
            if candidates:
                item=candidates[0]
                paired=source_spec["candidates"][item["candidate_index"]]["pair_partitions"]
                for index,partition in enumerate(paired):
                    verified=0
                    failures={}
                    for calibration_index in PHYSICAL_SAMPLE:
                        group=16+calibration_index//32
                        mask=calibration_index%32
                        result=matched_pair_geometry(
                            batch[calibration_index],decoder,
                            tuple(item["quartet"]),partition,
                            root=world,arm=arm,group=group,mask=mask,visit=int(visit),
                            trials=SHAM_PROPOSALS)
                        if result["admissible"]:
                            checked=verify_pair_geometry(
                                batch[calibration_index],decoder,result)
                            if checked["physical_coordinates_recomputed"]!=192:
                                raise ValueError("original source 2x2 pair geometry independent audit failed")
                            verified+=1
                        else:
                            reason=result["reason"]
                            failures[reason]=failures.get(reason,0)+1
                    tested.append({
                        "candidate_index":item["candidate_index"],
                        "quartet":item["quartet"],
                        "partition_index":index,
                        "partition":partition,
                        "full_source_calibration_support_count":item["calibration_support_count"],
                        "matched_physical_sample_count":verified,
                        "matched_physical_sample_fraction":verified/len(PHYSICAL_SAMPLE),
                        "physical_sample_cases":len(PHYSICAL_SAMPLE),
                        "invalid_sample_reason_counts":failures,
                    })
            best=min(tested,key=lambda x:(
                -x["matched_physical_sample_count"],x["partition_index"])) if tested else None
            entries[visit]={
                "original_source_visit":int(visit),
                "source_candidate_budget":len(counts),
                "positive_support_counts_by_quartet":counts,
                "calibration_case_count":len(batch),
                "minimum_source_co_support":MIN_CALIBRATION_CO_SUPPORT,
                "physical_sample_absolute_indices":list(PHYSICAL_SAMPLE),
                "tested_partitions":tested,
                "best_label_blind_partition":best,
                "selected_using_task_labels":False,
            }
        all_sites[arm]=entries
        available=[entry["best_label_blind_partition"]|{"visit":int(visit)}
                   for visit,entry in entries.items()
                   if entry["best_label_blind_partition"] is not None]
        if available:
            chosen[arm]=min(available,key=lambda x:(
                -x["matched_physical_sample_count"],
                -x["full_source_calibration_support_count"],
                x["visit"],x["candidate_index"],x["partition_index"]))
        else:
            chosen[arm]=None
    output={
        "schema":SCHEMA,"world_index":world,"world_id":orig["world_id"],
        "parent_source_fit_manifest_sha256":packsha,
        "original_geometry_candidate_roster_sha256":roster["roster_sha256"],
        "original_source_calibration_codes_sha256":file_hash(codes),
        "original_source_calibration_code_audit_sha256":file_hash(code_audit),
        "source_calibration_group_ids":list(CALIBRATION_GROUPS),
        "original_discovery_group_ids":list(range(16)),
        "previous_nec03v_scored_groups_excluded":list(range(24,32)),
        "original_source_query_group_ids_unscored":list(range(32,40)),
        "calibration_cases_per_arm":256,
        "physical_sample_indices":list(PHYSICAL_SAMPLE),
        "source_candidate_minimum_support":MIN_CALIBRATION_CO_SUPPORT,
        "source_candidate_max_tested_per_site":MAX_CALIBRATION_CANDIDATES_PER_SITE,
        "physical_sham_trial_budget":SHAM_PROPOSALS,
        "source_sites":all_sites,
        "selected_physical_source_candidate_per_arm":chosen,
        "source_query_was_originally_seen_by_SAE_fit":True,
        "new_native_model_forwards":0,
        "new_task_labels_read":0,"model_logits_read":0,
        "new_factorial_or_rescue_behavior_scored":False,
        "optimizer_updates":0,
        "recruited_reserved_or_confirmation_reads":0,
        "selection_type":"label_blind_source_calibration_support_and_matched_physical_coverage_only",
        "functional_candidate_selection_still_unrun":True,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    output["selection_roster_sha256"]=_source_sha(output)
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-pack",type=Path,required=True)
    p.add_argument("--calibration-codes",type=Path,required=True)
    p.add_argument("--independent-calibration-audit",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    if args.out.exists():
        raise ValueError("source calibration selection must use fresh output")
    result=shortlist(args.original_pack,args.calibration_codes,args.independent_calibration_audit)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(encode(result)+"\n",encoding="utf-8")
    print(encode({"world_index":result["world_index"],
                  "selection_roster_sha256":result["selection_roster_sha256"],
                  "label_blind_selected_per_arm":result["selected_physical_source_candidate_per_arm"],
                  "new_native_model_forwards":0,
                  "new_task_labels_read":0,
                  "scientific_gate":"REDESIGN_OR_PILOT_ONLY"}))
    return result


if __name__=="__main__":
    main()
