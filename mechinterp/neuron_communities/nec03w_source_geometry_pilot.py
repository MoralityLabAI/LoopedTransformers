"""NEC-03W original source-only physical candidate scan, zero neural inference.

Runs source-fit candidate discovery and a capped ORIGINAL NEC03V-EXPOSED
source group24 mechanical pairing scan. This is label-blind engineering and
cannot be used for selecting a later functional candidate by task outcome.

Original source groups16..23 and32..39 remain excluded from intervention
scoring. Their positive code *ID support counts* can be disclosed as unlabeled
feasibility only, after the quartet menu has been frozen from groups0..15.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .nec03v_geometry import read_pack,file_hash
from .nec03w_factorial import (
    build_roster, matched_pair_geometry, encode, digest,
)
from .nec03w_independent import (
    verify_candidate_roster,verify_pair_geometry,
)

SCHEMA="NEC-03W/original-source-only-mechanical-pair-feasibility/v1"
MAX_SOURCE_MECHANICAL_CASES=32
SITE_SHAM_PROPOSALS=96


def pilot(pack:Path,*,case_count=MAX_SOURCE_MECHANICAL_CASES,
          trials=SITE_SHAM_PROPOSALS):
    if type(case_count) is not int or not 1<=case_count<=MAX_SOURCE_MECHANICAL_CASES:
        raise ValueError("only original previously scored source group24 first32 cases")
    if trials!=SITE_SHAM_PROPOSALS:
        raise ValueError("frozen source-only physical search proposals changed")
    manifest,shards,pack_sha=read_pack(pack)
    roster=build_roster(pack)
    root=manifest["world_index"]
    if roster["source_pack_manifest_sha256"]!=pack_sha:
        raise ValueError("candidate source pack was changed after selection")
    records=[]
    site_stats={}
    for arm in ("composed","additive"):
        site_stats[arm]={}
        shard=shards[arm]
        for visit,entry in roster["arms"][arm]["visits"].items():
            source=shard["cases"][:case_count]
            if not entry["candidates"]:
                site_stats[arm][visit]={"candidate_menu_empty":True,
                                         "physical_pair_control_passes":0}
                continue
            first=entry["candidates"][0]
            candidate=first["quartet"]
            pairs=first["pair_partitions"][0]
            admitted=0
            errors={}
            for row in source:
                if row["group"]!=24:
                    raise ValueError("source-only mechanical pilot is original group24 only")
                result=matched_pair_geometry(
                    row["visits"][visit],
                    shard["visits"][visit]["decoder32"],
                    tuple(candidate),pairs,
                    root=root,arm=arm,group=row["group"],mask=row["mask"],
                    visit=int(visit),trials=trials,
                )
                if result["admissible"]:
                    independent=verify_pair_geometry(
                        row["visits"][visit],
                        shard["visits"][visit]["decoder32"],
                        result,
                    )
                    if independent["physical_coordinates_recomputed"]!=192:
                        raise AssertionError("source-only independent physical 2x2 audit incomplete")
                    admitted+=1
                else:
                    reason=result.get("reason","unknown")
                    errors[reason]=errors.get(reason,0)+1
                records.append({
                    "root":root,"arm":arm,"visit":int(visit),
                    "group":row["group"],"mask":row["mask"],
                    "candidate_group":candidate,
                    "pairing":pairs,
                    "physical_receipt":result,
                })
            site_stats[arm][visit]={
                "source_discovery_ID_count":first["source_discovery_count"],
                "original_source_calibration_ID_count_unlabeled":first[
                    "source_calibration_ID_support_only"],
                "original_source_query_ID_count_unlabeled":first[
                    "source_query_ID_support_only"],
                "candidate_count":entry["candidate_count"],
                "mechanical_original_group24_cases":case_count,
                "physical_pair_control_passes":admitted,
                "physical_pair_control_coverage":admitted/case_count,
                "inadmissible_reasons":errors,
                "selected_source_candidate_index":0,
                "selected_source_pair_partition_index":0,
                "selection_requires_future_functional_calibration":True,
            }
    return {
        "schema":SCHEMA,
        "world_index":root,
        "original_world_id":manifest["world_id"],
        "source_pack_manifest_sha256":pack_sha,
        "original_source_evidence":manifest["evidence"],
        "source_candidate_roster_sha256":roster["roster_sha256"],
        "roster":roster,
        "source_mechanical_group":24,
        "source_mechanical_max_cases":case_count,
        "source_mechanical_sham_trial_budget":trials,
        "site_results":site_stats,
        "source_mechanical_receipts":records,
        "source_fit_candidates_selected_by_behavior":False,
        "future_functional_calibration_started":False,
        "source_query_behavior_opened":False,
        "model_forwards":0,"task_labels_read":0,
        "behavior_logits_read":0,"optimizer_updates":0,
        "recruited_or_confirmation_reads":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--case-count",type=int,default=MAX_SOURCE_MECHANICAL_CASES)
    a=parser.parse_args(argv)
    if a.out.exists():
        raise FileExistsError("NEC03W source mechanical output must be fresh")
    output=pilot(a.pack,case_count=a.case_count)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(encode(output)+"\n",encoding="utf-8")
    summary={
        "world_index":output["world_index"],
        "original_world_id":output["original_world_id"],
        "candidate_roster_sha256":output["source_candidate_roster_sha256"],
        "source_pack_sha256":output["source_pack_manifest_sha256"],
        "site_physical_feasibility":{
            arm:{v:{"eligible":x["physical_pair_control_passes"],
                    "count":x.get("mechanical_original_group24_cases",0)}
                 for v,x in rows.items()}
            for arm,rows in output["site_results"].items()},
        "model_forwards":0,"task_labels_read":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    print(encode(summary))
    return output


if __name__=="__main__":
    main()
