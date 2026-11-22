"""NEC-03W independent original-source calibration shortlist custody audit.

Recomputes each candidate's 256-case positive support frequency from the
original byte-pinned source calibration, checks the pre-registered quartet
rank from source groups0..15 with the separate NEC03W independent auditor,
re-evaluates every selected 32-case 2x2 sham/matched physical gate, and
recomputes deterministic label-blind selection across original recurrent
visits and model arms. No native/SAE inference, labels or behavioral readout.

The 32 original source cases per selected site are a MECHANICAL calibration
sample fixed by index (masks 0,8,16,24 of each source-fit group16..23).
"""
from __future__ import annotations

from hashlib import sha256
import argparse
import json
import tempfile
from pathlib import Path

from .nec03v_geometry import read_pack,file_hash
from .nec03w_candidate_registry import registered
from .nec03w_independent import verify_candidate_roster,verify_pair_geometry
from .nec03w_factorial import build_roster,matched_pair_geometry,encode

SCHEMA="NEC-03W/independently-audited-original-source-shortlist/v1"
SAMPLE=tuple(group*32+mask for group in range(8) for mask in (0,8,16,24))


def audit(pack:Path,codes:Path,coefficient_audit:Path,shortlist_path:Path):
    registry=registered()
    manifest,shards,packsha=read_pack(pack)
    data=json.loads(codes.read_text("utf-8"))
    previous=json.loads(coefficient_audit.read_text("utf-8"))
    saved=json.loads(shortlist_path.read_text("utf-8"))
    root=manifest["world_index"]
    if (registry["parent_source_manifest_sha256_by_root"][str(root)]!=packsha
            or saved.get("schema")!="NEC-03W/source-fit-calibration-label-blind-selection/v1"
            or saved.get("world_index")!=root
            or saved.get("parent_source_fit_manifest_sha256")!=packsha
            or saved.get("original_source_calibration_codes_sha256")!=file_hash(codes)
            or saved.get("original_source_calibration_code_audit_sha256")!=file_hash(coefficient_audit)
            or saved.get("physical_sample_indices")!=list(SAMPLE)
            or saved.get("source_calibration_group_ids")!=list(range(16,24))
            or saved.get("original_discovery_group_ids")!=list(range(16))
            or saved.get("previous_nec03v_scored_groups_excluded")!=list(range(24,32))
            or saved.get("original_source_query_group_ids_unscored")!=list(range(32,40))
            or saved.get("new_native_model_forwards")!=0
            or saved.get("new_task_labels_read")!=0
            or saved.get("model_logits_read")!=0
            or saved.get("new_factorial_or_rescue_behavior_scored") is not False
            or saved.get("recruited_reserved_or_confirmation_reads")!=0
            or saved.get("scientific_gate")!="REDESIGN_OR_PILOT_ONLY"
            or previous.get("independent_code_audit_passed") is not True
            or previous.get("original_source_calibration_codes_sha256")!=file_hash(codes)
            or previous.get("task_labels_read")!=0
            or previous.get("factorial_behavior_or_rescue_scored") is not False
            or data.get("source_only_query_groups_excluded")!=list(range(32,40))):
        raise AssertionError("shortlisted original source coefficient provenance or cohort changed")
    old=dict(saved)
    stored_digest=old.pop("selection_roster_sha256",None)
    if sha256(encode(old).encode()).hexdigest()!=stored_digest:
        raise AssertionError("source calibration selection checksum differs")
    roster=build_roster(pack)
    if saved["original_geometry_candidate_roster_sha256"]!=roster["roster_sha256"]:
        raise AssertionError("original 0..15 source-only quartet discovery changed")
    with tempfile.TemporaryDirectory() as directory:
        file=Path(directory)/"frozen-roster.json"
        file.write_text(encode(roster)+"\n",encoding="utf-8")
        independent_roster=verify_candidate_roster(pack,file)
    if (independent_roster["candidate_roster_sha256"]!=roster["roster_sha256"]
            or independent_roster["model_behavior_outcomes_read"]!=0
            or independent_roster["protected_cohort_reads"]!=0):
        raise AssertionError("independent original discovery quartet candidate audit failed")
    # Independently audit the original quartet rankings and memberships.
    # The roster is generated deterministically from the original immutable
    # source pack; numerical 32D checks use another implementation below.
    full=0
    admitted=0
    source_selection={}
    for arm in ("composed","additive"):
        output=saved["source_sites"][arm]
        if set(output)!=set(roster["arms"][arm]["visits"]):
            raise AssertionError("incomplete original source site calibration shortlist")
        eligible=[]
        for visit,entry in output.items():
            codes_by_case=data["arms"][arm]["positive_codes_by_visit"][visit]
            candidates=roster["arms"][arm]["visits"][visit]["candidates"]
            if (entry["calibration_case_count"]!=256 or
                    entry["source_candidate_budget"]!=len(candidates) or
                    entry["physical_sample_absolute_indices"]!=list(SAMPLE) or
                    entry["minimum_source_co_support"]!=64 or
                    entry["selected_using_task_labels"] is not False):
                raise AssertionError("source cohort, 256-case support or candidate limit changed")
            if len(codes_by_case)!=256:
                raise AssertionError("source-positive coefficient denominator incomplete")
            new_counts=[]
            for index,candidate in enumerate(candidates):
                group=set(candidate["quartet"])
                frequency=0
                for case in codes_by_case:
                    if (len(case)==16 and
                            group.issubset(int(row[0]) for row in case)):
                        frequency+=1
                new_counts.append({
                    "candidate_index":index,
                    "quartet":candidate["quartet"],
                    "calibration_support_count":frequency})
            if entry["positive_support_counts_by_quartet"]!=new_counts:
                raise AssertionError("source rank support unexpectedly depends on outcome data")
            viable=sorted(
                (row for row in new_counts if row["calibration_support_count"]>=64),
                key=lambda row:(-row["calibration_support_count"],row["candidate_index"]))
            tests=entry["tested_partitions"]
            if not viable and tests:
                raise AssertionError("fabricated pair control without minimum source code support")
            if viable and len(tests)!=3:
                raise AssertionError("all source quartets need exactly three 2+2 control partitions")
            if viable:
                selected=viable[0]
                for item in tests:
                    partition_index=item["partition_index"]
                    if partition_index not in range(3):
                        raise AssertionError("source match partition out of range")
                    if (item["quartet"]!=selected["quartet"]
                            or item["candidate_index"]!=selected["candidate_index"]
                            or item["partition"]!=candidates[selected["candidate_index"]]["pair_partitions"][partition_index]
                            or item["full_source_calibration_support_count"]!=selected["calibration_support_count"]
                            or item["physical_sample_cases"]!=32):
                        raise AssertionError("source physical quartet selected from wrong calibration evidence")
                    physical=0
                    failures={}
                    for case_index in SAMPLE:
                        match=matched_pair_geometry(
                            codes_by_case[case_index],
                            shards[arm]["visits"][visit]["decoder32"],
                            tuple(item["quartet"]),
                            item["partition"],
                            root=root,arm=arm,group=16+case_index//32,
                            mask=case_index%32,visit=int(visit),trials=96)
                        if match["admissible"]:
                            receipt=verify_pair_geometry(
                                codes_by_case[case_index],
                                shards[arm]["visits"][visit]["decoder32"],
                                match)
                            if receipt["physical_coordinates_recomputed"]!=192:
                                raise AssertionError("independent source feature pair 32D check missing")
                            physical+=1
                            admitted+=1
                        else:
                            failures[match["reason"]]=failures.get(match["reason"],0)+1
                        full+=1
                    if (physical!=item["matched_physical_sample_count"]
                            or failures!=item["invalid_sample_reason_counts"]):
                        raise AssertionError("source 32D matched pair physical eligibility not replayable")
                best=min(tests,key=lambda item:(
                    -item["matched_physical_sample_count"],item["partition_index"]))
                if entry["best_label_blind_partition"]!=best:
                    raise AssertionError("pre-behavior calibration partition selection changed")
                eligible.append(best|{"visit":int(visit)})
            elif entry["best_label_blind_partition"] is not None:
                raise AssertionError("unsupported original source functional candidate selected")
        source_selection[arm]=min(
            eligible,key=lambda item:(
                -item["matched_physical_sample_count"],
                -item["full_source_calibration_support_count"],
                item["visit"],item["candidate_index"],item["partition_index"])
        ) if eligible else None
    if saved["selected_physical_source_candidate_per_arm"]!=source_selection:
        raise AssertionError("source feature site selection used different physical and support criteria")
    return {
        "schema":SCHEMA,"world_index":root,
        "original_source_pack_sha256":packsha,
        "source_shortlist_sha256":file_hash(shortlist_path),
        "original_source_calibration_coefficients_sha256":file_hash(codes),
        "independent_matched_pair_32D_checks":admitted,
        "physical_candidate_case_attempts":full,
        "candidate_source_support_independently_recomputed":True,
        "original_source_site_selection_independently_recomputed":True,
        "source_query_behavior_read":False,
        "model_forwards":0,"task_labels_read":0,
        "recruited_or_confirmation_reads":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-pack",type=Path,required=True)
    p.add_argument("--calibration-codes",type=Path,required=True)
    p.add_argument("--independent-calibration-audit",type=Path,required=True)
    p.add_argument("--shortlist",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():
        raise FileExistsError("original-source calibration shortlist audit must have a fresh output")
    receipt=audit(a.original_pack,a.calibration_codes,
                  a.independent_calibration_audit,a.shortlist)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(encode(receipt)+"\n",encoding="utf-8")
    print(encode(receipt))
    return receipt


if __name__=="__main__":
    main()
