"""NEC-03V-B original eight-root source-only developmental aggregation.

Only accepts eight complete, independently replayed original-source case-arm
results, each bound to its original frozen source geometry cohort. Root is the
statistical unit; roots 3 and 5 retain original fixed ITT zero effects.
No new model inference, source selection, optimizer updates, or p-values.
"""
from __future__ import annotations

import argparse
from hashlib import sha1
import json
import math
from pathlib import Path

from .nec03v_source_attest import canonical,sha256_file

SCHEMA="NEC-03V/eight-original-source-root-descriptive-aggregate/v1"
ROOTS=tuple(range(8))
ITT_ZERO=(3,5)
ARMS=("composed","additive")
SCORE_MODES=("full_affine","group_025","sham_025","group_05","sham_05","group_1","sham_1")


def frozen():
    base=Path(__file__).parent
    lock=json.loads((base/"NEC03V_BEHAVIOR_AGGREGATE_LOCK.json").read_text("utf-8"))
    panel=json.loads((base/"NEC03V_BEHAVIOR_SOURCE_PANEL_LOCK.json").read_text("utf-8"))
    if (lock.get("schema")!="NEC-03V/frozen-eight-source-root-statistical-aggregator/v1"
            or lock.get("roots")!=list(ROOTS)
            or lock.get("original_ITT_fixed_zero_roots")!=list(ITT_ZERO)
            or lock.get("mean_root_denominator")!=8
            or lock.get("meaningful_effect")!=.02
            or lock.get("population_inference") is not False
            or lock.get("scientific_gate")!="REDESIGN_OR_PILOT_ONLY"
            or panel.get("fixed_ITT_zero_roots")!=list(ITT_ZERO)):
        raise ValueError("source-only root statistic frozen register changed")
    for relative,hash_ in lock["git_blob_sha1"].items():
        bytes_=(base.parents[1]/relative).read_bytes()
        if sha1(b"blob "+str(len(bytes_)).encode()+b"\0"+bytes_).hexdigest()!=hash_:
            raise ValueError("source eight-root statistic code changed")
    return panel


def aggregate(root_folders:list[Path]):
    panel=frozen()
    if len(root_folders)!=8:
        raise ValueError("complete original eight-root source data required")
    effects={}
    counts={}
    source_reports={}
    for folder in root_folders:
        p=folder/"source_behavior_report.json"
        q=folder/"source_behavior_independent.json"
        raw=folder/"source_behavior_raw.jsonl"
        report=json.loads(p.read_text("utf-8"))
        audit=json.loads(q.read_text("utf-8"))
        root=report["world_index"]
        if type(root) is not int or root not in ROOTS or root in effects:
            raise ValueError("duplicate, recruited or incomplete source world")
        by_arm=report["correctness_totals"]
        if (report["schema"]!="NEC-03V/source-only-group-removal-behavior/v1"
                or report["world_id"]!=f"dev5-{root:03d}"
                or report["source_cases_per_arm"]!=256
                or report["original_source_groups"]!=list(range(24,32))
                or report["source_pack_manifest_sha256"]!=
                   panel["roots"][str(root)]["source_pack_manifest_sha256"]
                or report["frozen_source_group_roster_sha256"]!=
                   panel["roots"][str(root)]["frozen_group_selection_sha256"]
                or report["raw_behavior_sha256"]!=sha256_file(raw)
                or report["source_only_native_forward_calls"]!=112
                or report["recruited_or_embargoed_world_queries"]!=0
                or report["development_evaluation_or_reserved_queries"]!=0
                or report["old_confirmation_queries"]!=0
                or report["new_optimizer_updates"]!=0
                or report["scientific_gate"]!="REDESIGN_OR_PILOT_ONLY"
                or report["confirmatory_population_claim"] is not False
                or audit.get("schema")!="NEC-03V/independent-original-source-group-removal-behavior/v1"
                or audit.get("world_index")!=root
                or audit.get("independent_behavior_replay_passed") is not True
                or audit.get("source_behavior_report_sha256")!=sha256_file(p)
                or audit.get("source_behavior_raw_sha256")!=sha256_file(raw)
                or audit.get("original_source_case_arm_records")!=512
                or audit.get("independent_replayed_native_forwards")!=112
                or audit.get("independent_correctness_totals")!=by_arm
                or audit.get("population_claim_permitted") is not False
                or audit.get("scientific_gate")!="REDESIGN_OR_PILOT_ONLY"):
            raise ValueError("source-only root independent audit or original panel SHA missing")
        if set(by_arm)!=set(ARMS) or any(
            set(by_arm[arm])!=set(SCORE_MODES) or
            any(type(n) is not int or not 0<=n<=256 for n in by_arm[arm].values())
            for arm in ARMS):
            raise ValueError("incorrect fixed 256-case source denominator")
        numerator=((by_arm["composed"]["sham_1"]-by_arm["composed"]["group_1"])
                   -(by_arm["additive"]["sham_1"]-by_arm["additive"]["group_1"]))
        raw_effect=numerator/256
        effect=0. if root in ITT_ZERO else raw_effect
        if (not math.isclose(raw_effect,report["raw_root_composed_minus_additive_correctness_DID"],abs_tol=1e-12)
                or not math.isclose(effect,report["registered_ITT_root_effect"],abs_tol=1e-12)
                or not math.isclose(effect,audit["original_primary_ITT_effect"],abs_tol=1e-12)
                or report["original_fixed_ITT_zero"]!=(root in ITT_ZERO)):
            raise ValueError("independent source root effect and original ITT zero disagree")
        effects[root]=effect
        counts[root]=by_arm
        source_reports[root]={
            "root":root,
            "effect":effect,"raw_effect":raw_effect,"numerator":numerator,
            "original_fixed_ITT_zero":root in ITT_ZERO,
            "physical_case_eligibility":report["physical_gate"]["eligible_by_arm"],
            "behavior_report_sha256":sha256_file(p),
            "independent_audit_sha256":sha256_file(q)}
    if set(effects)!=set(ROOTS):
        raise ValueError("all original eight source roots must be present")
    total=math.fsum(effects.values())
    mean=total/8
    loo={str(r):(total-effects[r])/7 for r in ROOTS}
    positive=[r for r in ROOTS if effects[r]>0]
    nonzero=[r for r in ROOTS if effects[r]!=0]
    largest_positive=max((effects[r] for r in positive),default=0.)
    # Root contributions are descriptive, and the number of roots is small.
    positive_sum=math.fsum(effects[r] for r in positive)
    return {
        "schema":SCHEMA,"source_worlds":list(ROOTS),
        "source_cases_per_arm":256,
        "original_independent_source_eligible_roots":[0,1,2,4,6,7],
        "fixed_ITT_zero_roots":list(ITT_ZERO),
        "roots":{str(k):source_reports[k] for k in ROOTS},
        "single_registered_primary":"8-root equal-macro composed-minus-additive sham-minus-coherent-group correctness",
        "source_only_ITT_root_macro_effect":mean,
        "minimum_meaningful_effect":.02,
        "above_meaningful_effect_floor":mean>=.02,
        "leave_one_root_out_effect":loo,
        "positive_root_count":len(positive),
        "nonzero_ITT_root_count":len(nonzero),
        "largest_root_share_of_positive_sum":largest_positive/positive_sum if positive_sum>0 else None,
        "source_correctness_totals":{str(r):counts[r] for r in ROOTS},
        "root_level_population_p_value_computed":False,
        "root_level_equivalence_claim":False,
        "recruited_or_confirmation_worlds_read":0,
        "new_model_forward_calls":0,
        "new_training_updates":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
        "promotion_status":"requires independent source-only sham behavioral-resolution audit and new prospective preregistration",
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-dir",type=Path,action="append",required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():
        raise ValueError("aggregate output must be fresh")
    out=aggregate(a.root_dir)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(canonical(out)+"\n",encoding="utf-8")
    print(canonical(out))
    return out


if __name__=="__main__":
    main()
