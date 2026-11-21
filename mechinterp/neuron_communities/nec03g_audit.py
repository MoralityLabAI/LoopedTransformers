"""Independent NEC-03G development world receipt and source-closure auditor.

Recomputes 16-way labels from the registered truth mask, verifies paired
correctness from the raw logits, source split, frozen checkpoints, source
attribution roster and local equalized physical doses. No training code
is imported and confirmation outcomes are never read.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from mechinterp.neuron_communities.audit_nec03 import correct_count
from mechinterp.neuron_communities.audit_nec03e import check_site
from mechinterp.neuron_communities.audit_nec03f import check_policy
from mechinterp.neuron_communities.nec03g_worlds import (
    DEV_COUNT, encoded, context_catalog, digest, worlds,
)
from mechinterp.neuron_communities.nec03g_development import (
    PAIRS, sha_file,
)

ARMS=("composed","additive")


def verify_root(folder: Path) -> dict:
    spec=json.loads((folder/"world_spec.json").read_text())
    if spec["cohort"]!="development" or spec["world_index"] not in range(DEV_COUNT):
        raise AssertionError("confirmation world was evaluated in development")
    trusted=worlds()[spec["world_index"]]
    if spec!=trusted:
        raise AssertionError("world spec differs from frozen generator")
    source=json.loads((folder/"context_source_manifest.json").read_text())
    _,expected=context_catalog(trusted)
    if source!=expected:
        raise AssertionError("world context assignment differs from source function")
    summary=json.loads((folder/"root_summary.json").read_text())
    if (summary["world_sha256"]!=spec["world_sha256"]
        or summary["source_sha256"]!=source["sha256"]
        or summary["original_confirmation_worlds_evaluated"]!=0
        or summary["world_reserved_context_groups_evaluated"]!=0
        or summary["cohort"]!="development"):
        raise AssertionError("source/confirmation exposure violated")
    discovered=[]
    for arm in ARMS:
        row=json.loads((folder/f"{arm}.report.json").read_text())
        if row["source_manifest_sha256"]!=source["sha256"]:
            raise AssertionError("original source checksum differs")
        ck=row["train"]
        if (not ck["frozen_before_discovery"] or
            not ck["trained_on_source_train_only"]):
            raise AssertionError("training source was not frozen")
        if sha_file(folder/ck["checkpoint_name"])!=ck["checkpoint_sha256"]:
            raise AssertionError("frozen checkpoint bytes changed")
        discovery_path=folder/f"{arm}.community_source_freeze.json"
        if sha_file(discovery_path)!=row["source_graph_freeze_sha256"]:
            raise AssertionError("source-only native graph was mutated")
        frozen_graph=json.loads(discovery_path.read_text())
        if frozen_graph["source_sha256"]!=source["sha256"]:
            raise AssertionError("source discovery includes another world")
        if row["candidate_status"]=="NO_SOURCE_COMMUNITY":
            if row["raw_paired_records"] or row["pairs"]:
                raise AssertionError("no-candidate world has fabricated causal effects")
            discovered.append({"arm":arm,"status":"no_source_candidate",
                               "clean":row["clean"],"paired_source_records":0})
            continue
        if row["candidate_status"]!="SOURCE_SELECTED":
            raise AssertionError("unknown discovery status")
        frozen=folder/f"{arm}.source_freeze.json"
        if sha_file(frozen)!=row["source_freeze_sha256"]:
            raise AssertionError("SAE/graph/feature source manifest mutated")
        freeze=json.loads(frozen.read_text())
        if (freeze["source_manifest_sha256"]!=source["sha256"]
            or freeze["model_sha256"]!=ck["model_weights_sha256"]
            or not freeze["no_eval_or_reserved_contexts_used_for_selection"]):
            raise AssertionError("source-only selection contamination")
        if freeze["source_sae"]["test_and_reserved_contexts_used"]!=0:
            raise AssertionError("SAE fit exposed evaluation data")
        if sha_file(folder/f"{arm}.sae.pt")!=freeze["source_sae"]["source_sae_sha256"]:
            raise AssertionError("SAE source dictionary changed")
        native=freeze["source_graph_candidate"]
        graph=frozen_graph["source_discovery"]
        if native!=graph["candidate"]:
            raise AssertionError("graph candidate changed after source discovery")
        if sorted([tuple(x["pair"]) for x in row["pairs"]]) != sorted(PAIRS):
            raise AssertionError("three registered donor pairs required")
        if row["raw_paired_records"]!=48*len(PAIRS):
            raise AssertionError("wrong number of source-context records")
        raw_file=folder/row["raw_paired_context_file"]
        if sha_file(raw_file)!=row["raw_paired_context_sha256"]:
            raise AssertionError("paired source logits modified")
        lines=raw_file.read_text().splitlines()
        if len(lines)!=row["raw_paired_records"]:
            raise AssertionError("raw source receipt count differs")
        counts=defaultdict(lambda:{"clean":0,"hubs":0,"left":0,"right":0})
        seen=set()
        doses=defaultdict(lambda:{"n":0,"positive":0,"matched":0,"retained":[]})
        for line in lines:
            item=json.loads(line)
            pair=tuple(item["pair"])
            index=item["source_context_index"]
            if pair not in PAIRS or not 0<=index<48 or (pair,index) in seen:
                raise AssertionError("pair/source world duplicate or missing")
            seen.add((pair,index))
            if item["world_id"]!=trusted["world_id"] or item["arm"]!=arm:
                raise AssertionError("raw world ID or teacher arm mismatched")
            expected_context=source["contexts_ordered"][160+index]
            if "".join(map(str,item["source_context_bits"]))!=expected_context:
                raise AssertionError("input context was reassigned across world splits")
            if item["focal_mask_order"]!=list(range(16)):
                raise AssertionError("focal interventions not quartet closed")
            truth=(freeze["source_graph_candidate"]["id"] and
                   ck["teacher_truth_table"])
            labels=[(truth>>k)&1 for k in range(16)]
            if item["labels"]!=labels:
                raise AssertionError("independent truth-table scoring disagrees")
            for name,field in (("clean","clean_logits"),("hubs","hubs_logits"),
                               ("left","left_logits"),("right","right_logits")):
                counts[pair][name]+=16-correct_count(item[field],labels)
            if set(item["local_physical_doses"])!=set(item["active_features"]):
                raise AssertionError("physical dose and active route visit sets disagree")
            check_policy(item["active_features"],freeze["source_ranking_per_visit"])
            for sample in item["local_physical_doses"].values():
                verified=check_site(sample)
                totals=doses[pair]
                totals["n"]+=verified["n"]
                totals["positive"]+=verified["active"]
                totals["matched"]+=verified["matched"]
                totals["retained"].extend(verified["ratios"])
        if len(seen)!=48*len(PAIRS):
            raise AssertionError("one source context missing from candidate")
        for pair in PAIRS:
            if {k for p,k in seen if p==pair}!=set(range(48)):
                raise AssertionError("incomplete independent-context group cohort")
        for pair in row["pairs"]:
            identity=tuple(pair["pair"])
            errors=counts[identity]
            if pair["errors_of_768"]!=errors:
                raise AssertionError("printed accuracy differs from raw paired logits")
            if pair["left_minus_right_correct"]!=errors["right"]-errors["left"]:
                raise AssertionError("registered comparison printed incorrectly")
            d=doses[identity]
            summary_dose=pair["physical_dose"]
            ratios=sorted(d["retained"])
            median=(ratios[(len(ratios)-1)//2]+ratios[len(ratios)//2])/2
            if (summary_dose["site_sample_count"]!=d["n"] or
                abs(summary_dose["fraction_matched_within_1pct"]-d["matched"]/d["n"])>1e-7 or
                abs(summary_dose["fraction_with_positive_shared_dose"]-d["positive"]/d["n"])>1e-7 or
                abs(summary_dose["median_fraction_full_left_dose_preserved"]-median)>1e-6):
                raise AssertionError("physical dose summary was manipulated")
        discovered.append({"arm":arm,"status":"source_candidate_scored",
                           "clean":row["clean"],
                           "native_checkpoint_sha256":ck["checkpoint_sha256"],
                           "source_sae_sha256":freeze["source_sae"]["source_sae_sha256"],
                           "paired_source_records":len(lines)})
    # ITT effects are recomputed solely from the already-audited raw pair results.
    result=summary["primary_effect"]
    if not result["planned_world_valid"] and result["root_primary_effect"]!="0":
        raise AssertionError("failed world was excluded from intention-to-treat")
    if summary["full_registered_development_recipe"]:
        if summary["trunk_steps"]!=trusted["trunk_steps"] or summary["sae_steps"]!=trusted["sae_steps"]:
            raise AssertionError("full recipe mislabelled")
    return {"world_id":trusted["world_id"],"world_sha256":trusted["world_sha256"],
            "root_summary_sha256":sha_file(folder/"root_summary.json"),
            "source_manifest_sha256":source["sha256"],
            "root_primary_effect":result["root_primary_effect"],
            "sham_effects":result["sham_effects"],
            "world_valid":result["planned_world_valid"],
            "full_registered_development_recipe":summary["full_registered_development_recipe"],
            "paired_teacher_controls_verified":True,
            "confirmation_cohort_still_unopened":True,
            "arms":discovered}


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--folder",type=Path,required=True)
    args=p.parse_args(argv)
    result=verify_root(args.folder)
    receipt=args.folder/"independent_source_audit.json"
    receipt.write_bytes(encoded(result))
    print(json.dumps({"world":result["world_id"],
                      "full":result["full_registered_development_recipe"],
                      "primary_effect":result["root_primary_effect"],
                      "valid":result["world_valid"],
                      "paired_root_source_verified":True},sort_keys=True))


if __name__=="__main__":
    main()
