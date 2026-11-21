"""NEC-03H independent root-level teacher, site, raw-output and dose auditor.

The audit computes source labels from the unchanged NEC-03G world truth mask,
verifies every 16-state scored context and the physical site-dose telemetry,
and independently checks the new v2 intent-to-treat decision. Confirmation
roots and the 48 reserved contexts are forbidden.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path

from mechinterp.neuron_communities.audit_nec03 import correct_count
from mechinterp.neuron_communities.audit_nec03e import check_site
from mechinterp.neuron_communities.audit_nec03f import check_policy
from mechinterp.neuron_communities.nec03g_worlds import (
    encoded, context_catalog, worlds, registry, additive_control_truth,
)
from mechinterp.neuron_communities.nec03h_calibration_audit import sha_file

SCHEMA="NEC-03H/source-site-fallback-900steps-development-v2"
PAIRS=(("native","sham"),("native","topk"),("topk","topk_alt"))
ARMS=("composed","additive")


def audit_arm(folder: Path,world: dict,source: dict, arm: str) -> dict:
    row=json.loads((folder/f"{arm}.report.json").read_text())
    ck=row["train"]
    if (row["world_sha256"]!=world["world_sha256"]
        or row["source_manifest_sha256"]!=source["sha256"]):
        raise AssertionError("arm source identity differs")
    if (ck["world_sha256"]!=world["world_sha256"]
        or ck["training_seed"]!=world["trunk_seed"]+(100000 if arm=="additive" else 0)
        or not ck["source_training_only"]
        or not ck["source_native_weights_frozen_before_evaluation"]):
        raise AssertionError("native source training provenance invalid")
    if sha_file(folder/ck["checkpoint_name"])!=ck["checkpoint_sha256"]:
        raise AssertionError("trained checkpoint bytes altered")
    graph_file=folder/f"{arm}.community_source_freeze.json"
    if sha_file(graph_file)!=row["source_graph_freeze_sha256"]:
        raise AssertionError("source-only graph immutable receipt corrupted")
    graph=json.loads(graph_file.read_text())
    if (graph["source_sha256"]!=source["sha256"]
        or graph["model_sha256"]!=ck["final_model_sha256"]
        or graph["world_sha256"]!=world["world_sha256"]):
        raise AssertionError("native-source selector/checkpoint mismatch")
    choice=graph["source_discovery"]
    if (choice["source_contexts"]!=80
        or choice["development_or_reserved_contexts_exposed_for_selection"]
        or choice["source_selection_uses_labels"]):
        raise AssertionError("source selection accessed evaluation outcomes")
    native=choice["candidate"]
    expected_kind=("fallback" if native and choice["fallback_used"] else
                   "graph" if native else "none")
    if row["source_site_kind"]!=expected_kind:
        raise AssertionError("source-site kind or fallback status differs")
    if choice["fallback_used"] and native is not None:
        selected=world["focal_positions"][2:4]
        fallback=choice["fallback"]
        if fallback["scheduled_visits"]!=selected:
            raise AssertionError("fallback recurrent visit schedule altered")
        nodes=native["nodes"]
        if len(set(nodes))!=4 or len(nodes)!=4:
            raise AssertionError("fallback native donor cardinality differs")
        visits=sorted(int(x.split(":")[0]) for x in nodes)
        if visits!=sorted([v for v in selected for _ in range(2)]):
            raise AssertionError("fallback sites do not match fixed focal schedule")
        if set(nodes)&set(choice["graph"]["hubs"]):
            raise AssertionError("source fallback retained an excluded hub")
        if any(choice["graph"]["frequency"].get(x,0)<2 for x in nodes):
            raise AssertionError("fallback neuron source frequency below minimum")
    if choice["graph_has_qualifying_community"] and choice["fallback_used"]:
        raise AssertionError("fallback used despite source graph candidate")
    if row["candidate_status"]=="NO_SOURCE_COMMUNITY":
        if native is not None or row["source_site_kind"]!="none":
            raise AssertionError("falsified no-source-community label")
        if row["pairs"] or row["raw_paired_records"]:
            raise AssertionError("no-source-site report has invented causal data")
        return {"arm":arm,"clean":row["clean"],"status":"no_source_site",
                "source_site_kind":"none","paired_records":0,
                "graph_has_qualifying_community":choice["graph_has_qualifying_community"],
                "checkpoint_sha256":ck["checkpoint_sha256"],
                "pairs":{}}
    if row["candidate_status"]!="SOURCE_SELECTED":
        raise AssertionError("unsupported source candidate disposition")
    frozen_path=folder/f"{arm}.source_freeze.json"
    if sha_file(frozen_path)!=row["source_freeze_sha256"]:
        raise AssertionError("SAE rank/donor-selection freeze modified")
    frozen=json.loads(frozen_path.read_text())
    if (frozen["source_manifest_sha256"]!=source["sha256"]
        or frozen["model_sha256"]!=ck["final_model_sha256"]
        or frozen["world_sha256"]!=world["world_sha256"]
        or not frozen["no_eval_or_reserved_contexts_used_for_selection"]):
        raise AssertionError("source selected SAE mutated or leaked")
    if frozen["source_graph_candidate"]!=native:
        raise AssertionError("native candidate changed after source selection")
    if frozen["source_sae"]["test_and_reserved_contexts_used"]!=0:
        raise AssertionError("TopK SAE used development/heldout contexts")
    if sha_file(folder/f"{arm}.sae.pt")!=frozen["source_sae"]["source_sae_sha256"]:
        raise AssertionError("source-frozen learned TopK checkpoint changed")
    pairs={tuple(x["pair"]):x for x in row["pairs"]}
    if set(pairs)!=set(PAIRS) or len(row["pairs"])!=3:
        raise AssertionError("wrong source-selected causal comparison family")
    if row["raw_paired_records"]!=48*3:
        raise AssertionError("incorrect 48-context × 3-pair causal record count")
    path=folder/row["raw_paired_context_file"]
    if sha_file(path)!=row["raw_paired_context_sha256"]:
        raise AssertionError("raw context-pair evidence modified")
    records=path.read_text().splitlines()
    if len(records)!=48*3:
        raise AssertionError("missing raw paired source records")
    truth=(world["truth_table_16"] if arm=="composed"
           else additive_control_truth(world["truth_table_16"]))
    expected_y=[(truth>>i)&1 for i in range(16)]
    seen=set()
    counts=defaultdict(lambda:{"clean":0,"hubs":0,"left":0,"right":0})
    physical=defaultdict(lambda:{"n":0,"matched":0,"positive":0,"ratios":[]})
    for line in records:
        item=json.loads(line)
        pair=tuple(item["pair"])
        index=item["source_context_index"]
        if pair not in PAIRS or index not in range(48) or (pair,index) in seen:
            raise AssertionError("unknown or duplicated causal context/pair")
        seen.add((pair,index))
        if item["world_id"]!=world["world_id"] or item["arm"]!=arm:
            raise AssertionError("wrong source world and teacher on raw record")
        if ("".join(map(str,item["source_context_bits"]))!=
            source["contexts_ordered"][160+index]):
            raise AssertionError("non-development source context accessed")
        if item["focal_mask_order"]!=list(range(16)) or item["labels"]!=expected_y:
            raise AssertionError("teacher labels or counterfactual closure corrupted")
        for mode,key in (("clean","clean_logits"),("hubs","hubs_logits"),
                         ("left","left_logits"),("right","right_logits")):
            counts[pair][mode]+=16-correct_count(item[key],expected_y)
        if set(item["local_physical_doses"])!=set(item["active_features"]):
            raise AssertionError("active SAE policy and dose-visit sets differ")
        check_policy(item["active_features"],frozen["source_ranking_per_visit"])
        for dose in item["local_physical_doses"].values():
            measured=check_site(dose)
            x=physical[pair]
            x["n"]+=measured["n"]
            x["matched"]+=measured["matched"]
            x["positive"]+=measured["active"]
            x["ratios"].extend(measured["ratios"])
    if seen!={(pair,i) for pair in PAIRS for i in range(48)}:
        raise AssertionError("incomplete independent-context causal grid")
    verified={}
    clean_per_pair=set()
    for pair,r in pairs.items():
        n=counts[pair]
        if r["errors_of_768"]!=n:
            raise AssertionError("class counts fail independent raw-logit audit")
        if r["left_minus_right_correct"]!=n["right"]-n["left"]:
            raise AssertionError("reported signed pair correctness differs")
        clean_per_pair.add(n["clean"])
        d=physical[pair]
        rr=sorted(d["ratios"])
        median=(rr[(len(rr)-1)//2]+rr[len(rr)//2])/2
        outcome=r["physical_dose"]
        if (d["n"]!=outcome["site_sample_count"]
            or abs(outcome["fraction_matched_within_1pct"]-d["matched"]/d["n"])>1e-7
            or abs(outcome["fraction_with_positive_shared_dose"]-d["positive"]/d["n"])>1e-7
            or abs(outcome["median_fraction_full_left_dose_preserved"]-median)>1e-6):
            raise AssertionError("dose signal summary fails raw-source check")
        signal=(d["matched"]/d["n"]>=.99 and d["positive"]/d["n"]>=.80
                and median>=.25)
        if bool(outcome["candidate_physically_comparable_and_signal_sufficient"])!=signal:
            raise AssertionError("physical signal gate mislabeled")
        verified["/".join(pair)]={
            "effect":(n["right"]-n["left"])/768,
            "signal_eligible":signal,
            "errors":n,
            "physically_audited":True}
    if len(clean_per_pair)!=1 or row["clean"]["correct"]!=768-next(iter(clean_per_pair)):
        raise AssertionError("independent raw clean score differs from printed head")
    return {"arm":arm,"clean":row["clean"],"status":"source_candidate_scored",
            "source_site_kind":row["source_site_kind"],
            "graph_has_qualifying_community":choice["graph_has_qualifying_community"],
            "paired_records":len(records),
            "checkpoint_sha256":ck["checkpoint_sha256"],
            "sae_checkpoint_sha256":frozen["source_sae"]["source_sae_sha256"],
            "source_freeze_sha256":row["source_freeze_sha256"],
            "pairs":verified}


def verify_root(folder: Path) -> dict:
    world=json.loads((folder/"world_spec.json").read_text())
    index=world["world_index"]
    if index not in range(8,16) or world["cohort"]!="development":
        raise AssertionError("new NEC-03H v2 cannot evaluate old/confirmation roots")
    expected=worlds()[index]
    if world!=expected:
        raise AssertionError("world specification differs from unchanged SHA registry")
    source=json.loads((folder/"context_source_manifest.json").read_text())
    _,registered=context_catalog(world)
    if source!=registered:
        raise AssertionError("context source partition or world digest differs")
    root=json.loads((folder/"root_summary.json").read_text())
    if (root["schema"]!=SCHEMA or root["world_id"]!=world["world_id"]
        or root["world_sha256"]!=world["world_sha256"]
        or root["world_source_sha256"]!=source["sha256"]
        or root["original_NEC03G_world_registry_sha256"]!=registry()["registry_sha256"]
        or root["reserved_context_groups_evaluated"]!=0
        or root["original_confirmation_worlds_evaluated"]!=0
        or not root["source_sites_frozen_before_development_readout"]):
        raise AssertionError("root violates original cohort/source/confirmation contract")
    if root["full_registered_v2_recipe"] and (
            root["trunk_steps"]!=900 or root["sae_steps"]!=480):
        raise AssertionError("full v2 label on shortened training")
    arms={arm:audit_arm(folder,world,source,arm) for arm in ARMS}
    e=root["primary_effect"]
    comp=arms["composed"]
    additive=arms["additive"]
    p=comp["pairs"].get("topk/topk_alt")
    eligible=(comp["clean"]["accuracy"]>=.9
              and comp["status"]=="source_candidate_scored"
              and bool(p and p["signal_eligible"]))
    if e["planned_world_valid"]!=eligible:
        raise AssertionError("composed primary ITT admission was not applied")
    expected_effect=p["effect"] if eligible else 0
    if abs(float(e["root_primary_effect"])-expected_effect)>1e-10:
        raise AssertionError("source raw primary effect mismatches reported ITT")
    add=additive["pairs"].get("topk/topk_alt")
    addvalid=(additive["clean"]["accuracy"]>=.9
              and additive["status"]=="source_candidate_scored"
              and bool(add and add["signal_eligible"]))
    native=comp["pairs"].get("native/sham")
    natvalid=bool(native and native["signal_eligible"])
    if (e["additive_sham_valid"]!=addvalid
        or e["native_sham_valid"]!=natvalid):
        raise AssertionError("unavailable sham incorrectly marked valid")
    shams=[
        add["effect"] if addvalid else 1.0,
        native["effect"] if natvalid else 1.0]
    if len(e["sham_effects"])!=2 or any(
            abs(float(a)-b)>1e-10 for a,b in zip(e["sham_effects"],shams)):
        raise AssertionError("null calibration sham direction/upper bound altered")
    if e["null_floor_calibration_eligible"]!=(eligible and addvalid and natvalid):
        raise AssertionError("null feasibility independently classified incorrectly")
    if root["native_clean_scores"]!={arm:row["clean"] for arm,row in arms.items()}:
        raise AssertionError("root clean score differs from audited teacher reports")
    return {
        "schema":SCHEMA,"world_id":world["world_id"],"world_index":index,
        "world_sha256":world["world_sha256"],
        "source_sha256":source["sha256"],
        "root_summary_sha256":sha_file(folder/"root_summary.json"),
        "full_registered_v2_recipe":root["full_registered_v2_recipe"],
        "native_900_checkpoint_byte_hashes_verified":True,
        "sae_checkpoint_and_source_policy_hashes_verified":True,
        "all_48_contexts_each_three_pair_scored_from_raw_16state_logits":True,
        "all_local_physical_doses_and_active_feature_ranks_recomputed":True,
        "original_NEC03G_negative_pilot_unchanged":True,
        "reserved_contexts_evaluated":0,
        "confirmation_worlds_evaluated":0,
        "primary_effect":e["root_primary_effect"],
        "primary_eligible":eligible,
        "null_floor_calibration_eligible":eligible and addvalid and natvalid,
        "source_site_kind_by_arm":{
            arm:row["source_site_kind"] for arm,row in arms.items()},
        "arms":arms}


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--folder",type=Path,required=True)
    args=p.parse_args(argv)
    rec=verify_root(args.folder)
    (args.folder/"independent_source_audit.json").write_bytes(encoded(rec))
    print(json.dumps({"world":rec["world_id"],
                      "full":rec["full_registered_v2_recipe"],
                      "root_effect":rec["primary_effect"],
                      "primary_eligible":rec["primary_eligible"],
                      "null_shams_eligible":rec["null_floor_calibration_eligible"]},
                      sort_keys=True))


if __name__=="__main__":
    main()
