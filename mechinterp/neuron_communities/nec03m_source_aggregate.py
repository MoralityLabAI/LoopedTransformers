"""NEC-03M independent-source-audit aggregation across four new roots.

Only source-trained native and SAE evidence is admitted. Exactly four
development roots dev5-000..003, two teacher arms per root; no scoring
of new development-evaluation, reserved, embargoed or four-input
confirmation contexts.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    registration,worlds,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry

INDICES=(0,1,2,3)
SCHEMA="NEC-03M/independent-four-root-source-competence-aggregate/v1"


def digest_file(p:Path):
    h=sha256()
    with p.open("rb") as f:
        for z in iter(lambda:f.read(1<<20),b""):
            h.update(z)
    return h.hexdigest()


def aggregate(root:Path,out:Path,lock:Path):
    registry=registration()
    candidates=list(root.glob("nec03m-source-world-*"))
    if len(candidates)!=4 or {int(p.name.rsplit("-",1)[1]) for p in candidates}!=set(INDICES):
        raise AssertionError("source cohort must contain exactly four distinct registered five-input worlds")
    allroots=[]
    for index in INDICES:
        path=root/f"nec03m-source-world-{index}"
        run=json.loads((path/"source_pilot_receipt.json").read_text())
        audit=json.loads((path/"independent_source_audit.json").read_text())
        w=worlds()[index]
        if (run["schema"]!="NEC-03M/five-input-40-40-source-native-sae-competence-v1"
            or run["world_sha256"]!=w["world_sha256"]
            or run["world_index"]!=index
            or run["five_input_registry_sha256"]!=registry["registry_sha256"]
            or run["original_four_input_confirmation_spec_sha256"]!=
                old_registry()["confirmation_full_spec_sha256"]
            or run["NEC03M_source_lock_sha256"]!=digest_file(lock)
            or not run["source_results_only"]
            or run["development_evaluation_context_groups_scored"]!=0
            or run["reserved_context_groups_scored"]!=0
            or run["new_embargoed_worlds_scored"]!=0
            or run["original_confirmation_worlds_scored"]!=0):
            raise AssertionError("source world differs from registered training source or confirmation boundary")
        if (audit["raw_receipt_sha256"]!=digest_file(path/"source_pilot_receipt.json")
            or audit["world_sha256"]!=w["world_sha256"]
            or not audit["native_checkpoint_and_SAE_bytes_verified"]
            or not audit["native_source_training_labels_and_all_2560_input_scores_replayed"]
            or not audit["source_SAE_reconstruction_and_40_group_attribution_rank_recomputed"]
            or not audit["source_direction_active_candidates_independently_reenumerated"]
            or audit["development_evaluation_contexts_scored"]!=0
            or audit["five_input_embargoed_worlds_scored"]!=0
            or audit["old_four_input_confirmation_worlds_scored"]!=0):
            raise AssertionError("missing independent 900-step new-world source and angular audit")
        if run["root_ready_for_future_development_scoring"]!=audit["source_root_ready_for_development"]:
            raise AssertionError("feasibility claim differs from root independent audit")
        per_arm={}
        for arm in ("composed","additive"):
            m=audit["arms"][arm]
            if m["source_correct"] not in range(2561) or m["source_total"]!=2560:
                raise AssertionError("bad in-sample native teacher count")
            per_arm[arm]={
                "source_correct_of_2560":m["source_correct"],
                "source_train_competent":m["source_correct"]>=int(.9*2560),
                "site_kind":m["source_site_kind"],
                "source_SAE_present":m["source_SAE_present"],
                "source_SAE_MSE":m.get("source_SAE_MSE"),
                "source_angle_coverage":m.get("source_angular_coverage"),
                "source_control_physically_feasible":m["angle_source_feasible"],
                "source_model_eligible":m["root_model_source_eligible"],
                "native_checkpoint_sha256":m["native_checkpoint_sha256"],
                "sae_checkpoint_sha256":m.get("sae_checkpoint_sha256"),
                "source_rank_sha256":m.get("source_rank_sha256"),
            }
        allroots.append({
            "world_id":w["world_id"],"world_index":index,
            "source_root_ready":audit["source_root_ready_for_development"],
            "source_receipt_sha256":audit["raw_receipt_sha256"],
            "arms":per_arm})
    ready=sum(x["source_root_ready"] for x in allroots)
    allarms=[x["arms"][arm] for x in allroots for arm in ("composed","additive")]
    result={
        "schema":SCHEMA,
        "evidence_class":"source-only five-input model and sparse-null engineering pilot",
        "worlds":allroots,
        "development_roots_registered":4,
        "native_teacher_models_trained_from_new_seeds":8,
        "native_training_updates_per_model":900,
        "new_five_input_source_registry_sha256":registry["registry_sha256"],
        "original_confirmation_commitment_sha256":
            old_registry()["confirmation_full_spec_sha256"],
        "source_native_models_at_least_90pct_train_accuracy":sum(
            x["source_train_competent"] for x in allarms),
        "source_SAE_models_fitted":sum(x["source_SAE_present"] for x in allarms),
        "source_angle_controls_feasible":sum(x["source_control_physically_feasible"] for x in allarms),
        "whole_worlds_source_ready":ready,
        "root_sample_size_for_source_only":4,
        "no_development_generalization_claim":True,
        "new_development_evaluation_contexts_scored":0,
        "reserved_contexts_scored":0,
        "five_input_embargoed_worlds_scored":0,
        "original_four_input_confirmation_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY"
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec03m_five_input_source_competence_audit.json").write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("NEC03M_FOUR_FIVE_INPUT_WORLD_SOURCE_AUDITS_COMPLETE")
    return result


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lock",type=Path,
                   default=Path(__file__).with_name("NEC03M_SOURCE_LOCK.json"))
    a=p.parse_args(argv)
    aggregate(a.root,a.out,a.lock)


if __name__=="__main__":
    main()
