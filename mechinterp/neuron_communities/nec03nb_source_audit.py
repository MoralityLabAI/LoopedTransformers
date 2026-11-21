"""NEC-03N-B independent checkpoint-anchored source-only audit.

Reload immutable original NEC03M 900-step checkpoints and new fixed-reset
900-additional-step checkpoints, recompute all old/new 2560 original
training-source Boolean correctness outcomes, original/final weight digests
and full true-class confusion. Independently verify the continued source
minibatch index sequence and the *explicitly reset* AdamW condition.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from mechinterp.neuron_communities.nec03nb_checkpoint_reset import (
    SCHEMA,ROOTS,OLD_STEPS,ADDITIONAL_STEPS,TOTAL_STEPS,
    SOURCE_STATES,CLEAN_SOURCE_FLOOR,source_confusion,batch_digest,sha_file,
)
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from research.pef07.pef07 import Looper,model_digest


def read(path:Path):
    return json.loads(path.read_text())


def audit_one(original:Path,measured:Path,lock_path:Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data=read(measured/"checkpoint_reset_source_report.json")
    index=data["world_index"]
    if index not in ROOTS:
        raise AssertionError("unauthorized new/embargoed/confirmation world in replay")
    world=worlds()[index]
    catalog,source=context_catalog(world)
    prior=read(original/"source_pilot_receipt.json")
    if (read(original/"world_spec.json")!=world
        or read(original/"source_split_manifest.json")!=source
        or prior["world_sha256"]!=world["world_sha256"]
        or prior["five_input_registry_sha256"]!=registration()["registry_sha256"]
        or prior["source_manifest_sha256"]!=source["source_sha256"]):
        raise AssertionError("original five-input native teacher/source assignment changed")
    if (data["schema"]!=SCHEMA
        or data["world_sha256"]!=world["world_sha256"]
        or data["original_900_receipt_sha256"]!=sha_file(original/"source_pilot_receipt.json")
        or data["registered_source_reset_lock_sha256"]!=sha_file(lock_path)
        or data["five_input_world_registry_sha256"]!=registration()["registry_sha256"]
        or data["original_four_input_confirmation_spec_sha256"]!=
            old_registry()["confirmation_full_spec_sha256"]
        or data["original_NEC03M_run"]!=44894980345
        or data["original_exact_continuation_1800_failure_run"]!=44895917012
        or data["old_optimizer_state_reset_at900"] is not True
        or data["exact_continuous_AdamW_1800_claim"] is not False
        or data["old_native_steps"]!=900
        or data["additional_AdamW_reset_steps"]!=900
        or data["source_histories"]!=SOURCE_STATES
        or data["new_sae_models_fit"]!=0
        or data["new_interventions_scored"]!=0
        or data["new_five_input_development_evaluation_context_groups_scored"]!=0
        or data["reserved_context_groups_scored"]!=0
        or data["five_input_embargoed_worlds_scored"]!=0
        or data["original_four_input_confirmation_worlds_scored"]!=0):
        raise AssertionError("source-only reset optimizer protocol/cohort changed")
    arms={}
    for arm in ("composed","additive"):
        report=data["arms"][arm]
        prior_arm=prior["arms"][arm]["native"]
        seed=world["native_model_seed"]+(100000 if arm=="additive" else 0)
        if (report["arm"]!=arm or report["seed"]!=seed
            or not report["reset_optimizer_state_at_checkpoint"]
            or report["original_optimizer_state_reconstructed"]
            or report["source_batch_stream_offset"]!=900
            or report["source_index_generator_seed"]!=seed*7919+59
            or report["new_optimizer_update_count"]!=900
            or report["total_updates"]!=1800
            or not report["original_source_score_only"]
            or report["old_frozen900_checkpoint_sha256"]!=prior_arm["checkpoint_sha256"]
            or report["old_native_model_sha256"]!=prior_arm["final_model_weights_sha256"]):
            raise AssertionError("old checkpoint reset condition or source index seed changed")
        old_file=original/prior_arm["checkpoint_file"]
        new_file=measured/report["new_checkpoint_file"]
        if (sha_file(old_file)!=prior_arm["checkpoint_sha256"]
            or sha_file(new_file)!=report["new_checkpoint_sha256"]):
            raise AssertionError("original or new model checkpoint file bytes altered")
        old_state=torch.load(old_file,map_location="cpu",weights_only=True)
        new_state=torch.load(new_file,map_location="cpu",weights_only=True)
        if (new_state["schema"]!=SCHEMA
            or new_state["world_sha256"]!=world["world_sha256"]
            or new_state["arm"]!=arm or new_state["model_seed"]!=seed
            or new_state["new_reset_AdamW_steps"]!=900
            or new_state["original_learning_updates_before_warmstart"]!=900
            or new_state["prior_frozen900_model_sha256"]!=prior_arm["final_model_weights_sha256"]
            or new_state["prior_frozen900_checkpoint_file_sha256"]!=prior_arm["checkpoint_sha256"]
            or new_state["model_weights_sha256"]!=report["new_native_model_sha256"]
            or not new_state["checkpoint_source_only"]):
            raise AssertionError("new checkpoint falsely describes its actual original parent/model")
        original_model=Looper()
        original_model.load_state_dict(old_state["model"],strict=True)
        original_model.eval()
        if model_digest(original_model)!=prior_arm["final_model_weights_sha256"]:
            raise AssertionError("old original native checkpoint does not have reported weights")
        new_model=Looper()
        new_model.load_state_dict(new_state["model"],strict=True)
        new_model.eval()
        if model_digest(new_model)!=report["new_native_model_sha256"]:
            raise AssertionError("new warm-optimizer checkpoint weight digest differs")
        old_score=source_confusion(original_model,catalog["train"],world,arm=arm)
        new_score=source_confusion(new_model,catalog["train"],world,arm=arm)
        if (old_score!=report["old_source_score"]
            or new_score!=report["new_source_score"]
            or old_score["correct"]!=prior_arm["source_correct"]
            or abs(old_score["accuracy"]-prior_arm["source_accuracy"])>1e-10):
            raise AssertionError("independent complete source label/logit scores differ")
        rng=torch.Generator().manual_seed(seed*7919+59)
        for _ in range(900):
            torch.randint(SOURCE_STATES,(128,),generator=rng)
        first=last=None
        for i in range(900):
            ids=torch.randint(SOURCE_STATES,(128,),generator=rng)
            if first is None:first=batch_digest(ids)
            last=batch_digest(ids)
        if (first!=report["first_new_batch_index_sha256"]
            or last!=report["last_new_batch_index_sha256"]):
            raise AssertionError("new source-training minibatch index stream changed")
        if (report["old_competence"]!=(old_score["accuracy"]>=CLEAN_SOURCE_FLOOR)
            or report["new_competence"]!=(new_score["accuracy"]>=CLEAN_SOURCE_FLOOR)):
            raise AssertionError("reported 90% native source competence gate differs")
        arms[arm]={
            "arm":arm,
            "original_900_correct":old_score["correct"],
            "fixed_reset_1800_correct":new_score["correct"],
            "original_900_accuracy":old_score["accuracy"],
            "fixed_reset_1800_accuracy":new_score["accuracy"],
            "correctness_delta":new_score["correct"]-old_score["correct"],
            "old_model_checkpoint_sha256":prior_arm["checkpoint_sha256"],
            "new_model_checkpoint_sha256":report["new_checkpoint_sha256"],
            "all_2560_original_source_cases_recomputed":True,
            "original_training_stage_weights_verified":True,
            "new_optimizer_was_reset_at_checkpoint":True,
        }
    eligible=all(x["fixed_reset_1800_accuracy"]>=CLEAN_SOURCE_FLOOR for x in arms.values())
    if data["both_arms_source_competent"]!=eligible:
        raise AssertionError("whole model root competence inconsistent")
    r={
        "schema":"NEC-03N-B/independent-900frozen-reset900-source-audit/v1",
        "world_id":world["world_id"],"world_index":index,
        "world_sha256":world["world_sha256"],
        "original_900_receipt_sha256":sha_file(original/"source_pilot_receipt.json"),
        "registered_source_lock_sha256":sha_file(lock_path),
        "new_1800_native_report_sha256":sha_file(
            measured/"checkpoint_reset_source_report.json"),
        "full_900_old_and_new_training_prediction_check":True,
        "original_900_models_reused_as_immutable_anchored_weights":True,
        "native_optimizer_moments_reset_at_900":True,
        "exact_continuous_1800_claim":False,
        "two_native_models_audited":True,
        "source_training_groups_used":80,
        "five_input_development_evaluation_groups_scored":0,
        "reserved_context_groups_scored":0,
        "five_input_embargoed_worlds_scored":0,
        "original_four_input_confirmation_worlds_scored":0,
        "both_arms_competent":eligible,
        "arms":arms,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (measured/"independent_warmstart_audit.json").write_text(
        json.dumps(r,sort_keys=True,indent=2)+"\n")
    print("NEC03NB_WARMSTART_SOURCE_MODEL_AUDIT_PASS")
    return r


def aggregate(roots:Path,out:Path):
    data=[]
    for idx in ROOTS:
        folder=roots/f"nec03nb-warm-source-root-{idx}"
        a=read(folder/"independent_warmstart_audit.json")
        if (a["world_index"]!=idx or
            a["new_1800_native_report_sha256"]!=sha_file(
                folder/"checkpoint_reset_source_report.json")
            or not a["two_native_models_audited"]
            or a["original_four_input_confirmation_worlds_scored"]!=0):
            raise AssertionError("missing or edited independent warmstart audit")
        data.append(a)
    result={
        "schema":"NEC-03N-B/four-world-fixed-reset-native-competence/v1",
        "root_statistical_unit":"one five-input Boolean truth function and focal position quintet; two arms paired within root",
        "number_of_five_input_world_roots":4,
        "native_model_pairs":4,
        "native_model_arms":8,
        "original_step900_native_models_verified":8,
        "additional_updates_per_arm":900,
        "new_optimizer_moments_reset_at_original900":True,
        "exact_continuous1800_claim":False,
        "original_900_composed_competent":sum(r["arms"]["composed"]["original_900_accuracy"]>=.9 for r in data),
        "new_1800_composed_competent":sum(r["arms"]["composed"]["fixed_reset_1800_accuracy"]>=.9 for r in data),
        "original_900_additive_competent":sum(r["arms"]["additive"]["original_900_accuracy"]>=.9 for r in data),
        "new_1800_additive_competent":sum(r["arms"]["additive"]["fixed_reset_1800_accuracy"]>=.9 for r in data),
        "whole_roots_both_competent":sum(r["both_arms_competent"] for r in data),
        "new_SAE_dictionaries_trained":0,
        "new_causal_interventions_scored":0,
        "development_evaluation_worlds_scored":0,
        "reserved_contexts_scored":0,
        "embargoed_worlds_scored":0,
        "original_four_input_confirmation_worlds_scored":0,
        "original_NEC03M_900_source_model_results_unchanged":True,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
        "worlds":data,
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec03nb_four_root_fixed_reset_900_vs_1800.json").write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("NEC03NB_FOUR_ROOT_SOURCE_RESET_CALIBRATION_AUDIT_COMPLETE")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--old-root",type=Path)
    p.add_argument("--new-root",type=Path)
    p.add_argument("--lock",type=Path,
                   default=Path(__file__).with_name("NEC03NB_RESET_LOCK.json"))
    p.add_argument("--roots",type=Path)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.roots is not None and a.out is not None:
        aggregate(a.roots,a.out)
    elif a.old_root and a.new_root:
        audit_one(a.old_root,a.new_root,a.lock)
    else:
        p.error("supply --old-root --new-root or --roots --out")
