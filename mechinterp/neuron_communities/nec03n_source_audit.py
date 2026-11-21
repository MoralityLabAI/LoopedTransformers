"""NEC-03N independent audit of full five-input source-only 900->1800 replay.

For each of the four ALREADY OPENED training-source worlds, verify the
original 900-step checkpoint and source recipe, ensure that the reported
midpoint weight hash exactly matches the immutable NEC-03M model, and
reload/re-score every 2,560 training-source history at update1800.
No development-evaluation, reserved, embargoed or confirmation scoring.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from mechinterp.neuron_communities.nec03n_native1800 import (
    SCHEMA,INDICES,OLD_STEPS,NEW_STEPS,
    SOURCE_HISTORY_COUNT,source_confusion,file_sha,
)
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,labels,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from research.pef07.pef07 import Looper,model_digest


def load(path:Path):
    return json.loads(path.read_text())


def audit_root(source_root:Path,new_root:Path,lock_path:Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    old=load(source_root/"source_pilot_receipt.json")
    r=load(new_root/"native1800_source_calibration.json")
    index=r["world_index"]
    if index not in INDICES:
        raise AssertionError("new five-input unregistered or embargoed root scored")
    world=worlds()[index]
    catalog,source=context_catalog(world)
    if (r["schema"]!=SCHEMA or r["world_sha256"]!=world["world_sha256"]
        or r["old_900_receipt_sha256"]!=file_sha(source_root/"source_pilot_receipt.json")
        or r["new_1800_source_lock_sha256"]!=file_sha(lock_path)
        or r["new_five_input_registry_sha256"]!=registration()["registry_sha256"]
        or r["old_original_four_input_confirmation_sha256"]!=
            old_registry()["confirmation_full_spec_sha256"]
        or not r["all_prior_900_source_weights_replayed_exactly"]
        or r["native_1800_optimizer_steps_per_arm"]!=NEW_STEPS
        or r["in_sample_source_history_count"]!=SOURCE_HISTORY_COUNT
        or r["new_dev_evaluation_groups_scored"]!=0
        or r["reserved_groups_scored"]!=0 or r["embargoed_worlds_scored"]!=0
        or r["old_confirmation_worlds_scored"]!=0
        or r["new_source_AE_models_fitted"]!=0
        or r["new_causal_interventions_scored"]!=0):
        raise AssertionError("source-only cohort, model provenance or new 1800 lock altered")
    if (old["world_sha256"]!=world["world_sha256"]
        or old["source_manifest_sha256"]!=source["source_sha256"]
        or old["development_evaluation_context_groups_scored"]!=0
        or old["new_embargoed_worlds_scored"]!=0
        or old["original_confirmation_worlds_scored"]!=0
        or load(source_root/"world_spec.json")!=world
        or load(source_root/"source_split_manifest.json")!=source):
        raise AssertionError("original frozen 900 source world changed")
    scores={}
    for arm in ("composed","additive"):
        prior=old["arms"][arm]["native"]
        current=r["arms"][arm]
        if (current["arm"]!=arm
            or current["source_training_steps"]!=NEW_STEPS
            or current["source_training_histories"]!=SOURCE_HISTORY_COUNT
            or not current["all_900_checkpoint_tensors_matched_exactly"]
            or current["old_900_model_sha256"]!=prior["final_model_weights_sha256"]
            or current["replayed_900_model_sha256"]!=prior["final_model_weights_sha256"]
            or current["original_900_checkpoint_sha256"]!=prior["checkpoint_sha256"]
            or current["original_initial_model_sha256"]!=prior["initial_model_weights_sha256"]
            or current["native_model_seed"]!=world["native_model_seed"]+
                (100000 if arm=="additive" else 0)
            or current["old_900_source_score"]["correct"]!=prior["source_correct"]
            or current["old_900_source_score"]["total"]!=SOURCE_HISTORY_COUNT
            or not current["same_AdamW_state_continued_through_1800"]):
            raise AssertionError("intermediate optimizer/checkpoint replay historical identity changed")
        if file_sha(source_root/prior["checkpoint_file"])!=prior["checkpoint_sha256"]:
            raise AssertionError("original source-900 checkpoint bytes differ")
        payload=torch.load(source_root/prior["checkpoint_file"],
                           map_location="cpu",weights_only=True)
        orig_model=Looper()
        orig_model.load_state_dict(payload["model"],strict=True)
        orig_model.eval()
        if model_digest(orig_model)!=current["replayed_900_model_sha256"]:
            raise AssertionError("original source900 intermediate model weights differ")
        check=source_confusion(orig_model,catalog["train"],world,arm=arm)
        if check!=current["old_900_source_score"]:
            raise AssertionError("original 900-point predictions differ")
        if file_sha(new_root/current["new_checkpoint_file"])!=current["new_checkpoint_sha256"]:
            raise AssertionError("new 1800-checkpoint bytes differ")
        learned=torch.load(new_root/current["new_checkpoint_file"],
                           map_location="cpu",weights_only=True)
        if (learned["world_sha256"]!=world["world_sha256"]
            or learned["teacher"]!=arm or learned["steps"]!=NEW_STEPS
            or learned["intermediate_900_weight_sha256"]!=prior["final_model_weights_sha256"]
            or learned["model_weights_sha256"]!=current["new_1800_model_sha256"]):
            raise AssertionError("new native checkpoint belongs to another source world/teacher")
        fresh=Looper()
        fresh.load_state_dict(learned["model"],strict=True)
        fresh.eval()
        if model_digest(fresh)!=current["new_1800_model_sha256"]:
            raise AssertionError("new source model tensor digest differs")
        independent=source_confusion(fresh,catalog["train"],world,arm=arm)
        if independent!=current["new_1800_source_score"]:
            raise AssertionError("new 1800-model source predictions differ from independent replay")
        if current["new_1800_source_competent"]!=(independent["accuracy"]>=.9):
            raise AssertionError("new source native competence gate mislabeled")
        scores[arm]={
            "source_correct_step900":check["correct"],
            "source_correct_step1800":independent["correct"],
            "source_class_recall_step900":check["class_recall"],
            "source_class_recall_step1800":independent["class_recall"],
            "training_correctness_delta":independent["correct"]-check["correct"],
            "native_competent_step900":check["accuracy"]>=.9,
            "native_competent_step1800":independent["accuracy"]>=.9,
            "original900_checkpoint_sha256":prior["checkpoint_sha256"],
            "new1800_checkpoint_sha256":current["new_checkpoint_sha256"],
            "frozen_900_optimizer_stream_replayed_exactly":True,
        }
    result={
        "schema":"NEC-03N/independent-source-only-milestone-audit/v1",
        "world_id":world["world_id"],"world_index":index,
        "original_five_input_world_sha256":world["world_sha256"],
        "original_source_pilot_sha256":file_sha(source_root/"source_pilot_receipt.json"),
        "new_source_calibration_sha256":file_sha(new_root/"native1800_source_calibration.json"),
        "original_900_models_identical_at_milestone":True,
        "all_2560_source_training_input_predictions_recomputed":True,
        "native_1800_checkpoints_and_weight_digests_verified":True,
        "old_training_source_split_unchanged":True,
        "retrained_native_models":2,
        "new_SAE_models_fitted":0,
        "new_development_evaluation_groups_scored":0,
        "reserved_groups_scored":0,
        "embargoed_worlds_scored":0,
        "old_confirmation_worlds_scored":0,
        "arms":scores,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (new_root/"independent_milestone_audit.json").write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("NEC03N_INDEPENDENT_SOURCE_CHECKPOINT_MILESTONE_AUDIT_PASS")
    return result


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--old-root",type=Path,required=True)
    p.add_argument("--new-root",type=Path,required=True)
    p.add_argument("--lock",type=Path,
                   default=Path(__file__).with_name("NEC03N_SOURCE_LOCK.json"))
    a=p.parse_args(argv)
    audit_root(a.old_root,a.new_root,a.lock)


if __name__=="__main__":
    main()
