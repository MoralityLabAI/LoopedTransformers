"""NEC-03N-B: pure-stdlib replay of already independently audited source roots.

Recover four immutable successful root artifacts from original run 44898361742
without retraining. Every original 900 and new reset-optimizer 1800 source
prediction is re-scored against the exact registered five-input truth
table. Verify both source-score confusion matrices, checkpoint byte SHA256,
original independent PyTorch audit receipts and original confirmation seal.

This is source-training in-sample evidence on already opened worlds.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,additive_control_truth,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry

SCHEMA="NEC-03N-B/stdlib-replay-of-four-checkpoint-reset-root-artifacts/v1"
OLD_RUN=44898361742
ROOTS=tuple(range(4))


def read(path):
    return json.loads(path.read_text())


def digest_file(path):
    h=sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def truth_confusion(labels,truth):
    if len(labels)!=2560 or any(x not in (0,1) for x in labels):
        raise AssertionError("incomplete source-only predictions")
    y=[(truth>>focal)&1 for _ in range(80) for focal in range(32)]
    matrix=[[0,0],[0,0]]
    for a,b in zip(y,labels):
        matrix[a][b]+=1
    return {
        "correct":matrix[0][0]+matrix[1][1],
        "total":2560,
        "accuracy":(matrix[0][0]+matrix[1][1])/2560,
        "confusion_truth_rows_pred_columns":matrix,
        "per_class_recall":[matrix[i][i]/sum(matrix[i]) if sum(matrix[i]) else None
                            for i in range(2)],
    }


def audit_artifact(folder:Path,source_lock:Path,index:int):
    w=worlds()[index]
    score=read(folder/"checkpoint_reset_source_report.json")
    audited=read(folder/"independent_warmstart_audit.json")
    if (score["schema"]!="NEC-03N-B/frozen900-checkpoint-reset-AdamW-900-source/v1"
        or score["world_id"]!=w["world_id"] or score["world_index"]!=index
        or score["world_sha256"]!=w["world_sha256"]
        or score["registered_source_reset_lock_sha256"]!=digest_file(source_lock)
        or score["five_input_world_registry_sha256"]!=registration()["registry_sha256"]
        or score["original_four_input_confirmation_spec_sha256"]!=
            old_registry()["confirmation_full_spec_sha256"]
        or score["original_NEC03M_run"]!=44894980345
        or score["original_exact_continuation_1800_failure_run"]!=44895917012
        or score["old_optimizer_state_reset_at900"] is not True
        or score["exact_continuous_AdamW_1800_claim"] is not False
        or score["old_native_steps"]!=900 or score["additional_AdamW_reset_steps"]!=900
        or score["new_sae_models_fit"]!=0
        or score["new_interventions_scored"]!=0
        or score["new_five_input_development_evaluation_context_groups_scored"]!=0
        or score["reserved_context_groups_scored"]!=0
        or score["five_input_embargoed_worlds_scored"]!=0
        or score["original_four_input_confirmation_worlds_scored"]!=0):
        raise AssertionError("checkpoint-anchored source study population/optimizer conditions moved")
    if (audited["world_id"]!=w["world_id"] or audited["world_index"]!=index
        or audited["original_900_receipt_sha256"]!=score["original_900_receipt_sha256"]
        or audited["new_1800_native_report_sha256"]!=
            digest_file(folder/"checkpoint_reset_source_report.json")
        or audited["registered_source_lock_sha256"]!=digest_file(source_lock)
        or audited["five_input_development_evaluation_groups_scored"]!=0
        or audited["original_four_input_confirmation_worlds_scored"]!=0
        or not audited["two_native_models_audited"]
        or not audited["original_900_models_reused_as_immutable_anchored_weights"]
        or audited["exact_continuous_1800_claim"] is not False):
        raise AssertionError("registered independent neural source audit/receipt differs")
    allresults={}
    for arm in ("composed","additive"):
        r=score["arms"][arm]
        a=audited["arms"][arm]
        truth=w["truth_table_32"] if arm=="composed" else additive_control_truth(w["truth_table_32"])
        for name in ("old_source_score","new_source_score"):
            reported=r[name]
            independently=truth_confusion(reported["source_prediction_labels"],truth)
            for key in independently:
                reported_value=reported[key]
                actual=independently[key]
                if isinstance(actual,float):
                    if not math.isclose(reported_value,actual,rel_tol=0,abs_tol=1e-10):
                        raise AssertionError("source true-label correctness changed in "+name)
                elif isinstance(actual,list) and key=="per_class_recall":
                    if any(not math.isclose(x,y,rel_tol=0,abs_tol=1e-10)
                           for x,y in zip(reported_value,actual)):
                        raise AssertionError("class-specific source accuracy changed")
                elif reported_value!=actual:
                    raise AssertionError("source predictions and confusion matrix disagree")
            if reported["training_source_only"] is not True:
                raise AssertionError("reported score is not in-sample source-only")
        if (r["old_source_score"]["correct"]!=a["original_900_correct"]
            or r["new_source_score"]["correct"]!=a["fixed_reset_1800_correct"]
            or a["new_model_checkpoint_sha256"]!=r["new_checkpoint_sha256"]
            or a["old_model_checkpoint_sha256"]!=r["old_frozen900_checkpoint_sha256"]
            or a["new_optimizer_was_reset_at_checkpoint"] is not True
            or a["all_2560_original_source_cases_recomputed"] is not True):
            raise AssertionError("independent checkpoint-weight audit contradicts source labels")
        path=folder/r["new_checkpoint_file"]
        if digest_file(path)!=r["new_checkpoint_sha256"]:
            raise AssertionError("new checkpoint data bytes differ from registered receipt")
        if r["seed"]!=w["native_model_seed"]+(100000 if arm=="additive" else 0):
            raise AssertionError("source-model training seed changed")
        if (r["new_optimizer_update_count"]!=900
            or r["total_updates"]!=1800
            or r["source_batch_stream_offset"]!=900
            or r["original_optimizer_state_reconstructed"] is not False):
            raise AssertionError("new AdamW reset experiment misrepresented as full optimizer continuation")
        if r["new_competence"]!=(r["new_source_score"]["accuracy"]>=.9):
            raise AssertionError("new model source-competence floor moved")
        allresults[arm]={
            "arm":arm,
            "original_source_correct":r["old_source_score"]["correct"],
            "new_source_correct":r["new_source_score"]["correct"],
            "new_correctness_delta":r["new_source_score"]["correct"]-r["old_source_score"]["correct"],
            "original_source_accuracy":r["old_source_score"]["accuracy"],
            "new_source_accuracy":r["new_source_score"]["accuracy"],
            "new_source_competent":r["new_competence"],
            "new_checkpoint_file_sha256":r["new_checkpoint_sha256"],
            "original_checkpoint_file_sha256":r["old_frozen900_checkpoint_sha256"],
            "independent_checkpoint_and_source_logits_checked":True,
        }
    both=all(x["new_source_competent"] for x in allresults.values())
    if (score["both_arms_source_competent"]!=both
        or audited["both_arms_competent"]!=both):
        raise AssertionError("root-level training-only competence status moved")
    return {
        "world_id":w["world_id"],"index":index,
        "original_five_input_world_sha256":w["world_sha256"],
        "source_training_world_prior_run":44894980345,
        "checkpoint_reset_training_run":OLD_RUN,
        "registered_source_lock_sha256":digest_file(source_lock),
        "root_source_report_sha256":digest_file(folder/"checkpoint_reset_source_report.json"),
        "root_independent_neural_source_audit_sha256":digest_file(folder/"independent_warmstart_audit.json"),
        "both_arms_source_competent":both,
        "arms":allresults,
        "source_only":True,
        "development_evaluation_contexts_scored":0,
        "original_four_input_confirmation_worlds_scored":0,
    }


def aggregate(roots:Path,out:Path,source_lock:Path):
    expected={f"nec03nb-warm-source-root-{i}" for i in ROOTS}
    present={p.name for p in roots.iterdir() if p.is_dir()}
    if present!=expected:
        raise AssertionError("aggregator requires exactly the four original intact root artifacts")
    rows=[audit_artifact(roots/f"nec03nb-warm-source-root-{i}",
                         source_lock,i) for i in ROOTS]
    obj={
        "schema":SCHEMA,
        "root_unit":"one previously seen five-focal-input Boolean truth/position world; composed/additive arms paired",
        "original_frozen900_checkpoint_run":44894980345,
        "new_reset_optimizer_900_steps_run":OLD_RUN,
        "four_old_worlds":4,
        "original_and_new_native_model_arms_audited":8,
        "old_source_competent_models":sum(
            x["arms"][arm]["original_source_accuracy"]>=.9
            for x in rows for arm in ("composed","additive")),
        "new_source_competent_models":sum(
            x["arms"][arm]["new_source_competent"]
            for x in rows for arm in ("composed","additive")),
        "old_both_arms_competent_roots":sum(
            all(x["arms"][arm]["original_source_accuracy"]>=.9
                for arm in ("composed","additive")) for x in rows),
        "new_both_arms_competent_roots":sum(x["both_arms_source_competent"] for x in rows),
        "all_8_model_checkpoint_byte_hashes_reverified":True,
        "all_8_old_and_new_2560_source_class_predictions_rescored":True,
        "optimizer_reset_at_original_frozen900":True,
        "exact_1800_uninterrupted_AdamW_claim":False,
        "new_SAE_training_models":0,
        "new_causal_interventions":0,
        "new_development_evaluation_scored":0,
        "new_reserved_context_groups_scored":0,
        "new_five_input_embargoed_models_scored":0,
        "original_four_input_confirmation_models_scored":0,
        "original_NEC03M_source_model_and_SAE_results_unchanged":True,
        "evidence_class":"in-sample source-training recovery calibration only",
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "worlds":rows,
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec03nb_pure_replay_root_aggregate.json").write_text(
        json.dumps(obj,sort_keys=True,indent=2)+"\n")
    print("NEC03NB_FOUR_CHECKPOINT_RESET_ROOTS_SOURCE_ONLY_STD_LIB_REPLAY_PASSED")
    return obj


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lock",type=Path,
                   default=Path(__file__).with_name("NEC03NB_RESET_LOCK.json"))
    args=p.parse_args()
    aggregate(args.root,args.out,args.lock)
