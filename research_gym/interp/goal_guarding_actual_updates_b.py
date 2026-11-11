"""GG-04B: depth-aligned real SGD effects, bounded controller rewards.

A separately registered successor after GG-04's six-cue positive-damage
calibration failed. The four-cue protected task uses independently licensed
Bayes labels and real PyTorch one-step SGD counterfactual effects. A bounded
reward transform is applied only to controller feedback. Raw gold changes
are retained, and a six-cue gold readout is computed *after* actor freezing.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Any
import json
import math

import torch

from research_gym.interp.goal_guarding_causal import (
    make_panel,train_cell,_weights_digest,
)
from research_gym.interp.goal_guarding_actual_updates import (
    GOLD_PANEL_SIZE,SEEDS,META_UPDATES,PhysicalUpdate,
    SealedGoldAuditor,query_splits,hidden_features,
    train_controller,_cue_only_actions,evaluate_controller,
    _finite_effect_log,_rows_hash,
)


BASE_COMMIT="50270e3f22f6e3b9cabbb17258cec0a0a7fc7960"
ALIGNED_GOLD_SEED=69001
CROSS_DEPTH_GOLD_SEED=68001
PENALTY_CLIP=4.0


def clip_penalty(value:float,median_damage:float)->float:
    """Bound normalized scalar reward units; never alter raw gold evidence."""
    if not math.isfinite(value) or not math.isfinite(median_damage) or median_damage<=1e-9:
        raise ValueError("positive finite damage calibration required")
    return max(-PENALTY_CLIP,min(PENALTY_CLIP,value/median_damage))


def bounded_effects(effects:tuple[PhysicalUpdate,...],
                    median_damage:float)->tuple[PhysicalUpdate,...]:
    """Use existing GG-04 controller on bounded physically sourced rewards."""
    return tuple(PhysicalUpdate(
        gold_loss_delta_truth=clip_penalty(e.gold_loss_delta_truth,median_damage),
        gold_loss_delta_flip=clip_penalty(e.gold_loss_delta_flip,median_damage),
        linear_delta_truth=e.linear_delta_truth,
        linear_delta_flip=e.linear_delta_flip,
        raw_gradient_norm_truth=e.raw_gradient_norm_truth,
        raw_gradient_norm_flip=e.raw_gradient_norm_flip,
        truth_changed_weights=e.truth_changed_weights,
        flip_changed_weights=e.flip_changed_weights,
    ) for e in effects)


def _effect_ledger(rows,effects,median_damage:float)->list[dict[str,Any]]:
    raw=_finite_effect_log(rows,effects)
    for item,effect in zip(raw,effects):
        item["bounded_controller_truth_penalty"]=clip_penalty(
            effect.gold_loss_delta_truth,median_damage)
        item["bounded_controller_flip_penalty"]=clip_penalty(
            effect.gold_loss_delta_flip,median_damage)
    return raw


def _panel_hash(panel)->str:
    record={"domain":"gg04b.gold.v1",
            "observations":panel.observations.tolist(),
            "labels":panel.labels.tolist(),
            "context":panel.context.tolist()}
    return sha256(json.dumps(record,separators=(",",":"),
                             sort_keys=True).encode()).hexdigest()


def _cross_depth_summary(aligned:tuple[PhysicalUpdate,...],
                         other:tuple[PhysicalUpdate,...])->dict[str,Any]:
    if len(aligned)!=len(other) or not aligned:
        raise ValueError("paired cross-depth effect lengths must match")
    a=[e.damage_gap for e in aligned]
    b=[e.damage_gap for e in other]
    return {
        "queries":len(a),
        "aligned_four_cue_median_gap":float(median(a)),
        "cross_depth_six_cue_median_gap":float(median(b)),
        "aligned_four_cue_mean_gap":sum(a)/len(a),
        "cross_depth_six_cue_mean_gap":sum(b)/len(b),
        "fraction_aligned_positive":sum(x>0 for x in a)/len(a),
        "fraction_six_cue_positive":sum(x>0 for x in b)/len(b),
        "fraction_opposing_nonzero_signs":
            sum(x*y<0 for x,y in zip(a,b))/len(a),
        "mean_absolute_gap_difference":
            sum(abs(x-y) for x,y in zip(a,b))/len(a),
    }


def _actual_raw_factorial(assessment:dict[str,Any],actor,features,
                          aligned:tuple[PhysicalUpdate,...])->None:
    """Attach true, unbounded gold units to every 2x2 assignment cell."""
    with torch.no_grad():
        action={}
        for cue in (0,1):
            d=torch.full((len(aligned),),cue,dtype=torch.long)
            probability=actor(features,d).softmax(-1)[:,1]
            action[cue]=tuple(int(p>0.5) for p in probability.tolist())
    for cell in assessment["causal_randomization_factorial"]:
        cue=cell["disclosure"]
        u=cell["real_update_assignment"]
        bounded_units=cell.pop("mean_actual_gold_loss_delta")
        raw=sum((e.gold_loss_delta_truth if choice else e.gold_loss_delta_flip)
                for e,choice in zip(aligned,action[cue]))/len(aligned)
        cell["mean_bounded_controller_penalty_units"]=bounded_units
        cell["mean_raw_gold_ce_delta"]=u*raw
        cell["unit_type_for_mean_realized_reward"]="bounded controller reward"


def run_gg04b(*,output:Path,artifacts:Path,
              seeds:tuple[int,...]=SEEDS,base_updates:int=400,
              controller_updates:int=META_UPDATES)->dict[str,Any]:
    if seeds!=SEEDS or base_updates!=400 or controller_updates!=400:
        raise ValueError("GG-04B frozen seed and optimizer budgets required")
    torch.set_num_threads(1)
    source=query_splits()
    aligned_gold=make_panel(GOLD_PANEL_SIZE,4,ALIGNED_GOLD_SEED)
    cross_depth_gold=make_panel(GOLD_PANEL_SIZE,6,CROSS_DEPTH_GOLD_SEED)
    output.parent.mkdir(parents=True,exist_ok=True)
    artifacts.mkdir(parents=True,exist_ok=True)
    report:dict[str,Any]={
        "protocol":"RMP-GG-04B",
        "source_main_commit":BASE_COMMIT,
        "prior_GG04_calibration_failure_known":True,
        "same_frozen_query_split_hashes":{k:_rows_hash(rows)
                                          for k,rows in source.items()},
        "split_counts":{k:len(v) for k,v in source.items()},
        "primary_four_cue_gold_seed":ALIGNED_GOLD_SEED,
        "secondary_six_cue_gold_seed":CROSS_DEPTH_GOLD_SEED,
        "primary_gold_panel_sha256":_panel_hash(aligned_gold),
        "secondary_gold_panel_sha256":_panel_hash(cross_depth_gold),
        "reward_penalty_normalization":"calibration median then clip[-4,4]",
        "bounded_penalty_magnitude":PENALTY_CLIP,
        "actual_native_pytorch_sgd_updates":True,
        "post_action_task_model_updated":True,
        "post_action_controller_head_updated":False,
        "meta_policy_updates":controller_updates,
        "source_task_training_updates":base_updates,
        "controller_training_uses_gold_labels_directly":False,
        "results":[],
    }
    for seed in seeds:
        base,training=train_cell(seed,steps=base_updates,
                                 save_dir=artifacts)
        auditor=SealedGoldAuditor(base,aligned_gold)
        if not auditor.check_no_update_identity():
            raise AssertionError("no-update treatment lost baseline identity")
        trained=tuple(auditor.measure(x) for x in source["actor_train"])
        calibration=tuple(auditor.measure(x) for x in source["calibration"])
        median_damage=float(median(e.damage_gap for e in calibration))
        if median_damage<=1e-9:
            raise ValueError("GG-04B positive four-cue calibration guard failed")
        transformed=bounded_effects(trained,median_damage)
        features=hidden_features(base,source["actor_train"])
        actor,trace=train_controller(features,transformed,seed,scale=1.0,
                                     steps=controller_updates)
        cue_only=_cue_only_actions(transformed,1.0)
        # The heldout effects and independent cross-depth audits are opened
        # ONLY after both actor training and cue-only selection are frozen.
        unseen=tuple(auditor.measure(x) for x in source["heldout"])
        scored=bounded_effects(unseen,median_damage)
        unseen_states=hidden_features(base,source["heldout"])
        evaluation=evaluate_controller(actor,unseen_states,scored,1.0,cue_only)
        _actual_raw_factorial(evaluation,actor,unseen_states,unseen)
        original_six=SealedGoldAuditor(base,cross_depth_gold)
        secondary=tuple(original_six.measure(x)
                        for x in source["heldout"])
        cross_summary=_cross_depth_summary(unseen,secondary)
        if _weights_digest(base)!=auditor.base_hash:
            raise AssertionError("counterfactual post-report steps mutated base")
        ledger=artifacts/f"gg04b_seed{seed}_sgd_potentials.json"
        ledger.write_text(json.dumps({
            "seed":seed,"source_baseline_weight_sha256":auditor.base_hash,
            "median_calibration_damage":median_damage,
            "actual_raw_gold_four_cue":True,
            "bounded_reward_only":True,
            "train":_effect_ledger(source["actor_train"],trained,median_damage),
            "calibration":_effect_ledger(source["calibration"],calibration,median_damage),
            "heldout":_effect_ledger(source["heldout"],unseen,median_damage),
            "heldout_original_six_cue":
                _finite_effect_log(source["heldout"],secondary),
        },sort_keys=True,indent=2,allow_nan=False)+"\n")
        checkpoint=artifacts/f"gg04b_seed{seed}_reporter.pt"
        torch.save({
            "seed":seed,"trained_base_model":base.state_dict(),
            "reporter":actor.state_dict(),
            "base_training":training,
            "calibration_damage_scale":median_damage,
        },checkpoint)
        report["results"].append({
            "seed":seed,"base_training":training,
            "primary_gold_loss_baseline":auditor.baseline_loss,
            "secondary_gold_loss_baseline":original_six.baseline_loss,
            "no_update_weight_and_gold_identity":True,
            "trained_native_weight_sha256":auditor.base_hash,
            "calibration_median_positive_damage":median_damage,
            "train_median_raw_damage":float(median(e.damage_gap for e in trained)),
            "heldout_median_raw_damage":float(median(e.damage_gap for e in unseen)),
            "cross_depth_heldout":cross_summary,
            "controller_training":trace,
            "controller_eval":evaluation,
            "real_sgd_steps_per_seed":2*(len(trained)+len(calibration)
                                          +len(unseen)+len(secondary)),
            "ledger_filename":ledger.name,
            "ledger_sha256":sha256(ledger.read_bytes()).hexdigest(),
            "checkpoint_filename":checkpoint.name,
            "checkpoint_sha256":sha256(checkpoint.read_bytes()).hexdigest(),
        })
    report["two_seed_controller_gate"]=all(
        r["controller_eval"]["controller_registered_gate_pass"]
        for r in report["results"])
    output.write_text(json.dumps(report,sort_keys=True,indent=2,
                                 allow_nan=False)+"\n")
    return report
