"""PEF-06 frozen PEF-05 coalition-intervention and noise-latch calibration gym.

Posthoc read-only audit of all four PEF-05 seeds and baseline/protect/sham.
Every 256-point binary history is evaluated as a NATURAL recurrent
trajectory with fixed extension (0,1). Individual flips at positions
0..5 and every unordered pair of such flips preserve task bits 6,7.

Boolean XOR-of-AND latch F_3 calibrates the difference between single
and pairwise influences; this is a mathematical null, NOT a claimed
function learned by the recurrent model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import torch

from research.pef03.pef03 import (
    BitMemoryLooper, fit_ridge, hash_weights, labels, predict_ridge, tensor
)
from research.pef05.pef05 import SEEDS, ARMS, pair_split
from research.pef04.pef04 import _canonical_bytes

SOURCE_RUN_ID=43687506414
OLD_POSITIONS=tuple(range(6))
CUBE_BITS=8
REFERENCE_LATCH_PAIRS=((0,1),(2,3),(4,5))
CONTINUATION=(0,1)
ALL_MASKS=256


def latch(word:int)->int:
    """F3 = XOR of AND pairs in the six earliest inputs."""
    if type(word) is not int or not 0<=word<ALL_MASKS:
        raise ValueError("eight-bit word out of range")
    return sum(((word>>(2*j))&3)==3 for j in range(3))%2


def source_histories()->torch.Tensor:
    return torch.tensor([
        [(word>>position)&1 for position in range(CUBE_BITS)]
        for word in range(ALL_MASKS)
    ],dtype=torch.long)


def reference_latch_certificate()->dict:
    allzero=0
    singles=[latch(1<<i) for i in OLD_POSITIONS]
    pairs={f"{i},{j}":latch((1<<i)|(1<<j))
           for i,j in combinations(OLD_POSITIONS,2)}
    numerator=sum(3**(6-z.bit_count())
                  for z in range(64) if latch(z)==1)
    global_influence=sum(
        latch(word)!=latch(word^(1<<j))
        for word in range(64) for j in range(6)
    )
    if (latch(allzero)!=0 or any(singles) or
        [pairs[f"{i},{j}"] for i,j in REFERENCE_LATCH_PAIRS]!=[1,1,1] or
        numerator!=676 or global_influence!=192):
        raise AssertionError("exact theoretical latch certificate failed")
    return {
        "latch":"XOR_j(earliest_bit_2j AND earliest_bit_2j_plus1)",
        "zero_single_flip_outputs":singles,
        "zero_double_flip_outputs_by_pair":pairs,
        "zero_sensitive_complete_pairs":[list(v) for v in REFERENCE_LATCH_PAIRS],
        "zero_local_single_sensitivity":0,
        "zero_local_block_sensitivity":3,
        "uniform_source_sensitivity_total_over_64_states":192,
        "uniform_source_average_single_sensitivity":3,
        "bsc_p_one_fourth_zero_activation_numerator_over_4096":numerator,
        "bsc_p_one_fourth_zero_activation_exact":str(Fraction(numerator,4096)),
        "not_a_claim_the_trained_model_computes_the_latch":True,
    }


@torch.no_grad()
def forward_natural_01(model:BitMemoryLooper,histories:torch.Tensor):
    state=model.memory(histories)
    for next_bit in CONTINUATION:
        token=torch.full((len(histories),),next_bit,dtype=torch.long)
        state=model.step(state,model.bit_embedding(token))
    logits=model.head(state)
    if not torch.isfinite(state).all() or not torch.isfinite(logits).all():
        raise AssertionError("nonfinite frozen recurrent state or task logits")
    return state.detach(),logits.detach()


def finite_difference_audit(states:torch.Tensor,logits:torch.Tensor)->dict:
    """Complete unselected all-history finite interactions for six old bits."""
    if tuple(states.shape)!=(ALL_MASKS,16) or tuple(logits.shape)!=(ALL_MASKS,4):
        raise ValueError("expected complete 256-by-16 memory and 256-by-4 logits")
    torch.set_num_threads(1)
    words=torch.arange(ALL_MASKS,dtype=torch.long)
    task=2*((words>>6)&1)+((words>>7)&1)
    # The output head was trained on the last two bits before extension.
    pred=logits.argmax(dim=1)
    original_correct=int((pred==task).sum())
    individual={}
    for i in OLD_POSITIONS:
        j=words^(1<<i)
        if not torch.equal(task[j],task):
            raise AssertionError("old-bit intervention changed supervised target")
        d_hidden=states[j]-states
        d_logits=logits[j]-logits
        individual[str(i)]={
            "position":i,
            "mean_hidden_l2":float(d_hidden.norm(dim=1).mean()),
            "max_hidden_l2":float(d_hidden.norm(dim=1).max()),
            "mean_logits_l2":float(d_logits.norm(dim=1).mean()),
            "predicted_task_class_changes":int((pred[j]!=pred).sum()),
            "total_histories":ALL_MASKS,
        }
    pair_rows=[]
    for i,j in combinations(OLD_POSITIONS,2):
        fi=words^(1<<i)
        fj=words^(1<<j)
        fij=words^(1<<i)^(1<<j)
        if not torch.equal(task[fij],task):
            raise AssertionError("two-old-bit intervention changed task label")
        dh=states[fij]-states[fi]-states[fj]+states
        dl=logits[fij]-logits[fi]-logits[fj]+logits
        pair_only=(
            (pred[fi]==pred)&(pred[fj]==pred)&(pred[fij]!=pred)
        )
        pair_rows.append({
            "pair_positions":[i,j],
            "mean_second_difference_hidden_l2":float(dh.norm(dim=1).mean()),
            "max_second_difference_hidden_l2":float(dh.norm(dim=1).max()),
            "mean_second_difference_logits_l2":float(dl.norm(dim=1).mean()),
            "max_second_difference_logits_l2":float(dl.norm(dim=1).max()),
            "pair_only_task_class_changes":int(pair_only.sum()),
            "any_joint_task_class_changes":int((pred[fij]!=pred).sum()),
            "single_i_task_class_changes":int((pred[fi]!=pred).sum()),
            "single_j_task_class_changes":int((pred[fj]!=pred).sum()),
            "total_histories":ALL_MASKS,
        })
    if len(pair_rows)!=15:
        raise AssertionError("not all 15 old-bit pairs were audited")
    return {
        "full_finite_cube_states":ALL_MASKS,
        "old_bit_positions_used":list(OLD_POSITIONS),
        "supervised_last_two_bit_task_correct_count":original_correct,
        "supervised_last_two_bit_task_denominator":ALL_MASKS,
        "single_bit_interventions":individual,
        "two_bit_interventions":pair_rows,
        "number_of_pair_only_class_changes_across_all_15_pairs":sum(
            item["pair_only_task_class_changes"] for item in pair_rows
        ),
        "sum_of_mean_hidden_second_difference_l2":sum(
            item["mean_second_difference_hidden_l2"] for item in pair_rows
        ),
        "symmetric_basepoint_two_bit_intervention_counted_from_both_masks":True,
        "all_interventions_are_natural_forward_trajectories":True,
        "all_supervised_source_task_bits_held_constant":True,
        "using_full_256_point_universe_posthoc_not_history_disjoint":True,
    }


@torch.no_grad()
def downstream_latch_probe(model,split:dict)->dict:
    """Fit posthoc F3 oldest-six-bit label probe on PEF05 original probe split."""
    probe=tensor(split["probe"])
    heldout=tensor(split["heldout"])
    hprobe,_=forward_natural_01(model,probe)
    htest,_=forward_natural_01(model,heldout)
    target_probe=torch.tensor([
        latch(sum(int(bit)<<j for j,bit in enumerate(row)))
        for row in probe.tolist()
    ],dtype=torch.long)
    target_test=torch.tensor([
        latch(sum(int(bit)<<j for j,bit in enumerate(row)))
        for row in heldout.tolist()
    ],dtype=torch.long)
    fit=fit_ridge(hprobe,target_probe,penalty=1.0)
    guessed=predict_ridge(htest,fit)
    correct=int((guessed==target_test).sum())
    counts=[int((target_test==v).sum()) for v in (0,1)]

    # The maximum heldout number of correct F3 responses if a decoder
    # depends only on the newer seven bits. Each matched oldest-bit
    # pair contributes one correct if bit1=1, two if bit1=0.
    source_only_max=0
    for a,b in zip(target_test[0::2].tolist(),target_test[1::2].tolist()):
        source_only_max+=2 if a==b else 1
    return {
        "frozen_probe_train_size":len(probe),
        "posthoc_full_cube_family_heldout_size":len(heldout),
        "correct_latch_labels":correct,
        "majority_latch_baseline_correct":max(counts),
        "optimal_last_seven_bits_only_baseline_correct":source_only_max,
        "original_latch_heldout_label_counts":[counts[0],counts[1]],
        "ridge_penalty":1.0,
        "probe_does_not_update_trunk":True,
        "posthoc_prior_source_family_reused":True,
    }


def run(archive:Path,out:Path)->dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    summary=json.loads((archive/"summary.json").read_text())
    ledger=json.loads((archive/"split_manifest.json").read_text())
    split,manifest=pair_split()
    if summary["registered_decision"]!="criterion_not_met":
        raise AssertionError("source PEF05 registered decision changed")
    if summary["registered_seed_order"]!=list(SEEDS):
        raise AssertionError("source model seeds not complete")
    if manifest!=ledger or ledger["sha256"]!=summary["source_split_sha256"]:
        raise AssertionError("original source split has changed")
    out.mkdir(parents=True,exist_ok=True)
    calibration=reference_latch_certificate()
    full=source_histories()
    rows=[]
    for seed in SEEDS:
        book=json.loads((archive/f"seed{seed}.sealed_ledger.json").read_text())
        if not book["all_three_arms_frozen_before_downstream_probe"]:
            raise AssertionError("source PEF05 weights not sealed before heldout")
        torch.manual_seed(seed)
        random_init=BitMemoryLooper().eval()
        control_hash=hash_weights(random_init)
        with torch.no_grad():
            random_states,random_logits=forward_natural_01(random_init,full)
            random_metrics=finite_difference_audit(random_states,random_logits)
            random_h8_state=random_init.memory(full)
            random_h8_metrics=finite_difference_audit(
                random_h8_state,random_init.head(random_h8_state))
            random_probe=downstream_latch_probe(random_init,split)
        if hash_weights(random_init)!=control_hash:
            raise AssertionError("random initialization changed during audit")
        for arm in ARMS:
            original=book["arms"][arm]["final"]
            checkpoint=archive/original["file"]
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest!=original["checkpoint_sha256"]:
                raise AssertionError("PEF05 original checkpoint digest mismatch")
            source=torch.load(checkpoint,weights_only=True,map_location="cpu")
            if (source["seed"],source["arm"],source["step"])!=(seed,arm,250):
                raise AssertionError("checkpoint metadata mismatch")
            model=BitMemoryLooper()
            model.load_state_dict(source["model"],strict=True)
            model.eval()
            before=hash_weights(model)
            if before!=original["weights_sha256"]:
                raise AssertionError("checkpoint tensor content mismatch")
            with torch.no_grad():
                hidden,logits=forward_natural_01(model,full)
                metrics=finite_difference_audit(hidden,logits)
                original_h8=model.memory(full)
                h8_metrics=finite_difference_audit(
                    original_h8,model.head(original_h8))
                probe=downstream_latch_probe(model,split)
            if hash_weights(model)!=before:
                raise AssertionError("frozen model mutated during analysis")
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=digest:
                raise AssertionError("original source checkpoint mutated")
            data={
                "seed":seed,"arm":arm,
                "original_model_checkpoint_sha256":digest,
                "original_model_tensor_sha256":before,
                "random_control_initial_weights_sha256":control_hash,
                "original_pef05_registered_negative_decision_preserved":True,
                "frozen_model_read_only":True,
                "frozen_model_finite_difference":metrics,
                "frozen_model_h8_original_task_finite_difference_posthoc":h8_metrics,
                "frozen_model_downstream_coalition_latch_probe":probe,
                "random_trunk_finite_difference_control":random_metrics,
                "random_trunk_h8_original_task_posthoc_control":random_h8_metrics,
                "random_trunk_coalition_latch_probe_control":random_probe,
                "primary_h10_output_head_applied_after_unsupervised_extension":True,
                "h8_task_head_control_added_after_exploratory_h10_review":True,
            }
            (out/f"seed{seed}.{arm}.json").write_bytes(_canonical_bytes(data))
            rows.append({
                "seed":seed,"arm":arm,
                "original_checkpoint_sha256":digest,
                "continued_h10_head_original_task_label_agreement_256":
                    metrics["supervised_last_two_bit_task_correct_count"],
                "trained_h8_head_original_task_correct_256":
                    h8_metrics["supervised_last_two_bit_task_correct_count"],
                "h8_pair_only_class_changes_all15":
                    h8_metrics["number_of_pair_only_class_changes_across_all_15_pairs"],
                "h8_single_old_bit_class_changes_total":sum(
                    x["predicted_task_class_changes"]
                    for x in h8_metrics["single_bit_interventions"].values()),
                "single_old_bit_class_changes_total":sum(
                    x["predicted_task_class_changes"]
                    for x in metrics["single_bit_interventions"].values()),
                "pair_only_class_changes_all15":metrics[
                    "number_of_pair_only_class_changes_across_all_15_pairs"],
                "total_mean_hidden_second_difference":metrics[
                    "sum_of_mean_hidden_second_difference_l2"],
                "oldest_six_bit_latch_decode_correct_48":probe["correct_latch_labels"],
                "latch_suffix_only_ceiling_48":probe[
                    "optimal_last_seven_bits_only_baseline_correct"],
                "random_latch_decode_correct_48":random_probe["correct_latch_labels"],
            })
    aggregate={
        "campaign":"PEF-06",
        "selected_three_openai_math_results":[119,127,132],
        "source_original_pef05_registered_actions_run":SOURCE_RUN_ID,
        "source_original_negative_trial_unchanged":True,
        "posthoc_full_cube_not_new_heldout":True,
        "source_model_count":len(rows),
        "all_frozen_source_checkpoints_sha256_verified":True,
        "boolean_calibration":calibration,
        "seed_arms":rows,
    }
    (out/"summary.json").write_bytes(_canonical_bytes(aggregate))
    print(json.dumps(aggregate,indent=2,sort_keys=True),flush=True)
    return aggregate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    run(args.archive,args.out)


if __name__=="__main__":
    main()
