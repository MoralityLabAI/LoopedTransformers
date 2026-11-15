"""PEF-04C: exact deterministic replay of posthoc training memory path.

This analysis is selected after PEF-04's negative preregistered result.
Training is recomputed with the ORIGINAL fixed seed/data/optimizer setup.
Every replay MUST reproduce the original final frozen parameter SHA256
bit for bit. Intermediate scores are observational diagnostics, never
used to select checkpoints, tune a model or change the registered result.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from research.pef03.pef03 import (
    BitMemoryLooper, hash_weights, fit_ridge, predict_ridge, labels, tensor
)
from research.pef04.pef04 import (
    BATCH, SEEDS, paired_split, natural_memory, _pair_scoring, _canonical_bytes
)

SCAN_STEPS = (0,10,25,50,75,100,125,150,175,200,225,250)
TRAIN_STEPS = 250


@torch.no_grad()
def checkpoint_metrics(model, probe, heldout, step):
    model.eval()
    h_probe=natural_memory(model,probe,(0,0))
    h_eval=natural_memory(model,heldout,(0,0))
    fitted=fit_ridge(h_probe,probe[:,0],penalty=1.0)
    decoded=predict_ridge(h_eval,fitted)
    scored=_pair_scoring(decoded,heldout[:,0],h_eval)
    d=h_eval[1::2]-h_eval[0::2]
    avg=d.norm(dim=1).mean()
    task=int((model.logits(heldout).argmax(1)==labels(heldout)).sum())
    return {
        "optimizer_step":step,
        "original_task_correct_out_of_48":task,
        "oldest_bit_correct_out_of_48":scored["correct_histories"],
        "both_members_correct_pairs_out_of_24":scored["both_members_correct_pairs"],
        "pair_prediction_flip_count_out_of_24":scored[
            "prediction_changes_across_oldest_bit_flip_pairs"],
        "mean_causal_oldest_pair_l2":float(avg),
        "relative_causal_pair_l2_vs_state_norm":float(
            avg/h_eval.norm(dim=1).mean().clamp_min(1e-12)),
        "pair_difference_directional_coherence":float(
            d.mean(dim=0).norm()/avg.clamp_min(1e-12)),
    }


def replay_training(seed,source_row,probe,heldout,train_pool):
    torch.manual_seed(seed)
    model=BitMemoryLooper().train()
    initialized=hash_weights(model)
    if initialized!=source_row["initial_weights_sha256"]:
        raise AssertionError(f"seed{seed} initializer differs from original")
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0.0)
    generator=torch.Generator(device="cpu").manual_seed(seed*7919+59)
    rows=[]
    first=last=None
    for step in range(TRAIN_STEPS+1):
        if step in SCAN_STEPS:
            snapshot=checkpoint_metrics(model,probe,heldout,step)
            rows.append(snapshot)
        if step==TRAIN_STEPS:
            break
        model.train()
        histories=train_pool[torch.randint(len(train_pool),(BATCH,),generator=generator)]
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model.logits(histories),labels(histories))
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite replay loss")
        if step==0:
            first=float(loss.detach())
        loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(gradient):
            raise AssertionError("nonfinite replay gradient")
        optimizer.step()
        last=float(loss.detach())
    model.eval()
    digest=hash_weights(model)
    if digest!=source_row["frozen_weights_sha256"]:
        raise AssertionError(f"exact post-training checkpoint replay failed for seed {seed}")
    if abs(first-source_row["initial_loss"])>1e-7:
        raise AssertionError("initial registered loss mismatched")
    if abs(last-source_row["final_loss"])>1e-7:
        raise AssertionError("terminal registered loss mismatched")
    return {
        "seed":seed,
        "exact_initial_weight_hash":initialized,
        "exact_final_weight_hash":digest,
        "original_final_checkpoint_weight_replay_passed":True,
        "replay_updates":TRAIN_STEPS,
        "all_intermediate_evaluations_read_only":True,
        "snapshots":rows,
    }


def run(source:Path,output:Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    split,manifest=paired_split()
    registered=json.loads((source/"summary.json").read_text())
    ledger=json.loads((source/"sealed_training_ledger.json").read_text())
    saved_split=json.loads((source/"split_manifest.json").read_text())
    if registered["seed_order"] != list(SEEDS) or not registered["registered_scoring_run"]:
        raise AssertionError("unexpected source seed set")
    if registered["registered_transfer_decision"] != "criterion_not_met":
        raise AssertionError("original preregistered decision differs")
    if saved_split!=manifest or ledger["split_manifest_sha256"]!=manifest["split_sha256"]:
        raise AssertionError("registered data split mismatch")
    by_seed={r["seed"]:r for r in ledger["training"]}
    by_score={r["seed"]:r for r in registered["result_per_seed"]}
    if set(by_seed)!=set(SEEDS) or set(by_score)!=set(SEEDS):
        raise AssertionError("incomplete source model inventory")
    train_pool=tensor(split["train"])
    probe=tensor(split["probe"])
    heldout=tensor(split["heldout"])
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for seed in SEEDS:
        r=by_seed[seed]
        ckpt=source/r["checkpoint_file"]
        if hashlib.sha256(ckpt.read_bytes()).hexdigest()!=r["checkpoint_sha256"]:
            raise AssertionError(f"seed {seed} source checkpoint SHA256 incorrect")
        x=replay_training(seed,r,probe,heldout,train_pool)
        at_end=x["snapshots"][-1]
        scored=by_score[seed]["downstream_arms"]["h10_00"]["source_frozen_trunk"]
        if (at_end["oldest_bit_correct_out_of_48"]!=scored["correct_histories"] or
            at_end["both_members_correct_pairs_out_of_24"]!=scored["both_members_correct_pairs"] or
            at_end["original_task_correct_out_of_48"]!=by_score[seed]["head_task_correct_heldout"]):
            raise AssertionError("replayed terminal score differs from preregistered result")
        if abs(at_end["mean_causal_oldest_pair_l2"]-
               scored["mean_natural_pair_hidden_l2"])>1e-5:
            raise AssertionError("replayed final memory geometry differs from registered")
        x["original_registered_terminal_scores_reproduced"]=True
        below=[row["optimizer_step"] for row in x["snapshots"]
               if row["oldest_bit_correct_out_of_48"]<36]
        x["posthoc_first_sampled_below_36_out_of_48"]=(below[0] if below else None)
        x["steps_are_fixed_retrospective_snapshots_not_an_early_stop_policy"]=True
        (output/f"seed{seed}.trajectory.json").write_bytes(_canonical_bytes(x))
        records.append(x)
    summary={
        "campaign":"PEF-04C",
        "retrospective_training_path_selected_after_preregistered_negative_result":True,
        "source_registered_run":43683120934,
        "original_registered_decision_preserved":"criterion_not_met",
        "exact_original_terminal_checkpoint_weights_reproduced_all_six":True,
        "original_terminal_readout_scores_reproduced_all_six":True,
        "optimizer_steps_scanned":list(SCAN_STEPS),
        "no_hyperparameter_or_checkpoint_selection":True,
        "results":[{
            "seed":x["seed"],
            "first_sampled_below_36":x["posthoc_first_sampled_below_36_out_of_48"],
            "at_step_125":next(v for v in x["snapshots"] if v["optimizer_step"]==125),
            "at_step_150":next(v for v in x["snapshots"] if v["optimizer_step"]==150),
            "at_step_250":x["snapshots"][-1],
        } for x in records],
    }
    (output/"trajectory_summary.json").write_bytes(_canonical_bytes(summary))
    print(json.dumps(summary,sort_keys=True,indent=2),flush=True)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv)
    run(args.source,args.out)


if __name__=="__main__":
    main()
