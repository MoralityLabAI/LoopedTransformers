"""PEF-05 independent read-only archive replay of the original frozen trial.

Rebuild SHA256-pair-closed train, probe and heldout sets independently;
load original branch checkpoints, independently refit old-bit ridge
readouts, reconstruct train-only signed margins and recompute the
original preregistered negative verdict. No model training and no
dependency on research.pef05.pef05 scoring functions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import torch

from research.pef03.pef03 import (
    BitMemoryLooper, fit_ridge, hash_weights, labels, predict_ridge, tensor
)

SEEDS=(101,109,127,157)
ARMS=("baseline","protect","sham")


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _independent_source_split():
    suffixes=list(product((0,1),repeat=7))
    suffixes.sort(key=lambda s:(
        hashlib.sha256(b"pef05-v1-pair:"+bytes(s)).digest(),bytes(s)
    ))
    rows={}
    for split,lo,hi in (("train",0,80),("probe",80,104),("heldout",104,128)):
        rows[split]=[(b,*suffix) for suffix in suffixes[lo:hi] for b in (0,1)]
    sets=[set(v) for v in rows.values()]
    if len(set.union(*sets))!=256 or sum(len(s) for s in sets)!=256:
        raise AssertionError("independent source split overlapped")
    for name,values in rows.items():
        for left,right in zip(values[0::2],values[1::2]):
            if (left[0],right[0])!=(0,1) or left[1:]!=right[1:]:
                raise AssertionError("original source was not pair-closed")
    return {key:tensor(v) for key,v in rows.items()}


@torch.no_grad()
def _natural(model,inputs,ext):
    state=model.memory(inputs)
    for bit in ext:
        token=torch.full((len(inputs),),bit,dtype=torch.long)
        state=model.step(state,model.bit_embedding(token))
    return state


def _check_model(path,expected,seed,arm,step):
    content=path.read_bytes()
    digest=hashlib.sha256(content).hexdigest()
    if digest!=expected["checkpoint_sha256"]:
        raise AssertionError(f"checkpoint digest mismatch {seed}/{arm}/{step}")
    data=torch.load(path,weights_only=True,map_location="cpu")
    if (data["seed"],data["arm"],data["step"])!=(seed,arm,step):
        raise AssertionError("model role and visit counter mismatch")
    model=BitMemoryLooper()
    model.load_state_dict(data["model"],strict=True)
    model.eval()
    if hash_weights(model)!=expected["weights_sha256"]:
        raise AssertionError("frozen model tensor hash mismatch")
    if data["weights_sha256"]!=expected["weights_sha256"]:
        raise AssertionError("checkpoint embedded tensor hash disagrees")
    if data["optimizer_sha256"]!=expected["optimizer_sha256"]:
        raise AssertionError("checkpoint embedded optimizer hash disagrees")
    return model,digest


@torch.no_grad()
def _independent_probe(model,fit_bits,heldout,ext):
    probe=_natural(model,fit_bits,ext)
    test=_natural(model,heldout,ext)
    fitted=fit_ridge(probe,fit_bits[:,0],penalty=1.0)
    prediction=predict_ridge(test,fitted)
    correct=int((prediction==heldout[:,0]).sum())
    pair_correct=int((
        (prediction[0::2]==0)&(prediction[1::2]==1)
    ).sum())
    pair_flip=int((prediction[0::2]!=prediction[1::2]).sum())
    distance=(test[1::2]-test[0::2]).norm(dim=1)
    task_correct=int((model.logits(heldout).argmax(1)==labels(heldout)).sum())
    return {
        "correct_histories":correct,
        "both_members_correct_pairs":pair_correct,
        "prediction_changes_across_oldest_bit_flip_pairs":pair_flip,
        "mean_natural_pair_hidden_l2":float(distance.mean()),
        "min_natural_pair_hidden_l2":float(distance.min()),
        "task_correct":task_correct,
    }


@torch.no_grad()
def _signed_train_margin(model,source_pairs,anchor):
    z=_natural(model,source_pairs,(0,1))
    return float(((z[1::2]-z[0::2])@anchor).mean())


def audit(archive:Path,output:Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    split=_independent_source_split()
    registered=_load_json(archive/"summary.json")
    manifest=_load_json(archive/"split_manifest.json")
    if not registered["registered_scoring_run"]:
        raise AssertionError("archive does not contain a registered trial")
    if registered["registered_decision"]!="criterion_not_met":
        raise AssertionError("unexpected original registered verdict")
    if registered["registered_seed_order"]!=list(SEEDS):
        raise AssertionError("wrong original seed protocol")
    if registered["source_split_sha256"]!=manifest["sha256"]:
        raise AssertionError("registered source manifest ID inconsistent")
    if manifest["history_counts"]!={"train":160,"probe":48,"heldout":48}:
        raise AssertionError("wrong registered source sizes")
    for name,values in split.items():
        if len(values)!=manifest["history_counts"][name]:
            raise AssertionError("independent split allocation incorrect")
        expected_suffixes=manifest["assignment_by_suffix"]
        for row in values.tolist():
            if expected_suffixes["".join(map(str,row[1:]))]!=name:
                raise AssertionError("independent split disagrees with source manifest")
    output.mkdir(parents=True,exist_ok=True)
    summary_rows=[]
    for seed in SEEDS:
        ledger=_load_json(archive/f"seed{seed}.sealed_ledger.json")
        original=_load_json(archive/f"seed{seed}.evaluation.json")
        source_receipt=next(r for r in registered["results"] if r["seed"]==seed)
        if source_receipt!=original:
            raise AssertionError("individual source evaluations differ from summary")
        if not ledger["all_three_arms_frozen_before_downstream_probe"]:
            raise AssertionError("source training was not sealed before probing")
        common=ledger["common"]
        original_model,_=_check_model(archive/common["file"],common,seed,"common",125)
        source_states=_natural(original_model,split["train"],(0,1))
        avg=(source_states[1::2]-source_states[0::2]).mean(dim=0)
        magnitude=float(avg.norm())
        anchor=avg/magnitude
        if abs(magnitude-ledger["train_only_anchor_l2"])>2e-5:
            raise AssertionError("independent anchor norm does not replay")
        anchor_pairs=split["train"][:32]
        baseline_margin=_signed_train_margin(original_model,anchor_pairs,anchor)
        heldout=split["heldout"]
        # Test deterministic exact baseline for every possible source-only
        # suffix predictor, which must output the same bit per pair.
        if not torch.all(heldout[0::2,1:]==heldout[1::2,1:]):
            raise AssertionError("heldout pair closure violated")
        if int((heldout[:,0]==0).sum())!=24:
            raise AssertionError("oldest-bit source was not fair in heldout")
        arm_results={}
        for arm in ARMS:
            book=ledger["arms"][arm]
            if arm!="baseline":
                rows=book["surgery"]
                if len(rows)!=25 or [r["step"] for r in rows]!=list(range(126,151)):
                    raise AssertionError("missing stepwise surgery records")
                for row in rows:
                    if row["matched_removed_gradient_norm_error"]>1e-5:
                        raise AssertionError("norm-matched sham/protection failed")
                    if arm=="protect" and row["active"] and row["memory_inner_after"]>1e-5:
                        raise AssertionError("stored gradient violated memory guard")
            elif book["surgery"]:
                raise AssertionError("baseline performed surgery")
            if abs(book["source_only_margin_step125"]-baseline_margin)>2e-5:
                raise AssertionError("source train-only step125 score did not replay")
            final=book["final"]
            midway=book["intermediate"]
            model_mid,_=_check_model(archive/midway["file"],midway,seed,arm,150)
            model_final,digest=_check_model(archive/final["file"],final,seed,arm,250)
            for model,label in ((model_mid,"source_only_margin_step150"),
                                (model_final,"source_only_margin_step250")):
                score=_signed_train_margin(model,anchor_pairs,anchor)
                if abs(score-book[label])>5e-5:
                    raise AssertionError(f"{seed}/{arm} signed score {label} mismatch")
            evaluated=_independent_probe(model_final,split["probe"],heldout,(0,1))
            scored=source_receipt["arms"][arm]["h10_01"]
            for field in ("correct_histories","both_members_correct_pairs",
                          "prediction_changes_across_oldest_bit_flip_pairs"):
                if evaluated[field]!=scored[field]:
                    raise AssertionError(f"{seed}/{arm}/{field} independent decoder mismatch")
            if abs(evaluated["mean_natural_pair_hidden_l2"]-
                   scored["mean_natural_pair_hidden_l2"])>5e-5:
                raise AssertionError("independent decoder distance mismatch")
            if evaluated["task_correct"]!=source_receipt["arms"][arm]["original_task_correct_out_of_48"]:
                raise AssertionError("independent four-class task mismatch")
            arm_results[arm]={
                "final_checkpoint_sha256":digest,
                "independent_oldest_bit_correct":evaluated["correct_histories"],
                "independent_both_members_correct":evaluated["both_members_correct_pairs"],
                "independent_task_correct":evaluated["task_correct"],
                "independent_mean_causal_pair_l2":evaluated["mean_natural_pair_hidden_l2"],
                "independent_train_margin_step150":book["source_only_margin_step150"],
                "independent_train_margin_step250":book["source_only_margin_step250"],
                "gradient_surgery_active_updates":book["active_surgery_updates"],
                "model_frozen_and_read_only":True,
            }
        p=arm_results["protect"]
        sh=arm_results["sham"]
        vulnerable_pass=bool(
            p["independent_oldest_bit_correct"]>=sh["independent_oldest_bit_correct"]+8 and
            p["independent_both_members_correct"]>=sh["independent_both_members_correct"]+4 and
            p["independent_task_correct"]>=40 and
            p["independent_task_correct"]>=sh["independent_task_correct"]-3
        )
        if vulnerable_pass != source_receipt["vulnerable_rescue_rule_met"]:
            raise AssertionError("independently derived primary seed verdict mismatch")
        control_pass=(
            p["independent_oldest_bit_correct"]>=sh["independent_oldest_bit_correct"]-4
        ) if seed==109 else None
        if control_pass!=source_receipt["retention_seed_noninferiority"]:
            raise AssertionError("independent control verdict mismatch")
        summary_rows.append({
            "seed":seed,
            "original_source_initial_source_sha256":common["checkpoint_sha256"],
            "independent_sham_protect_and_baseline":arm_results,
            "vulnerable_seed_rescue":vulnerable_pass,
            "strong_retention_control_noninferiority":control_pass,
        })
        (output/f"seed{seed}.independent_replay.json").write_text(
            json.dumps(summary_rows[-1],indent=2,sort_keys=True)+"\n"
        )
    rescue=sum(r["vulnerable_seed_rescue"] for r in summary_rows if r["seed"] in (101,127,157))
    controls=next(r["strong_retention_control_noninferiority"] for r in summary_rows if r["seed"]==109)
    if rescue!=0 or not controls or registered["vulnerable_rescue_count"]!=rescue:
        raise AssertionError("independent registered trial verdict mismatch")
    verdict={
        "campaign":"PEF-05-INDEPENDENT-REPLAY",
        "source_original_registered_run":43687506414,
        "retrospective_verification_not_a_second_registered_trial":True,
        "registered_negative_decision_preserved":True,
        "trial_checkpoints_sha256_verified":True,
        "model_states_loaded_and_refitted_independently":True,
        "pair_closed_split_recomputed_from_salt":True,
        "train_only_signed_margins_recomputed_from_frozen_weights":True,
        "all_readout_correct_counts_recomputed_from_frozen_weights":True,
        "memory_guard_and_sham_norm_receipts_verified":True,
        "vulnerable_rescue_seeds":rescue,
        "retention_control_passed":controls,
        "seeds":summary_rows,
    }
    (output/"independent_summary.json").write_text(
        json.dumps(verdict,indent=2,sort_keys=True)+"\n"
    )
    print(json.dumps({
        "campaign":verdict["campaign"],"verified_original_negative":True,
        "vulnerable_rescue":rescue,"retention_control":controls,
        "seeds":[{
            "seed":r["seed"],
            "correct":{
                key:r["independent_sham_protect_and_baseline"][key]["independent_oldest_bit_correct"]
                for key in ARMS
            }
        } for r in summary_rows]
    },indent=2,sort_keys=True))
    return verdict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    audit(args.archive,args.out)


if __name__=="__main__":
    main()
