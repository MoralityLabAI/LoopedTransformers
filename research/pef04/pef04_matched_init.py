"""PEF-04B: posthoc own-initialization versus trained memory replay.

Uses only original PEF-04 run 43683120934 checkpoints and manifests.
Checks reconstructed initial parameter hashes against the original
pre-evaluation training ledger. Never retrains any scored model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from research.pef03.pef03 import BitMemoryLooper, fit_ridge, predict_ridge, hash_weights, load_frozen, tensor
from research.pef04.pef04 import SEEDS, CONTINUATIONS, natural_memory, paired_split, _pair_scoring, _canonical_bytes


@torch.no_grad()
def arm_score(model, tr, te, extension):
    train_state=natural_memory(model,tr,extension)
    test_state=natural_memory(model,te,extension)
    head=fit_ridge(train_state,tr[:,0],penalty=1.0)
    pred=predict_ridge(test_state,head)
    base=_pair_scoring(pred,te[:,0],test_state)
    d=test_state[1::2]-test_state[0::2]
    mean=d.norm(dim=1).mean()
    base["relative_oldest_pair_l2"]=float(mean/test_state.norm(dim=1).mean().clamp_min(1e-12))
    base["oldest_bit_difference_direction_coherence"]=float(
        d.mean(dim=0).norm()/mean.clamp_min(1e-12))
    return base


@torch.no_grad()
def natural_trajectory(model, bits):
    state=model.memory_init.unsqueeze(0).expand(len(bits),-1)
    rows=[]
    for visit in range(10):
        token=(bits[:,visit] if visit < 8 else
               torch.zeros((len(bits),),dtype=torch.long))
        state=model.step(state,model.bit_embedding(token))
        delta=state[1::2]-state[0::2]
        rows.append({
            "visit":visit+1,
            "mean_oldest_bit_pair_l2":float(delta.norm(dim=1).mean()),
            "mean_state_l2":float(state.norm(dim=1).mean()),
            "pair_difference_direction_coherence":float(
                delta.mean(dim=0).norm()/delta.norm(dim=1).mean().clamp_min(1e-12))
        })
    return rows


def replay(source:Path,output:Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    summary=json.loads((source/"summary.json").read_text())
    ledger=json.loads((source/"sealed_training_ledger.json").read_text())
    saved_split=json.loads((source/"split_manifest.json").read_text())
    catalog,split=paired_split()
    if summary["seed_order"] != list(SEEDS) or not summary["registered_scoring_run"]:
        raise AssertionError("unexpected preregistered source cohort")
    if summary["registered_transfer_decision"] != "criterion_not_met":
        raise AssertionError("original decision mismatch")
    if saved_split != split or ledger["split_manifest_sha256"] != split["split_sha256"]:
        raise AssertionError("source split changed")
    by_seed={r["seed"]:r for r in ledger["training"]}
    by_result={r["seed"]:r for r in summary["result_per_seed"]}
    if set(by_seed)!=set(SEEDS) or set(by_result)!=set(SEEDS):
        raise AssertionError("missing source checkpoint")
    probe,heldout=tensor(catalog["probe"]),tensor(catalog["heldout"])
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in SEEDS:
        frozen=by_seed[seed]
        ck=source/frozen["checkpoint_file"]
        if hashlib.sha256(ck.read_bytes()).hexdigest()!=frozen["checkpoint_sha256"]:
            raise AssertionError("frozen checkpoint digest mismatch")
        torch.manual_seed(seed)
        initial=BitMemoryLooper().eval()
        initial_sha=hash_weights(initial)
        if initial_sha!=frozen["initial_weights_sha256"]:
            raise AssertionError("same-seed initial weights do not match original receipt")
        trained=load_frozen(ck,seed)
        trained_sha=hash_weights(trained)
        if trained_sha!=frozen["frozen_weights_sha256"]:
            raise AssertionError("trained weights do not match receipt")
        arms={}
        with torch.no_grad():
            for arm,continuation in CONTINUATIONS.items():
                init=arm_score(initial,probe,heldout,continuation)
                final=arm_score(trained,probe,heldout,continuation)
                original=by_result[seed]["downstream_arms"][arm]["source_frozen_trunk"]
                for key in ("correct_histories","both_members_correct_pairs",
                            "prediction_changes_across_oldest_bit_flip_pairs"):
                    if final[key]!=original[key]:
                        raise AssertionError(f"source {seed} {arm} {key} changed")
                if abs(final["mean_natural_pair_hidden_l2"]-
                       original["mean_natural_pair_hidden_l2"])>1e-5:
                    raise AssertionError("source causal pair distances differ")
                arms[arm]={
                    "own_initialization":init,
                    "same_weights_after_training":final,
                    "change_in_correct_histories":final["correct_histories"]-init["correct_histories"],
                    "registered_trained_result_replayed":True,
                }
            init_trace=natural_trajectory(initial,heldout)
            train_trace=natural_trajectory(trained,heldout)
        for label,trace,armtype in (
            ("initial",init_trace,"own_initialization"),
            ("trained",train_trace,"same_weights_after_training")
        ):
            reference=arms["h10_00"][armtype]["mean_natural_pair_hidden_l2"]
            if abs(trace[-1]["mean_oldest_bit_pair_l2"]-reference)>1e-5:
                raise AssertionError(f"{label} causal trajectory inconsistent")
        if hash_weights(initial)!=initial_sha or hash_weights(trained)!=trained_sha:
            raise AssertionError("evaluation changed model weights")
        if hashlib.sha256(ck.read_bytes()).hexdigest()!=frozen["checkpoint_sha256"]:
            raise AssertionError("evaluation changed checkpoint")
        row={
            "campaign":"PEF-04B",
            "seed":seed,
            "posthoc":True,
            "original_frozen_checkpoint_sha256":frozen["checkpoint_sha256"],
            "initial_weight_sha256_verified":initial_sha,
            "trained_weight_sha256_verified":trained_sha,
            "pair_closure_and_source_split_verified":True,
            "no_trunk_retraining":True,
            "arms":arms,
            "natural_oldest_bit_distance_by_visit":{"initial":init_trace,"trained":train_trace},
        }
        (output/f"seed{seed}.matched.json").write_bytes(_canonical_bytes(row))
        rows.append(row)
    report={
        "campaign":"PEF-04B",
        "posthoc_after_original_negative_result":True,
        "registered_source_run_id":43683120934,
        "original_decision_preserved":"criterion_not_met",
        "all_initial_weight_hashes_verified":True,
        "all_original_trained_scores_replayed":True,
        "no_trunk_retraining":True,
        "seed_results":[{
            "seed":r["seed"],
            "initial_correct":r["arms"]["h10_00"]["own_initialization"]["correct_histories"],
            "trained_correct":r["arms"]["h10_00"]["same_weights_after_training"]["correct_histories"],
            "initial_joint_pairs":r["arms"]["h10_00"]["own_initialization"]["both_members_correct_pairs"],
            "trained_joint_pairs":r["arms"]["h10_00"]["same_weights_after_training"]["both_members_correct_pairs"],
            "initial_hidden_pair_l2":r["arms"]["h10_00"]["own_initialization"]["mean_natural_pair_hidden_l2"],
            "trained_hidden_pair_l2":r["arms"]["h10_00"]["same_weights_after_training"]["mean_natural_pair_hidden_l2"]
        } for r in rows],
    }
    (output/"matched_summary.json").write_bytes(_canonical_bytes(report))
    print(json.dumps(report,indent=2,sort_keys=True))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    replay(args.source,args.out)


if __name__=="__main__":
    main()
