"""NW-01 original three-seed CPU recurrent-witness pilot.

Development training uses independently synthesized public roots; final readout
uses the original SHA-locked NOR42 first 64 source pairs exactly once. Source
world pairing, sham matching and intervention selection are fixed in
REGISTRATION.md. No native program truth is inferred from public logits here.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import statistics
import sys
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.nor42_recurrent_witness_v0 import data as src
from experiments.nor42_recurrent_witness_v0.model import (
    RecurrentWitnessVerifier,root_paired_public,root_paired_targets,
    matched_world_donor,STEPS,LATENT,witness_vectors
)

TRAIN_SEEDS=(11,23,47)
UPDATES=200
BATCH_ROOTS=32
TRAIN_ROOTS=512
LR=0.002
WEIGHT_DECAY=0.0001
READOUT_DEPTHS=(1,2,4,8)
PATCH_DEPTHS=(1,2,4,6)
T_CRIT_15=2.131449545559323
SHAM_SEED=202611420203


def sha(path:Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_metric(pred:torch.Tensor,truth:torch.Tensor):
    p=pred.flatten()
    correct=((p>=.5).float()==truth.float()).float()
    return {"accuracy":float(correct.mean().item()),
            "brier":float(((p-truth)**2).mean().item()),
            "mean_confidence":float(p.mean().item())}


def training_batch(feats:torch.Tensor,betas:torch.Tensor,
                   g:torch.Generator):
    """32 roots, two worlds and two info conditions per root, balanced."""
    idx=torch.randint(feats.shape[0],(BATCH_ROOTS,),generator=g)
    x=feats[idx].repeat_interleave(4,dim=0)
    beta=betas[idx]
    witness=torch.zeros((BATCH_ROOTS*4,3),dtype=torch.float32)
    witness[2::4,0]=1
    witness[3::4,0]=1
    witness[3::4,1]=beta/2
    witness[3::4,2]=beta.abs()/2
    y=torch.tensor([0.,1.,0.,1.]*BATCH_ROOTS)
    return x,witness,y


def train_one(seed:int,features:torch.Tensor,betas:torch.Tensor):
    torch.manual_seed(seed)
    model=RecurrentWitnessVerifier(src.PUBLIC_DIM,LATENT,STEPS)
    model.train()
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    rng=torch.Generator().manual_seed(seed+390002)
    losses=[]
    for t in range(UPDATES):
        x,w,y=training_batch(features,betas,rng)
        optimizer.zero_grad(set_to_none=True)
        logits=model(x,w).logits[:,-1]
        loss=nn.functional.binary_cross_entropy_with_logits(logits,y)
        loss.backward()
        optimizer.step()
        if t in (0,19,49,99,149,199):
            losses.append({"update":t+1,"loss":float(loss.detach().item())})
    model.eval()
    return model,losses


def sham_witness(roots:list[src.PublicRoot]):
    """Matched per-root sham, same false signal for clean AND hidden world."""
    tensor=torch.zeros((len(roots)*2,3),dtype=torch.float32)
    tensor[:,0]=1
    for i,root in enumerate(roots):
        n=int.from_bytes(hashlib.sha256(
            f"{SHAM_SEED}|{root.identity}".encode()).digest(),"big")
        val=(-2,-1,0,1,2)[n%5]
        tensor[2*i:2*i+2,1]=val/2
        tensor[2*i:2*i+2,2]=abs(val)/2
    return tensor


def interval_from_blocks(records,field:str):
    block=[statistics.mean(r[field] for r in records
                           if r["replicate"]==rep) for rep in src.REPS]
    assert len(block)==16
    m=statistics.mean(block)
    stdev=statistics.stdev(block)
    half=T_CRIT_15*stdev/math.sqrt(16)
    return {"mean":m,"descriptive_t95":[m-half,m+half],
            "positive_blocks":sum(v>0 for v in block),
            "replicate_block_means":block}


@torch.no_grad()
def evaluate_seed(model:RecurrentWitnessVerifier,seed:int,
                  roots:list[src.PublicRoot],outdir:Path):
    n=len(roots)
    pub=torch.tensor([r.features for r in roots],dtype=torch.float32)
    betas=torch.tensor([r.hidden_beta for r in roots],dtype=torch.float32)
    pair_x=root_paired_public(pub)
    y=root_paired_targets(n)
    w_public=witness_vectors(betas,"public")
    w_auth=witness_vectors(betas,"authentic")
    w_sham=sham_witness(roots)
    traces={
        mode:model(pair_x,w) for mode,w in
        (("public",w_public),("authenticated",w_auth),("sham",w_sham))
    }
    metrics={}
    probabilities={}
    for mode,trace in traces.items():
        probs=trace.logits.sigmoid()
        probabilities[mode]=probs
        metrics[mode]={str(depth):paired_metric(probs[:,depth-1],y)
                       for depth in READOUT_DEPTHS}
    # A hidden world and a clean world have identical PUBLIC or SHAM
    # input, so identical trajectories are required at every loop.
    for mode in ("public","sham"):
        tr=traces[mode]
        assert torch.max((tr.states[0::2]-tr.states[1::2]).abs()).item()<=1e-7
        assert torch.max((tr.logits[0::2]-tr.logits[1::2]).abs()).item()<=1e-7
        assert metrics[mode]["8"]["accuracy"]==.5
        assert metrics[mode]["8"]["brier"]>=.25-1e-7
    matched_input_parity={
        mode:{
            "max_recurrent_state_difference":float(
                (traces[mode].states[0::2]-traces[mode].states[1::2]).abs().max().item()),
            "max_readout_logit_difference":float(
                (traces[mode].logits[0::2]-traces[mode].logits[1::2]).abs().max().item())
        } for mode in ("public","sham")
    }
    authentic=traces["authenticated"]
    logits=authentic.logits[:,-1].detach()
    state_distances={}
    for mode,trace in traces.items():
        dif=trace.states[1::2]-trace.states[0::2]
        norms=torch.linalg.vector_norm(dif,dim=2)
        state_distances[mode]={
            str(depth):float(norms[:,depth-1].mean().item())
            for depth in READOUT_DEPTHS
        }
    donor_results={}
    norm_sham_doses={}
    for depth in PATCH_DEPTHS:
        k=depth-1
        states=authentic.states[:,k,:].detach()
        donors=matched_world_donor(states)
        patch=model(pair_x,w_auth,patch_after=k,donor_state=donors)
        same=model(pair_x,w_auth,patch_after=k,donor_state=states)
        donor_logits=logits.reshape(n,2).flip(1).reshape(-1)
        actual_patch=patch.logits[:,-1]
        assert torch.max((actual_patch-donor_logits).abs()).item()<1e-5
        assert torch.max((same.logits[:,-1]-logits).abs()).item()<1e-6
        gap=torch.linalg.vector_norm(donors-states,dim=1,keepdim=True)
        rng=torch.Generator().manual_seed(SHAM_SEED+seed*17+depth*113)
        random_direction=torch.randn(
            states.shape,generator=rng,dtype=states.dtype)
        random_direction/=torch.linalg.vector_norm(
            random_direction,dim=1,keepdim=True).clamp_min(1e-10)
        sham_state=states+random_direction*gap
        matched_sham=model(pair_x,w_auth,patch_after=k,donor_state=sham_state)
        target_change=(actual_patch-logits).abs()
        sham_change=(matched_sham.logits[:,-1]-logits).abs()
        donor_results[str(depth)]={
            "mean_abs_donor_logit_effect":float(target_change.mean().item()),
            "mean_abs_norm_matched_random_effect":float(sham_change.mean().item()),
            "mean_donor_minus_random":float(
                (target_change-sham_change).mean().item()),
            "max_donor_trace_error":float(
                (actual_patch-donor_logits).abs().max().item()),
            "max_same_world_identity_error":float(
                (same.logits[:,-1]-logits).abs().max().item())
        }
        norm_sham_doses[str(depth)]={
            "mean_state_l2_dose":float(gap.mean().item()),
            "matched_random_l2_dose":float(
                torch.linalg.vector_norm(sham_state-states,dim=1).mean().item())
        }
        # Record each root's paired mean donor/random effect.
        if depth==4:
            donor_by_root=target_change.reshape(n,2).mean(1).numpy()
            sham_by_root=sham_change.reshape(n,2).mean(1).numpy()
    final_probs=probabilities["authenticated"][:,-1]
    separations=(final_probs[1::2]-final_probs[0::2]).numpy()
    per_root=[]
    for i,root in enumerate(roots):
        metrics_root={
            "root_id":root.identity,"replicate":root.replicate,
            "degree":root.degree,"family":root.family,
            "public_probability":float(probabilities["public"][2*i,-1].item()),
            "auth_clean_probability":float(final_probs[2*i].item()),
            "auth_hidden_probability":float(final_probs[2*i+1].item()),
            "auth_pair_probability_delta":float(separations[i]),
            "sham_probability":float(probabilities["sham"][2*i,-1].item()),
            "auth_state_distance_pass8":float(
                torch.linalg.vector_norm(
                    authentic.states[2*i+1,-1]-authentic.states[2*i,-1]).item()),
            "patch4_abs_donor_effect":float(donor_by_root[i]),
            "patch4_abs_random_effect":float(sham_by_root[i]),
            "patch4_donor_minus_random":float(donor_by_root[i]-sham_by_root[i])
        }
        per_root.append(metrics_root)
    root_summary={
        "paired_probability_separation":interval_from_blocks(
            per_root,"auth_pair_probability_delta"),
        "patch4_donor_minus_random":interval_from_blocks(
            per_root,"patch4_donor_minus_random"),
        "public_probability_deviation":interval_from_blocks(
            [{**r,"public_abs_deviation":abs(r["public_probability"]-.5)}
             for r in per_root],"public_abs_deviation")
    }
    saved=outdir/f"nw01_seed{seed}_activations.npz"
    np.savez_compressed(saved,
        public_logits=traces["public"].logits.detach().numpy(),
        auth_logits=authentic.logits.detach().numpy(),
        sham_logits=traces["sham"].logits.detach().numpy(),
        public_states=traces["public"].states.detach().numpy(),
        auth_states=authentic.states.detach().numpy(),
        sham_states=traces["sham"].states.detach().numpy(),
        root_ids=np.array([r.identity for r in roots],dtype="U64"),
        roots=np.array([r.replicate for r in roots],dtype=np.int16))
    return {
        "training_seed":seed,
        "depth_outcomes":metrics,
        "matched_pair_numeric_firewall":matched_input_parity,
        "state_distances":state_distances,
        "causal_donor_swaps":donor_results,
        "intervention_doses":norm_sham_doses,
        "source_block_descriptives":root_summary,
        "per_root":per_root,
        "raw_activation_file":saved.name,
        "raw_activation_sha256":sha(saved),
        "raw_activation_bytes":saved.stat().st_size
    }


def run(archive:Path,out:Path):
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=True)
    roots=src.validate_archive(archive)
    fit=src.synthetic_public_roots(TRAIN_ROOTS)
    if len(set(r.identity for r in roots)&set(r.identity for r in fit)):
        raise ValueError("train/test identity overlap")
    train_x=torch.tensor([r.features for r in fit],dtype=torch.float32)
    train_beta=torch.tensor([r.hidden_beta for r in fit],dtype=torch.float32)
    by_seed=[]
    for seed in TRAIN_SEEDS:
        model,losses=train_one(seed,train_x,train_beta)
        one=evaluate_seed(model,seed,roots,out)
        one["training_loss_snapshots"]=losses
        ckpt=out/f"nw01_seed{seed}_weights.pt"
        torch.save(model.state_dict(),ckpt)
        one["checkpoint_file"]=ckpt.name
        one["checkpoint_sha256"]=sha(ckpt)
        by_seed.append(one)
        print(f"NW01 seed={seed} public={one['depth_outcomes']['public']['8']['accuracy']:.3f} "
              f"witnessed={one['depth_outcomes']['authenticated']['8']['accuracy']:.3f} "
              f"sham={one['depth_outcomes']['sham']['8']['accuracy']:.3f} "
              f"patch4={one['causal_donor_swaps']['4']['mean_donor_minus_random']:+.4f}",
              flush=True)
    g1=all(
        seed["depth_outcomes"]["public"]["8"]["accuracy"]==.5
        and seed["depth_outcomes"]["sham"]["8"]["accuracy"]==.5
        and all(seed["state_distances"]["public"][str(d)]==0
                and seed["state_distances"]["sham"][str(d)]==0
                for d in READOUT_DEPTHS)
        for seed in by_seed)
    g2=sum(s["depth_outcomes"]["authenticated"]["8"]["accuracy"]>=.95
           and s["depth_outcomes"]["authenticated"]["8"]["brier"]<=.05
           for s in by_seed)>=2
    g3=sum(s["state_distances"]["authenticated"]["8"]>.25 and
           s["source_block_descriptives"]["paired_probability_separation"]["mean"]>.60
           for s in by_seed)>=2
    g4=sum(s["causal_donor_swaps"]["4"]["mean_abs_donor_logit_effect"]>=.25
           and s["causal_donor_swaps"]["4"]["mean_donor_minus_random"]>=.10
           and all(z["max_donor_trace_error"]<1e-5
                   and z["max_same_world_identity_error"]<1e-6
                   for z in s["causal_donor_swaps"].values())
           for s in by_seed)>=2
    files=[Path(__file__),Path(src.__file__),Path(sys.modules[
        "experiments.nor42_recurrent_witness_v0.model"].__file__)]
    source_files={p.name:sha(p) for p in files}
    g5=len(roots)==64 and len({r.identity for r in roots})==64 and (
        set(r.replicate for r in roots)==set(src.REPS)) and len(by_seed)==3 and all(
        (out/s["raw_activation_file"]).exists()
        and sha(out/s["raw_activation_file"])==s["raw_activation_sha256"]
        for s in by_seed)
    gates={
        "paired_public_and_sham_information_firewall":g1,
        "authenticated_witness_readout_reachability":g2,
        "recurrent_witness_state_retention":g3,
        "full_state_donor_specificity_vs_equal_norm_sham":g4,
        "source_model_and_evalmath_root_integrity":g5
    }
    summary={
        "schema":"nw01.original_three_seed_cpu_paired_witness_v1",
        "registration_first_commit":"42983fba14f3a616e9697523bce8a533fa8c6a84",
        "evidence_level":"retrospective neural model pilot on publicly released NOR42 roots",
        "source_first_sha256":src.SOURCE_SHA,
        "first_grade_sha256":src.GRADE_SHA,
        "original_nor42_zip_sha256":src.ORIGINAL_ZIP_SHA,
        "source_commit":src.ORIGINAL_COMMIT,
        "source_units":64,"replicate_blocks":16,
        "paired_true_world_rows":128,
        "synthetic_train_roots":TRAIN_ROOTS,
        "training_seeds":list(TRAIN_SEEDS),
        "train_updates_per_seed":UPDATES,
        "batch_roots_per_update":BATCH_ROOTS,
        "model_class":"tied_weight_GRUCell_recurrent_verifier",
        "recurrent_steps":STEPS,"latent_dim":LATENT,
        "public_feature_dim":src.PUBLIC_DIM,
        "public_only_feature_roster_sha256":src.feature_manifest(roots),
        "model_source_file_sha256":source_files,
        "by_seed":by_seed,
        "registered_five_gates":gates,
        "pilot_pass":all(gates.values()),
        "limitations":"No full RRC25 neural clean-continuation learning. Synthetic witness delta is constructed from preexisting original NOR42 beta, not obtained by external authenticated oracle. Matched state swap is full-state causal transport, not BPD component identification. 64 sources share 16 generator blocks."
    }
    raw=(json.dumps(summary,sort_keys=True,indent=2)+"\n").encode()
    path=out/"nw01_original_pilot.json"
    path.write_bytes(raw)
    print("NW01 ORIGINAL THREE-SEED SCIENCE PILOT",summary["pilot_pass"])
    print("NW01 REGISTERED GATES",gates)
    print("NW01 ORIGINAL RECEIPT",len(raw),hashlib.sha256(raw).hexdigest())
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    run(args.archive,args.out)

if __name__=="__main__":
    main()
