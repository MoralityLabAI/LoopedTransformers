"""PEF-05: prospective training-only memory-gradient preservation.

The protocol was committed before this implementation. Four source seeds
branch from their identical step-125 model and AdamW optimizer state.
Only updates 126..150 are modified; sham matches removed gradient norm.
The heldout probe uses fresh within-PEF05 pair-closed suffixes and natural
two-visit (0,1) continuation. Failure is retained as research evidence.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import math
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F

from research.pef03.pef03 import (
    BitMemoryLooper, fit_ridge, hash_weights, labels, predict_ridge, tensor
)
from research.pef04.pef04 import (
    _canonical_bytes, _digest_tensor_sequence, _pair_scoring
)

SEEDS=(101,109,127,157)
VULNERABLE=(101,127,157)
CONTROL=109
ARMS=("baseline","protect","sham")
PAIR_COUNTS={"train":80,"probe":24,"heldout":24}
SALT=b"pef05-v1-pair:"
CONT=(0,1)
SECONDARY=(1,0)
TOTAL_STEPS=250
PREFIX_STEPS=125
WINDOW_START=126
WINDOW_END=150
BATCH=64
ANCHOR_PAIRS=16


def pair_split():
    suffixes=list(product((0,1),repeat=7))
    suffixes.sort(key=lambda s:(hashlib.sha256(SALT+bytes(s)).digest(),bytes(s)))
    catalog={k:[] for k in PAIR_COUNTS}
    assigned={}
    offset=0
    for split,n in PAIR_COUNTS.items():
        for suffix in suffixes[offset:offset+n]:
            assigned["".join(map(str,suffix))]=split
            catalog[split].extend(((0,*suffix),(1,*suffix)))
        offset+=n
    if offset!=128:
        raise AssertionError("missing seven-bit suffix")
    for split,rows in catalog.items():
        if len(rows)!=2*PAIR_COUNTS[split]:
            raise AssertionError("wrong source allocation")
        for a,b in zip(rows[0::2],rows[1::2]):
            if (a[0],b[0])!=(0,1) or a[1:]!=b[1:]:
                raise AssertionError("pair closure violation")
    if len(set.union(*(set(rows) for rows in catalog.values())))!=256:
        raise AssertionError("split is not disjoint and exhaustive")
    receipt={
        "campaign":"PEF-05",
        "salt":SALT.decode(),
        "history_counts":{k:len(v) for k,v in catalog.items()},
        "pair_counts":PAIR_COUNTS,
        "sorted_suffixes":["".join(map(str,s)) for s in suffixes],
        "assignment_by_suffix":assigned,
        "pair_closed":True,
        "primary_full_context_extension":[0,1],
        "old_eight_bit_universe_was_analyzed_by_PEF04":True,
    }
    receipt["sha256"]=hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    return {k:tuple(v) for k,v in catalog.items()},receipt


def natural_state(model, histories, extension=CONT):
    if histories.ndim!=2 or histories.shape[1]!=8:
        raise ValueError("model needs eight chronological bits")
    state=model.memory(histories)
    for bit in extension:
        if bit not in (0,1):
            raise ValueError("binary extension required")
        inp=torch.full((len(histories),),int(bit),dtype=torch.long)
        state=model.step(state,model.bit_embedding(inp))
    return state


def _hash_optimizer(opt):
    state=opt.state_dict()
    h=hashlib.sha256()
    for g in state["param_groups"]:
        h.update(_canonical_bytes(g))
    for index in sorted(state["state"]):
        h.update(str(index).encode())
        for name,value in sorted(state["state"][index].items()):
            h.update(name.encode())
            if isinstance(value,torch.Tensor):
                x=value.detach().cpu().contiguous()
                h.update(str((x.dtype,tuple(x.shape))).encode())
                h.update(bytes(x.reshape(-1).view(torch.uint8).tolist()))
            else:
                h.update(_canonical_bytes(value))
    return h.hexdigest()


def _flat_grad(params):
    return torch.cat([
        (p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        for p in params
    ])


def _install_flat_grad(params,values):
    i=0
    for p in params:
        n=p.numel()
        if p.grad is None:
            p.grad=torch.zeros_like(p)
        p.grad.copy_(values[i:i+n].view_as(p))
        i+=n
    if i!=len(values):
        raise AssertionError("gradient shape mismatch")


def _memory_gradient(model,anchor,pair_examples,params):
    state=natural_state(model,pair_examples,CONT)
    score=((state[1::2]-state[0::2]) @ anchor).mean()
    grads=torch.autograd.grad(score,params,allow_unused=True)
    m=torch.cat([
        (v.detach() if v is not None else torch.zeros_like(p)).reshape(-1)
        for p,v in zip(params,grads)
    ])
    return float(score.detach()),m


def project(g,m,*,arm,seed,step):
    """Double-precision orthogonal projection, verified in stored FP32.

    The original signed-gradient target and the sham norm-matching
    tolerance remain unchanged. A minimal extra positive safety
    displacement along m ensures rounding of the stored FP32 gradient
    cannot invert a zero inner product into a positive memory loss.
    """
    if arm not in ("protect","sham") or g.ndim!=1 or m.shape!=g.shape:
        raise ValueError("projection arguments invalid")
    if not torch.isfinite(g).all() or not torch.isfinite(m).all():
        raise AssertionError("nonfinite gradient")
    gd=g.detach().double()
    md=m.detach().double()
    gn=float(gd.norm())
    mn=float(md.norm())
    before=float(torch.dot(gd,md))
    base={
        "step":step,"active":False,"removed_l2":0.0,
        "gradient_before_l2":gn,"gradient_after_l2":gn,
        "memory_inner_before":before,
        "memory_inner_after":before,
        "matched_removed_gradient_norm_error":0.0,
        "projection_numeric_safety_displacement":0.0,
    }
    if gn<=1e-12 or mn<=1e-12:
        return g.clone(),base
    pm=md/mn
    c=min(gn,max(0.0,float(torch.dot(gd,pm))))
    if c<=1e-10:
        return g.clone(),base
    safety_displacement=0.0
    if arm=="protect":
        q=pm
        exact=gd-c*q
        result=exact.to(g.dtype)
        after=float(torch.dot(result.double(),md))
        if after>1e-5:
            # Additional margin only addresses rounding on the FP32
            # gradient actually handed to AdamW. It is <~1e-6 for
            # unit-clipped task gradients in normal circumstances.
            pad=max(1e-7*gn,(after+1e-5)/mn + 1e-7*gn)
            for _ in range(8):
                result=(exact-pad*pm).to(g.dtype)
                after=float(torch.dot(result.double(),md))
                if after<=1e-5:
                    safety_displacement=pad
                    break
                pad*=2
            else:
                raise AssertionError(
                    f"first-order stored gradient sign violation at step {step}: "
                    f"after={after}, ||m||={mn}, ||g||={gn}, c={c}"
                )
    else:
        rng=torch.Generator(device="cpu").manual_seed(seed*100003+step)
        noise=torch.randn(g.shape,generator=rng,dtype=torch.float64,device=g.device)
        unit_g=gd/gn
        perp=noise-torch.dot(noise,unit_g)*unit_g
        if float(perp.norm())<=1e-10:
            raise AssertionError("degenerate sham random direction")
        perp=perp/perp.norm()
        cosine=min(1.0,c/gn)
        q=cosine*unit_g+math.sqrt(max(0.0,1-cosine**2))*perp
        q=q/q.norm()
        if abs(float(torch.dot(gd,q))-c)>1e-5:
            raise AssertionError("sham projection direction mismatch")
        result=(gd-c*q).to(g.dtype)
        after=float(torch.dot(result.double(),md))
    target=math.sqrt(max(0.0,gn*gn-c*c))
    error=abs(float(result.double().norm())-target)
    if error>1e-5:
        raise AssertionError(
            f"sham/protect norm reduction incorrect at step {step}: "
            f"error={error}, safety={safety_displacement}"
        )
    if arm=="protect" and after>1e-5:
        raise AssertionError("protection failed first-order stored gradient sign test")
    return result,{
        "step":step,"active":True,"removed_l2":c,
        "gradient_before_l2":gn,"gradient_after_l2":float(result.double().norm()),
        "memory_inner_before":before,
        "memory_inner_after":after,
        "matched_removed_gradient_norm_error":error,
        "projection_numeric_safety_displacement":safety_displacement,
    }

def train_update(model,opt,examples,*,arm,seed,step,anchor=None,source_pairs=None):
    model.train()
    opt.zero_grad(set_to_none=True)
    loss=F.cross_entropy(model.logits(examples),labels(examples))
    if not torch.isfinite(loss):
        raise AssertionError("nonfinite task loss")
    loss.backward()
    params=list(model.parameters())
    norm=torch.nn.utils.clip_grad_norm_(params,1.0)
    if not torch.isfinite(norm):
        raise AssertionError("nonfinite task gradients")
    audit=None
    if arm in ("protect","sham") and WINDOW_START<=step<=WINDOW_END:
        if anchor is None or source_pairs is None:
            raise AssertionError("training-only preservation anchor missing")
        score,m=_memory_gradient(model,anchor,source_pairs,params)
        update,audit=project(_flat_grad(params),m,arm=arm,seed=seed,step=step)
        _install_flat_grad(params,update)
        audit["train_only_margin_before_update"]=score
    opt.step()
    return float(loss.detach()),audit


@torch.no_grad()
def anchor_score(model,pairs,anchor):
    h=natural_state(model,pairs,CONT)
    return float(((h[1::2]-h[0::2])@anchor).mean())


def _seal(model,opt,root,seed,arm,step,split_sha):
    name=f"seed{seed}.{arm}.step{step}.pt"
    path=root/name
    data={
        "model":model.state_dict(),"optimizer":opt.state_dict(),
        "seed":seed,"arm":arm,"step":step,"split_sha256":split_sha,
        "weights_sha256":hash_weights(model),
        "optimizer_sha256":_hash_optimizer(opt),
    }
    torch.save(data,path)
    return {
        "file":name,"checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "weights_sha256":data["weights_sha256"],
        "optimizer_sha256":data["optimizer_sha256"],
        "step":step,"arm":arm,
    }


def _verified_model(root,receipt,seed,arm,step,split_sha):
    path=root/receipt["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt["checkpoint_sha256"]:
        raise AssertionError("frozen checkpoint byte hash mismatch")
    data=torch.load(path,weights_only=True,map_location="cpu")
    if data["seed"]!=seed or data["arm"]!=arm or data["step"]!=step or data["split_sha256"]!=split_sha:
        raise AssertionError("checkpoint metadata mismatch")
    model=BitMemoryLooper().eval()
    model.load_state_dict(data["model"])
    if hash_weights(model)!=receipt["weights_sha256"]:
        raise AssertionError("source model weights altered")
    return model


@torch.no_grad()
def _evaluate(model,probe,heldout):
    arms={}
    for name,extra in (("h10_01",CONT),("h8",()),("h10_10",SECONDARY)):
        p=natural_state(model,probe,extra)
        e=natural_state(model,heldout,extra)
        ridge=fit_ridge(p,probe[:,0],penalty=1.0)
        pred=predict_ridge(e,ridge)
        arms[name]={
            **_pair_scoring(pred,heldout[:,0],e),
            "frozen_linear_head_sha256":_digest_tensor_sequence(ridge),
        }
    arms["original_task_correct_out_of_48"]=int(
        (model.logits(heldout).argmax(1)==labels(heldout)).sum())
    return arms


def run(root:Path,*,seeds=SEEDS,prefix_steps=PREFIX_STEPS,
        total_steps=TOTAL_STEPS):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not (0<prefix_steps<total_steps) or len(seeds)!=len(set(seeds)):
        raise ValueError("invalid run configuration")
    if any(seed not in SEEDS for seed in seeds):
        raise ValueError("unsupported seed")
    root.mkdir(parents=True,exist_ok=True)
    catalog,split=pair_split()
    (root/"split_manifest.json").write_bytes(_canonical_bytes(split))
    train,probe,heldout=[tensor(catalog[x]) for x in ("train","probe","heldout")]
    anchor_examples=train[:2*ANCHOR_PAIRS]
    results=[]
    all_frozen=[]
    for seed in seeds:
        torch.manual_seed(seed)
        model=BitMemoryLooper().train()
        initial_weight_sha=hash_weights(model)
        opt=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0)
        g=torch.Generator(device="cpu").manual_seed(seed*7919+59)
        batches=torch.randint(len(train),(total_steps,BATCH),generator=g)
        batch_sha=hashlib.sha256(
            bytes(batches.to(torch.uint8).flatten().tolist())).hexdigest()
        for step in range(1,prefix_steps+1):
            train_update(model,opt,train[batches[step-1]],
                         arm="baseline",seed=seed,step=step)
        pre_weights=hash_weights(model)
        pre_optimizer=_hash_optimizer(opt)
        common=_seal(model,opt,root,seed,"common",prefix_steps,split["sha256"])
        with torch.no_grad():
            states=natural_state(model,train,CONT)
            signed=(states[1::2]-states[0::2]).mean(dim=0)
            magnitude=float(signed.norm())
            if magnitude<1e-10:
                raise AssertionError("source-only anchor degenerate")
            anchor=(signed/signed.norm()).detach()
        anchor_sha=_digest_tensor_sequence((anchor,))
        margin_start=anchor_score(model,anchor_examples,anchor)
        arms={}
        for arm in ARMS:
            fork=deepcopy(model)
            optimizer=torch.optim.AdamW(fork.parameters(),lr=0.003,weight_decay=0)
            optimizer.load_state_dict(deepcopy(opt.state_dict()))
            if hash_weights(fork)!=pre_weights or _hash_optimizer(optimizer)!=pre_optimizer:
                raise AssertionError("three branches started with different weights/moments")
            surgery=[]
            intermediate=None
            loss_end=0.0
            mid_margin=None
            for step in range(prefix_steps+1,total_steps+1):
                loss_end,entry=train_update(
                    fork,optimizer,train[batches[step-1]],
                    arm=arm,seed=seed,step=step,
                    anchor=anchor,source_pairs=anchor_examples)
                if entry is not None:
                    surgery.append(entry)
                if step==WINDOW_END:
                    mid_margin=anchor_score(fork,anchor_examples,anchor)
                    intermediate=_seal(fork,optimizer,root,seed,arm,step,split["sha256"])
            if intermediate is None and total_steps>=WINDOW_END:
                raise AssertionError("missing step-150 checkpoint")
            end_margin=anchor_score(fork,anchor_examples,anchor)
            final=_seal(fork,optimizer,root,seed,arm,total_steps,split["sha256"])
            arms[arm]={
                "arm":arm,
                "common_weight_sha256":pre_weights,
                "common_optimizer_sha256":pre_optimizer,
                "batch_stream_sha256":batch_sha,
                "intermediate":intermediate,
                "final":final,
                "source_only_margin_step125":margin_start,
                "source_only_margin_step150":mid_margin,
                "source_only_margin_step250":end_margin,
                "terminal_training_loss":loss_end,
                "surgery":surgery,
                "active_surgery_updates":sum(x["active"] for x in surgery),
            }
            print("PEF05_SEALED",seed,arm,final["weights_sha256"],flush=True)
        # Important: freeze all branches of this seed before any probe/head
        # fitting or heldout source evaluation.
        ledger={
            "seed":seed,"source_split_sha256":split["sha256"],
            "original_initial_weights_sha256":initial_weight_sha,
            "common":common,
            "common_weights_sha256":pre_weights,
            "common_optimizer_sha256":pre_optimizer,
            "train_only_anchor_l2":magnitude,
            "train_only_anchor_sha256":anchor_sha,
            "supervised_batch_stream_sha256":batch_sha,
            "all_three_arms_frozen_before_downstream_probe":True,
            "arms":arms,
        }
        (root/f"seed{seed}.sealed_ledger.json").write_bytes(_canonical_bytes(ledger))
        all_frozen.append(ledger)
        scored={}
        for arm in ARMS:
            frozen=arms[arm]["final"]
            check=_verified_model(root,frozen,seed,arm,total_steps,split["sha256"])
            before=hash_weights(check)
            scored[arm]=_evaluate(check,probe,heldout)
            if hash_weights(check)!=before:
                raise AssertionError("probe mutated a frozen model")
            if hashlib.sha256((root/frozen["file"]).read_bytes()).hexdigest()!=frozen["checkpoint_sha256"]:
                raise AssertionError("probe changed sealed checkpoint bytes")
        torch.manual_seed(seed)
        own_init=BitMemoryLooper().eval()
        if hash_weights(own_init)!=initial_weight_sha:
            raise AssertionError("seed initializer not reproducible")
        precursor=_verified_model(root,common,seed,"common",prefix_steps,split["sha256"])
        controls={"initial":_evaluate(own_init,probe,heldout),
                  "common_before_intervention":_evaluate(precursor,probe,heldout)}
        pp=scored["protect"]["h10_01"]
        sh=scored["sham"]["h10_01"]
        pt=scored["protect"]["original_task_correct_out_of_48"]
        st=scored["sham"]["original_task_correct_out_of_48"]
        rescue=bool(
            pp["correct_histories"]>=sh["correct_histories"]+8 and
            pp["both_members_correct_pairs"]>=sh["both_members_correct_pairs"]+4 and
            pt>=40 and pt>=st-3)
        srecord={
            "seed":seed,"arms":scored,
            "controls":controls,
            "vulnerable_rescue_rule_met":rescue,
            "retention_seed_noninferiority":(
                pp["correct_histories"]>=sh["correct_histories"]-4
                if seed==CONTROL else None),
            "all_source_checkpoints_frozen_before_test":True,
            "source_only_correct_out_of_48":24,
            "source_only_joint_pairs_out_of_24":0,
            "heldout_history_count":48,
            "paired_counterfactual_count":24,
            "train_only_anchor_sha256":anchor_sha,
        }
        (root/f"seed{seed}.evaluation.json").write_bytes(_canonical_bytes(srecord))
        results.append(srecord)
    n=sum(r["vulnerable_rescue_rule_met"] for r in results if r["seed"] in VULNERABLE)
    control=next((r["retention_seed_noninferiority"] for r in results if r["seed"]==CONTROL),None)
    registered=(tuple(seeds)==SEEDS and prefix_steps==PREFIX_STEPS
                and total_steps==TOTAL_STEPS)
    decision=(
        "criterion_met" if n>=2 and control is True else "criterion_not_met"
    ) if registered else "unregistered_debug"
    summary={
        "campaign":"PEF-05",
        "registered_scoring_run":registered,
        "registered_decision":decision,
        "registered_seed_order":list(SEEDS),
        "seeds_run":list(seeds),
        "training_steps":total_steps,
        "source_split_sha256":split["sha256"],
        "primary_arm":"natural_h10_01",
        "retrospectively_selected_training_window_updates":[WINDOW_START,WINDOW_END],
        "vulnerable_rescue_count":n,
        "retention_control_noninferiority":control,
        "all_seeds_forked_before_probe":True,
        "older_PEF04_checkpoints_were_not_reused":True,
        "results":results,
    }
    (root/"summary.json").write_bytes(_canonical_bytes(summary))
    print(json.dumps({
        "campaign":"PEF-05","registered":registered,
        "decision":decision,"rescue_count":n,"retention_control":control,
        "seeds":[{
            "seed":r["seed"],
            "correct":{a:r["arms"][a]["h10_01"]["correct_histories"] for a in ARMS},
            "pairs":{a:r["arms"][a]["h10_01"]["both_members_correct_pairs"] for a in ARMS},
            "task":{a:r["arms"][a]["original_task_correct_out_of_48"] for a in ARMS},
        } for r in results],
    },indent=2,sort_keys=True),flush=True)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seed",type=int,choices=SEEDS)
    parser.add_argument("--debug-prefix",type=int,default=PREFIX_STEPS)
    parser.add_argument("--debug-total",type=int,default=TOTAL_STEPS)
    args=parser.parse_args(argv)
    run(args.out,seeds=SEEDS if args.seed is None else (args.seed,),
        prefix_steps=args.debug_prefix,total_steps=args.debug_total)


if __name__=="__main__":
    main()
