"""LGA-04 Stage A: gradient-matched vs value-only horizon critic calibration.

This module only generates calibration seeds 31052/31053. Fresh source
31054 is reserved for a distinct CI job after source checkpoint upload.
All original 44K teachers, train-only SAEs and LGA03 critic initializations
are byte-locked to the FIRST original source-run artifacts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from mechinterp.loopscope04.gradient_flow import fixed_data,tensor_int64_sha,per_token_ce
from mechinterp.loopscope05.fit_and_freeze import digest,load_frozen,tensor_fingerprint
from mechinterp.loopscope05.terminal_sae_test import make_model,terminal_gradients
from mechinterp.loopscope06.fit_value import load_value,HorizonValue
from mechinterp.loopscope06.eval_value import gradient_value

SCHEMA="LGA-04/calibration-only-matched-sobolev-critic-v1"
TEACHER_SHA="d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266"
TEACHER_META_SHA="7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d"
CRITIC_SHA="4d1cb63d301f6cc1b1c90fea999a82de0fb4f077b1f1f2e3bb6f0047aead937a"
CRITIC_META_SHA="267ffabb76e3a9fd7e42cf10b4eb31268d635f3bfe67a9c875ef461437ab189d"
TRAIN_SEED=31052
VAL_SEED=31053
RESERVED_TEST_SEED=31054
TEACHER_SEEDS=(103,211,307)
ARMS=("value_only_continue","value_plus_gradient")
STEPS=100
CHECKPOINTS=(0,50,100)
BATCH=8
GRADIENT_LAMBDA=1.0
OLD_SOURCE=((31041,512),(31042,128),(31043,256),(31044,256),
            (31048,768),(31049,192),(31050,256))


def protocol_sha():
    return digest((Path(__file__).resolve().parent/"PROTOCOL.md").read_bytes())


def calibration_cases(smoke=False):
    n_train=16 if smoke else 256
    n_val=8 if smoke else 64
    x,y=fixed_data(n_train,TRAIN_SEED)
    v,z=fixed_data(n_val,VAL_SEED)
    forbidden=set()
    for sd,n in OLD_SOURCE:
        past,_=fixed_data(n,sd)
        forbidden.update(tuple(row) for row in past.tolist())
    for split in (x,v):
        vals=[tuple(row) for row in split.tolist()]
        if len(vals)!=len(set(vals)) or any(row in forbidden for row in vals):
            raise ValueError("new calibration duplicated an earlier source")
        forbidden.update(vals)
    return (x,y),(v,z),{
        "train":{"n_sequences":n_train,"generator_seed":TRAIN_SEED,
                 "x_sha256":tensor_int64_sha(x),"y_sha256":tensor_int64_sha(y),
                 "x":x.tolist(),"y":y.tolist()},
        "validation":{"n_sequences":n_val,"generator_seed":VAL_SEED,
                      "x_sha256":tensor_int64_sha(v),"y_sha256":tensor_int64_sha(z),
                      "x":v.tolist(),"y":z.tolist()},
        "new_test_generated":False,"new_test_labels_accessed":False,
        "reserved_test_seed":RESERVED_TEST_SEED,
    }


def source_cache(teacher,x,y,chunk=16):
    """Original teacher's full four-loop states, terminal CE and dSUMCE/dh."""
    states=[]
    ce=[]
    gradients=[]
    for i in range(0,len(x),chunk):
        bx=x[i:i+chunk];by=y[i:i+chunk]
        hidden,gs=terminal_gradients(teacher,bx,by)
        with torch.no_grad():
            target=per_token_ce(teacher.head(hidden[4]),by)
        states.append(torch.stack(hidden[:4],dim=0).detach().cpu())
        gradients.append(torch.stack(gs,dim=0).detach().cpu())
        ce.append(target.detach().cpu())
    cache={
        "h":torch.cat(states,dim=1).float().contiguous(),
        "target":torch.cat(ce,dim=0).float().contiguous(),
        "true_grad":torch.cat(gradients,dim=1).float().contiguous(),
    }
    if cache["h"].shape!=(4,len(x),81,48) or cache["true_grad"].shape!=cache["h"].shape:
        raise ValueError("teacher-forced source derivative cache lost shape")
    return cache


def evaluate_model(net,cache,mu,std,rms,chunk=16):
    net.eval()
    token_err=0.
    derivative_err=0.
    n=0
    for k in range(4):
        for i in range(0,len(cache["target"]),chunk):
            h=cache["h"][k,i:i+chunk].detach().requires_grad_(True)
            target=cache["target"][i:i+chunk]
            true_grad=cache["true_grad"][k,i:i+chunk]
            with torch.enable_grad():
                estimate=net(h,k,mu,std)
                gp=torch.autograd.grad(estimate.sum(),h)[0]
            token_err+=float((estimate.detach()-target).square().mean())*len(h)
            derivative_err+=float(((gp.detach()-true_grad)/rms).square().mean())*len(h)
            n+=len(h)
    return {"value_token_MSE":token_err/n,
            "normalized_gradient_MSE":derivative_err/n,
            "total_calibration_horizons":n}


def fit_pair(teacher,original_critic,mu,std,train,val,seed_id,smoke=False):
    """Same start, minibatch schedule and 100 updates for two frozen arms."""
    g_scale=train["true_grad"].square().mean().sqrt().clamp_min(1e-5).detach()
    grid=[]
    saved={}
    common_start={k:v.detach().cpu().clone() for k,v in original_critic.state_dict().items()}
    steps=3 if smoke else STEPS
    checks=(0,3) if smoke else CHECKPOINTS
    for arm in ARMS:
        net=HorizonValue()
        net.load_state_dict(common_start,strict=True)
        net.train()
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        rng=torch.Generator().manual_seed(77000+seed_id)
        history=[]
        for step in range(steps+1):
            if step in checks:
                metrics=evaluate_model(net,val,mu,std,g_scale)
                history.append({"updates":step,**metrics})
            if step==steps:break
            indices=torch.randint(len(train["target"]),(BATCH,),generator=rng)
            k=int(torch.randint(0,4,(1,),generator=rng))
            h=train["h"][k,indices].detach().requires_grad_(True)
            correct=train["target"][indices]
            true_grad=train["true_grad"][k,indices]
            out=net(h,k,mu,std)
            val_loss=F.mse_loss(out,correct)
            val_loss=val_loss+.1*(out.mean(-1)-correct.mean(-1)).square().mean()
            if arm=="value_plus_gradient":
                predicted=torch.autograd.grad(out.sum(),h,create_graph=True,
                                              retain_graph=True)[0]
                loss=val_loss+GRADIENT_LAMBDA*F.mse_loss(predicted/g_scale,true_grad/g_scale)
            else:
                loss=val_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if not torch.isfinite(loss.detach()):
                raise ArithmeticError("source-only Sobolev training loss nonfinite")
        checkpoint={name:v.detach().cpu().clone() for name,v in net.state_dict().items()}
        saved[arm]=checkpoint
        grid.append({
            "arm":arm,"optimizer_updates":steps,
            "weight_sha256":tensor_fingerprint(checkpoint),
            "starting_weight_sha256":tensor_fingerprint(common_start),
            "validation_checkpoints":history,
        })
    return saved,grid,g_scale,common_start


def fit(source_bundle,source_meta,prior_critic,prior_meta,out_bundle,out_meta,smoke=False):
    torch.set_num_threads(2)
    old_raw=Path(source_bundle).read_bytes()
    oldmeta_raw=Path(source_meta).read_bytes()
    prior_raw=Path(prior_critic).read_bytes()
    prior_meta_raw=Path(prior_meta).read_bytes()
    if not smoke and (digest(old_raw)!=TEACHER_SHA or digest(oldmeta_raw)!=TEACHER_META_SHA
                      or digest(prior_raw)!=CRITIC_SHA or digest(prior_meta_raw)!=CRITIC_META_SHA):
        raise ValueError("original frozen LGA02/LGA03 checkpoint artifact SHA mismatch")
    _,frozen=load_frozen(source_bundle,source_meta)
    (tx,ty),(vx,vy),cohorts=calibration_cases(smoke)
    results={}
    bundle={}
    teacher_seeds=(103,) if smoke else TEACHER_SEEDS
    for sid in teacher_seeds:
        teacher=make_model(frozen,sid)
        original,_,mu,std,_=load_value(prior_critic,prior_meta,sid)
        original.eval().requires_grad_(False)
        train_data=source_cache(teacher,tx,ty)
        val_data=source_cache(teacher,vx,vy)
        fitted,grid,scale,start=fit_pair(teacher,original,mu,std,train_data,val_data,sid,smoke)
        cache_sha={
            "train_hidden":tensor_fingerprint({"h":train_data["h"]}),
            "train_true_terminal_gradient":tensor_fingerprint({"g":train_data["true_grad"]}),
            "train_final_CE":tensor_fingerprint({"ce":train_data["target"]}),
            "val_hidden":tensor_fingerprint({"h":val_data["h"]}),
            "val_true_terminal_gradient":tensor_fingerprint({"g":val_data["true_grad"]}),
            "val_final_CE":tensor_fingerprint({"ce":val_data["target"]})
        }
        bundle[str(sid)]={"models":fitted,"start":start,
                          "mu":mu.detach().cpu(),"std":std.detach().cpu(),
                          "grad_scale":scale.detach().cpu()}
        results[str(sid)]={
            "source_teacher_sha256":tensor_fingerprint(frozen["models"][str(sid)]),
            "original_critic_initial_sha256":tensor_fingerprint(start),
            "train_gradient_RMS":float(scale),
            "calibration_gradient_cache_sha256":cache_sha,
            "paired_training":grid,
            "fit_grad_lambda":GRADIENT_LAMBDA,
            "train_source_n":len(tx),"val_source_n":len(vx)
        }
        print("LGA04_CALIBRATED",sid,"gradient_MSE",{
            r["arm"]:round(r["validation_checkpoints"][-1]["normalized_gradient_MSE"],5)
            for r in grid},flush=True)
    out_bundle=Path(out_bundle);out_bundle.parent.mkdir(parents=True,exist_ok=True)
    torch.save(bundle,out_bundle)
    result={
        "schema":SCHEMA,"mode":"smoke" if smoke else "full",
        "protocol_sha256":protocol_sha(),
        "old_teacher_SAE_bundle_sha256":digest(old_raw),
        "old_teacher_SAE_meta_sha256":digest(oldmeta_raw),
        "old_LGA03_critic_bundle_sha256":digest(prior_raw),
        "old_LGA03_critic_receipt_sha256":digest(prior_meta_raw),
        "calibration_source":cohorts,
        "original_teacher_seeds":list(teacher_seeds),
        "training_steps":3 if smoke else STEPS,
        "training_arms":list(ARMS),
        "gradient_loss_lambda":GRADIENT_LAMBDA,
        "paired_training_receipts":results,
        "frozen_bundle_sha256":digest(out_bundle.read_bytes()),
        "test_data_opened":False,"torch":torch.__version__,
    }
    out_meta=Path(out_meta);out_meta.parent.mkdir(parents=True,exist_ok=True)
    out_meta.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("LGA04_FROZEN",result["frozen_bundle_sha256"],flush=True)
    return result


def load_pair(bundle_path,meta_path,sid):
    raw=Path(bundle_path).read_bytes()
    meta=json.loads(Path(meta_path).read_bytes())
    if (meta["schema"]!=SCHEMA or meta["protocol_sha256"]!=protocol_sha()
            or meta["frozen_bundle_sha256"]!=digest(raw)
            or meta["test_data_opened"] is not False):
        raise ValueError("paired critics changed since old-only fitting")
    data=torch.load(bundle_path,map_location="cpu",weights_only=True)
    if str(sid) not in data:raise ValueError("missing original frozen critic seed")
    saved=data[str(sid)]
    original_sha=tensor_fingerprint(saved["start"])
    row=meta["paired_training_receipts"][str(sid)]
    if original_sha!=row["original_critic_initial_sha256"]:
        raise ValueError("different starting critic for paired source arms")
    nets={}
    for arm in ARMS:
        state=saved["models"][arm]
        recorded=next(r for r in row["paired_training"] if r["arm"]==arm)
        if tensor_fingerprint(state)!=recorded["weight_sha256"] or recorded["starting_weight_sha256"]!=original_sha:
            raise ValueError("source calibration-trained critic weights replaced")
        net=HorizonValue().eval()
        net.load_state_dict(state,strict=True)
        net.requires_grad_(False)
        nets[arm]=net
    start=HorizonValue().eval()
    start.load_state_dict(saved["start"],strict=True)
    start.requires_grad_(False)
    return nets,start,saved["mu"],saved["std"],saved["grad_scale"]


def main():
    p=argparse.ArgumentParser()
    for item in ("source-bundle","source-meta","prior-critic","prior-meta","out-bundle","out-meta"):
        p.add_argument("--"+item,type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    fit(a.source_bundle,a.source_meta,a.prior_critic,a.prior_meta,a.out_bundle,a.out_meta,a.smoke)


if __name__=="__main__":
    main()
