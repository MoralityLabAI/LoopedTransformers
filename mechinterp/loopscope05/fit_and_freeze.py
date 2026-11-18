"""LGA-02 stage A: source-reproduced 44K TRMs and train-only SAE dictionaries.

The protocol was committed before this file or any LGA-02 data.
All test-corpus generation belongs to a SEPARATE job after uploading the
frozen state_dict bundle and old-train/validation source receipts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from mechinterp.loopscope04.gradient_flow import (
    TinyRecursiveModel,train_one,fixed_data,model_state,
    tensor_int64_sha
)

MODEL_SEEDS=(103,211,307)
TRAIN_SEED=31041
VAL_SEED=31042
OLD_TEST_SEED=31043
OLD_SOURCE_ACCURACY={103:0.9873167506884784,
                     211:0.984712584875524,
                     307:0.9866415972355753}
SAE_WIDTH=96
K=4
TRAIN_STEPS=200
SAE_STEPS=150
SCHEMA="LGA-02/source-reproduced-trained-model-and-SAE-v1"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def protocol_sha():
    return digest((Path(__file__).parent/"PROTOCOL.md").read_bytes())


def tensor_fingerprint(state):
    import struct
    h=hashlib.sha256()
    for name in sorted(state):
        t=state[name].detach().cpu().to(torch.float32).contiguous()
        vals=t.reshape(-1).tolist()
        h.update(name.encode()+b"\0")
        h.update(str(tuple(t.shape)).encode()+b"\0")
        h.update(struct.pack("<"+"f"*len(vals),*vals))
    return h.hexdigest()


class TopKSAE(nn.Module):
    def __init__(self,d=48,width=96,k=4):
        super().__init__()
        self.encoder=nn.Linear(d,width)
        self.decoder=nn.Linear(width,d)
        self.k=k

    def encode(self,h):
        a=F.relu(self.encoder(h))
        v,idx=a.topk(self.k,dim=-1,sorted=True)
        return torch.zeros_like(a).scatter(-1,idx,v)

    def forward(self,h):
        codes=self.encode(h)
        return self.decoder(codes),codes


def hidden_tokens(model,x,depth,chunk=32):
    model.eval()
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),chunk):
            s=model.reason_states(x[i:i+chunk])[depth]
            out.append(s.detach().reshape(-1,48).cpu())
    return torch.cat(out,dim=0)


def fit_sae(train,val,init_seed,steps=SAE_STEPS):
    if train.ndim!=2 or val.ndim!=2 or train.shape[1:]!=(48,) or val.shape[1:]!=(48,):
        raise ValueError("train and validation source hidden states must have width 48")
    torch.manual_seed(init_seed)
    mu=train.mean(0)
    std=train.std(0,unbiased=False).clamp_min(1e-4)
    x=((train-mu)/std).detach()
    y=((val-mu)/std).detach()
    sae=TopKSAE()
    optimizer=torch.optim.AdamW(sae.parameters(),lr=.003,weight_decay=1e-5)
    best=float("inf")
    chosen=None
    best_step=None
    history=[]
    for step in range(1,steps+1):
        indices=torch.randint(len(x),(256,))
        xx=x[indices]
        prediction,code=sae(xx)
        loss=F.mse_loss(prediction,xx)+1e-4*code.abs().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step==1 or step%25==0 or step==steps:
            sae.eval()
            with torch.no_grad():
                err=float(F.mse_loss(sae(y)[0],y))
            history.append({"step":step,"val_normalized_MSE":err})
            if err<best:
                best=err
                chosen=model_state(sae)
                best_step=step
            sae.train()
    sae.load_state_dict(chosen,strict=True)
    sae.eval()
    return sae,mu,std,{"best_step":best_step,"best_val_MSE":best,
                       "history":history,"train_tokens":len(x),"val_tokens":len(y),
                       "sae_init_seed":init_seed,
                       "sae_tensor_sha256":tensor_fingerprint(sae.state_dict()),
                       "normalizer_sha256":tensor_fingerprint({"mu":mu,"std":std})}


def train_and_freeze(path_bundle,path_meta,smoke=False):
    torch.set_num_threads(2)
    seeds=MODEL_SEEDS[:1] if smoke else MODEL_SEEDS
    steps=8 if smoke else TRAIN_STEPS
    sae_steps=20 if smoke else SAE_STEPS
    train_x,train_y=fixed_data(64 if smoke else 512,TRAIN_SEED)
    val_x,val_y=fixed_data(32 if smoke else 128,VAL_SEED)
    old_x,old_y=fixed_data(32 if smoke else 256,OLD_TEST_SEED)
    bundle={"models":{},"saes":{},"deep":{}}
    receipts={}
    for seed in seeds:
        model,weights,optimizer=train_one(seed,"final_only",train_x,train_y,
                                          updates=steps,
                                          check_steps=(steps,))
        with torch.no_grad():
            original_acc=float((model(old_x).argmax(-1)==old_y).float().mean())
        if not smoke and abs(original_acc-OLD_SOURCE_ACCURACY[seed])>.01:
            raise ValueError("original LGA-01 trained checkpoint source reproduction failed")
        state=model_state(model)
        bundle["models"][str(seed)]=state
        rs={"original_LGA01_accuracy":OLD_SOURCE_ACCURACY[seed] if not smoke else None,
            "reproduced_accuracy":original_acc,
            "original_source_reproduction_pass":bool(smoke or
                    abs(original_acc-OLD_SOURCE_ACCURACY[seed])<=.01),
            "model_state_sha256":tensor_fingerprint(state),
            "old_training_update_step":steps,
            "optimizer_ledger":optimizer,"saes":{}}
        for depth in (1,2):
            train_state=hidden_tokens(model,train_x,depth)
            val_state=hidden_tokens(model,val_x,depth)
            init_seed=71000+10*seed+depth
            sae,mu,std,info=fit_sae(train_state,val_state,init_seed,steps=sae_steps)
            bundle["saes"][f"{seed}|{depth}"]={
                "state":model_state(sae),
                "mu":mu.detach().clone(),"std":std.detach().clone()
            }
            rs["saes"][str(depth)]=info
        receipts[str(seed)]=rs
        print("LGA02_FROZEN_TEACHER",seed,"source_accuracy",round(original_acc,6),
              "SAE_val",[round(rs["saes"][str(k)]["best_val_MSE"],5)
                         for k in (1,2)],flush=True)
    deep,_,_=train_one(103,"deep_020",train_x,train_y,
                        updates=steps,check_steps=(steps,))
    bundle["deep"]["103"]=model_state(deep)
    with torch.no_grad():
        deep_acc=float((deep(old_x).argmax(-1)==old_y).float().mean())
    if not smoke and abs(deep_acc-0.9822)>.015:
        raise ValueError("old source deep-supervision checkpoint parity failed")
    path_bundle=Path(path_bundle)
    path_bundle.parent.mkdir(parents=True,exist_ok=True)
    torch.save(bundle,path_bundle)
    raw=path_bundle.read_bytes()
    data={
        "schema":SCHEMA,"mode":"smoke" if smoke else "full",
        "protocol_sha256":protocol_sha(),
        "source_training_inputs":{
            "seed":TRAIN_SEED,"n":len(train_x),
            "x_sha256":tensor_int64_sha(train_x),
            "y_sha256":tensor_int64_sha(train_y)},
        "source_validation_inputs":{
            "seed":VAL_SEED,"n":len(val_x),
            "x_sha256":tensor_int64_sha(val_x),
            "y_sha256":tensor_int64_sha(val_y)},
        "prior_test_used_only_for_reproduction":{
            "seed":OLD_TEST_SEED,"n":len(old_x),
            "x_sha256":tensor_int64_sha(old_x),
            "y_sha256":tensor_int64_sha(old_y)},
        "new_test_generated":False,"new_test_labels_read":False,
        "teacher_init_seeds":list(seeds),
        "teacher_training_steps":steps,
        "sae_training_steps":sae_steps,
        "teacher_receipts":receipts,
        "deep_020_seed103_state_sha256":tensor_fingerprint(bundle["deep"]["103"]),
        "deep_020_old_test_accuracy":deep_acc,
        "source_bundle_sha256":digest(raw),
        "source_bundle_filename":path_bundle.name,
        "torch_version":torch.__version__
    }
    path_meta=Path(path_meta)
    path_meta.parent.mkdir(parents=True,exist_ok=True)
    path_meta.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
    print("LGA02_SOURCE_FROZEN",data["source_bundle_sha256"],flush=True)
    return data


def load_frozen(bundle_path,meta_path):
    raw=Path(bundle_path).read_bytes()
    meta=json.loads(Path(meta_path).read_bytes())
    if (meta["schema"]!=SCHEMA or meta["protocol_sha256"]!=protocol_sha()
            or digest(raw)!=meta["source_bundle_sha256"]
            or meta["new_test_generated"] is not False
            or meta["new_test_labels_read"] is not False):
        raise ValueError("original source-trained checkpoint/SAE byte lock failed")
    bundle=torch.load(bundle_path,map_location="cpu",weights_only=True)
    for seed in meta["teacher_init_seeds"]:
        sid=str(seed)
        state=bundle["models"][sid]
        if tensor_fingerprint(state)!=meta["teacher_receipts"][sid]["model_state_sha256"]:
            raise ValueError("source teacher tensor fingerprint changed")
        for depth in (1,2):
            record=bundle["saes"][f"{seed}|{depth}"]
            info=meta["teacher_receipts"][sid]["saes"][str(depth)]
            if tensor_fingerprint(record["state"])!=info["sae_tensor_sha256"]:
                raise ValueError("source SAE tensor fingerprint changed")
            if tensor_fingerprint({"mu":record["mu"],"std":record["std"]})!=info["normalizer_sha256"]:
                raise ValueError("source SAE normalizer changed")
    return meta,bundle


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bundle",required=True,type=Path)
    p.add_argument("--meta",required=True,type=Path)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    train_and_freeze(a.bundle,a.meta,smoke=a.smoke)


if __name__=="__main__":
    main()
