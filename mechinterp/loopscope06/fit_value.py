"""LGA-03: fit horizon-conditioned terminal-loss value potentials on OLD calibration.

Original tied teachers and train-only TopK4 SAE dictionaries are immutable
from LGA-02 original Actions run 44222863870. All 31050 heldout test data
is generated only in a later workflow job after this artifact is uploaded.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from mechinterp.loopscope04.gradient_flow import (
    fixed_data,per_token_ce,tensor_int64_sha
)
from mechinterp.loopscope05.fit_and_freeze import (
    digest,load_frozen,tensor_fingerprint
)
from mechinterp.loopscope05.terminal_sae_test import make_model

SCHEMA="LGA-03/calibration-horizon-value-potential-v1"
SOURCE_BUNDLE_SHA="d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266"
SOURCE_META_SHA="7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d"
TRAIN_SEED=31048
VAL_SEED=31049
TEST_SEED=31050
TRAIN_N=768
VAL_N=192
SEEDS=(103,211,307)
UPDATES=300
CHECKPOINTS=(0,50,100,150,200,250,300)


def protocol_sha():
    return digest((Path(__file__).resolve().parent/"PROTOCOL.md").read_bytes())


class HorizonValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth=nn.Embedding(4,8)
        self.c1=nn.Conv1d(56,96,3,padding=1,padding_mode="circular")
        self.c2=nn.Conv1d(96,48,3,padding=1,padding_mode="circular")
        self.c3=nn.Conv1d(48,1,1)
        self.activation=nn.Softplus()

    def forward(self,h,k,mu,std):
        if h.ndim!=3 or h.shape[1:]!=(81,48):
            raise ValueError("value model requires full [batch,81,48] physical state")
        if not 0<=int(k)<=3:
            raise ValueError("value horizon must be current loop k 0..3")
        n=h.shape[0]
        embed=self.depth(torch.full((n,81),int(k),device=h.device,dtype=torch.long))
        normalized=(h-mu)/std
        x=torch.cat((normalized,embed),dim=-1).transpose(1,2)
        x=F.gelu(self.c1(x))
        x=F.gelu(self.c2(x))
        return self.activation(self.c3(x).transpose(1,2).squeeze(-1))


def source_cases(n,seed):
    return fixed_data(n,seed)


def calibration_data():
    tx,ty=source_cases(TRAIN_N,TRAIN_SEED)
    vx,vy=source_cases(VAL_N,VAL_SEED)
    # Check every old source independently; current new test seed 31050 is not opened here.
    used=set()
    for oldseed,n in ((31041,512),(31042,128),(31043,256),(31044,256)):
        previous,_=fixed_data(n,oldseed)
        for v in previous.tolist():
            used.add(tuple(v))
    for x in (tx,vx):
        if len({tuple(r) for r in x.tolist()})!=len(x):
            raise ValueError("duplicate independent calibration sequences")
        if any(tuple(r) in used for r in x.tolist()):
            raise ValueError("value training/validation reused old task source")
        used.update(tuple(r) for r in x.tolist())
    return (tx,ty),(vx,vy),{
        "train":{"seed":TRAIN_SEED,"n":TRAIN_N,
                 "x_sha256":tensor_int64_sha(tx),"y_sha256":tensor_int64_sha(ty)},
        "validation":{"seed":VAL_SEED,"n":VAL_N,
                      "x_sha256":tensor_int64_sha(vx),"y_sha256":tensor_int64_sha(vy)},
        "new_test_generated":False
    }


def old_hidden_and_terminal_ce(model,x,y):
    states=[]
    final=[]
    with torch.no_grad():
        for i in range(0,len(x),32):
            layers=model.reason_states(x[i:i+32])
            states.append(torch.stack(layers[:4]).detach().float())
            final.append(per_token_ce(model.head(layers[-1]),y[i:i+32]).detach().float())
    return torch.cat(states,dim=1),torch.cat(final,dim=0)


def validation_loss(net,h,y,mu,std):
    net.eval()
    with torch.no_grad():
        totals=[]
        mse=[]
        for k in range(4):
            for i in range(0,len(y),32):
                pred=net(h[k,i:i+32],k,mu,std)
                target=y[i:i+32]
                error=F.smooth_l1_loss(pred,target,beta=.5)
                error=error+.1*(pred.mean(-1)-target.mean(-1)).square().mean()
                totals.append(float(error)*len(target))
                mse.append(float((pred-target).square().mean())*len(target))
    return sum(totals)/(4*len(y)),sum(mse)/(4*len(y))


def fit_one(model,train_x,train_y,val_x,val_y,seed_id,smoke=False):
    train_h,train_ce=old_hidden_and_terminal_ce(model,train_x,train_y)
    val_h,val_ce=old_hidden_and_terminal_ce(model,val_x,val_y)
    mu=train_h.mean(dim=(0,1,2))
    std=train_h.std(dim=(0,1,2),unbiased=False).clamp_min(1e-4)
    mean_terminal=float(train_ce.mean())
    torch.manual_seed(76000+seed_id)
    net=HorizonValue()
    initial={k:v.detach().clone() for k,v in net.state_dict().items()}
    optimizer=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.0001)
    gen=torch.Generator().manual_seed(76000+seed_id)
    budget=8 if smoke else UPDATES
    checks=(0,4,8) if smoke else CHECKPOINTS
    records=[]
    best=None
    best_val=float("inf")
    selected=None
    for step in range(budget+1):
        if step in checks:
            valobjective,valmse=validation_loss(net,val_h,val_ce,mu,std)
            records.append({"update":step,"val_huber_objective":valobjective,
                            "val_token_MSE":valmse})
            if valobjective<best_val:
                best_val=valobjective
                best={k:v.detach().clone() for k,v in net.state_dict().items()}
                selected=step
        if step==budget:
            break
        net.train()
        ids=torch.randint(len(train_y),(16,),generator=gen)
        k=int(torch.randint(0,4,(1,),generator=gen))
        h=train_h[k,ids]
        target=train_ce[ids]
        pred=net(h,k,mu,std)
        loss=F.smooth_l1_loss(pred,target,beta=.5)
        loss=loss+.1*(pred.mean(-1)-target.mean(-1)).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    net.load_state_dict(best,strict=True)
    net.eval().requires_grad_(False)
    baseline_mse=float((val_ce-mean_terminal).square().mean())
    return {
        "selected":{k:v.detach().cpu() for k,v in best.items()},
        "initial":{k:v.detach().cpu() for k,v in initial.items()},
        "mu":mu.cpu(),"std":std.cpu(),
        "constant_ce":mean_terminal
    },{
        "seed":seed_id,"train_sequences":len(train_x),
        "validation_sequences":len(val_x),
        "selected_update":selected,
        "validation_grid":records,
        "val_selected_objective":best_val,
        "val_constant_MSE":baseline_mse,
        "value_selected_state_sha256":tensor_fingerprint(best),
        "value_initial_state_sha256":tensor_fingerprint(initial),
        "normalization_sha256":tensor_fingerprint({"mu":mu,"std":std}),
    }


def train(bundle_path,source_meta_path,out_bundle,out_receipt,smoke=False):
    torch.set_num_threads(2)
    source_raw=Path(bundle_path).read_bytes()
    meta_raw=Path(source_meta_path).read_bytes()
    if not smoke and (digest(source_raw)!=SOURCE_BUNDLE_SHA or
                      digest(meta_raw)!=SOURCE_META_SHA):
        raise ValueError("original frozen LGA02 source artifact bytes changed")
    oldmeta,source=load_frozen(bundle_path,source_meta_path)
    if not smoke and oldmeta["mode"]!="full":
        raise ValueError("LGA03 calibration requires three original full teacher seeds")
    (tx,ty),(vx,vy),rosters=calibration_data()
    seeds=SEEDS[:1] if smoke else SEEDS
    saved={}
    receipts={}
    for sid in seeds:
        teacher=make_model(source,sid)
        values,details=fit_one(teacher,tx,ty,vx,vy,sid,smoke)
        saved[str(sid)]=values
        receipts[str(sid)]=details
        print("LGA03_VALUE_FROZEN",sid,"step",details["selected_update"],
              "old_val_MSE",round(next(x["val_token_MSE"] for x in details["validation_grid"]
                                      if x["update"]==details["selected_update"]),5),
              flush=True)
    out_bundle=Path(out_bundle);out_bundle.parent.mkdir(parents=True,exist_ok=True)
    torch.save(saved,out_bundle)
    receipt={
        "schema":SCHEMA,"mode":"smoke" if smoke else "full",
        "protocol_sha256":protocol_sha(),
        "source_LGA02_bundle_sha256":digest(source_raw),
        "source_LGA02_train_receipt_sha256":digest(meta_raw),
        "value_training_source":rosters,
        "new_test_seed_reserved":TEST_SEED,
        "new_test_labels_read":False,
        "teacher_seeds":list(seeds),
        "value_train_steps":8 if smoke else UPDATES,
        "source_fits":receipts,
        "value_bundle_sha256":digest(out_bundle.read_bytes()),
        "torch_version":torch.__version__
    }
    out_receipt=Path(out_receipt);out_receipt.parent.mkdir(parents=True,exist_ok=True)
    out_receipt.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    return receipt


def load_value(out_bundle,out_receipt,seed_id):
    raw=Path(out_bundle).read_bytes()
    meta=json.loads(Path(out_receipt).read_bytes())
    if (meta["schema"]!=SCHEMA or meta["protocol_sha256"]!=protocol_sha() or
        meta["value_bundle_sha256"]!=digest(raw) or
        meta["value_training_source"]["new_test_generated"] is not False or
        meta["new_test_labels_read"] is not False):
        raise ValueError("calibration-frozen value model or source fingerprint changed")
    payload=torch.load(out_bundle,weights_only=True,map_location="cpu")
    if str(seed_id) not in payload:
        raise ValueError("original independent teacher value critic unavailable")
    row=payload[str(seed_id)]
    info=meta["source_fits"][str(seed_id)]
    if (tensor_fingerprint(row["selected"])!=info["value_selected_state_sha256"] or
        tensor_fingerprint(row["initial"])!=info["value_initial_state_sha256"] or
        tensor_fingerprint({"mu":row["mu"],"std":row["std"]})!=info["normalization_sha256"]):
        raise ValueError("value potential weights, init or calibration normalizer changed")
    model=HorizonValue().eval()
    model.load_state_dict(row["selected"],strict=True)
    null=HorizonValue().eval()
    null.load_state_dict(row["initial"],strict=True)
    model.requires_grad_(False)
    null.requires_grad_(False)
    return model,null,row["mu"],row["std"],float(row["constant_ce"])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-bundle",type=Path,required=True)
    p.add_argument("--source-meta",type=Path,required=True)
    p.add_argument("--value-bundle",type=Path,required=True)
    p.add_argument("--value-receipt",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    r=train(a.source_bundle,a.source_meta,a.value_bundle,a.value_receipt,a.smoke)
    print("LGA03_CALIBRATION_LOCKED",r["value_bundle_sha256"],
          r["new_test_labels_read"],flush=True)


if __name__=="__main__":
    main()
