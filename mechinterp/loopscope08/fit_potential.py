"""LGA-05 stage A: train scalar potentials on ORIGINAL recurrent motion only.

No new heldout 31057 data is created here. The teachers and old SAEs are
unchanged, exact-byte-pinned to first LGA-02 scientific source run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from mechinterp.loopscope04.gradient_flow import fixed_data,tensor_int64_sha
from mechinterp.loopscope05.fit_and_freeze import digest,load_frozen,tensor_fingerprint
from mechinterp.loopscope05.terminal_sae_test import make_model
from mechinterp.loopscope06.fit_value import HorizonValue

SCHEMA="LGA-05/calibration-only-recurrent-energy-potentials-v1"
SOURCE_SHA="d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266"
SOURCE_META_SHA="7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d"
TRAIN_SEED=31055
VAL_SEED=31056
TEST_SEED=31057
TEACHER_SEEDS=(103,211,307)
ARMS=("shared","conditioned")
UPDATES=200
CHECKS=(0,100,200)
OLD_SOURCES=((31041,512),(31042,128),(31043,256),(31044,256),
             (31048,768),(31049,192),(31050,256),
             (31052,256),(31053,64),(31054,256))


def protocol_sha():
    return digest((Path(__file__).resolve().parent/"PROTOCOL.md").read_bytes())


def old_calibration(smoke=False):
    ntrain=16 if smoke else 256
    nval=8 if smoke else 64
    x,y=fixed_data(ntrain,TRAIN_SEED)
    v,w=fixed_data(nval,VAL_SEED)
    seen=set()
    for sd,n in OLD_SOURCES:
        old,_=fixed_data(n,sd)
        seen.update(tuple(row) for row in old.tolist())
    for group in (x,v):
        rows=[tuple(row) for row in group.tolist()]
        if len(set(rows))!=len(rows) or any(row in seen for row in rows):
            raise ValueError("new recurrent-energy train/validation reused an earlier source")
        seen.update(rows)
    return (x,y),(v,w),{
        "train":{"seed":TRAIN_SEED,"n":ntrain,
                 "x_sha256":tensor_int64_sha(x),"y_sha256":tensor_int64_sha(y),
                 "x":x.tolist(),"y":y.tolist()},
        "validation":{"seed":VAL_SEED,"n":nval,
                      "x_sha256":tensor_int64_sha(v),"y_sha256":tensor_int64_sha(w),
                      "x":v.tolist(),"y":w.tolist()},
        "new_test_generated":False,"new_test_labels_available":False,
        "new_test_seed_reserved":TEST_SEED
    }


def trajectory_cache(model,x,chunk=32):
    states=[]
    for i in range(0,len(x),chunk):
        with torch.no_grad():
            snapshots=model.reason_states(x[i:i+chunk])
        states.append(torch.stack(snapshots).detach().cpu())
    all_states=torch.cat(states,dim=1).contiguous().float()
    if all_states.shape!=(5,len(x),81,48):
        raise ValueError("source tied four-pass trajectory geometry changed")
    return all_states


def potential_gradient(model,h,k,mu,std,create_graph=False):
    """Exact whole-state derivative of SUM of all 81 token energy terms."""
    state=h.detach().requires_grad_(True)
    with torch.enable_grad():
        output=model(state,k,mu,std)
        g=torch.autograd.grad(output.sum(),state,
                              create_graph=create_graph)[0]
    if not bool(torch.isfinite(g.detach()).all()):
        raise ArithmeticError("nonfinite potential gradient")
    return output,g


def calibration_diagnostics(net,states,mu,std,scale,arm,chunk=16):
    net.eval()
    n=0
    err=0.
    positive=0
    energy=0.
    for k in range(4):
        for i in range(0,states.shape[1],chunk):
            h=states[k,i:i+chunk]
            hnext=states[k+1,i:i+chunk]
            original=hnext-h
            embed=0 if arm=="shared" else k
            _,gp=potential_gradient(net,h,embed,mu,std)
            with torch.no_grad():
                before=net(h,embed,mu,std).sum(-1)
                after=net(hnext,embed,mu,std).sum(-1)
                drops=(before-after)/81
            err+=float(F.mse_loss(gp.detach()/scale,-original/scale))*len(h)
            positive+=int((drops>0).sum())
            energy+=float(drops.sum())
            n+=len(h)
    return {"gradient_normalized_MSE":err/n,
            "native_energy_descent_fraction":positive/n,
            "mean_energy_drop_per_token":energy/n,
            "n_source_sequence_horizons":n}


def fit_one(seed,model,train_states,val_states,smoke=False):
    mu=train_states[:4].mean(dim=(0,1,2))
    std=train_states[:4].std(dim=(0,1,2),unbiased=False).clamp_min(1e-4)
    v=train_states[1:]-train_states[:-1]
    scale=v.square().mean().sqrt().clamp_min(1e-5)
    torch.manual_seed(78000+seed)
    reference=HorizonValue()
    initial={k:t.detach().cpu().clone() for k,t in reference.state_dict().items()}
    budget=3 if smoke else UPDATES
    checks=(0,3) if smoke else CHECKS
    outcome={}
    receipt={}
    for arm in ARMS:
        net=HorizonValue()
        net.load_state_dict(initial,strict=True)
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        rng=torch.Generator().manual_seed(79000+seed)
        records=[]
        for step in range(budget+1):
            if step in checks:
                val=calibration_diagnostics(net,val_states,mu,std,scale,arm)
                records.append({"update":step,**val})
            if step==budget:
                break
            ids=torch.randint(len(train_states[0]),(8,),generator=rng)
            k=int(torch.randint(0,4,(1,),generator=rng))
            h=train_states[k,ids]
            native=v[k,ids]
            index=0 if arm=="shared" else k
            _,gp=potential_gradient(net,h,index,mu,std,create_graph=True)
            loss=F.mse_loss(-gp/scale,native/scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if not bool(torch.isfinite(loss.detach())):
                raise ArithmeticError("source gradient-target training produced nonfinite loss")
        saved={name:t.detach().cpu().clone() for name,t in net.state_dict().items()}
        outcome[arm]=saved
        receipt[arm]={"optimizer_steps":budget,
                      "same_initial_weights_sha256":tensor_fingerprint(initial),
                      "trained_weights_sha256":tensor_fingerprint(saved),
                      "validation_grid":records}
    payload={
        "models":outcome,"initial":initial,"mu":mu.detach().cpu(),
        "std":std.detach().cpu(),"native_RMS":scale.detach().cpu()
    }
    checksums={
        "initial_sha256":tensor_fingerprint(initial),
        "normalizer_sha256":tensor_fingerprint({"mu":mu,"std":std}),
        "native_RMS":float(scale),
        "native_field_train_sha256":tensor_fingerprint({"v":v}),
        "train_state_sha256":tensor_fingerprint({"h":train_states}),
        "validation_state_sha256":tensor_fingerprint({"h":val_states}),
        "arms":receipt
    }
    return payload,checksums


def fit(source_bundle,source_meta,bundle_path,manifest_path,smoke=False):
    torch.set_num_threads(2)
    frozen_raw=Path(source_bundle).read_bytes()
    frozen_meta=Path(source_meta).read_bytes()
    if not smoke and (digest(frozen_raw)!=SOURCE_SHA or digest(frozen_meta)!=SOURCE_META_SHA):
        raise ValueError("original LGA02 teacher model/SAE source SHA mismatch")
    _,frozen=load_frozen(source_bundle,source_meta)
    (tx,ty),(vx,vy),splits=old_calibration(smoke)
    seeds=(103,) if smoke else TEACHER_SEEDS
    bundle={};receipts={}
    for sid in seeds:
        teacher=make_model(frozen,sid)
        tr=trajectory_cache(teacher,tx)
        val=trajectory_cache(teacher,vx)
        packed,measure=fit_one(sid,teacher,tr,val,smoke)
        bundle[str(sid)]=packed
        receipts[str(sid)]=measure
        print("LGA05_CALIBRATED",sid,
              {a:round(measure["arms"][a]["validation_grid"][-1]["gradient_normalized_MSE"],5)
               for a in ARMS},flush=True)
    bundle_path=Path(bundle_path);bundle_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(bundle,bundle_path)
    report={
        "schema":SCHEMA,"mode":"smoke" if smoke else "full",
        "protocol_sha256":protocol_sha(),
        "original_teacher_source_bundle_sha256":digest(frozen_raw),
        "original_teacher_source_manifest_sha256":digest(frozen_meta),
        "calibration_splits":splits,
        "teacher_seeds":list(seeds),
        "source_fit_steps":3 if smoke else UPDATES,
        "paired_arms":list(ARMS),
        "teacher_potential_receipts":receipts,
        "source_potential_bundle_sha256":digest(bundle_path.read_bytes()),
        "test_labels_read":False,"torch_version":torch.__version__
    }
    manifest_path=Path(manifest_path);manifest_path.parent.mkdir(parents=True,exist_ok=True)
    manifest_path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("LGA05_POTENTIALS_FROZEN",report["source_potential_bundle_sha256"],flush=True)
    return report


def load_potentials(bundle_path,manifest_path,sid):
    raw=Path(bundle_path).read_bytes()
    meta=json.loads(Path(manifest_path).read_bytes())
    if (meta["schema"]!=SCHEMA or meta["protocol_sha256"]!=protocol_sha() or
        meta["source_potential_bundle_sha256"]!=digest(raw) or
        meta["calibration_splits"]["new_test_generated"] is not False or
        meta["test_labels_read"] is not False):
        raise ValueError("old calibration source potential/novel test chronology changed")
    saved=torch.load(bundle_path,map_location="cpu",weights_only=True)[str(sid)]
    info=meta["teacher_potential_receipts"][str(sid)]
    if tensor_fingerprint(saved["initial"])!=info["initial_sha256"] or tensor_fingerprint(
        {"mu":saved["mu"],"std":saved["std"]})!=info["normalizer_sha256"]:
        raise ValueError("source potential initial weights or normalizer changed")
    models={}
    for arm in ARMS:
        state=saved["models"][arm]
        if (tensor_fingerprint(state)!=info["arms"][arm]["trained_weights_sha256"] or
            info["arms"][arm]["same_initial_weights_sha256"]!=info["initial_sha256"]):
            raise ValueError("paired learned potential model weights differ from source")
        network=HorizonValue().eval()
        network.load_state_dict(state,strict=True)
        network.requires_grad_(False)
        models[arm]=network
    null=HorizonValue().eval()
    null.load_state_dict(saved["initial"],strict=True)
    null.requires_grad_(False)
    return models,null,saved["mu"],saved["std"],saved["native_RMS"]


def main():
    p=argparse.ArgumentParser()
    for n in ("source-bundle","source-meta","potential-bundle","potential-meta"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    fit(a.source_bundle,a.source_meta,a.potential_bundle,a.potential_meta,a.smoke)


if __name__=="__main__":
    main()
