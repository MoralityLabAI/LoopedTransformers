"""LGA-03: fresh heldout value-gradient and physical native causal experiment.

The stage-A value weights/initialization and original LGA02 teacher+SAE
bundle are sealed in independent artifacts before the fresh test generator
seed 31050 is opened. Only frozen hidden activations choose test edits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct

import torch
from torch.nn import functional as F

from mechinterp.loopscope04.gradient_flow import (
    fixed_data,head_rowspace,per_token_ce,tensor_int64_sha
)
from mechinterp.loopscope05.fit_and_freeze import digest,load_frozen
from mechinterp.loopscope05.terminal_sae_test import (
    make_model,make_sae,terminal_gradients,active_components
)
from mechinterp.loopscope06.fit_value import (
    load_value,protocol_sha,SCHEMA as CAL_SCHEMA,
    SOURCE_BUNDLE_SHA,SOURCE_META_SHA,TRAIN_SEED,VAL_SEED,
    TRAIN_N,VAL_N,SEEDS
)

SCHEMA="LGA-03/fresh-horizon-value-gradient-and-native-causal-v1"
TEST_SEED=31050
TEST_N=256
SMOKE_TEST_SEED=61050
DEPTHS=(0,1,2,3)
INTERVENTION_DEPTHS=(1,2)
FIXED_GOALS=(0,10,20,30,40,50,60,70)
BOOT=2000
CAUSAL_N=64
CAUSAL_ARMS=("native","sae_active","sae_remainder","head_null",
             "random","value_down","oracle_down")
CONTRASTS=(("value_down","random"),("value_down","native"),
           ("sae_active","sae_remainder"),("head_null","random"))


def stable_seed(*parts):
    string="LGA03|"+"|".join(str(x) for x in parts)
    return int.from_bytes(hashlib.sha256(string.encode()).digest()[:8],"big")&((1<<63)-1)


def sample_test(smoke=False):
    n=8 if smoke else TEST_N
    sd=SMOKE_TEST_SEED if smoke else TEST_SEED
    x,y=fixed_data(n,sd)
    seen=set()
    for prior_sd,size in ((31041,512),(31042,128),(31043,256),
                          (31044,256),(TRAIN_SEED,TRAIN_N),(VAL_SEED,VAL_N)):
        old,_=fixed_data(size,prior_sd)
        for row in old.tolist():
            seen.add(tuple(row))
    if len({tuple(row) for row in x.tolist()})!=n:
        raise ValueError("source new test strings duplicate each other")
    if any(tuple(row) in seen for row in x.tolist()):
        raise ValueError("source new test reused earlier train/calibration/heldout string")
    return x,y,{"seed":sd,"n_sequences":n,
                "input_sha256":tensor_int64_sha(x),
                "target_sha256":tensor_int64_sha(y),
                "no_source_overlap":True}


def gradient_value(net,h,k,mu,std):
    """Gradient of sum of all 81 critic outputs, with frozen critic weights."""
    h=h.detach().requires_grad_(True)
    with torch.enable_grad():
        output=net(h,k,mu,std)
        gradient=torch.autograd.grad(output.sum(),h,create_graph=False)[0]
    if gradient.shape!=h.shape or output.shape!=h.shape[:2]:
        raise ValueError("horizon critic gradient or output shape changed")
    if not bool(torch.isfinite(gradient).all()):
        raise ArithmeticError("nonfinite candidate value gradient")
    return output.detach(),gradient.detach()


def _cos(a,b):
    dot=(a*b).sum(-1)
    return dot/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-12)


def profile(model,net,initial,mu,std,const_ce,x,y):
    """Whole-sequence 81-token metrics for four remaining-loop horizons."""
    rows={str(k):[] for k in DEPTHS}
    model.eval()
    for i in range(0,len(x),8):
        xb=x[i:i+8];yb=y[i:i+8]
        states,g_future=terminal_gradients(model,xb,yb)
        with torch.no_grad():
            final=per_token_ce(model.head(states[-1]),yb)
        for k in DEPTHS:
            h=states[k].detach()
            proposed=states[k+1]-h
            out=model.head(h).detach()
            local=(torch.softmax(out,-1)-F.one_hot(yb,10).float())@model.head.weight
            value,g_value=gradient_value(net,h,k,mu,std)
            _,g_init=gradient_value(initial,h,k,mu,std)
            future=g_future[k]
            c_value=_cos(g_value,future)
            c_initial=_cos(g_init,future)
            c_local=_cos(local,future)
            c_native_future=_cos(proposed,-future)
            c_native_value=_cos(proposed,-g_value)
            c_future_local=_cos(future,local)
            for j in range(len(xb)):
                rows[str(k)].append({
                    "cos_value_future":float(c_value[j].mean()),
                    "cos_initial_future":float(c_initial[j].mean()),
                    "cos_local_future":float(c_local[j].mean()),
                    "gain_value_over_initial":float((c_value[j]-c_initial[j]).mean()),
                    "gain_value_over_local":float((c_value[j]-c_local[j]).mean()),
                    "cos_native_negative_future":float(c_native_future[j].mean()),
                    "cos_native_negative_value":float(c_native_value[j].mean()),
                    "cos_future_local":float(c_future_local[j].mean()),
                    "value_pred_token_MSE":float((value[j]-final[j]).square().mean()),
                    "constant_pred_token_MSE":float((final[j]-const_ce).square().mean()),
                    "value_grad_l2":float(g_value[j].norm(dim=-1).mean()),
                    "oracle_grad_l2":float(future[j].norm(dim=-1).mean()),
                })
    if any(len(v)!=len(x) for v in rows.values()):
        raise ValueError("fresh test sequence profile incomplete")
    return rows


def bootstrap(vals,seed_id,draws=BOOT):
    if len(vals)<2 or not all(math.isfinite(v) for v in vals):
        raise ValueError("bootstrapping requires independent finite full sequences")
    rng=random.Random(seed_id)
    n=len(vals)
    z=sorted(sum(vals[rng.randrange(n)] for _ in range(n))/n
             for __ in range(draws))
    return [z[int(.025*draws)],z[int(.975*draws)-1]]


def summarize(profile_rows,model_seed,smoke=False):
    data={}
    passed_depths=0
    for k in DEPTHS:
        group=profile_rows[str(k)]
        means={metric:sum(z[metric] for z in group)/len(group) for metric in group[0]}
        ci={}
        for metric in ("gain_value_over_initial","gain_value_over_local"):
            ci[metric]=bootstrap(
                [row[metric] for row in group],
                stable_seed(43050,model_seed,k,metric),draws=100 if smoke else BOOT)
        lower=all(ci[metric][0]>0 for metric in ci)
        val_better=means["value_pred_token_MSE"]<means["constant_pred_token_MSE"]
        depth_pass=lower and val_better
        passed_depths+=int(depth_pass)
        data[str(k)]={"n_whole_sequences":len(group),"mean":means,
                      "bootstrap95":ci,"depth_pass":depth_pass,
                      "rows":group}
    return data,passed_depths


def safe_normalize(z,radius):
    n=float(z.norm())
    if n<=1e-8 or radius<=1e-8:
        return None
    direction=(z*float(radius/n)).detach().float()
    if abs(float(direction.norm())-radius)>1e-5:
        raise ArithmeticError("edit arm did not preserve exact physical L2 dose")
    return direction


def edit_sha(edit):
    z=edit.reshape(-1).detach().float().cpu().tolist()
    return digest(struct.pack("<"+"f"*len(z),*z))


def native_interventions(model,value_model,mu,std,saes,x,y,model_seed,smoke=False):
    """Edits chosen without labels except explicit terminal-gradient oracle."""
    n=min(4 if smoke else CAUSAL_N,len(x))
    basis,_=head_rowspace(model)
    source={}
    for k in INTERVENTION_DEPTHS:
        sae,sae_mu,sae_std=saes[k]
        cases=[]
        for idx in range(n):
            xb=x[idx:idx+1];yb=y[idx:idx+1]
            with torch.no_grad():
                states=model.reason_states(xb)
            current=states[k][0].detach()
            step=states[k+1][0]-current
            _,g_value=gradient_value(value_model,current.unsqueeze(0),k,mu,std)
            value_grad=g_value[0]
            parts=active_components(sae,sae_mu,sae_std,current,step)
            headnull=step.double()-(step.double()@basis)@basis.T
            goals=[]
            schedule=[]
            for j,tok in enumerate(FIXED_GOALS):
                native=step[tok]
                radius=.25*float(native.norm())
                gen=torch.Generator().manual_seed(stable_seed(
                    "isotropic",model_seed,k,idx,j))
                random_vec=torch.randn(48,dtype=torch.float32,generator=gen)
                directions={
                    "native":native,
                    "sae_active":parts["active"][tok],
                    "sae_remainder":parts["remainder"][tok],
                    "head_null":headnull[tok].float(),
                    "random":random_vec,
                    "value_down":-value_grad[tok],
                }
                source_valid=bool(parts["admitted"][tok]) and all(
                    float(v.norm())>1e-8 for v in directions.values()) and radius>1e-8
                item={"slot":j,"position":tok,"admitted":source_valid,
                      "radius":radius,"arms":{}}
                if source_valid:
                    for arm,v in directions.items():
                        unit=safe_normalize(v,radius)
                        if unit is None:
                            raise ArithmeticError("admitted original physical direction became zero")
                        schedule.append((j,tok,arm,unit))
                else:
                    item["reason"]="source_degenerate_active_remainder_headnull_or_value_gradient"
                goals.append(item)
            # Oracle labels enter AFTER fixed source plans exist.
            _,future=terminal_gradients(model,xb,yb)
            oracle=future[k][0]
            for j,tok in enumerate(FIXED_GOALS):
                if goals[j]["admitted"]:
                    oracle_edit=safe_normalize(-oracle[tok],goals[j]["radius"])
                    if oracle_edit is not None:
                        schedule.append((j,tok,"oracle_down",oracle_edit))
                    else:
                        goals[j]["oracle_available"]=False
            batch=[current]
            for j,tok,arm,direction in schedule:
                h=current.clone()
                h[tok]+=direction
                batch.append(h)
            z=torch.stack(batch)
            with torch.no_grad():
                for _ in range(4-k):
                    z=model.block(z)
                logits=model.head(z)
                target=yb.repeat(len(z),1)
                CE=per_token_ce(logits,target)
            clean_token_ce=CE[0].tolist()
            clean_logits=logits[0]
            max_closure=float((clean_logits-states[-1][0].new_tensor(
                model.head(states[-1])[0])).abs().max())
            if max_closure>1e-5:
                raise ArithmeticError("original source zero-edit native forward mismatch")
            for n_arm,(j,tok,arm,direction) in enumerate(schedule,1):
                true_digit=int(yb[0,tok])
                competitor=clean_logits[tok].clone()
                competitor[true_digit]=-float("inf")
                rival=int(competitor.argmax())
                target_logits=logits[n_arm,tok]
                base=clean_logits[tok]
                measured=float(CE[n_arm].mean()-CE[0].mean())
                entry={
                    "dose":float(direction.norm()),
                    "edit_sha256":edit_sha(direction),
                    "full_token_CE":CE[n_arm].tolist(),
                    "native_final_sequence_CE":float(CE[n_arm].mean()),
                    "clean_final_sequence_CE":float(CE[0].mean()),
                    "global_CE_change":measured,
                    "goal_logits":target_logits.tolist(),
                    "clean_goal_logits":base.tolist(),
                    "truth_digit":true_digit,"clean_rival":rival,
                    "goal_CE_change":float(CE[n_arm,tok]-CE[0,tok]),
                    "goal_margin_change":float(
                        (target_logits[true_digit]-target_logits[rival])-
                        (base[true_digit]-base[rival])),
                    "learned_linear_global_CE_change":float(
                        (g_value[0,tok].double()@direction.double())/81),
                    "oracle_linear_global_CE_change":float(
                        (oracle[tok].double()@direction.double())/81),
                }
                goals[j]["arms"][arm]=entry
            cases.append({
                "new_test_sequence_index":idx,
                "clean_token_CE":clean_token_ce,
                "original_goals":goals,
            })
        source[str(k)]=cases
    return source


def summarize_causal(cases,seed_id,smoke=False):
    result={}
    for k in INTERVENTION_DEPTHS:
        rows=[]
        for source in cases[str(k)]:
            original=source["original_goals"]
            admitted=[g for g in original if g["admitted"]]
            if len(original)!=8:
                raise ValueError("original intervention goal roster must retain all eight")
            means={}
            for a,b in CONTRASTS:
                label=f"{a}_minus_{b}"
                values=[g["arms"][a]["global_CE_change"]-
                        g["arms"][b]["global_CE_change"] for g in admitted]
                means[label]=sum(values)/len(values) if values else 0.
            rows.append({"source_sequence_index":source["new_test_sequence_index"],
                         "original_goals":8,"admitted_goals":len(admitted),
                         "paired_contrasts":means})
        summary={}
        for a,b in CONTRASTS:
            label=f"{a}_minus_{b}"
            vals=[row["paired_contrasts"][label] for row in rows]
            ci=bootstrap(vals,stable_seed("causal",43050,seed_id,k,label),
                         draws=100 if smoke else BOOT)
            summary[label]={"mean":sum(vals)/len(vals),
                            "bootstrap95":ci,
                            "nominal_direction":"POSITIVE" if ci[0]>0 else
                                                "NEGATIVE" if ci[1]<0 else "INCONCLUSIVE"}
        result[str(k)]={
            "source_sequences":len(rows),"original_goals":len(rows)*8,
            "admitted_goals":sum(row["admitted_goals"] for row in rows),
            "contrasts":summary,"per_sequence":rows,"raw_cases":cases[str(k)],
        }
    return result


def run(source_bundle,source_meta,value_bundle,value_receipt,out,smoke=False):
    torch.set_num_threads(2)
    source_raw=Path(source_bundle).read_bytes()
    source_meta_raw=Path(source_meta).read_bytes()
    value_raw=Path(value_bundle).read_bytes()
    value_meta_raw=Path(value_receipt).read_bytes()
    meta,source=load_frozen(source_bundle,source_meta)
    vm=json.loads(value_meta_raw)
    if (vm["schema"]!=CAL_SCHEMA or vm["mode"]!=("smoke" if smoke else "full")
        or vm["source_LGA02_bundle_sha256"]!=digest(source_raw)
        or vm["source_LGA02_train_receipt_sha256"]!=digest(source_meta_raw)
        or vm["value_bundle_sha256"]!=digest(value_raw)
        or vm["new_test_labels_read"] is not False):
        raise ValueError("source-frozen value potential or original LGA02 teacher changed")
    x,y,test_meta=sample_test(smoke)
    receipt={}
    seed_passes=0
    for seed_id in vm["teacher_seeds"]:
        original=make_model(source,seed_id)
        fitted,initial,mu,std,const_ce=load_value(value_bundle,value_receipt,seed_id)
        profiles=profile(original,fitted,initial,mu,std,const_ce,x,y)
        statistics,passes=summarize(profiles,seed_id,smoke)
        saes={k:make_sae(source,seed_id,k) for k in INTERVENTION_DEPTHS}
        native=native_interventions(original,fitted,mu,std,saes,x,y,seed_id,smoke)
        contrasts=summarize_causal(native,seed_id,smoke)
        passed=bool(passes>=2)
        seed_passes+=int(passed)
        receipt[str(seed_id)]={"value_gradient_profiles":statistics,
                               "n_positive_depths":passes,
                               "registered_seed_pass":passed,
                               "causal_interventions":contrasts}
        print("LGA03_SCORED",seed_id,"future_alignment",[
            round(statistics[str(k)]["mean"]["cos_value_future"],4)
            for k in DEPTHS],"gain_vs_local",[
            round(statistics[str(k)]["mean"]["gain_value_over_local"],4)
            for k in DEPTHS],"pass",passed,flush=True)
    verdict=("PILOT_ONLY" if smoke else
             "POSITIVE" if seed_passes>=2 else "FAIL")
    result={
        "schema":SCHEMA,"mode":"smoke" if smoke else "full",
        "protocol_sha256":protocol_sha(),
        "original_LGA02_source_bundle_sha256":digest(source_raw),
        "original_LGA02_train_receipt_sha256":digest(source_meta_raw),
        "source_frozen_value_bundle_sha256":digest(value_raw),
        "source_frozen_value_receipt_sha256":digest(value_meta_raw),
        "heldout":test_meta,"new_test_source":{
            "x":x.tolist(),"y":y.tolist()},
        "n_independent_teacher_seeds":len(vm["teacher_seeds"]),
        "n_original_heldout_sequences":len(x),
        "registered_primary_status":verdict,
        "n_teacher_seed_passes":seed_passes,
        "results":receipt,
        "limitations":"Value model learned externally on old calibration labels; test-local and terminal gradients use new labels only as retrospective diagnostics. Oracle edit is label-dependent."
    }
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("LGA03_NATIVE_COMPLETE",verdict,seed_passes,flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ("source-bundle","source-meta","value-bundle","value-receipt","out"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    run(a.source_bundle,a.source_meta,a.value_bundle,a.value_receipt,a.out,a.smoke)


if __name__=="__main__":
    main()
