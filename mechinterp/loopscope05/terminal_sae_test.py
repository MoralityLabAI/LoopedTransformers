"""LGA-02 stage B: native future-gradient and equal-dose SAE/head-null replay.

All source checkpoints and depth-specific SAE dictionaries are loaded from
a preceding train-only CI artifact. The new heldout generator seed 31044
is opened only in this stage. Correct target labels are diagnostic endpoints
and NEVER select active SAE supports or physical intervention directions.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import torch
from torch import nn
from torch.nn import functional as F

from mechinterp.loopscope04.gradient_flow import (
    TinyRecursiveModel,fixed_data,per_token_ce,head_rowspace,
    derived_seed,tensor_int64_sha
)
from mechinterp.loopscope05.fit_and_freeze import (
    TopKSAE,load_frozen,digest,protocol_sha
)

SCHEMA="LGA-02/future-gradient-SAE-and-native-causal-receipt-v1"
NEW_TEST_SEED=31044
NEW_TEST_N=256
OLD_SEEDS=(31041,31042,31043)
BOOT=2000
GOALS=(0,10,20,30,40,50,60,70)
CAUSAL_SAMPLES=64
ARMS=("full","sae_active","sae_remainder","head_null","random")
DEPTHS=(1,2)


def seed(*parts):
    x="LGA02|"+"|".join(map(str,parts))
    return int.from_bytes(hashlib.sha256(x.encode()).digest()[:8],"big") & ((1<<63)-1)


def bootstrap(values,root,draws=BOOT):
    gen=random.Random(root)
    n=len(values)
    if n<2 or any(not math.isfinite(v) for v in values):
        raise ValueError("registered cluster bootstrap requires >=2 finite values")
    distribution=sorted(sum(values[gen.randrange(n)] for j in range(n))/n
                        for i in range(draws))
    return [distribution[int(.025*draws)],distribution[int(.975*draws)-1]]


def fixed_new_test(smoke=False):
    n=16 if smoke else NEW_TEST_N
    x,y=fixed_data(n,NEW_TEST_SEED)
    existing=set()
    for s,c in zip(OLD_SEEDS,(64,32,32) if smoke else (512,128,256)):
        prior,_=fixed_data(c,s)
        existing.update(tuple(row) for row in prior.tolist())
    if any(tuple(row) in existing for row in x.tolist()):
        raise ValueError("new source sample exactly repeats an earlier split")
    if len({tuple(row) for row in x.tolist()})!=len(x):
        raise ValueError("new test source contains a duplicate")
    return x,y,{
        "seed":NEW_TEST_SEED,"n_sequences":len(x),
        "x_sha256":tensor_int64_sha(x),
        "y_sha256":tensor_int64_sha(y),
        "old_source_test_seed":31043,
        "no_exact_sequence_overlap":True,
    }


def make_model(bundle,seed_id,deep=False):
    model=TinyRecursiveModel(48,4)
    if deep:
        model.load_state_dict(bundle["deep"]["103"],strict=True)
    else:
        model.load_state_dict(bundle["models"][str(seed_id)],strict=True)
    model.eval().requires_grad_(False)
    return model


def make_sae(bundle,seed_id,depth):
    data=bundle["saes"][f"{seed_id}|{depth}"]
    sae=TopKSAE()
    sae.load_state_dict(data["state"],strict=True)
    sae.eval().requires_grad_(False)
    return sae,data["mu"],data["std"]


def active_components(sae,mu,std,h,delta,future_grad=None):
    """Physical SAE active-span projection in a source-defined rank-4 cell.

    h/delta are arrays of shape [n,48] or [...,48]; all operations are
    batched and use FP64 small-Gram solves with rank admission per token.
    Dictionary and active Top4 indices depend only on physical input h.
    """
    shape=h.shape[:-1]
    H=h.reshape(-1,48)
    D=delta.reshape(-1,48)
    x=((H-mu)/std).float()
    with torch.no_grad():
        code=F.relu(sae.encoder(x))
        values,ids=code.topk(4,dim=-1,sorted=True)
        physical=(sae.decoder.weight.detach().double()*
                  std.double().reshape(-1,1)).T  # [96,48]
        A=physical[ids].transpose(1,2).contiguous() # [N,48,4]
        gram=A.transpose(1,2)@A
        evals=torch.linalg.eigvalsh(gram)
        nonsingular=(evals[:,0]>1e-16*evals[:,-1].clamp_min(1e-20))
        gram_stable=gram+torch.eye(4,dtype=torch.float64).view(1,4,4)*1e-12
        coeff=torch.linalg.solve(gram_stable,
                                 (A.transpose(1,2)@D.double().unsqueeze(-1)))
        active=(A@coeff).squeeze(-1)
        remaining=D.double()-active
        active_norm=active.norm(dim=-1)
        remaining_norm=remaining.norm(dim=-1)
        admitted=(values[:,3]>0)&nonsingular&(
                 active_norm>1e-8)&(remaining_norm>1e-8)
        recode=sae.encode(x)
        reconstructed=sae.decoder(recode)
        mse=(reconstructed-x).square().mean(-1)
        next_ids=None
        if future_grad is not None:
            G=future_grad.reshape(-1,48).double()
            cosine_active=-(active*G).sum(-1)/(
                active_norm*G.norm(dim=-1)).clamp_min(1e-12)
            cosine_remainder=-(remaining*G).sum(-1)/(
                remaining_norm*G.norm(dim=-1)).clamp_min(1e-12)
        else:
            cosine_active=torch.zeros(len(H))
            cosine_remainder=torch.zeros(len(H))
        original_norm=D.double().square().sum(-1).clamp_min(1e-18)
        energy=active.square().sum(-1)/original_norm
    return {
        "active":active.reshape(*shape,48).float(),
        "remainder":remaining.reshape(*shape,48).float(),
        "admitted":admitted.reshape(*shape),
        "ids":ids.reshape(*shape,4),
        "active_energy_fraction":energy.reshape(*shape).float(),
        "SAE_MSE":mse.reshape(*shape),
        "future_cos_active":cosine_active.reshape(*shape).float(),
        "future_cos_remainder":cosine_remainder.reshape(*shape).float(),
    }


def terminal_gradients(model,x,y):
    """True derivative through remaining F blocks, holding source weights fixed.

    Sum over all output-token CE so each physical goal gradient includes
    cross-token future influence from the entire 81-token final task.
    """
    h0=model.initial_state(x).detach().requires_grad_(True)
    states=[h0]
    for _ in range(4):
        states.append(model.block(states[-1]))
    logits=model.head(states[-1])
    final_sum=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1),reduction="sum")
    grads=torch.autograd.grad(final_sum,states[:4],retain_graph=False,
                              create_graph=False)
    return [h.detach() for h in states],[g.detach() for g in grads]


def geometry_batch(model,saes,x,y,depths=(0,1,2,3)):
    """Per-sequence trajectory; gradient labels only enter this diagnostic."""
    states,future=terminal_gradients(model,x,y)
    head_basis,rank=head_rowspace(model)
    result={}
    outputs=[model.head(s).detach() for s in states]
    losses=[per_token_ce(z,y).detach() for z in outputs]
    for k in depths:
        h=states[k]
        delta=states[k+1]-h
        local=(torch.softmax(outputs[k],-1)-F.one_hot(y,10).float())@model.head.weight
        gf=future[k]
        def cosine(a,b):
            return (a*b).sum(-1)/(
                a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-12)
        local_cos=cosine(delta,-local)
        future_cos=cosine(delta,-gf)
        disagreement=cosine(local,gf)
        head_visible=(delta.double()@head_basis).square().sum(-1)/(
            delta.double().square().sum(-1)).clamp_min(1e-18)
        gf_headnull=gf.double()-(gf.double()@head_basis)@head_basis.T
        row={
            "local_cos":local_cos,
            "future_cos":future_cos,
            "future_minus_local_cos":future_cos-local_cos,
            "local_future_gradient_cos":disagreement,
            "future_predicted_CE_gain_sum":-(gf*delta).sum(-1),
            "actual_current_to_next_CE_gain":losses[k]-losses[k+1],
            "head_visible_update_energy":head_visible.float(),
            "future_gradient_headnull_norm_fraction":(
                gf_headnull.norm(dim=-1)/
                gf.double().norm(dim=-1).clamp_min(1e-12)).float(),
            "update_norm":delta.norm(dim=-1),
            "future_grad_norm":gf.norm(dim=-1),
        }
        if k in saes:
            sae,mu,std=saes[k]
            parts=active_components(sae,mu,std,h,delta,gf)
            next_state=((states[k+1]-mu)/std).float().reshape(-1,48)
            with torch.no_grad():
                next_ids=F.relu(sae.encoder(next_state)).topk(4,dim=-1)[1].reshape(
                    *h.shape[:-1],4)
                overlap=(parts["ids"].unsqueeze(-1)==next_ids.unsqueeze(-2)).any(
                    dim=-1).float().mean(-1)
            row.update({
                "sae_active_update_energy":parts["active_energy_fraction"],
                "sae_active_future_cos":parts["future_cos_active"],
                "sae_remainder_future_cos":parts["future_cos_remainder"],
                "sae_reconstruction_MSE":parts["SAE_MSE"],
                "sae_active_support_overlap":overlap,
                "sae_physical_admitted":parts["admitted"].float(),
            })
        result[str(k)]={
            "head_rank":rank,
            "rows":[{name:float(z[i].mean()) for name,z in row.items()}
                    for i in range(len(x))]
        }
    return result


def full_profiles(model,saes,x,y,chunk=8):
    out={str(k):[] for k in range(4)}
    for i in range(0,len(x),chunk):
        current=geometry_batch(model,saes,x[i:i+chunk],y[i:i+chunk])
        for k in range(4):
            out[str(k)].extend(current[str(k)]["rows"])
    for k in range(4):
        if len(out[str(k)])!=len(x):
            raise ValueError("future-gradient profile lost a heldout sequence")
    return out


def analyze_profiles(profiles,seed_id,smoke=False):
    report={}
    positive=0
    for k in range(4):
        rows=profiles[str(k)]
        fields=list(rows[0])
        means={f:sum(r[f] for r in rows)/len(rows) for f in fields}
        ci={}
        for key in ("future_cos","local_cos","future_minus_local_cos"):
            data=[r[key] for r in rows]
            ci[key]=bootstrap(data,seed(42044,seed_id,k,key),
                              draws=100 if smoke else BOOT)
        if ci["future_cos"][0]>0:
            positive+=1
        report[str(k)]={
            "means":means,
            "bootstrap95":ci,
            "n_whole_sequences":len(rows),
            "per_sequence":rows,
        }
    return report,positive


def zero_edit_and_causal(model,saes,x,y,seed_id,smoke=False):
    """Native 4-k recurrence under equal physical update radius.

    Per-sequence outcome: paired final GLOBAL CE effects, plus goal CE/margin.
    Every physical edit direction is determined from source states without y.
    """
    model.eval()
    source_count=min(8 if smoke else CAUSAL_SAMPLES,len(x))
    basis,rank=head_rowspace(model)
    report={}
    for k in DEPTHS:
        sae,mu,std=saes[k]
        cases=[]
        for i in range(source_count):
            xb=x[i:i+1];yb=y[i:i+1]
            # Only the model activation and SAE readout enter edit design.
            with torch.no_grad():
                states=model.reason_states(xb)
            current=states[k][0]
            update=(states[k+1]-states[k])[0]
            part=active_components(sae,mu,std,current,update)
            headnull=update.double()-(update.double()@basis)@basis.T
            frame=[]
            plans=[]
            for j,tok in enumerate(GOALS):
                radius=float(.25*update[tok].norm())
                active=part["active"][tok]
                rem=part["remainder"][tok]
                hn=headnull[tok].float()
                v=update[tok]
                eligible=bool(part["admitted"][tok]) and radius>1e-8 and float(hn.norm())>1e-8
                case={"position":tok,"goal_index":j,"admitted":eligible,
                      "dose":radius,"arms":{}}
                if not eligible:
                    case["reason"]="degenerate_source_sae_or_headnull_projection"
                    plans.append(case)
                    continue
                generator=torch.Generator().manual_seed(
                    seed("random_physical",seed_id,k,i,j))
                random_vector=torch.randn(48,generator=generator,dtype=torch.float32)
                names={"full":v,"sae_active":active,"sae_remainder":rem,
                       "head_null":hn,"random":random_vector}
                edits={name:t*radius/float(t.norm()) for name,t in names.items()}
                if any(abs(float(z.norm())-radius)>1e-5 for z in edits.values()):
                    raise ArithmeticError("source-control physical intervention doses differ")
                case["arms"]={name:{"dose":float(edit.norm())} for name,edit in edits.items()}
                plans.append(case)
                for name,edit in edits.items():
                    physical=current.clone()
                    physical[tok]+=edit
                    frame.append((j,tok,name,physical,edit))
            # After editing directions and admission are frozen, evaluate the
            # label-conditioned TRUE terminal gradient solely for diagnostics.
            _,future_gradients=terminal_gradients(model,xb,yb)
            g_future=future_gradients[k][0]
            # Always include clean source state for paired global CE.
            source_batch=[current]+[row[3] for row in frame]
            h=torch.stack(source_batch)
            with torch.no_grad():
                for _ in range(4-k):
                    h=model.block(h)
                logits=model.head(h)
                losses=per_token_ce(logits,yb.expand(len(h),-1))
            clean_all=losses[0].mean()
            clean_logits=logits[0]
            for idx,(j,tok,name,physical,edit) in enumerate(frame,1):
                truth=int(yb[0,tok])
                rival=clean_logits[tok].clone()
                rival[truth]=-float("inf")
                competitor=int(rival.argmax())
                z=logits[idx,tok]
                margin=float(z[truth]-z[competitor])
                clean_margin=float(clean_logits[tok,truth]-clean_logits[tok,competitor])
                entry=plans[j]["arms"][name]
                plans[j]["truth_digit"]=truth
                plans[j]["frozen_clean_rival"]=competitor
                entry.update({
                    "native_goal_full_logits":logits[idx,tok].tolist(),
                    "clean_goal_full_logits":clean_logits[tok].tolist(),
                    "native_global_mean_CE":float(losses[idx].mean()),
                    "clean_global_mean_CE":float(clean_all),
                    "native_goal_CE":float(losses[idx,tok]),
                    "clean_goal_CE":float(losses[0,tok]),
                    "linear_terminal_global_CE_change":float(
                        (g_future[tok].double()@edit.double())/81),
                    "global_CE_change":float(losses[idx].mean()-clean_all),
                    "goal_CE_change":float(losses[idx,tok]-losses[0,tok]),
                    "goal_margin_change":margin-clean_margin,
                    "edit_physical_l2":float(edit.norm()),
                })
            cases.append({"test_sequence_index":i,"goals":plans})
        report[str(k)]=cases
    return report


def causal_stats(cases,seed_id,smoke=False):
    result={}
    n_iter=100 if smoke else BOOT
    for k in DEPTHS:
        original=cases[str(k)]
        rowlist=[]
        for row in original:
            admitted=[g for g in row["goals"] if g["admitted"]]
            contrasts={}
            for a,b in (("sae_active","sae_remainder"),("head_null","random")):
                contrasts[f"{a}_minus_{b}"]=(
                    sum(g["arms"][a]["global_CE_change"]-
                        g["arms"][b]["global_CE_change"] for g in admitted)/len(admitted)
                    if admitted else 0.)
            rowlist.append({
                "eligible_goals":len(admitted),
                "original_goals":len(row["goals"]),
                "contrasts":contrasts,
                "arms_mean":{name:(
                    sum(g["arms"][name]["global_CE_change"] for g in admitted)/len(admitted)
                    if admitted else 0.)
                    for name in ARMS},
            })
        n_admitted=sum(z["eligible_goals"] for z in rowlist)
        means={}
        for key in rowlist[0]["contrasts"]:
            data=[r["contrasts"][key] for r in rowlist]
            interval=bootstrap(data,seed("causal",seed_id,k,key),draws=n_iter)
            means[key]={"mean":sum(data)/len(data),"bootstrap95":interval,
                        "signed_classification":"POSITIVE" if interval[0]>0 else
                        "NEGATIVE" if interval[1]<0 else "INCONCLUSIVE"}
        result[str(k)]={
            "n_source_sequences":len(rowlist),
            "n_original_goals":len(rowlist)*len(GOALS),
            "n_admitted_goal_comparisons":n_admitted,
            "contrasts":means,
            "source_sequence_rows":rowlist,
            "raw_cases":original,
        }
    return result


def run(bundle_path,meta_path,out_path,smoke=False):
    torch.set_num_threads(2)
    meta,bundle=load_frozen(bundle_path,meta_path)
    if meta["mode"]!=("smoke" if smoke else "full"):
        raise ValueError("source state bundle mode does not match evaluation")
    x,y,manifest=fixed_new_test(smoke)
    seeds=meta["teacher_init_seeds"]
    data={}
    for seed_id in seeds:
        model=make_model(bundle,seed_id)
        saes={k:make_sae(bundle,seed_id,k) for k in DEPTHS}
        profiles=full_profiles(model,saes,x,y)
        analysis,positive=analyze_profiles(profiles,seed_id,smoke)
        causal=zero_edit_and_causal(model,saes,x,y,seed_id,smoke)
        summary=causal_stats(causal,seed_id,smoke)
        old_good=meta["teacher_receipts"][str(seed_id)]["original_source_reproduction_pass"]
        passed=bool(old_good and positive>=2)
        data[str(seed_id)]={
            "gradient_by_loop":analysis,
            "positive_future_gradient_depths":positive,
            "old_model_source_parity_pass":old_good,
            "registered_seed_positive":passed,
            "native_causal":summary,
        }
        print("LGA02_EVALUATED",seed_id,"future_cosine",[
            round(analysis[str(k)]["means"]["future_cos"],4) for k in range(4)],
            "local_cosine",[
            round(analysis[str(k)]["means"]["local_cos"],4) for k in range(4)],
            "SAE_active_energy",[
            round(analysis[str(k)]["means"].get("sae_active_update_energy",0),4)
            for k in (1,2)],"passed",passed,flush=True)
    deep_summary=None
    if not smoke:
        deep=make_model(bundle,103,deep=True)
        dep=full_profiles(deep,{},x,y)
        deep_summary={str(k):{
            "mean_future_cos":sum(q["future_cos"] for q in dep[str(k)])/len(x),
            "mean_local_cos":sum(q["local_cos"] for q in dep[str(k)])/len(x),
        } for k in range(4)}
        print("LGA02_DEEP103",deep_summary,flush=True)
    passes=sum(data[str(s)]["registered_seed_positive"] for s in seeds)
    classification=("PILOT_ONLY" if smoke else
                    "POSITIVE" if passes>=2 else "FAIL_OR_NOT_ADMITTED")
    output={
        "schema":SCHEMA,"mode":"smoke" if smoke else "full",
        "protocol_sha256":protocol_sha(),
        "source_training_bundle_sha256":digest(Path(bundle_path).read_bytes()),
        "source_training_manifest_sha256":digest(Path(meta_path).read_bytes()),
        "fresh_test":manifest,
        "primary_seeds":seeds,"native_remaining_loops":list(DEPTHS),
        "registered_primary_status":classification,
        "n_primary_seed_passes":passes,
        "new_test_labels_used_only_for_diagnostics":True,
        "results":data,"deep_020_seed103_secondary":deep_summary,
        "limits":"Local and terminal gradients are evaluation-label dependent; SAE dictionaries are train-only. Physical native edits are outcome-blind matched-dose interventions."
    }
    out_path=Path(out_path);out_path.parent.mkdir(parents=True,exist_ok=True)
    out_path.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    print("LGA02_TEST_COMPLETE",classification,"passing",passes,flush=True)
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--meta",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    run(a.bundle,a.meta,a.out,smoke=a.smoke)


if __name__=="__main__":
    main()
